package chutes

import (
	"bytes"
	"context"
	"crypto/mlkem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/inference/codec/openaiapi"
	stream "github.com/looprig/inference/stream"

	"github.com/looprig/inference/failure"
)

// wrappedEOFBody serves data and then reports end of input as a WRAPPED io.EOF,
// which is what any reader that adds framing context on the way out produces.
// trackedStreamBody cannot express this: bytes.Reader always ends with a bare
// io.EOF, which is exactly why the equality comparison this test targets looked
// correct for so long.
type wrappedEOFBody struct{ reader *bytes.Reader }

func (b *wrappedEOFBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if errors.Is(err, io.EOF) {
		return n, fmt.Errorf("framed body: %w", io.EOF)
	}
	return n, err
}

func (*wrappedEOFBody) Close() error { return nil }

// TestPumpTreatsAWrappedEOFAsAnUnterminatedFinish pins the io.EOF-comparison
// defect in pump: `err == io.EOF` classified a wrapped end of input as a
// transport FAULT rather than an end of stream, so the completed generation was
// discarded as a *failure.NetworkError instead of being relayed to the terminal
// gate that judges it.
//
// bufio.Scanner is what makes the branch reachable at all: Scanner.Err()
// suppresses only a BARE io.EOF (it compares with ==), so a wrapped one is
// surfaced as a scan error. And sseEventReader.next checks that error BEFORE
// flushing a pending partial event, so the loss is real and not merely a
// mislabelled error: the final event is dropped outright.
//
// The stream below therefore ends with an UNTERMINATED final event — no
// trailing blank line — and that event is the one carrying finish_reason. Under
// a bare EOF the reader flushes it and the stream completes; under a wrapped
// EOF it was discarded and a finished turn was reported as a network failure.
// The bare-EOF subtest is the control: identical bytes, unwrapped terminal.
func TestPumpTreatsAWrappedEOFAsAnUnterminatedFinish(t *testing.T) {
	t.Parallel()

	contentChunk := `{"model":"test-model","choices":[{"index":0,"delta":{"content":"hello"}}]}`
	finishChunk := `{"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`

	for _, tc := range []struct {
		name    string
		wrapEOF bool
	}{
		{"bare io.EOF (control)", false},
		{"wrapped io.EOF", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// No plaintext [DONE]: the sealed finish_reason is the only terminal,
			// so how the body's end of input is classified is the whole test.
			respDK, sse := sealedChutesStream(t, []string{contentChunk, finishChunk}, false)
			sse = strings.TrimRight(sse, "\n")

			var body io.ReadCloser
			if tc.wrapEOF {
				body = &wrappedEOFBody{reader: bytes.NewReader([]byte(sse))}
			} else {
				body = io.NopCloser(bytes.NewReader([]byte(sse)))
			}

			hc := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       body,
				}, nil
			})}
			client := New("https://api.example.test", "test-key", WithHTTPClient(hc))

			enclaveKey, err := mlkem.GenerateKey768()
			if err != nil {
				t.Fatalf("GenerateKey768(enclave) error = %v", err)
			}
			session := &attestedSession{
				key:        enclaveKey.EncapsulationKey().Bytes(),
				instanceID: "instance-1",
				nonces:     []string{"nonce-1"},
			}

			reader, err := client.chatStream(context.Background(), "chute-1", session, []byte(`{"stream":true}`), respDK)
			if err != nil {
				t.Fatalf("chatStream() error = %v", err)
			}
			defer func() { _ = reader.Close() }()

			var terminalErr error
			for {
				if _, terminalErr = reader.Next(); terminalErr != nil {
					break
				}
			}

			var networkErr *failure.NetworkError
			if errors.As(terminalErr, &networkErr) {
				t.Fatalf("Next() error = %v; a clean finish was reported as a transport fault", terminalErr)
			}
			if !errors.Is(terminalErr, io.EOF) {
				t.Fatalf("Next() error = %v, want io.EOF", terminalErr)
			}
			result, ok := reader.Result()
			if !ok {
				t.Fatal("Result() = unavailable; the completed generation was discarded")
			}
			if result.FinishReason != stream.FinishReasonStop {
				t.Errorf("Result().FinishReason = %q, want %q", result.FinishReason, stream.FinishReasonStop)
			}
		})
	}
}

// TestPumpClassifiesAWrappedEOFAsAnEndNotAFault is a regression guard on the
// other end of the same stream, and it comes with a MEASURED caveat: it passes
// under both the old `err == io.EOF` and the new errors.Is, so it does not by
// itself justify that change.
//
// The measurement, run rather than reasoned: pump's misclassification of a
// wrapped end of input as *failure.NetworkError is not observable from outside
// this package in either direction. When the stream carried a finish_reason the
// shared decoder terminates on it and never reads the pipe's terminal at all;
// when it did not, openaiapi's own terminal gate raises *openaiapi.
// StreamDecodeError and that supersedes whatever the pipe was closed with. So
// the externally visible damage from the equality comparison was entirely the
// event ssereader.next dropped, which the sibling test above covers.
//
// The comparison in pump is corrected anyway, as latent correctness: nothing
// guarantees the gate keeps superseding the pipe error, and "the bug is
// currently invisible" is not a reason to keep it. What this test pins is the
// property that must hold either way — a body that ended normally is never
// reported as a transport fault.
func TestPumpClassifiesAWrappedEOFAsAnEndNotAFault(t *testing.T) {
	t.Parallel()

	contentChunk := `{"model":"test-model","choices":[{"index":0,"delta":{"content":"hello"}}]}`
	respDK, sse := sealedChutesStream(t, []string{contentChunk}, false)

	hc := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       &wrappedEOFBody{reader: bytes.NewReader([]byte(sse))},
		}, nil
	})}
	client := New("https://api.example.test", "test-key", WithHTTPClient(hc))

	enclaveKey, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatalf("GenerateKey768(enclave) error = %v", err)
	}
	session := &attestedSession{key: enclaveKey.EncapsulationKey().Bytes(), instanceID: "instance-1", nonces: []string{"nonce-1"}}

	reader, err := client.chatStream(context.Background(), "chute-1", session, []byte(`{"stream":true}`), respDK)
	if err != nil {
		t.Fatalf("chatStream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	var terminalErr error
	for {
		if _, terminalErr = reader.Next(); terminalErr != nil {
			break
		}
	}

	var networkErr *failure.NetworkError
	if errors.As(terminalErr, &networkErr) {
		t.Fatalf("Next() error = %v; a body that ended normally was reported as a transport fault", terminalErr)
	}
	var decodeErr *openaiapi.StreamDecodeError
	if !errors.As(terminalErr, &decodeErr) {
		t.Fatalf("Next() error = %T (%v), want *openaiapi.StreamDecodeError", terminalErr, terminalErr)
	}
	if _, ok := reader.Result(); ok {
		t.Error("Result() after an unterminated stream = present, want unavailable")
	}
}
