package chutes

import (
	"context"
	"crypto/mlkem"
	"encoding/base64"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/looprig/inference/codec/openaiapi"
	stream "github.com/looprig/inference/stream"

	"github.com/looprig/llm/e2e"
)

// TestChatStreamTerminalAuthority pins the one thing the pump must never get
// wrong: a stream that carried a real end-of-generation signal completes, and a
// stream whose body simply stopped is reported as truncated.
//
// The Chutes transport tunnels the upstream OpenAI SSE bytes verbatim inside
// sealed e2e frames (pump writes the decrypted `data: {...}` straight to the
// pipe), so the upstream's own finish_reason reaches openaiapi's terminal gate
// through the tunnel. A bare io.EOF on the HTTP body therefore carries no
// authority of its own: it must be relayed as-is and judged by that gate, never
// papered over with a manufactured [DONE].
func TestChatStreamTerminalAuthority(t *testing.T) {
	t.Parallel()

	contentChunk := `{"model":"test-model","choices":[{"index":0,"delta":{"content":"hello"}}]}`
	finishChunk := `{"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`

	tests := []struct {
		name string
		// chunks are the upstream OpenAI SSE payloads sealed into e2e frames.
		chunks []string
		// plaintextDone appends the gateway's unsealed `data: [DONE]` terminal.
		plaintextDone bool
		wantTruncated bool
		wantFinish    stream.FinishReason
	}{
		{
			name:          "sealed finish_reason completes without a plaintext DONE",
			chunks:        []string{contentChunk, finishChunk},
			plaintextDone: false,
			wantFinish:    stream.FinishReasonStop,
		},
		{
			name:          "plaintext DONE completes a stream with no finish_reason",
			chunks:        []string{contentChunk},
			plaintextDone: true,
		},
		{
			name:          "sealed finish_reason and plaintext DONE complete",
			chunks:        []string{contentChunk, finishChunk},
			plaintextDone: true,
			wantFinish:    stream.FinishReasonStop,
		},
		{
			name:          "body ending mid-generation is truncated, not complete",
			chunks:        []string{contentChunk},
			plaintextDone: false,
			wantTruncated: true,
		},
		{
			name:          "body ending before any frame is truncated, not complete",
			chunks:        nil,
			plaintextDone: false,
			wantTruncated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			respDK, sse := sealedChutesStream(t, tt.chunks, tt.plaintextDone)
			body := newTrackedStreamBody(sse, false)
			t.Cleanup(func() {
				if body.closeCalls.Load() == 0 {
					_ = body.Close()
				}
			})

			hc := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       body,
				}, nil
			})}
			client := New("https://api.example.test", "test-key", WithHTTPClient(hc))
			done := make(chan struct{}, 1)
			var doneCalls atomic.Int32
			client.streamDone = func() {
				doneCalls.Add(1)
				done <- struct{}{}
			}

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

			var terminalErr error
			for {
				if _, terminalErr = reader.Next(); terminalErr != nil {
					break
				}
			}
			if err := reader.Close(); err != nil {
				t.Fatalf("StreamReader.Close() error = %v", err)
			}
			assertPumpLifecycle(t, done, &doneCalls, body)

			if tt.wantTruncated {
				if errors.Is(terminalErr, io.EOF) {
					t.Fatal("Next() error = EOF, want a truncation error")
				}
				var decodeErr *openaiapi.StreamDecodeError
				if !errors.As(terminalErr, &decodeErr) {
					t.Fatalf("Next() error = %T (%v), want *openaiapi.StreamDecodeError", terminalErr, terminalErr)
				}
				if _, ok := reader.Result(); ok {
					t.Error("Result() after truncation = present, want unavailable")
				}
				return
			}

			if !errors.Is(terminalErr, io.EOF) {
				t.Fatalf("Next() error = %v, want EOF", terminalErr)
			}
			result, ok := reader.Result()
			if !ok {
				t.Fatal("Result() after clean EOF = unavailable, want present")
			}
			if result.FinishReason != tt.wantFinish {
				t.Errorf("Result().FinishReason = %q, want %q", result.FinishReason, tt.wantFinish)
			}
		})
	}
}

// sealedChutesStream builds a Chutes streaming body: the plaintext e2e_init
// event, one sealed e2e frame per upstream OpenAI SSE payload, and optionally
// the gateway's unsealed `data: [DONE]` terminal.
func sealedChutesStream(t *testing.T, chunks []string, plaintextDone bool) (*mlkem.DecapsulationKey768, string) {
	t.Helper()
	respDK, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatalf("GenerateKey768(response) error = %v", err)
	}
	shared, initCT := respDK.EncapsulationKey().Encapsulate()
	streamKey, err := e2e.DeriveKey(shared, initCT, []byte("e2e-stream-v1"))
	if err != nil {
		t.Fatalf("DeriveKey() error = %v", err)
	}

	body := chutesEvent(t, map[string]string{"e2e_init": base64.StdEncoding.EncodeToString(initCT)})
	for _, chunk := range chunks {
		frame, err := e2e.SealFrame(streamKey, []byte("data: "+chunk))
		if err != nil {
			t.Fatalf("SealFrame() error = %v", err)
		}
		body += chutesEvent(t, map[string]string{"e2e": base64.StdEncoding.EncodeToString(frame)})
	}
	if plaintextDone {
		body += "data: [DONE]\n\n"
	}
	return respDK, body
}
