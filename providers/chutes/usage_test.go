package chutes

import (
	"bytes"
	"context"
	"crypto/mlkem"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/llm/e2e"
)

const streamLifecycleTimeout = 2 * time.Second

func TestChatStreamUsageResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		usage      string
		wantUsage  *content.Usage
		wantErr    bool
		earlyClose bool
	}{
		{
			name:  "clean encrypted stream preserves normalized terminal usage",
			usage: `{"usage":{"prompt_tokens":11,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":4}}}`,
			wantUsage: &content.Usage{
				InputTokens:         6,
				OutputTokens:        7,
				CacheReadTokens:     3,
				CacheCreationTokens: 2,
				ReasoningTokens:     4,
			},
		},
		{
			name:    "malformed terminal usage leaves result unavailable",
			usage:   `{"usage":{"prompt_tokens":-1,"completion_tokens":7}}`,
			wantErr: true,
		},
		{
			name:      "present-zero terminal usage remains distinguishable",
			usage:     `{"usage":{}}`,
			wantUsage: &content.Usage{},
		},
		{
			name:       "caller early close stops pump and closes upstream body",
			earlyClose: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			respDK, initCT, frame := encryptedStreamFixture(t)
			prefix := chutesEvent(t, map[string]string{"e2e_init": base64.StdEncoding.EncodeToString(initCT)}) +
				chutesEvent(t, map[string]string{"e2e": base64.StdEncoding.EncodeToString(frame)})
			if !tt.earlyClose {
				prefix += "data: " + tt.usage + "\n\ndata: [DONE]\n\n"
			}
			body := newTrackedStreamBody(prefix, tt.earlyClose)
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
			if _, ok := reader.Result(); ok {
				t.Fatal("Result() before EOF = present, want unavailable")
			}

			if tt.earlyClose {
				if _, err := reader.Next(); err != nil {
					t.Fatalf("Next() before early Close error = %v", err)
				}
				if err := reader.Close(); err != nil {
					t.Fatalf("StreamReader.Close() error = %v", err)
				}
				assertPumpLifecycle(t, done, &doneCalls, body)
				return
			}

			var terminalErr error
			for {
				_, terminalErr = reader.Next()
				if terminalErr == nil {
					continue
				}
				break
			}
			if err := reader.Close(); err != nil {
				t.Fatalf("StreamReader.Close() error = %v", err)
			}
			assertPumpLifecycle(t, done, &doneCalls, body)

			if tt.wantErr {
				if errors.Is(terminalErr, io.EOF) {
					t.Fatal("Next() error = EOF, want malformed-usage error")
				}
				assertUsageNormalization(t, terminalErr, inference.UsageNormalizationFieldInputTokens)
				if _, ok := reader.Result(); ok {
					t.Fatal("Result() after error = present, want unavailable")
				}
				return
			}
			if !errors.Is(terminalErr, io.EOF) {
				t.Fatalf("Next() error = %v, want EOF", terminalErr)
			}
			result, ok := reader.Result()
			if !ok {
				t.Fatal("Result() after EOF = unavailable, want usage")
			}
			if result.Usage == nil || *result.Usage != *tt.wantUsage {
				t.Errorf("Result().Usage = %+v, want %+v", result.Usage, tt.wantUsage)
			}
		})
	}
}

func encryptedStreamFixture(t *testing.T) (*mlkem.DecapsulationKey768, []byte, []byte) {
	t.Helper()
	respDK, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatalf("GenerateKey768(response) error = %v", err)
	}
	streamShared, initCT := respDK.EncapsulationKey().Encapsulate()
	streamKey, err := e2e.DeriveKey(streamShared, initCT, []byte("e2e-stream-v1"))
	if err != nil {
		t.Fatalf("DeriveKey() error = %v", err)
	}
	plaintext := []byte(`data: {"model":"test-model","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}`)
	frame, err := e2e.SealFrame(streamKey, plaintext)
	if err != nil {
		t.Fatalf("SealFrame() error = %v", err)
	}
	return respDK, initCT, frame
}

func chutesEvent(t *testing.T, value map[string]string) string {
	t.Helper()
	payload, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return "data: " + string(payload) + "\n\n"
}

func assertUsageNormalization(t *testing.T, err error, field inference.UsageNormalizationField) {
	t.Helper()
	var usageErr *inference.UsageNormalizationError
	if !errors.As(err, &usageErr) {
		t.Fatalf("error = %T (%v), want *inference.UsageNormalizationError", err, err)
	}
	if usageErr.Field != field || usageErr.Reason != inference.UsageNormalizationReasonNegative {
		t.Errorf("UsageNormalizationError = {Field:%q Reason:%q}, want {Field:%q Reason:%q}",
			usageErr.Field, usageErr.Reason, field, inference.UsageNormalizationReasonNegative)
	}
}

func assertPumpLifecycle(t *testing.T, done <-chan struct{}, doneCalls *atomic.Int32, body *trackedStreamBody) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), streamLifecycleTimeout)
	defer cancel()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("stream pump did not exit within %s", streamLifecycleTimeout)
	}
	if got := doneCalls.Load(); got != 1 {
		t.Errorf("streamDone calls = %d, want 1", got)
	}
	if got := body.closeCalls.Load(); got != 1 {
		t.Errorf("upstream body Close calls = %d, want 1", got)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type trackedStreamBody struct {
	reader     *bytes.Reader
	block      bool
	closed     chan struct{}
	closeOnce  sync.Once
	closeCalls atomic.Int32
}

func newTrackedStreamBody(data string, block bool) *trackedStreamBody {
	return &trackedStreamBody{reader: bytes.NewReader([]byte(data)), block: block, closed: make(chan struct{})}
}

func (b *trackedStreamBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if n > 0 || !b.block {
		return n, err
	}
	<-b.closed
	return 0, io.ErrClosedPipe
}

func (b *trackedStreamBody) Close() error {
	b.closeCalls.Add(1)
	b.closeOnce.Do(func() { close(b.closed) })
	return nil
}
