package chutes

import (
	"context"
	"crypto/mlkem"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/llm/e2e"
)

func TestChatStreamUsageResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		usage     string
		wantUsage *content.Usage
		wantErr   bool
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
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

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				writeChutesEvent(t, w, map[string]string{"e2e_init": base64.StdEncoding.EncodeToString(initCT)})
				writeChutesEvent(t, w, map[string]string{"e2e": base64.StdEncoding.EncodeToString(frame)})
				if _, err := io.WriteString(w, "data: "+tt.usage+"\n\ndata: [DONE]\n\n"); err != nil {
					t.Errorf("write stream terminal events: %v", err)
				}
			}))
			defer srv.Close()

			enclaveKey, err := mlkem.GenerateKey768()
			if err != nil {
				t.Fatalf("GenerateKey768(enclave) error = %v", err)
			}
			client := New(srv.URL, "test-key", WithHTTPClient(srv.Client()))
			session := &attestedSession{
				key:        enclaveKey.EncapsulationKey().Bytes(),
				instanceID: "instance-1",
				nonces:     []string{"nonce-1"},
			}
			reader, err := client.chatStream(context.Background(), "chute-1", session, []byte(`{"stream":true}`), respDK)
			if err != nil {
				t.Fatalf("chatStream() error = %v", err)
			}
			t.Cleanup(func() {
				if err := reader.Close(); err != nil {
					t.Errorf("StreamReader.Close() error = %v", err)
				}
			})
			if _, ok := reader.Result(); ok {
				t.Fatal("Result() before EOF = present, want unavailable")
			}

			for {
				_, nextErr := reader.Next()
				if nextErr == nil {
					continue
				}
				if tt.wantErr {
					if errors.Is(nextErr, io.EOF) {
						t.Fatal("Next() error = EOF, want malformed-usage error")
					}
					if _, ok := reader.Result(); ok {
						t.Fatal("Result() after error = present, want unavailable")
					}
					return
				}
				if !errors.Is(nextErr, io.EOF) {
					t.Fatalf("Next() error = %v, want EOF", nextErr)
				}
				break
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

func writeChutesEvent(t *testing.T, w io.Writer, value map[string]string) {
	t.Helper()
	payload, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if _, err := w.Write(append(append([]byte("data: "), payload...), []byte("\n\n")...)); err != nil {
		t.Errorf("write SSE event: %v", err)
	}
}
