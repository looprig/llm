package gemini_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/llm/providers/gemini"
)

func TestGeminiStreamUsageResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		terminal  string
		wantUsage *content.Usage
		wantErr   bool
	}{
		{
			name:     "clean stream preserves normalized terminal usage",
			terminal: `{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"cachedContentTokenCount":3,"candidatesTokenCount":3,"thoughtsTokenCount":4,"totalTokenCount":18},"modelVersion":"gemini-2.5-flash"}`,
			wantUsage: &content.Usage{
				InputTokens:     8,
				OutputTokens:    7,
				CacheReadTokens: 3,
				ReasoningTokens: 4,
			},
		},
		{
			name:     "malformed terminal usage leaves result unavailable",
			terminal: `{"usageMetadata":{"promptTokenCount":-1}}`,
			wantErr:  true,
		},
		{
			name:      "present-zero terminal usage remains distinguishable",
			terminal:  `{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{},"modelVersion":"gemini-2.5-flash"}`,
			wantUsage: &content.Usage{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hello\"}]}}]}\n\n")
				fmt.Fprint(w, "data: "+tt.terminal+"\n\n")
			}))
			defer srv.Close()

			client := gemini.NewWithEndpoint(testKey, srv.URL)
			reader, err := client.Stream(context.Background(), geminiRequest("gemini-2.5-flash"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
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
					assertGeminiUsageNormalization(t, nextErr)
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

func assertGeminiUsageNormalization(t *testing.T, err error) {
	t.Helper()
	var usageErr *inference.UsageNormalizationError
	if !errors.As(err, &usageErr) {
		t.Fatalf("error = %T (%v), want *inference.UsageNormalizationError", err, err)
	}
	if usageErr.Field != inference.UsageNormalizationFieldInputTokens || usageErr.Reason != inference.UsageNormalizationReasonNegative {
		t.Errorf("UsageNormalizationError = {Field:%q Reason:%q}, want {Field:%q Reason:%q}",
			usageErr.Field, usageErr.Reason, inference.UsageNormalizationFieldInputTokens, inference.UsageNormalizationReasonNegative)
	}
}
