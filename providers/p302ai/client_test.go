package p302ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/p302ai"
)

func TestNewInvokeAndStream(t *testing.T) {
	t.Parallel()

	requestCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		if got := r.Header.Get("X-Provider"); got != "302ai" {
			t.Errorf("X-Provider = %q, want 302ai", got)
		}
		body, _ := io.ReadAll(r.Body)
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON error = %v", err)
		}
		streaming := string(request["stream"]) == "true"
		if streaming {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"streamed\"}}]}\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.Provider302AI), model.APIFormatOpenAI, srv.URL, "model")
	client, err := p302ai.New(selected, auth.APIKey("test-key"), p302ai.WithHeader("X-Provider", "302ai"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response == nil || response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one text block", response)
	}
	if response.Usage == nil || response.Usage.InputTokens != 2 || response.Usage.OutputTokens != 3 {
		t.Fatalf("usage = %#v, want input 2/output 3", response.Usage)
	}

	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("Stream.Next() error = %v", err)
	}
	textChunk, ok := chunk.(*content.TextChunk)
	if !ok || textChunk.Text != "streamed" {
		t.Fatalf("stream chunk = %#v, want TextChunk{streamed}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	if requestCount != 2 {
		t.Fatalf("request count = %d, want 2", requestCount)
	}
}

func TestNewCounterIsExplicitlyUnsupported(t *testing.T) {
	counter, err := p302ai.NewCounter(auth.APIKey("key"))
	if counter != nil || err == nil {
		t.Fatalf("NewCounter() = (%T, %v), want nil and typed error", counter, err)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) || supportErr.Provider != llm.Provider302AI {
		t.Fatalf("NewCounter() error = %T %v, want 302ai CounterSupportError", err, err)
	}
}
