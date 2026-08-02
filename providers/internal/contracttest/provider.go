// Package contracttest contains deterministic provider-package HTTP contracts.
// It is internal test infrastructure, not a public runtime API.
package contracttest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
)

type Constructor func(model.Model, auth.APIKey) (inference.Client, error)

// OpenAI verifies the common JSON/SSE, auth, usage, finish, and route contract
// for a provider using the OpenAI Chat Completions dialect.
func OpenAI(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	openAI(t, provider, key, "Authorization", "Bearer "+string(key), construct)
}

// OpenAIWithHeader is the same contract for providers such as Azure that use a
// documented API-key header instead of Authorization: Bearer.
func OpenAIWithHeader(t *testing.T, provider llm.Provider, key auth.APIKey, header string, construct Constructor) {
	openAI(t, provider, key, header, string(key), construct)
}

func openAI(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		if key == "" {
			if got := r.Header.Get(authHeader); got != "" {
				t.Errorf("%s = %q, want no auth header", authHeader, got)
			}
		} else if got := r.Header.Get(authHeader); got != authValue {
			t.Errorf("%s = %q, want %s", authHeader, got, authValue)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"stream\"}}]}\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"json"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model")
	client, err := construct(selected, key)
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
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
}

// Anthropic verifies native Messages JSON/SSE, x-api-key authentication, usage,
// and finish normalization for a provider that exposes Anthropic semantics.
func Anthropic(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != string(key) {
			t.Errorf("x-api-key = %q, want %q", got, key)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q, want 2023-06-01", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"model\":\"model\",\"usage\":{\"input_tokens\":4}}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"json"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":6}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatAnthropic, srv.URL, "model")
	client, err := construct(selected, key)
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
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Usage == nil || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("stream result = %+v ok=%v, want terminal usage", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
}
