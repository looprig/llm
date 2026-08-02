package compat_test

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
	"github.com/looprig/llm/providers/internal/compat"
)

func TestNewInvokeAppliesHeadersAndBodyPatch(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	headerCh := make(chan http.Header, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodyCh <- body
		headerCh <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := compat.New(selected, compat.Config{
		Authenticator: auth.Key("secret"),
		Headers:       http.Header{"X-Provider": []string{"test"}},
		PatchRequest: func(body map[string]json.RawMessage) error {
			body["reasoning_effort"] = json.RawMessage(`"high"`)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	resp, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp == nil || resp.Message == nil || len(resp.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one decoded message", resp)
	}

	headers := <-headerCh
	if got := headers.Get("Authorization"); got != "Bearer secret" {
		t.Errorf("Authorization = %q, want Bearer secret", got)
	}
	if got := headers.Get("X-Provider"); got != "test" {
		t.Errorf("X-Provider = %q, want test", got)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	if got := string(body["reasoning_effort"]); got != `"high"` {
		t.Errorf("reasoning_effort = %s, want %q", got, `"high"`)
	}
}

func TestNewStreamUsesOpenAIStreamingCodec(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := compat.New(selected, compat.Config{Authenticator: auth.Key("secret")})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	var chunks []content.Chunk
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("Stream.Next() error = %v", nextErr)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("stream chunks = %#v, want thinking and text", chunks)
	}
	if got, ok := chunks[0].(*content.ThinkingChunk); !ok || got.Thinking != "think" {
		t.Errorf("first chunk = %#v, want ThinkingChunk{think}", chunks[0])
	}
	if got, ok := chunks[1].(*content.TextChunk); !ok || got.Text != "done" {
		t.Errorf("second chunk = %#v, want TextChunk{done}", chunks[1])
	}
}

func TestNewStreamRejectsMalformedSSE(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {not-json}\n\n")
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := compat.New(selected, compat.Config{Authenticator: auth.Key("secret")})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()
	if _, err := reader.Next(); err == nil {
		t.Fatal("Stream.Next() error = nil, want malformed SSE error")
	}
}

func TestNewRejectsMissingAuthenticator(t *testing.T) {
	t.Parallel()
	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, "https://example.test/v1", "model")
	if client, err := compat.New(selected, compat.Config{}); client != nil || err == nil {
		t.Fatalf("New() = (%T, %v), want nil client and error", client, err)
	}
}

func TestNewProviderResolvesDefaultsAndOptions(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodyCh <- body
		if got := r.Header.Get("X-Test"); got != "yes" {
			t.Errorf("X-Test = %q, want yes", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, "", "model")
	client, err := compat.NewProvider(selected, "secret", compat.Definition{
		Provider:       llm.ProviderOpenRouter,
		DefaultBaseURL: srv.URL,
		DefaultPath:    "/chat/completions",
		Authentication: auth.AuthAPIKey,
	}, compat.WithHeader("X-Test", "yes"), compat.WithBodyField("reasoning_effort", "high"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	if string(body["reasoning_effort"]) != `"high"` {
		t.Errorf("reasoning_effort = %s, want %q", body["reasoning_effort"], `"high"`)
	}
}
