package gmicloud_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/gmicloud"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderGMICloud, auth.APIKey("gmi-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return gmicloud.New(selected, key)
	})
}

func TestNewUsesOpenAIChatBearerAuthentication(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/chat/completions"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("Authorization"), "Bearer gmi-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
		if got := r.Header.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key = %q, want no Anthropic authentication header", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "" {
			t.Errorf("anthropic-version = %q, want no Anthropic header", got)
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("request JSON error = %v", err)
		}
		if _, ok := body["messages"]; !ok {
			t.Error("request missing OpenAI messages")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"gmi","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGMICloud), model.APIFormatOpenAI, server.URL, "model")
	client, err := gmicloud.New(selected, auth.APIKey("gmi-key"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestDocumentedOptionsAreForwarded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Organization-ID"); got != "org-123" {
			t.Errorf("X-Organization-ID = %q, want org-123", got)
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("request JSON error = %v", err)
		}
		var topK int
		if err := json.Unmarshal(body["top_k"], &topK); err != nil || topK != 32 {
			t.Errorf("top_k = %d, err=%v, want 32", topK, err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"gmi","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGMICloud), model.APIFormatOpenAI, server.URL, "model")
	client, err := gmicloud.New(selected, auth.APIKey("gmi-key"), gmicloud.WithOrganizationID("org-123"), gmicloud.WithTopK(32))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestNewRejectsAnthropicModel(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGMICloud), model.APIFormatAnthropic, "", "model")
	client, err := gmicloud.New(selected, auth.APIKey("gmi-key"))
	if client != nil {
		t.Fatalf("New() returned %T for unsupported Anthropic model", client)
	}
	if err == nil {
		t.Fatal("New() error = nil, want API-format validation error")
	}
}
