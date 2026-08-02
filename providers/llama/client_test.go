package llama_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/llama"
)

func TestNewUsesHostedLlamaChatEndpoint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/chat/completions"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("Authorization"), "Bearer llama-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"llama","model":"Llama-4-Scout","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderLlama), model.APIFormatOpenAI, server.URL, "Llama-4-Scout")
	client, err := llama.New(selected, auth.APIKey("llama-key"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestNewUsesLLAMAAPIKEYWhenExplicitKeyIsEmpty(t *testing.T) {
	t.Setenv("LLAMA_API_KEY", "env-llama-key")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), "Bearer env-llama-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"llama","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderLlama), model.APIFormatOpenAI, server.URL, "model")
	client, err := llama.New(selected, "")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestNewUsesCanonicalHostedDefault(t *testing.T) {
	selected := model.CustomModel(model.ProviderName(llm.ProviderLlama), model.APIFormatOpenAI, "", "Llama-4-Scout")
	client, err := llama.New(selected, auth.APIKey("llama-key"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if client == nil {
		t.Fatal("New() = nil, want client")
	}
	if got, want := llama.DefaultBaseURL, "https://api.llama.com/compat/v1"; got != want {
		t.Fatalf("DefaultBaseURL = %q, want %q", got, want)
	}
}

func TestNewRejectsLocalLlamaCPPModel(t *testing.T) {
	selected := model.CustomModel(model.ProviderName(llm.ProviderLlamaCPP), model.APIFormatOpenAI, "http://127.0.0.1:8080/v1", "qwen")
	client, err := llama.New(selected, auth.APIKey("llama-key"))
	if client != nil {
		t.Fatalf("New() returned %T for llama.cpp model", client)
	}
	if err == nil {
		t.Fatal("New() error = nil, want provider mismatch")
	}
}
