package llamacpp_test

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
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/llamacpp"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderLlamaCPP, auth.APIKey(""), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return llamacpp.New(selected, key)
	})
}

func TestNewUsesLocalLlamaCPPEndpointWithoutAuthentication(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/chat/completions"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want no authentication", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"local","model":"qwen","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderLlamaCPP), model.APIFormatOpenAI, server.URL, "qwen")
	client, err := llamacpp.New(selected, "")
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

func TestNewAcceptsLegacyHostedLlamaIdentityForExplicitLocalConstruction(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(model.ProviderName(llm.ProviderLlama), model.APIFormatOpenAI, "http://127.0.0.1:8080/v1", "qwen")
	if _, err := llamacpp.New(selected, ""); err != nil {
		t.Fatalf("New(legacy llama identity) error = %v", err)
	}
}
