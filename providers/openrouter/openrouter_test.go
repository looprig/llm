package openrouter_test

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
	"github.com/looprig/llm/providers/openrouter"
)

func TestNewInvokeAddsOpenRouterOptions(t *testing.T) {
	t.Parallel()

	requestCh := make(chan *http.Request, 1)
	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read request body: %v", err), http.StatusInternalServerError)
			return
		}
		requestCh <- r
		bodyCh <- body
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"response-id","model":"anthropic/claude-sonnet-4","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenRouter),
		model.APIFormatOpenAI,
		srv.URL+"/api/v1/",
		"anthropic/claude-sonnet-4",
	)
	client, err := openrouter.New(
		selected,
		auth.APIKey("sk-or-test"),
		openrouter.WithHTTPReferer("https://looprig.example/"),
		openrouter.WithTitle("Looprig"),
		openrouter.WithUsage(true),
		openrouter.WithReasoning(openrouter.ReasoningOptions{
			Effort:    "high",
			MaxTokens: intPtr(256),
			Exclude:   boolPtr(false),
			Enabled:   boolPtr(true),
		}),
		openrouter.WithPromptCacheKey("session-123"),
		openrouter.WithProviderRouting(openrouter.ProviderRoutingOptions{
			Order:             []string{"anthropic", "openai"},
			AllowFallbacks:    boolPtr(false),
			RequireParameters: boolPtr(true),
			DataCollection:    "deny",
			ZDR:               boolPtr(false),
		}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	resp, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke() response = %+v, want a decoded message", resp)
	}

	req := <-requestCh
	if req.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", req.Method)
	}
	if got, want := req.URL.Path, "/api/v1/chat/completions"; got != want {
		t.Errorf("path = %q, want %q", got, want)
	}
	if got, want := req.Header.Get("Authorization"), "Bearer sk-or-test"; got != want {
		t.Errorf("Authorization = %q, want %q", got, want)
	}
	if got, want := req.Header.Get("HTTP-Referer"), "https://looprig.example/"; got != want {
		t.Errorf("HTTP-Referer = %q, want %q", got, want)
	}
	if got, want := req.Header.Get("X-OpenRouter-Title"), "Looprig"; got != want {
		t.Errorf("X-OpenRouter-Title = %q, want %q", got, want)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}

	var usage struct {
		Include bool `json:"include"`
	}
	decodeField(t, body, "usage", &usage)
	if !usage.Include {
		t.Error("usage.include = false, want true")
	}

	var reasoning openrouter.ReasoningOptions
	decodeField(t, body, "reasoning", &reasoning)
	if reasoning.Effort != "high" || reasoning.MaxTokens == nil || *reasoning.MaxTokens != 256 || reasoning.Exclude == nil || *reasoning.Exclude || reasoning.Enabled == nil || !*reasoning.Enabled {
		t.Errorf("reasoning = %+v, want all configured fields preserved", reasoning)
	}

	var provider openrouter.ProviderRoutingOptions
	decodeField(t, body, "provider", &provider)
	if len(provider.Order) != 2 || provider.Order[0] != "anthropic" || provider.Order[1] != "openai" {
		t.Errorf("provider.order = %#v, want [anthropic openai]", provider.Order)
	}
	if provider.AllowFallbacks == nil || *provider.AllowFallbacks {
		t.Errorf("provider.allow_fallbacks = %v, want explicit false", provider.AllowFallbacks)
	}
	if provider.RequireParameters == nil || !*provider.RequireParameters {
		t.Errorf("provider.require_parameters = %v, want explicit true", provider.RequireParameters)
	}
	if provider.DataCollection != "deny" {
		t.Errorf("provider.data_collection = %q, want deny", provider.DataCollection)
	}
	if provider.ZDR == nil || *provider.ZDR {
		t.Errorf("provider.zdr = %v, want explicit false", provider.ZDR)
	}

	var cacheKey string
	decodeField(t, body, "prompt_cache_key", &cacheKey)
	if cacheKey != "session-123" {
		t.Errorf("prompt_cache_key = %q, want session-123", cacheKey)
	}
}

func TestNewInvokeOmitsUnsetOptions(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodyCh <- body
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"response-id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := openrouter.New(selected, "sk-or-test")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	for _, field := range []string{"usage", "reasoning", "prompt_cache_key", "provider"} {
		if _, ok := body[field]; ok {
			t.Errorf("request body contains unset OpenRouter field %q", field)
		}
	}
}

func TestNewValidation(t *testing.T) {
	t.Parallel()

	openRouterModel := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, "https://openrouter.ai/api/v1", "model")
	if client, err := openrouter.New(openRouterModel, ""); client != nil || err == nil {
		t.Fatalf("New(empty key) = (%T, %v), want typed auth error and nil client", client, err)
	} else {
		var authErr *llm.AuthRequiredError
		if !errors.As(err, &authErr) {
			t.Fatalf("New(empty key) error = %T, want *llm.AuthRequiredError", err)
		}
		if authErr.Provider != llm.ProviderOpenRouter || authErr.Kind != auth.AuthAPIKey {
			t.Errorf("AuthRequiredError = {Provider:%q Kind:%q}, want {openrouter api_key}", authErr.Provider, authErr.Kind)
		}
	}

	wrongProvider := model.CustomModel(model.ProviderName(llm.ProviderLMStudio), model.APIFormatOpenAI, "http://localhost:1234/v1", "model")
	if client, err := openrouter.New(wrongProvider, "sk-or-test"); client != nil || err == nil {
		t.Fatalf("New(wrong provider) = (%T, %v), want typed validation error and nil client", client, err)
	} else {
		var validationErr *model.ValidationError
		if !errors.As(err, &validationErr) {
			t.Fatalf("New(wrong provider) error = %T, want *model.ValidationError", err)
		}
		if validationErr.Field != "Provider" {
			t.Errorf("ValidationError.Field = %q, want Provider", validationErr.Field)
		}
	}
}

func decodeField[T any](t *testing.T, body map[string]json.RawMessage, field string, target *T) {
	t.Helper()
	raw, ok := body[field]
	if !ok {
		t.Fatalf("request body lacks %q", field)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("request body field %q: %v", field, err)
	}
}

func boolPtr(value bool) *bool { return &value }

func intPtr(value int) *int { return &value }
