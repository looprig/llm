package snowflake_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/llm"
	snowflake "github.com/looprig/llm/providers/snowflake-cortex"
)

func TestMaxTokensUsesSnowflakeCompletionField(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("request JSON = %v", err)
		} else {
			var maxTokens int
			if err := json.Unmarshal(body["max_completion_tokens"], &maxTokens); err != nil || maxTokens != 42 {
				t.Errorf("max_completion_tokens = %d, err=%v, want 42", maxTokens, err)
			}
			if _, ok := body["max_tokens"]; ok {
				t.Error("request contains OpenAI max_tokens field")
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderSnowflakeCortex),
		model.APIFormatOpenAI,
		server.URL,
		"model",
		model.WithSampling(model.Sampling{MaxTokens: intPtr(42)}),
	)
	client, err := snowflake.New(selected, auth.APIKey("snowflake-token"), snowflake.WithAccount("account"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestConversationCompleteBecomesEmptyAssistantStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Errorf("request = %s %s, want POST /chat/completions", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer snowflake-token" {
			t.Errorf("Authorization = %q, want bearer token", r.Header.Get("Authorization"))
		}
		http.Error(w, `{"error":{"message":"conversation complete"}}`, http.StatusBadRequest)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderSnowflakeCortex), model.APIFormatOpenAI, server.URL, "model")
	client, err := snowflake.New(selected, auth.APIKey("snowflake-token"), snowflake.WithAccount("org-account"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want empty stop response", err)
	}
	if response == nil || response.Message == nil || response.Message.Role != content.RoleAssistant || response.FinishReason != stream.FinishReasonStop {
		t.Fatalf("response = %#v, want empty assistant stop", response)
	}
}

func TestConversationCompleteStreamBecomesCleanEmptyStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"message":"conversation complete"}`, http.StatusBadRequest)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderSnowflakeCortex), model.APIFormatOpenAI, server.URL, "model")
	client, err := snowflake.New(selected, auth.APIKey("snowflake-token"), snowflake.WithAccount("org-account"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v, want empty stop stream", err)
	}
	defer func() { _ = reader.Close() }()
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Stream.Next() error = %v, want EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Model != "model" || result.FinishReason != stream.FinishReasonStop {
		t.Fatalf("stream result = %#v, ok=%v, want model/stop", result, ok)
	}
}

func TestEmptyAssistantRoleIsNormalizedThroughClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("request JSON = %v", err)
		}
		if string(body["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"role\":\"\",\"content\":\"stream\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"model\":\"model\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"","content":"json"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderSnowflakeCortex), model.APIFormatOpenAI, server.URL, "model")
	client, err := snowflake.New(selected, auth.APIKey("snowflake-token"), snowflake.WithAccount("org-account"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Message == nil || response.Message.Role != content.RoleAssistant {
		t.Fatalf("response message = %#v, want assistant role", response.Message)
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
	if text, ok := chunk.(*content.TextChunk); !ok || text.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want stream text", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want EOF", err)
	}
}

func intPtr(value int) *int { return &value }
