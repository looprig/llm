package snowflake_test

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

func intPtr(value int) *int { return &value }
