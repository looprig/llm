package azure_test

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
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/azure"
)

func TestNewInvokeUsesAzureResponsesAndAPIKey(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodyCh <- body
		if got, want := r.URL.Path, "/openai/v1/responses"; got != want {
			http.Error(w, fmt.Sprintf("path = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		if got, want := r.Header.Get("api-key"), "azure-test-key"; got != want {
			http.Error(w, fmt.Sprintf("api-key = %q, want %q", got, want), http.StatusUnauthorized)
			return
		}
		if got := r.Header.Get("Authorization"); got != "" {
			http.Error(w, fmt.Sprintf("authorization = %q, want empty", got), http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_azure_1",
			"status":"completed",
			"model":"gpt-4.1",
			"output":[
				{"type":"reasoning","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"opaque"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"type":"function_call","call_id":"call_azure_1","name":"lookup","arguments":"{\"city\":\"NYC\"}"}
			],
			"usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":2},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":3}}
		}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderAzure),
		model.APIFormatOpenAIResponses,
		srv.URL+"/openai/v1",
		"gpt-4.1",
		model.WithTools(),
		model.WithThinking(),
		model.WithStructuredOutputWithTools(),
		model.WithSampling(model.Sampling{Effort: model.EffortMedium, MaxTokens: intPtr(128)}),
	)
	client, err := azure.New(selected, auth.APIKey("azure-test-key"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	resp, err := client.Invoke(context.Background(), inference.Request{
		Model:  selected,
		System: "system instruction",
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "Look up a city",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		}},
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
			Strict: true,
		},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke() response = %+v, want decoded response", resp)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("FinishReason = %v, want tool_use", resp.FinishReason)
	}
	if got := len(resp.Message.Blocks); got != 3 {
		t.Fatalf("decoded blocks = %d, want reasoning/text/tool", got)
	}
	if thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock); !ok || thinking.Thinking != "think" {
		t.Errorf("reasoning block = %#v, want think", resp.Message.Blocks[0])
	}
	if text, ok := resp.Message.Blocks[1].(*content.TextBlock); !ok || text.Text != "answer" {
		t.Errorf("text block = %#v, want answer", resp.Message.Blocks[1])
	}
	if tool, ok := resp.Message.Blocks[2].(*content.ToolUseBlock); !ok || tool.ID != "call_azure_1" || tool.Name != "lookup" {
		t.Errorf("tool block = %#v, want lookup call", resp.Message.Blocks[2])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 10 || resp.Usage.CacheReadTokens != 2 || resp.Usage.OutputTokens != 9 || resp.Usage.ReasoningTokens != 3 {
		t.Errorf("usage = %+v, want input=10 cache=2 output=9 reasoning=3", resp.Usage)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	var modelName string
	decodeField(t, body, "model", &modelName)
	if modelName != "gpt-4.1" {
		t.Errorf("model = %q, want deployment name gpt-4.1", modelName)
	}
	if _, ok := body["messages"]; ok {
		t.Error("Responses request contains Chat Completions messages field")
	}
	if _, ok := body["tools"]; !ok {
		t.Error("Responses request missing tools")
	}
	if _, ok := body["text"]; !ok {
		t.Error("Responses request missing structured-output text configuration")
	}
}

func TestStreamDecodesAzureResponsesEventsAndUsage(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-4.1\",\"status\":\"in_progress\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-4.1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}],\"usage\":{\"input_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n")
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses, srv.URL+"/openai/v1", "gpt-4.1", model.WithThinking())
	client, err := azure.New(selected, auth.APIKey("azure-test-key"))
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
		t.Fatalf("chunks = %#v, want reasoning/text", chunks)
	}
	if got, ok := chunks[0].(*content.ThinkingChunk); !ok || got.Thinking != "think" {
		t.Errorf("chunk 0 = %#v, want thinking", chunks[0])
	}
	if got, ok := chunks[1].(*content.TextChunk); !ok || got.Text != "answer" {
		t.Errorf("chunk 1 = %#v, want text", chunks[1])
	}
	result, ok := reader.Result()
	if !ok || result.Model != "gpt-4.1" || result.Usage == nil || result.Usage.InputTokens != 6 || result.Usage.CacheReadTokens != 1 {
		t.Errorf("stream result = %+v, ok=%v, want model/usage", result, ok)
	}
}

func TestNewValidation(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "resource")
	valid := model.CustomModel(model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses, "", "gpt-4.1")
	client, err := azure.New(valid, "")
	if client != nil {
		t.Fatalf("New(empty key) returned %T", client)
	}
	var authErr *llm.AuthRequiredError
	if !errors.As(err, &authErr) || authErr.Provider != llm.ProviderAzure || authErr.Kind != auth.AuthAPIKey {
		t.Fatalf("New(empty key) error = %T %v, want Azure auth error", err, err)
	}

	if client, err := azure.New(valid, "azure-test-key"); err != nil || client == nil {
		t.Fatalf("New(valid) = %T, %v, want client", client, err)
	}

	wrong := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "https://api.openai.com/v1", "gpt-4.1")
	if client, err := azure.New(wrong, "azure-test-key"); client != nil || err == nil {
		t.Fatalf("New(wrong provider) = %T, %v, want validation error", client, err)
	}
}

func TestMalformedAndHTTPErrorResponses(t *testing.T) {
	t.Parallel()

	t.Run("malformed JSON", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, "{")
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses, srv.URL+"/openai/v1", "gpt-4.1")
		client, err := azure.New(selected, "azure-test-key")
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
		if err == nil {
			t.Fatal("Invoke() error = nil, want malformed-response error")
		}
	})

	t.Run("HTTP error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "bad request", http.StatusBadRequest)
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses, srv.URL+"/openai/v1", "gpt-4.1")
		client, err := azure.New(selected, "azure-test-key")
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("Invoke() error = %T %v, want *failure.APIError", err, err)
		}
		if apiErr.Status != http.StatusBadRequest {
			t.Errorf("APIError.Status = %d, want %d", apiErr.Status, http.StatusBadRequest)
		}
	})
}

func decodeField(t *testing.T, body map[string]json.RawMessage, key string, out any) {
	t.Helper()
	raw, ok := body[key]
	if !ok {
		t.Fatalf("request body missing %q", key)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("decode %q: %v", key, err)
	}
}

func intPtr(value int) *int { return &value }
