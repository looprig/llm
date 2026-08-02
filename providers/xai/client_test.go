package xai_test

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
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/xai"
)

func TestChatCompletionsContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderXAI, "xai-test-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return xai.New(selected, key)
	})
}

func TestNewInvokeUsesResponsesAndXAIOptions(t *testing.T) {
	t.Parallel()
	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodyCh <- body
		if got, want := r.URL.Path, "/v1/responses"; got != want {
			http.Error(w, fmt.Sprintf("path = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		if got, want := r.Header.Get("Authorization"), "Bearer xai-test-key"; got != want {
			http.Error(w, fmt.Sprintf("Authorization = %q, want %q", got, want), http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_xai_1","status":"completed","model":"grok-4.5",
			"output":[
				{"type":"reasoning","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"opaque"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"type":"function_call","call_id":"call_xai_1","name":"lookup","arguments":"{\"city\":\"SF\"}"}
			],
			"usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":2},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":3}}
		}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderXAI),
		model.APIFormatOpenAIResponses,
		srv.URL+"/v1/",
		"grok-4.5",
		model.WithTools(),
		model.WithThinking(),
		model.WithStructuredOutputWithTools(),
		model.WithSampling(model.Sampling{Effort: model.EffortMedium, MaxTokens: intPtr(128)}),
	)
	client, err := xai.New(
		selected,
		auth.APIKey("xai-test-key"),
		xai.WithReasoning(xai.ReasoningOptions{Effort: "high"}),
		xai.WithServiceTier(xai.ServiceTierPriority),
		xai.WithPromptCacheKey("conv-123"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	resp, err := client.Invoke(context.Background(), inference.Request{
		Model:  selected,
		System: "system instruction",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}},
				ToolUseID: "call_previous",
			},
		},
		Tools: []inference.Tool{{
			Name:   "lookup",
			Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
		}},
		ToolChoice: inference.ToolChoiceRequired,
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
			Strict: true,
		},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp == nil || resp.Message == nil || resp.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("response = %+v, want message/tool_use", resp)
	}
	if len(resp.Message.Blocks) != 3 {
		t.Fatalf("decoded blocks = %d, want reasoning/text/tool", len(resp.Message.Blocks))
	}
	if thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock); !ok || thinking.Thinking != "think" || thinking.ProviderStateFormat != "openai-responses" {
		t.Errorf("thinking block = %#v, want xAI Responses provider state", resp.Message.Blocks[0])
	}
	if text, ok := resp.Message.Blocks[1].(*content.TextBlock); !ok || text.Text != "answer" {
		t.Errorf("text block = %#v, want answer", resp.Message.Blocks[1])
	}
	if tool, ok := resp.Message.Blocks[2].(*content.ToolUseBlock); !ok || tool.ID != "call_xai_1" || tool.Name != "lookup" || string(tool.Input) != `{"city":"SF"}` {
		t.Errorf("tool block = %#v, want xAI function call", resp.Message.Blocks[2])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 10 || resp.Usage.CacheReadTokens != 2 || resp.Usage.OutputTokens != 9 || resp.Usage.ReasoningTokens != 3 {
		t.Errorf("usage = %+v, want input=10/cache=2/output=9/reasoning=3", resp.Usage)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	if _, ok := body["messages"]; ok {
		t.Error("xAI Responses request contains Chat Completions messages")
	}
	for _, field := range []string{"input", "instructions", "tools", "text", "reasoning", "service_tier", "prompt_cache_key"} {
		if _, ok := body[field]; !ok {
			t.Errorf("request missing Responses field %q", field)
		}
	}
	var reasoning xai.ReasoningOptions
	decodeField(t, body, "reasoning", &reasoning)
	if reasoning.Effort != "high" {
		t.Errorf("reasoning = %+v, want high", reasoning)
	}
	var serviceTier string
	decodeField(t, body, "service_tier", &serviceTier)
	if serviceTier != string(xai.ServiceTierPriority) {
		t.Errorf("service_tier = %q, want priority", serviceTier)
	}
	var cacheKey string
	decodeField(t, body, "prompt_cache_key", &cacheKey)
	if cacheKey != "conv-123" {
		t.Errorf("prompt_cache_key = %q, want conv-123", cacheKey)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Error("xAI Responses request contains legacy reasoning_effort")
	}
}

func TestStreamNormalizesXAIReasoningEventAlias(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.created\",\"response\":{\"model\":\"grok-4.5\",\"status\":\"in_progress\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"think\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"grok-4.5\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}],\"usage\":{\"input_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n")
	}))
	defer srv.Close()
	selected := model.CustomModel(model.ProviderName(llm.ProviderXAI), model.APIFormatOpenAIResponses, srv.URL+"/v1", "grok-4.5", model.WithThinking())
	client, err := xai.New(selected, "xai-test-key")
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
	if !ok || result.Model != "grok-4.5" || result.FinishReason != stream.FinishReasonStop || result.Usage == nil || result.Usage.InputTokens != 6 || result.Usage.CacheReadTokens != 1 || result.Usage.ReasoningTokens != 1 {
		t.Errorf("stream result = %+v, ok=%v, want terminal xAI usage", result, ok)
	}
}

func TestNewValidationCounterSupportAndErrors(t *testing.T) {
	t.Parallel()
	valid := model.CustomModel(model.ProviderName(llm.ProviderXAI), model.APIFormatOpenAIResponses, "", "grok-4.5")
	if client, err := xai.New(valid, ""); client != nil || err == nil {
		t.Fatalf("New(empty key) = %T, %v, want auth error", client, err)
	} else {
		var authErr *llm.AuthRequiredError
		if !errors.As(err, &authErr) || authErr.Provider != llm.ProviderXAI || authErr.Kind != auth.AuthAPIKey {
			t.Fatalf("New(empty key) error = %T %v, want xAI auth error", err, err)
		}
	}
	if counter, err := xai.NewCounter("xai-test-key"); counter != nil || err == nil {
		t.Fatalf("NewCounter() = %T, %v, want explicit unsupported-counter error", counter, err)
	} else {
		var supportErr *llm.CounterSupportError
		if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderXAI || supportErr.Reason != llm.CounterSupportExactUnavailable {
			t.Fatalf("NewCounter() error = %T %v, want xAI unsupported counter", err, err)
		}
	}
	wrong := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "https://api.openai.com/v1", "gpt-5")
	if client, err := xai.New(wrong, "xai-test-key"); client != nil || err == nil {
		t.Fatalf("New(wrong provider) = %T, %v, want validation error", client, err)
	} else {
		var validationErr *model.ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "Provider" {
			t.Fatalf("wrong-provider error = %T %v, want provider validation", err, err)
		}
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, "{")
	}))
	defer srv.Close()
	selected := model.CustomModel(model.ProviderName(llm.ProviderXAI), model.APIFormatOpenAIResponses, srv.URL+"/v1", "grok-4.5")
	client, err := xai.New(selected, "xai-test-key")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err == nil {
		t.Fatal("malformed response error = nil")
	}

	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "provider failure", http.StatusBadRequest)
	}))
	defer srv2.Close()
	selected.BaseURL = srv2.URL + "/v1"
	client, err = xai.New(selected, "xai-test-key")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
		t.Fatalf("HTTP error = %T %v, want *failure.APIError(400)", err, err)
	}
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
