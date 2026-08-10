package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/openai"
)

func TestNewInvokeUsesResponsesAndNormalizesProviderFields(t *testing.T) {
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
		if got, want := r.Header.Get("Authorization"), "Bearer sk-openai-test"; got != want {
			http.Error(w, fmt.Sprintf("authorization = %q, want %q", got, want), http.StatusUnauthorized)
			return
		}
		if got, want := r.Header.Get("Content-Type"), "application/json"; got != want {
			http.Error(w, fmt.Sprintf("content-type = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_1",
			"status":"completed",
			"model":"gpt-5",
			"output":[
				{"type":"reasoning","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"opaque"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"city\":\"NYC\"}"}
			],
			"usage":{
				"input_tokens":12,
				"input_tokens_details":{"cached_tokens":2},
				"output_tokens":9,
				"output_tokens_details":{"reasoning_tokens":3}
			}
		}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAIResponses,
		srv.URL+"/v1/",
		"gpt-5",
		model.WithTools(),
		model.WithThinking(),
		model.WithStructuredOutputWithTools(),
		model.WithSampling(model.Sampling{Effort: model.EffortMedium, MaxTokens: intPtr(128)}),
	)
	client, err := openai.New(
		selected,
		auth.APIKey("sk-openai-test"),
		openai.WithReasoning(openai.ReasoningOptions{Effort: "high", Summary: "detailed"}),
		openai.WithServiceTier(openai.ServiceTierFlex),
		openai.WithMetadata(map[string]string{"request": "test"}),
		openai.WithPromptCacheKey("cache-123"),
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
				IsError:   true,
			},
		},
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "Look up a city",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
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
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke() response = %+v, want decoded response", resp)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("FinishReason = %v, want tool_use", resp.FinishReason)
	}
	if got := len(resp.Message.Blocks); got != 3 {
		t.Fatalf("decoded blocks = %d, want reasoning/text/tool", got)
	}
	if thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock); !ok || thinking.Thinking != "think" || thinking.ProviderStateFormat != "openai-responses" {
		t.Errorf("reasoning block = %#v, want Responses reasoning with provider state", resp.Message.Blocks[0])
	}
	if text, ok := resp.Message.Blocks[1].(*content.TextBlock); !ok || text.Text != "answer" {
		t.Errorf("text block = %#v, want answer", resp.Message.Blocks[1])
	}
	if tool, ok := resp.Message.Blocks[2].(*content.ToolUseBlock); !ok || tool.ID != "call_1" || tool.Name != "lookup" || string(tool.Input) != `{"city":"NYC"}` {
		t.Errorf("tool block = %#v, want lookup call", resp.Message.Blocks[2])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 10 || resp.Usage.CacheReadTokens != 2 || resp.Usage.OutputTokens != 9 || resp.Usage.ReasoningTokens != 3 {
		t.Errorf("usage = %+v, want input=10 cache=2 output=9 reasoning=3", resp.Usage)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	var instructions string
	decodeField(t, body, "instructions", &instructions)
	if instructions != "system instruction" {
		t.Errorf("instructions = %q, want system instruction", instructions)
	}
	if _, ok := body["messages"]; ok {
		t.Error("Responses request contains Chat Completions messages field")
	}
	var input []map[string]json.RawMessage
	decodeField(t, body, "input", &input)
	if len(input) != 2 {
		t.Fatalf("input items = %d, want user and function_call_output", len(input))
	}
	var toolChoice string
	decodeField(t, body, "tool_choice", &toolChoice)
	if toolChoice != "required" {
		t.Errorf("tool_choice = %q, want required", toolChoice)
	}
	var reasoning openai.ReasoningOptions
	decodeField(t, body, "reasoning", &reasoning)
	if reasoning.Effort != "high" || reasoning.Summary != "detailed" {
		t.Errorf("reasoning = %+v, want explicit high/detailed", reasoning)
	}
	var serviceTier string
	decodeField(t, body, "service_tier", &serviceTier)
	if serviceTier != string(openai.ServiceTierFlex) {
		t.Errorf("service_tier = %q, want flex", serviceTier)
	}
	var metadata map[string]string
	decodeField(t, body, "metadata", &metadata)
	if metadata["request"] != "test" {
		t.Errorf("metadata = %#v, want request=test", metadata)
	}
	var cacheKey string
	decodeField(t, body, "prompt_cache_key", &cacheKey)
	if cacheKey != "cache-123" {
		t.Errorf("prompt_cache_key = %q, want cache-123", cacheKey)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Error("Responses request contains Chat Completions reasoning_effort field")
	}
}

func TestChatCompletionsContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderOpenAI, "sk-openai-test", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return openai.New(selected, key)
	})
}

func TestChatCompletionsReasoningOptionUsesChatField(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("request JSON = %v", err)
			return
		}
		if _, ok := body["messages"]; !ok {
			t.Error("Chat request missing messages")
		}
		var effort string
		if err := json.Unmarshal(body["reasoning_effort"], &effort); err != nil || effort != "high" {
			t.Errorf("reasoning_effort = %q, err=%v, want high", effort, err)
		}
		if _, ok := body["reasoning"]; ok {
			t.Error("Chat request contains Responses reasoning object")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chat","model":"gpt-4.1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAI, server.URL+"/v1", "gpt-4.1")
	client, err := openai.New(selected, "sk-test", openai.WithReasoning(openai.ReasoningOptions{Effort: "high", Summary: "detailed"}))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestStreamDecodesResponsesEventsAndTerminalUsage(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.created\",\"response\":{\"model\":\"gpt-5\",\"status\":\"in_progress\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"city\\\":\\\"NYC\\\"}\"}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}],\"usage\":{\"input_tokens\":7,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n")
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, srv.URL+"/v1", "gpt-5", model.WithThinking())
	client, err := openai.New(selected, "sk-openai-test")
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
	if len(chunks) != 4 {
		t.Fatalf("chunks = %#v, want reasoning/text/tool start/tool args", chunks)
	}
	if got, ok := chunks[0].(*content.ThinkingChunk); !ok || got.Thinking != "think" {
		t.Errorf("chunk 0 = %#v, want thinking", chunks[0])
	}
	if got, ok := chunks[1].(*content.TextChunk); !ok || got.Text != "answer" {
		t.Errorf("chunk 1 = %#v, want text", chunks[1])
	}
	if got, ok := chunks[2].(*content.ToolUseChunk); !ok || got.ID != "call_1" || got.Name != "lookup" {
		t.Errorf("chunk 2 = %#v, want tool start", chunks[2])
	}
	if got, ok := chunks[3].(*content.ToolUseChunk); !ok || got.InputJSON != `{"city":"NYC"}` {
		t.Errorf("chunk 3 = %#v, want tool arguments", chunks[3])
	}
	result, ok := reader.Result()
	if !ok || result.Model != "gpt-5" || result.FinishReason != stream.FinishReasonToolUse || result.Usage == nil || result.Usage.InputTokens != 6 || result.Usage.CacheReadTokens != 1 {
		t.Errorf("stream result = %+v, ok=%v, want model/finish/usage", result, ok)
	}
}

func TestNewValidation(t *testing.T) {
	t.Parallel()

	valid := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "", "gpt-5")
	client, err := openai.New(valid, "")
	if client != nil {
		t.Fatalf("New(empty key) returned %T", client)
	}
	var authErr *llm.AuthRequiredError
	if !errors.As(err, &authErr) || authErr.Provider != llm.ProviderOpenAI || authErr.Kind != auth.AuthAPIKey {
		t.Fatalf("New(empty key) error = %T %v, want OpenAI auth error", err, err)
	}

	if client, err := openai.New(valid, "sk-test"); err != nil || client == nil {
		t.Fatalf("New(valid) = %T, %v, want client", client, err)
	}

	wrong := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, "https://api.anthropic.com/v1", "claude-sonnet-4")
	if client, err := openai.New(wrong, "sk-test"); client != nil || err == nil {
		t.Fatalf("New(wrong provider) = %T, %v, want validation error", client, err)
	} else {
		var validationErr *model.ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "Provider" {
			t.Fatalf("wrong-provider error = %T %v, want provider validation", err, err)
		}
	}
}

func TestWithRoundTripperRejectsNil(t *testing.T) {
	t.Parallel()
	defer func() {
		if recover() == nil {
			t.Fatal("WithRoundTripper(nil) did not panic")
		}
	}()
	openai.WithRoundTripper(nil)
}

func TestWithRoundTripperRoutesInvokeThroughCallerTransport(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAI,
		"https://caller-owned.invalid/v1",
		"gpt-4.1",
	)
	var calls int
	rt := openAIRoundTripperFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chat","model":"gpt-4.1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)),
			Request:    req,
		}, nil
	})

	client, err := openai.New(selected, "sk-test", openai.WithRoundTripper(rt))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	resp, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke() response = %+v, want decoded response", resp)
	}
	if calls != 1 {
		t.Fatalf("caller-owned RoundTripper calls = %d, want 1", calls)
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
		selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, srv.URL+"/v1", "gpt-5")
		client, err := openai.New(selected, "sk-test")
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
		selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, srv.URL+"/v1", "gpt-5")
		client, err := openai.New(selected, "sk-test")
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

type openAIRoundTripperFunc func(*http.Request) (*http.Response, error)

func (f openAIRoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
