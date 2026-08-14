package anthropic_test

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
	"github.com/looprig/llm/providers/anthropic"
)

func TestNewInvokeUsesNativeMessagesSemantics(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodyCh <- body
		if got, want := r.URL.Path, "/v1/messages"; got != want {
			http.Error(w, fmt.Sprintf("path = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		if got, want := r.Header.Get("x-api-key"), "sk-ant-test"; got != want {
			http.Error(w, fmt.Sprintf("x-api-key = %q, want %q", got, want), http.StatusUnauthorized)
			return
		}
		if got, want := r.Header.Get("anthropic-version"), "2023-06-01"; got != want {
			http.Error(w, fmt.Sprintf("anthropic-version = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		if got, want := r.Header.Get("anthropic-beta"), "prompt-caching-2024-07-31"; got != want {
			http.Error(w, fmt.Sprintf("anthropic-beta = %q, want %q", got, want), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4",
			"content":[
				{"type":"thinking","thinking":"think","signature":"sig"},
				{"type":"text","text":"answer"},
				{"type":"tool_use","id":"tool_1","name":"lookup","input":{"city":"NYC"}}
			],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":12,"cache_read_input_tokens":2,"cache_creation_input_tokens":1,"output_tokens":9}
		}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderAnthropic),
		model.APIFormatAnthropic,
		srv.URL+"/v1/",
		"claude-sonnet-4",
		model.WithTools(),
		model.WithThinkingDialect(model.ThinkingDialectAdaptive),
		model.WithStructuredOutputWithTools(),
		model.WithSampling(model.Sampling{Effort: model.EffortHigh, MaxTokens: intPtr(256)}),
	)
	client, err := anthropic.New(
		selected,
		auth.APIKey("sk-ant-test"),
		anthropic.WithThinking(anthropic.ThinkingOptions{Type: "adaptive", Effort: "high"}),
		anthropic.WithBetaHeaders("prompt-caching-2024-07-31"),
		anthropic.WithMetadataUserID("user-123"),
		anthropic.WithPromptCacheControl(anthropic.CacheControlOptions{Type: "ephemeral", TTL: "5m"}),
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
			&content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{&content.ToolUseBlock{
					ID: "tool_previous", Name: "lookup", Input: json.RawMessage(`{"city":"NYC"}`),
				}},
			}},
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}},
				ToolUseID: "tool_previous",
				IsError:   true,
			},
		},
		Tools: []inference.Tool{{
			Name:        "lookup",
			Description: "Look up a city",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"additionalProperties":false}`),
		}},
		ToolChoice: inference.ToolRequired(),
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
		t.Fatalf("decoded blocks = %d, want thinking/text/tool", len(resp.Message.Blocks))
	}
	if thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock); !ok || thinking.Thinking != "think" || thinking.Signature != "sig" {
		t.Errorf("thinking block = %#v, want native thinking/signature", resp.Message.Blocks[0])
	}
	if text, ok := resp.Message.Blocks[1].(*content.TextBlock); !ok || text.Text != "answer" {
		t.Errorf("text block = %#v, want answer", resp.Message.Blocks[1])
	}
	if tool, ok := resp.Message.Blocks[2].(*content.ToolUseBlock); !ok || tool.ID != "tool_1" || tool.Name != "lookup" || string(tool.Input) != `{"city":"NYC"}` {
		t.Errorf("tool block = %#v, want tool_use", resp.Message.Blocks[2])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 12 || resp.Usage.CacheReadTokens != 2 || resp.Usage.CacheCreationTokens != 1 || resp.Usage.OutputTokens != 9 {
		t.Errorf("usage = %+v, want input=12/cache_read=2/cache_creation=1/output=9", resp.Usage)
	}

	raw := <-bodyCh
	// The gate runs before every structural assertion below: the bytes that
	// reached the server must be a legal CreateMessageParams, or nothing the
	// assertions prove about them matters.
	gateRequest(t, raw)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	// This request sets no TransientMessages, so NOTHING in it is transient and
	// every message — including the live user turn — is "committed". A
	// breakpoint on the last message would move with every turn: a cache WRITE
	// every turn and never a read, strictly worse than not enabling caching.
	// The stable system/tools prefix is the only boundary worth having here, so
	// that is where the breakpoint must land.
	var systemBlocks []map[string]json.RawMessage
	decodeField(t, body, "system", &systemBlocks)
	if len(systemBlocks) != 1 {
		t.Fatalf("system blocks = %d, want cached text block", len(systemBlocks))
	}
	var systemText string
	decodeField(t, systemBlocks[0], "text", &systemText)
	if systemText != "system instruction" {
		t.Errorf("system text = %q, want system instruction", systemText)
	}
	var systemCache map[string]string
	decodeField(t, systemBlocks[0], "cache_control", &systemCache)
	if systemCache["type"] != "ephemeral" || systemCache["ttl"] != "5m" {
		t.Errorf("system cache_control = %#v, want ephemeral/5m", systemCache)
	}
	var messages []map[string]json.RawMessage
	decodeField(t, body, "messages", &messages)
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want user, assistant tool-use, and tool-result", len(messages))
	}
	var toolResult []map[string]json.RawMessage
	decodeField(t, messages[2], "content", &toolResult)
	var isError bool
	decodeField(t, toolResult[0], "is_error", &isError)
	if !isError {
		t.Error("tool_result.is_error = false, want true")
	}
	// The live turn must NOT carry the breakpoint (see above).
	if raw, exists := toolResult[0]["cache_control"]; exists {
		t.Errorf("live turn carries cache_control = %s; the breakpoint would move every turn", raw)
	}
	var thinking anthropic.ThinkingOptions
	decodeField(t, body, "thinking", &thinking)
	if thinking.Type != "adaptive" {
		t.Errorf("thinking = %+v, want adaptive", thinking)
	}
	var outputConfig map[string]json.RawMessage
	decodeField(t, body, "output_config", &outputConfig)
	var effort string
	decodeField(t, outputConfig, "effort", &effort)
	if effort != "high" {
		t.Errorf("output_config.effort = %q, want high", effort)
	}
	var metadata map[string]string
	decodeField(t, body, "metadata", &metadata)
	if metadata["user_id"] != "user-123" {
		t.Errorf("metadata = %#v, want user_id", metadata)
	}
	if _, ok := body["reasoning_effort"]; ok {
		t.Error("Anthropic Messages request contains OpenAI reasoning_effort")
	}
}

func TestStreamDecodesNativeSSEAndUsage(t *testing.T) {
	t.Parallel()
	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodyCh <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-4\",\"usage\":{\"input_tokens\":8,\"cache_read_input_tokens\":1,\"cache_creation_input_tokens\":0}}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"think\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"lookup\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":4}}\n\n")
		_, _ = io.WriteString(w, "data: {\"type\":\"message_stop\"}\n\n")
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, srv.URL+"/v1", "claude-sonnet-4", model.WithThinkingDialect(model.ThinkingDialectAdaptive))
	client, err := anthropic.New(selected, "sk-ant-test")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	// The streaming encoder is a separate mode of the same codec, so its body
	// gets the same gate. Notably it must carry "stream":true and still satisfy
	// CreateMessageParams, which is additionalProperties:false.
	gateRequest(t, <-bodyCh)

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
		t.Fatalf("chunks = %#v, want thinking/text/tool start/tool args", chunks)
	}
	if got, ok := chunks[0].(*content.ThinkingChunk); !ok || got.Thinking != "think" {
		t.Errorf("chunk 0 = %#v, want thinking", chunks[0])
	}
	if got, ok := chunks[1].(*content.TextChunk); !ok || got.Text != "answer" {
		t.Errorf("chunk 1 = %#v, want text", chunks[1])
	}
	if got, ok := chunks[2].(*content.ToolUseChunk); !ok || got.ID != "tool_1" || got.Name != "lookup" {
		t.Errorf("chunk 2 = %#v, want tool start", chunks[2])
	}
	if got, ok := chunks[3].(*content.ToolUseChunk); !ok || got.InputJSON != `{"city":"NYC"}` {
		t.Errorf("chunk 3 = %#v, want tool args", chunks[3])
	}
	result, ok := reader.Result()
	if !ok || result.Model != "claude-sonnet-4" || result.FinishReason != stream.FinishReasonToolUse || result.Usage == nil || result.Usage.InputTokens != 8 || result.Usage.CacheReadTokens != 1 || result.Usage.OutputTokens != 4 {
		t.Errorf("stream result = %+v, ok=%v, want terminal native usage", result, ok)
	}
}

func TestNewValidationAndErrors(t *testing.T) {
	t.Parallel()
	valid := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, "", "claude-sonnet-4")
	if client, err := anthropic.New(valid, ""); client != nil || err == nil {
		t.Fatalf("New(empty key) = %T, %v, want auth error", client, err)
	} else {
		var authErr *llm.AuthRequiredError
		if !errors.As(err, &authErr) || authErr.Provider != llm.ProviderAnthropic || authErr.Kind != auth.AuthAPIKey {
			t.Fatalf("New(empty key) error = %T %v, want Anthropic auth error", err, err)
		}
	}
	if client, err := anthropic.New(valid, "sk-ant-test"); err != nil || client == nil {
		t.Fatalf("New(valid) = %T, %v, want client", client, err)
	}
	wrong := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "https://api.openai.com/v1", "gpt-5")
	if client, err := anthropic.New(wrong, "sk-ant-test"); client != nil || err == nil {
		t.Fatalf("New(wrong provider) = %T, %v, want validation error", client, err)
	} else {
		var validationErr *model.ValidationError
		if !errors.As(err, &validationErr) || validationErr.Field != "Provider" {
			t.Fatalf("wrong-provider error = %T %v, want provider validation", err, err)
		}
	}

	for _, response := range []string{"{", `{"type":"error","error":{"message":"bad"}}`} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, response)
		}))
		selected := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, srv.URL+"/v1", "claude-sonnet-4")
		client, err := anthropic.New(selected, "sk-ant-test")
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		_, invokeErr := client.Invoke(context.Background(), inference.Request{Model: selected})
		if invokeErr == nil {
			t.Errorf("Invoke(%q) error = nil, want decode error", response)
		}
		srv.Close()
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "provider failure", http.StatusBadRequest)
	}))
	defer srv.Close()
	selected := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, srv.URL+"/v1", "claude-sonnet-4")
	client, err := anthropic.New(selected, "sk-ant-test")
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
