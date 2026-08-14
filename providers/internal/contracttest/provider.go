// Package contracttest contains deterministic provider-package HTTP contracts.
// It is internal test infrastructure, not a public runtime API.
package contracttest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/simple"
)

type Constructor func(model.Model, auth.APIKey) (inference.Client, error)

type OptionConstructor func(model.Model, auth.APIKey, ...simple.Option) (inference.Client, error)

// NoDefaultOpenCodeAttribution verifies that a gateway does not silently send
// OpenCode branding headers, while preserving an explicit caller override.
func NoDefaultOpenCodeAttribution(t *testing.T, provider llm.Provider, key auth.APIKey, header string, construct OptionConstructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := calls.Add(1)
		want := ""
		if call == 2 {
			want = "caller-value"
		}
		if got := r.Header.Get(header); got != want {
			t.Errorf("%s = %q on call %d, want %q", header, got, call, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model")
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() without attribution option error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() without attribution option error = %v", err)
	}

	client, err = construct(selected, key, simple.WithHeader(header, "caller-value"))
	if err != nil {
		t.Fatalf("New() with explicit attribution option error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() with explicit attribution option error = %v", err)
	}
}

// OpenAI verifies the common JSON/SSE, auth, usage, finish, and route contract
// for a provider using the OpenAI Chat Completions dialect.
func OpenAI(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	openAI(t, provider, key, "Authorization", "Bearer "+string(key), construct)
}

// OpenAIWithHeader is the same contract for providers such as Azure that use a
// documented API-key header instead of Authorization: Bearer.
func OpenAIWithHeader(t *testing.T, provider llm.Provider, key auth.APIKey, header string, construct Constructor) {
	openAI(t, provider, key, header, string(key), construct)
}

func openAI(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		if key == "" {
			if got := r.Header.Get(authHeader); got != "" {
				t.Errorf("%s = %q, want no auth header", authHeader, got)
			}
		} else if got := r.Header.Get(authHeader); got != authValue {
			t.Errorf("%s = %q, want %s", authHeader, got, authValue)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"stream\"}}]}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"model\":\"model\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}\n\n")
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"json"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model")
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response == nil || response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one text block", response)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	textChunk, ok := chunk.(*content.TextChunk)
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Model != "model" || result.FinishReason != stream.FinishReasonStop || result.Usage == nil || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("stream result = %+v, ok=%v, want model/stop/input=4/output=6", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
	openAIErrorResponses(t, provider, key, authHeader, authValue, construct)
	openAIToolStructured(t, provider, key, authHeader, authValue, construct)
}

func openAIToolStructured(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get(authHeader); got != authValue && key != "" {
			t.Errorf("%s = %q, want %q", authHeader, got, authValue)
		}
		var request map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("request JSON: %v", err)
		} else {
			if len(request["tools"]) == 0 {
				t.Error("request tools missing")
			}
			if len(request["response_format"]) == 0 {
				t.Error("request response_format missing")
			}
			if !bytes.Contains(request["messages"], []byte(`"tool_call_id":"call_1"`)) {
				t.Errorf("tool result missing from messages: %s", request["messages"])
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"done","tool_calls":[{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"value\":\"ok\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16,"completion_tokens_details":{"reasoning_tokens":2}}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model", model.WithStructuredOutputWithTools())
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "lookup"}}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.ToolUseBlock{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"value":"input"}`)}}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call_1"},
		},
		Tools:      []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`)}},
		Output:     &inference.OutputSchema{Name: "answer", Strict: true, Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)},
		ToolChoice: inference.ToolRequired(),
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %q, want tool_use", response.FinishReason)
	}
	if response.Usage == nil || response.Usage.InputTokens != 7 || response.Usage.OutputTokens != 9 || response.Usage.ReasoningTokens != 2 {
		t.Errorf("usage = %#v, want input=7 output=9 reasoning=2", response.Usage)
	}
	if len(response.Message.Blocks) != 2 {
		t.Fatalf("response blocks = %#v, want text and tool use", response.Message.Blocks)
	}
	if _, ok := response.Message.Blocks[1].(*content.ToolUseBlock); !ok {
		t.Errorf("second response block = %#v, want ToolUseBlock", response.Message.Blocks[1])
	}
}

// Anthropic verifies native Messages JSON/SSE, x-api-key authentication, usage,
// and finish normalization for a provider that exposes Anthropic semantics.
func Anthropic(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != string(key) {
			t.Errorf("x-api-key = %q, want %q", got, key)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q, want 2023-06-01", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"model\":\"model\",\"usage\":{\"input_tokens\":4}}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"json"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":6}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatAnthropic, srv.URL, "model")
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response == nil || response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one text block", response)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	textChunk, ok := chunk.(*content.TextChunk)
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Usage == nil || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("stream result = %+v ok=%v, want terminal usage", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
	anthropicErrorResponses(t, provider, key, "x-api-key", string(key), construct)
	anthropicToolContract(t, provider, key, "x-api-key", string(key), construct)
}

// AnthropicBearer verifies native Messages JSON/SSE for gateways that expose
// Anthropic semantics but deliberately authenticate the whole gateway with a
// bearer token (for example Cloudflare AI Gateway and Vercel AI Gateway).
func AnthropicBearer(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer "+string(key) {
			t.Errorf("Authorization = %q, want bearer auth", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"model\":\"model\",\"usage\":{\"input_tokens\":4}}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"json"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":6}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatAnthropic, srv.URL, "model")
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response == nil || response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one text block", response)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	textChunk, ok := chunk.(*content.TextChunk)
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Usage == nil || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("stream result = %+v ok=%v, want terminal usage", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
	anthropicErrorResponses(t, provider, key, "Authorization", "Bearer "+string(key), construct)
	anthropicToolContract(t, provider, key, "Authorization", "Bearer "+string(key), construct)
}

func anthropicToolContract(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Errorf("path = %q, want /messages", r.URL.Path)
		}
		if got := r.Header.Get(authHeader); got != authValue {
			t.Errorf("%s = %q, want %q", authHeader, got, authValue)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("request body: %v", err)
		} else {
			if !bytes.Contains(body, []byte(`"tools"`)) {
				t.Error("request tools missing")
			}
			if !bytes.Contains(body, []byte(`"tool_result"`)) {
				t.Errorf("tool result missing from request: %s", body)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"done"},{"type":"tool_use","id":"call_2","name":"lookup","input":{"value":"ok"}}],"stop_reason":"tool_use","usage":{"input_tokens":7,"output_tokens":9}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatAnthropic, srv.URL, "model", model.WithTools())
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "lookup"}}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.ToolUseBlock{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"value":"input"}`)}}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call_1"},
		},
		Tools: []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`)}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %q, want tool_use", response.FinishReason)
	}
	if response.Usage == nil || response.Usage.InputTokens != 7 || response.Usage.OutputTokens != 9 {
		t.Errorf("usage = %#v, want input=7 output=9", response.Usage)
	}
	if len(response.Message.Blocks) != 2 {
		t.Fatalf("response blocks = %#v, want text and tool use", response.Message.Blocks)
	}
	if _, ok := response.Message.Blocks[1].(*content.ToolUseBlock); !ok {
		t.Errorf("second response block = %#v, want ToolUseBlock", response.Message.Blocks[1])
	}
}

// Responses verifies the common OpenAI Responses JSON/SSE, bearer auth, usage,
// tool/finish normalization, and /responses route contract.
func Responses(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	responses(t, provider, key, "Authorization", "Bearer "+string(key), construct)
}

// ResponsesWithHeader verifies the Responses dialect for providers such as
// Azure that use a documented API-key header instead of Authorization: Bearer.
func ResponsesWithHeader(t *testing.T, provider llm.Provider, key auth.APIKey, header string, construct Constructor) {
	responses(t, provider, key, header, string(key), construct)
}

func responses(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		if got := r.Header.Get(authHeader); got != authValue {
			t.Errorf("%s = %q, want %q", authHeader, got, authValue)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"stream\"}\n\n")
			_, _ = fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"model\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":4,\"output_tokens\":6,\"total_tokens\":10}}}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp","object":"response","model":"model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"json"}]}],"usage":{"input_tokens":4,"output_tokens":6,"total_tokens":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAIResponses, srv.URL, "model")
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response == nil || response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("Invoke() response = %#v, want one text block", response)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "json" {
		t.Errorf("response text = %q, want json", got)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Errorf("usage = %#v, want input 4/output 6", response.Usage)
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
	textChunk, ok := chunk.(*content.TextChunk)
	if !ok || textChunk.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want TextChunk{stream}", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.Usage == nil || result.Usage.InputTokens != 4 || result.Usage.OutputTokens != 6 {
		t.Fatalf("stream result = %+v ok=%v, want terminal usage", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
	responsesFeatureContract(t, provider, key, authHeader, authValue, construct)
	responsesErrorResponses(t, provider, key, authHeader, authValue, construct)
}

func responsesFeatureContract(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q, want /responses", r.URL.Path)
		}
		if got := r.Header.Get(authHeader); got != authValue {
			t.Errorf("%s = %q, want %q", authHeader, got, authValue)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("request JSON: %v", err)
			return
		}
		if len(request["tools"]) == 0 || len(request["text"]) == 0 || len(request["reasoning"]) == 0 {
			t.Errorf("feature fields missing from Responses request: %s", body)
		}
		var toolChoice string
		if err := json.Unmarshal(request["tool_choice"], &toolChoice); err != nil || toolChoice != "required" {
			t.Errorf("tool_choice = %q, err=%v, want required", toolChoice, err)
		}
		if !bytes.Contains(request["input"], []byte(`"function_call_output"`)) {
			t.Errorf("function_call_output missing from Responses input: %s", request["input"])
		}
		if string(request["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			// No malformed frame here: this contract covers feature round-tripping
			// (tools, reasoning, text), and malformed SSE is now an authoritative
			// decode error rather than a skipped line. Malformed-frame handling is
			// asserted by the codec's own stream tests, not by a feature contract.
			_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\"}\n\n")
			_, _ = fmt.Fprint(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n")
			_, _ = fmt.Fprint(w, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_2\",\"name\":\"lookup\"}}\n\n")
			_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"value\\\":\\\"ok\\\"}\"}\n\n")
			_, _ = fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"model\":\"model\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_2\",\"name\":\"lookup\",\"arguments\":\"{\\\"value\\\":\\\"ok\\\"}\"}],\"usage\":{\"input_tokens\":8,\"input_tokens_details\":{\"cached_tokens\":2},\"output_tokens\":9,\"output_tokens_details\":{\"reasoning_tokens\":2}}}}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp","object":"response","model":"model","status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"think"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"json"}]},{"type":"function_call","call_id":"call_2","name":"lookup","arguments":"{\"value\":\"ok\"}"}],"usage":{"input_tokens":8,"input_tokens_details":{"cached_tokens":2},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":2}}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(provider),
		model.APIFormatOpenAIResponses,
		srv.URL,
		"model",
		model.WithThinking(),
		model.WithStructuredOutputWithTools(),
		model.WithSampling(model.Sampling{Effort: model.EffortHigh}),
	)
	request := inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "lookup"}}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.ToolUseBlock{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"value":"input"}`)}}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call_1"},
		},
		Tools:      []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`)}},
		Output:     &inference.OutputSchema{Name: "answer", Strict: true, Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)},
		ToolChoice: inference.ToolRequired(),
	}
	client, err := construct(selected, key)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), request)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.FinishReason != stream.FinishReasonToolUse || response.Usage == nil || response.Usage.InputTokens != 6 || response.Usage.CacheReadTokens != 2 || response.Usage.OutputTokens != 9 || response.Usage.ReasoningTokens != 2 {
		t.Fatalf("response finish/usage = %q/%+v, want tool/input=6/cache=2/output=9/reasoning=2", response.FinishReason, response.Usage)
	}
	if len(response.Message.Blocks) != 3 {
		t.Fatalf("response blocks = %#v, want reasoning/text/tool", response.Message.Blocks)
	}
	if _, ok := response.Message.Blocks[0].(*content.ThinkingBlock); !ok {
		t.Errorf("response block 0 = %#v, want ThinkingBlock", response.Message.Blocks[0])
	}
	if _, ok := response.Message.Blocks[2].(*content.ToolUseBlock); !ok {
		t.Errorf("response block 2 = %#v, want ToolUseBlock", response.Message.Blocks[2])
	}

	reader, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()
	var sawThinking, sawText, sawTool bool
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("Stream.Next() error = %v", nextErr)
		}
		switch chunk.(type) {
		case *content.ThinkingChunk:
			sawThinking = true
		case *content.TextChunk:
			sawText = true
		case *content.ToolUseChunk:
			sawTool = true
		}
	}
	if !sawThinking || !sawText || !sawTool {
		t.Fatalf("stream feature chunks = thinking:%v text:%v tool:%v", sawThinking, sawText, sawTool)
	}
	result, ok := reader.Result()
	if !ok || result.Model != "model" || result.FinishReason != stream.FinishReasonToolUse || result.Usage == nil || result.Usage.InputTokens != 6 || result.Usage.CacheReadTokens != 2 || result.Usage.OutputTokens != 9 || result.Usage.ReasoningTokens != 2 {
		t.Fatalf("stream result = %+v, ok=%v, want terminal tool/usage", result, ok)
	}
	if calls.Load() != 2 {
		t.Errorf("request count = %d, want 2", calls.Load())
	}
}

func openAIErrorResponses(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	t.Run("malformed response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, "{")
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model")
		client, err := construct(selected, key)
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err == nil {
			t.Fatal("Invoke() error = nil, want malformed response error")
		}
	})
	t.Run("HTTP error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if key == "" {
				if got := r.Header.Get(authHeader); got != "" {
					t.Errorf("%s = %q, want no auth header", authHeader, got)
				}
			} else if got := r.Header.Get(authHeader); got != authValue {
				t.Errorf("%s = %q, want %q", authHeader, got, authValue)
			}
			http.Error(w, "provider failure", http.StatusBadRequest)
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAI, srv.URL, "model")
		client, err := construct(selected, key)
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
			t.Fatalf("Invoke() error = %T %v, want *failure.APIError(400)", err, err)
		}
	})
}

func anthropicErrorResponses(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	for _, test := range []struct {
		name string
		body string
		code int
	}{
		{name: "malformed response", body: "{", code: http.StatusOK},
		{name: "HTTP error", body: "provider failure", code: http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get(authHeader); got != authValue {
					t.Errorf("%s = %q, want %q", authHeader, got, authValue)
				}
				w.WriteHeader(test.code)
				_, _ = io.WriteString(w, test.body)
			}))
			defer srv.Close()
			selected := model.CustomModel(model.ProviderName(provider), model.APIFormatAnthropic, srv.URL, "model")
			client, err := construct(selected, key)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
			if test.code == http.StatusOK {
				if err == nil {
					t.Fatal("Invoke() error = nil, want malformed response error")
				}
				return
			}
			var apiErr *failure.APIError
			if !errors.As(err, &apiErr) || apiErr.Status != test.code {
				t.Fatalf("Invoke() error = %T %v, want *failure.APIError(%d)", err, err, test.code)
			}
		})
	}
}

func responsesErrorResponses(t *testing.T, provider llm.Provider, key auth.APIKey, authHeader, authValue string, construct Constructor) {
	t.Helper()
	for _, test := range []struct {
		name string
		body string
		code int
	}{
		{name: "malformed response", body: "{", code: http.StatusOK},
		{name: "HTTP error", body: "provider failure", code: http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get(authHeader); got != authValue {
					t.Errorf("%s = %q, want %q", authHeader, got, authValue)
				}
				w.WriteHeader(test.code)
				_, _ = io.WriteString(w, test.body)
			}))
			defer srv.Close()
			selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAIResponses, srv.URL, "model")
			client, err := construct(selected, key)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
			if test.code == http.StatusOK {
				if err == nil {
					t.Fatal("Invoke() error = nil, want malformed response error")
				}
				return
			}
			var apiErr *failure.APIError
			if !errors.As(err, &apiErr) || apiErr.Status != test.code {
				t.Fatalf("Invoke() error = %T %v, want *failure.APIError(%d)", err, err, test.code)
			}
		})
	}
	t.Run("truncated SSE has no fabricated terminal result", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n")
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(provider), model.APIFormatOpenAIResponses, srv.URL, "model")
		client, err := construct(selected, key)
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer func() { _ = reader.Close() }()
		if _, err := reader.Next(); err != nil {
			t.Fatalf("first Stream.Next() error = %v", err)
		}
		// A body that stops before response.completed is a truncated answer,
		// so the Responses codec's terminal gate ends the read with a typed
		// failure. This assertion previously required io.EOF — the clean
		// exhaustion sentinel — which is precisely how a lost turn came to be
		// reported as a completed one: the caller sees a finished stream whose
		// content silently stops mid-sentence. Only the missing result trailer
		// was checked, and no caller is obliged to check it.
		_, err = reader.Next()
		if errors.Is(err, io.EOF) {
			t.Fatal("truncated Stream.Next() = EOF; a stream that never reached a terminal response event must fail, not exhaust cleanly")
		}
		var resultErr *stream.StreamResultError
		if !errors.As(err, &resultErr) {
			t.Fatalf("truncated Stream.Next() error = %T %v, want *stream.StreamResultError", err, err)
		}
		if _, ok := reader.Result(); ok {
			t.Fatal("truncated stream returned terminal result without response.completed")
		}
	})
}
