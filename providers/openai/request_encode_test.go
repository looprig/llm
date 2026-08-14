package openai_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
)

// Agentic-turn REQUEST coverage, gated against OpenAI's own request schemas.
//
// multimodal_request_test.go gates the single-user-message shapes. This file
// gates the shapes a real agent loop actually sends back: an assistant turn
// replayed as history, a tool call and its result, a reasoning item carrying
// provider state, and the token-limit field a reasoning model requires. Every
// one of those wire shapes changed recently, and each is a variant the encoder
// must spell exactly — a `function_call_output` without `output`, a
// `reasoning` without `id`, or a `max_tokens` sent to a reasoning model are all
// bodies the provider rejects outright.
//
// Standing caveat on what the gate can and cannot prove here: OpenAI's own
// specs close `additionalProperties` on almost nothing (3 of 54 object shapes
// in Chat Completions, 5 of 147 in Responses), so an INVENTED member is
// generally NOT caught. What is caught is a missing required member, a wrong
// type, a bad enum value, and a variant whose discriminator matches nothing.
// The explicit assertions below carry everything else.

// reasoningStateJSON is the ProviderState a Responses reasoning item round-trips
// through content.ThinkingBlock. Its shape is the codec's business; a test can
// only produce it the way a prior decode would have.
func reasoningStateJSON(id, encrypted string) json.RawMessage {
	raw, _ := json.Marshal(map[string]string{"id": id, "encrypted_content": encrypted})
	return raw
}

// agenticHistory is one complete prior turn plus a follow-up: user asks,
// assistant thinks / answers / calls a tool, the tool answers.
func agenticHistory() content.AgenticMessages {
	return content.AgenticMessages{
		&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "weather in NYC?"}},
		}},
		&content.AIMessage{Message: content.Message{
			Role: content.RoleAssistant,
			Blocks: []content.Block{
				content.NewThinkingBlock("consider the tools", "",
					reasoningStateJSON("rs_abc123", "opaque-blob"), "openai-responses"),
				&content.TextBlock{Text: "Let me look that up."},
				&content.ToolUseBlock{
					ID:    "call_1",
					Name:  "lookup",
					Input: json.RawMessage(`{"city":"NYC"}`),
				},
			},
		}},
		&content.ToolResultMessage{
			Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "sunny, 21C"}}},
			ToolUseID: "call_1",
		},
	}
}

func lookupTool() []inference.Tool {
	return []inference.Tool{{
		Name:        "lookup",
		Description: "Look up a city",
		Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}}
}

// --- Responses -------------------------------------------------------------

func responsesInputItems(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var decoded struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode request body: %v\nbody: %s", err, body)
	}
	return decoded.Input
}

func itemString(t *testing.T, item map[string]json.RawMessage, key string) string {
	t.Helper()
	raw, ok := item[key]
	if !ok {
		t.Fatalf("item %v has no %q", item, key)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("decode %q from %s: %v", key, raw, err)
	}
	return s
}

// TestResponsesEncodesAFullAgenticTurn gates the whole replayed turn and then
// pins each item variant's required members. The gate alone would accept a
// reasoning item that lost its summary or a tool result that lost its call_id
// only if the schema left them optional — it does not, which is exactly why
// the encoder spells them explicitly.
func TestResponsesEncodesAFullAgenticTurn(t *testing.T) {
	t.Parallel()

	body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
		responsesDir, "completed_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{Model: selected, System: "be brief", Messages: agenticHistory(), Tools: lookupTool()}
		},
		model.WithTools(), model.WithThinking(),
	)

	items := responsesInputItems(t, body)
	if len(items) != 5 {
		t.Fatalf("input items = %d, want user/reasoning/assistant-text/function_call/function_call_output\n%s", len(items), body)
	}

	// 0: the user message keeps the Item→InputMessage form (typed parts).
	if got := itemString(t, items[0], "role"); got != "user" {
		t.Errorf("item 0 role = %q, want user", got)
	}

	// 1: reasoning. ReasoningItem.required is id, summary, type — summary must
	// be present even when empty, and the id must be the provider's own.
	if got := itemString(t, items[1], "type"); got != "reasoning" {
		t.Fatalf("item 1 type = %q, want reasoning", got)
	}
	if got := itemString(t, items[1], "id"); got != "rs_abc123" {
		t.Errorf("reasoning id = %q, want the replayed provider id rs_abc123", got)
	}
	if _, ok := items[1]["summary"]; !ok {
		t.Error("reasoning item has no summary; ReasoningItem.required lists it")
	}
	if got := itemString(t, items[1], "encrypted_content"); got != "opaque-blob" {
		t.Errorf("encrypted_content = %q, want the replayed blob", got)
	}
	for _, leaked := range []string{"call_id", "output", "arguments", "name"} {
		if _, ok := items[1][leaked]; ok {
			t.Errorf("reasoning item leaked %q from another variant", leaked)
		}
	}

	// 2: replayed assistant text takes the EasyInputMessage form — role plus a
	// bare string, and NO `type`, because a typed InputMessage/OutputMessage
	// would need an id no neutral block carries.
	if _, ok := items[2]["type"]; ok {
		t.Errorf("assistant history item carries a type; want the id-free EasyInputMessage form: %v", items[2])
	}
	if got := itemString(t, items[2], "role"); got != "assistant" {
		t.Errorf("item 2 role = %q, want assistant", got)
	}
	if got := itemString(t, items[2], "content"); got != "Let me look that up." {
		t.Errorf("assistant content = %q, want the bare-string form", got)
	}

	// 3: function_call. arguments is a JSON-encoded STRING, never an object.
	if got := itemString(t, items[3], "type"); got != "function_call" {
		t.Fatalf("item 3 type = %q, want function_call", got)
	}
	if got := itemString(t, items[3], "call_id"); got != "call_1" {
		t.Errorf("call_id = %q, want call_1", got)
	}
	args := itemString(t, items[3], "arguments")
	var argsObj map[string]any
	if err := json.Unmarshal([]byte(args), &argsObj); err != nil {
		t.Fatalf("arguments %q is not a JSON-encoded string of an object: %v", args, err)
	}
	if argsObj["city"] != "NYC" {
		t.Errorf("arguments = %q, want city=NYC", args)
	}

	// 4: function_call_output always carries output AND call_id.
	if got := itemString(t, items[4], "type"); got != "function_call_output" {
		t.Fatalf("item 4 type = %q, want function_call_output", got)
	}
	if got := itemString(t, items[4], "call_id"); got != "call_1" {
		t.Errorf("output call_id = %q, want call_1", got)
	}
	if got := itemString(t, items[4], "output"); got != "sunny, 21C" {
		t.Errorf("output = %q, want the tool text", got)
	}
}

// TestResponsesEncodesEmptyToolResultWithAnExplicitOutput pins the empty case
// of the same variant rule: `output` is required, so an empty tool result must
// still spell "output":"" rather than omitting the member.
func TestResponsesEncodesEmptyToolResultWithAnExplicitOutput(t *testing.T) {
	t.Parallel()

	body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
		responsesDir, "completed_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{
				Model: selected,
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{
						Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
					}},
					&content.ToolResultMessage{
						Message:   content.Message{Role: content.RoleTool},
						ToolUseID: "call_9",
					},
				},
				Tools: lookupTool(),
			}
		},
		model.WithTools(),
	)

	items := responsesInputItems(t, body)
	last := items[len(items)-1]
	raw, ok := last["output"]
	if !ok {
		t.Fatalf("empty tool result = %v, want an explicit output member", last)
	}
	if string(raw) != `""` {
		t.Errorf("output = %s, want an empty JSON string", raw)
	}
}

// TestResponsesEncodesFunctionToolStrict pins the FunctionTool member OpenAI's
// spec marks required and Looprig used to omit entirely.
func TestResponsesEncodesFunctionToolStrict(t *testing.T) {
	t.Parallel()

	body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
		responsesDir, "completed_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{
				Model: selected,
				Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
					Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
				}}},
				Tools: lookupTool(),
			}
		},
		model.WithTools(),
	)

	var decoded struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if len(decoded.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(decoded.Tools))
	}
	raw, ok := decoded.Tools[0]["strict"]
	if !ok {
		t.Fatalf("tool = %v, want a strict member (FunctionTool.required)", decoded.Tools[0])
	}
	if string(raw) != "false" {
		t.Errorf("strict = %s, want false", raw)
	}
}

// --- Chat Completions ------------------------------------------------------

func chatMessages(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode request body: %v\nbody: %s", err, body)
	}
	return decoded.Messages
}

// TestChatEncodesAFullAgenticTurn is the Chat Completions counterpart: an
// assistant message carrying tool_calls, followed by the role:"tool" message
// that answers it. ChatCompletionRequestToolMessage.required is
// [role, content, tool_call_id], and a tool_calls entry requires
// [id, type, function]; the gate holds all of that.
func TestChatEncodesAFullAgenticTurn(t *testing.T) {
	t.Parallel()

	body := captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
		chatDir, "plain_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{Model: selected, System: "be brief", Messages: agenticHistory(), Tools: lookupTool()}
		},
		model.WithTools(),
	)

	msgs := chatMessages(t, body)
	if len(msgs) < 4 {
		t.Fatalf("messages = %d, want system/user/assistant/tool\n%s", len(msgs), body)
	}
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i] = itemString(t, m, "role")
	}
	if roles[0] != "system" || roles[1] != "user" {
		t.Errorf("leading roles = %v, want system then user", roles)
	}

	var assistant, tool map[string]json.RawMessage
	for i, r := range roles {
		switch r {
		case "assistant":
			assistant = msgs[i]
		case "tool":
			tool = msgs[i]
		}
	}
	if assistant == nil || tool == nil {
		t.Fatalf("roles = %v, want an assistant and a tool message", roles)
	}

	var calls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	raw, ok := assistant["tool_calls"]
	if !ok {
		t.Fatalf("assistant message = %v, want tool_calls", assistant)
	}
	if err := json.Unmarshal(raw, &calls); err != nil {
		t.Fatalf("decode tool_calls: %v", err)
	}
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Type != "function" || calls[0].Function.Name != "lookup" {
		t.Fatalf("tool_calls = %+v, want one function call to lookup", calls)
	}
	// arguments is a JSON-encoded string, not a nested object.
	var argsObj map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &argsObj); err != nil {
		t.Fatalf("arguments %q is not a JSON-encoded object: %v", calls[0].Function.Arguments, err)
	}
	if argsObj["city"] != "NYC" {
		t.Errorf("arguments = %q, want city=NYC", calls[0].Function.Arguments)
	}

	if got := itemString(t, tool, "tool_call_id"); got != "call_1" {
		t.Errorf("tool_call_id = %q, want call_1", got)
	}
	if _, ok := tool["content"]; !ok {
		t.Error("tool message has no content; ChatCompletionRequestToolMessage.required lists it")
	}
}

// TestChatTokenLimitSpelling pins the recently changed capability gate:
// max_tokens is deprecated and REJECTED by reasoning models, which take
// max_completion_tokens; plenty of OpenAI-compatible servers speaking this
// dialect still know only max_tokens, so a non-reasoning model keeps that
// spelling. Both members are individually legal in the schema, so this is an
// assertion the gate cannot make — only that whichever is emitted is an
// integer.
func TestChatTokenLimitSpelling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		opts    []model.ModelOption
		want    string
		notWant string
	}{
		{
			name:    "reasoning model uses max_completion_tokens",
			opts:    []model.ModelOption{model.WithThinking()},
			want:    "max_completion_tokens",
			notWant: "max_tokens",
		},
		{
			name:    "plain model keeps the legacy max_tokens",
			want:    "max_tokens",
			notWant: "max_completion_tokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			opts := append([]model.ModelOption{
				model.WithSampling(model.Sampling{MaxTokens: intPtr(256)}),
			}, tt.opts...)
			body := captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
				chatDir, "plain_text.json",
				func(selected model.Model) inference.Request {
					return inference.Request{
						Model: selected,
						Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
							Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
						}}},
					}
				},
				opts...,
			)

			var decoded map[string]json.RawMessage
			if err := json.Unmarshal(body, &decoded); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			raw, ok := decoded[tt.want]
			if !ok {
				t.Fatalf("body = %s, want a %s member", body, tt.want)
			}
			if string(raw) != "256" {
				t.Errorf("%s = %s, want 256", tt.want, raw)
			}
			if _, ok := decoded[tt.notWant]; ok {
				t.Errorf("body also carries %s; exactly one spelling may be sent", tt.notWant)
			}
		})
	}
}

// TestParameterlessToolCarriesAnEmptyObjectSchema pins the tool whose neutral
// inference.Tool declares no Schema at all. `parameters` is spec-typed
// `object` in both dialects, so the nil json.RawMessage must be substituted
// for {"type":"object"} rather than marshalled as `null`. Chat Completions
// used to emit the null; Responses always substituted.
func TestParameterlessToolCarriesAnEmptyObjectSchema(t *testing.T) {
	t.Parallel()

	parameterless := func(selected model.Model) inference.Request {
		return inference.Request{
			Model: selected,
			Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
				Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
			}}},
			Tools: []inference.Tool{{Name: "ping", Description: "takes no arguments"}},
		}
	}

	t.Run("chat", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
			chatDir, "plain_text.json", parameterless, model.WithTools())

		var decoded struct {
			Tools []struct {
				Function struct {
					Parameters json.RawMessage `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if got := string(decoded.Tools[0].Function.Parameters); got != `{"type":"object"}` {
			t.Errorf("parameters = %s, want an empty object schema", got)
		}
	})

	t.Run("responses", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
			responsesDir, "completed_text.json", parameterless, model.WithTools())

		var decoded struct {
			Tools []struct {
				Parameters json.RawMessage `json:"parameters"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if got := string(decoded.Tools[0].Parameters); got != `{"type":"object"}` {
			t.Errorf("parameters = %s, want an empty object schema", got)
		}
	})
}

// TestEncodesStructuredOutput gates the two dialects' different spellings of
// the same neutral inference.OutputSchema: Chat's response_format.json_schema
// and Responses' text.format. Both nest the caller's schema verbatim, so a
// misplaced member here is a request the provider rejects.
func TestEncodesStructuredOutput(t *testing.T) {
	t.Parallel()

	const schema = `{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`
	structured := func(selected model.Model) inference.Request {
		return inference.Request{
			Model: selected,
			Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
				Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
			}}},
			Output: &inference.OutputSchema{Name: "answer", Schema: json.RawMessage(schema), Strict: true},
		}
	}

	t.Run("chat", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
			chatDir, "plain_text.json", structured, model.WithStructuredOutput())

		var decoded struct {
			ResponseFormat struct {
				Type       string `json:"type"`
				JSONSchema struct {
					Name   string          `json:"name"`
					Strict bool            `json:"strict"`
					Schema json.RawMessage `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if decoded.ResponseFormat.Type != "json_schema" || decoded.ResponseFormat.JSONSchema.Name != "answer" || !decoded.ResponseFormat.JSONSchema.Strict {
			t.Errorf("response_format = %+v, want a strict json_schema named answer", decoded.ResponseFormat)
		}
		if string(decoded.ResponseFormat.JSONSchema.Schema) != schema {
			t.Errorf("schema = %s, want it nested verbatim", decoded.ResponseFormat.JSONSchema.Schema)
		}
	})

	t.Run("responses", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
			responsesDir, "completed_text.json", structured, model.WithStructuredOutput())

		var decoded struct {
			Text struct {
				Format struct {
					Type   string          `json:"type"`
					Name   string          `json:"name"`
					Strict bool            `json:"strict"`
					Schema json.RawMessage `json:"schema"`
				} `json:"format"`
			} `json:"text"`
		}
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if decoded.Text.Format.Type != "json_schema" || decoded.Text.Format.Name != "answer" || !decoded.Text.Format.Strict {
			t.Errorf("text.format = %+v, want a strict json_schema named answer", decoded.Text.Format)
		}
		if string(decoded.Text.Format.Schema) != schema {
			t.Errorf("schema = %s, want it nested verbatim", decoded.Text.Format.Schema)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if _, ok := raw["response_format"]; ok {
			t.Error("Responses body carries the Chat Completions member response_format")
		}
	})
}

// TestResponsesTokenLimitIsMaxOutputTokens is the Responses-side counterpart:
// that dialect has exactly one spelling, and neither Chat member is legal.
func TestResponsesTokenLimitIsMaxOutputTokens(t *testing.T) {
	t.Parallel()

	body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
		responsesDir, "completed_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{
				Model: selected,
				Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
					Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go"}},
				}}},
			}
		},
		model.WithThinking(), model.WithSampling(model.Sampling{MaxTokens: intPtr(256)}),
	)

	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if string(decoded["max_output_tokens"]) != "256" {
		t.Errorf("max_output_tokens = %s, want 256", decoded["max_output_tokens"])
	}
	for _, wrong := range []string{"max_tokens", "max_completion_tokens"} {
		if _, ok := decoded[wrong]; ok {
			t.Errorf("Responses body carries the Chat Completions member %q", wrong)
		}
	}
}

// --- named tool choice -----------------------------------------------------

// namedToolChoiceRequest forces the one declared tool, in whichever dialect the
// caller gates.
func namedToolChoiceRequest(selected model.Model) inference.Request {
	return inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "weather in NYC?"}},
		}}},
		Tools:      lookupTool(),
		ToolChoice: inference.ToolNamed("lookup"),
	}
}

// TestEncodesNamedToolChoice gates the forced-tool body in both OpenAI
// dialects. The two spellings differ by one level of nesting — Chat wraps the
// name in a `function` object, Responses puts it beside `type` — so a single
// shared expectation would silently pass the wrong one.
func TestEncodesNamedToolChoice(t *testing.T) {
	t.Parallel()

	t.Run("chat", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
			chatDir, "plain_text.json", namedToolChoiceRequest, model.WithTools())

		var decoded map[string]json.RawMessage
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		const want = `{"type":"function","function":{"name":"lookup"}}`
		if got := string(decoded["tool_choice"]); got != want {
			t.Errorf("tool_choice = %s, want %s", got, want)
		}
	})

	t.Run("responses", func(t *testing.T) {
		t.Parallel()
		body := captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
			responsesDir, "completed_text.json", namedToolChoiceRequest, model.WithTools())

		var decoded map[string]json.RawMessage
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		const want = `{"type":"function","name":"lookup"}`
		if got := string(decoded["tool_choice"]); got != want {
			t.Errorf("tool_choice = %s, want %s", got, want)
		}
	})
}

// TestToolChoiceGateStrength measures what each OpenAI request gate really
// enforces on tool_choice, by feeding it shapes the encoder never emits.
//
// Both dialects model tool_choice as an anyOf over a mode enum and several
// named-tool objects, and both of those objects declare required members — so
// a missing name IS caught, and so is the other dialect's nesting, because
// neither object accepts the other's members. What is NOT caught is anything
// cross-field: neither schema knows which tools the body declares, and
// OpenAI closes additionalProperties on almost nothing, so an extra member
// rides along on the Chat objects. inference.ValidateRequestFeatures carries
// the tool-must-be-declared constraint instead.
func TestToolChoiceGateStrength(t *testing.T) {
	t.Parallel()

	chatBody := func(toolChoice string) []byte {
		return []byte(`{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}],` +
			`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],` +
			`"tool_choice":` + toolChoice + `}`)
	}
	responsesBody := func(toolChoice string) []byte {
		return []byte(`{"model":"gpt-4.1","input":[{"role":"user","content":"hi"}],` +
			`"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"strict":false}],` +
			`"tool_choice":` + toolChoice + `}`)
	}

	cases := []struct {
		name       string
		format     string
		kind       string
		body       []byte
		wantReject bool
		because    string
	}{
		{
			name:   "chat: the shape the encoder emits",
			format: "openai", kind: chatRequestKind,
			body:    chatBody(`{"type":"function","function":{"name":"lookup"}}`),
			because: "ChatCompletionNamedToolChoice requires type and function.name",
		},
		{
			name:   "chat: name not wrapped in function",
			format: "openai", kind: chatRequestKind,
			body:       chatBody(`{"type":"function","name":"lookup"}`),
			wantReject: true,
			because:    "the Responses spelling omits the required `function` member",
		},
		{
			name:   "chat: function object with no name",
			format: "openai", kind: chatRequestKind,
			body:       chatBody(`{"type":"function","function":{}}`),
			wantReject: true,
			because:    "function.name is required",
		},
		{
			name:   "chat: unknown discriminator",
			format: "openai", kind: chatRequestKind,
			body:       chatBody(`{"type":"tool","name":"lookup"}`),
			wantReject: true,
			because:    "Anthropic's spelling matches no member of the anyOf",
		},
		{
			name:   "chat: extra member on the named choice",
			format: "openai", kind: chatRequestKind,
			body:    chatBody(`{"type":"function","function":{"name":"lookup"},"invented":true}`),
			because: "ChatCompletionNamedToolChoice does not close additionalProperties",
		},
		{
			name:   "chat: name that matches no declared tool",
			format: "openai", kind: chatRequestKind,
			body:    chatBody(`{"type":"function","function":{"name":"undeclared"}}`),
			because: "no cross-field constraint exists; ValidateRequestFeatures carries it",
		},
		{
			name:   "responses: the shape the encoder emits",
			format: "openai-responses", kind: responsesRequestKind,
			body:    responsesBody(`{"type":"function","name":"lookup"}`),
			because: "ToolChoiceFunction requires type and name",
		},
		{
			name:   "responses: name wrapped in function",
			format: "openai-responses", kind: responsesRequestKind,
			body:       responsesBody(`{"type":"function","function":{"name":"lookup"}}`),
			wantReject: true,
			because:    "the Chat spelling omits ToolChoiceFunction's required top-level name",
		},
		{
			name:   "responses: function with no name",
			format: "openai-responses", kind: responsesRequestKind,
			body:       responsesBody(`{"type":"function"}`),
			wantReject: true,
			because:    "name is required",
		},
		{
			name:   "responses: name that matches no declared tool",
			format: "openai-responses", kind: responsesRequestKind,
			body:    responsesBody(`{"type":"function","name":"undeclared"}`),
			because: "no cross-field constraint exists; ValidateRequestFeatures carries it",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate(tc.format, tc.kind, tc.body)
			if tc.wantReject && err == nil {
				t.Fatalf("gate accepted %s, want rejection (%s)", tc.body, tc.because)
			}
			if !tc.wantReject && err != nil {
				t.Fatalf("gate rejected %s (%v), want acceptance (%s)", tc.body, err, tc.because)
			}
		})
	}
}
