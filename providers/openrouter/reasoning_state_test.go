package openrouter

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
)

func TestReasoningDetailsRoundTripIsOpenRouterLocal(t *testing.T) {
	t.Parallel()

	details := `[{
        "type":"reasoning.summary","index":0,"format":"anthropic-claude-v1","summary":"first"
      },{"type":"reasoning.text","index":1,"format":"anthropic-claude-v1","text":"second","signature":"sig"},
      {"type":"reasoning.encrypted","index":2,"format":"anthropic-claude-v1","data":"opaque","signature":"encrypted-sig"}]`
	responseBody := []byte(`{"id":"r","model":"anthropic/claude-sonnet-4","choices":[{"message":{"role":"assistant","content":null,"reasoning_details":` + details + `,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	response, err := (requestCodec{}).DecodeResponse(responseBody)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	thinking, ok := response.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("first block = %T, want *content.ThinkingBlock", response.Message.Blocks[0])
	}
	if thinking.Thinking != "first\nsecond" {
		t.Errorf("Thinking = %q, want ordered readable details", thinking.Thinking)
	}
	if !thinking.ReplayableAs(openRouterReasoningDetailsFormat) {
		t.Fatalf("reasoning state format = %q, state = %s", thinking.ProviderStateFormat, thinking.ProviderState)
	}
	assertJSONEqual(t, thinking.ProviderState, []byte(details))
	assertRawJSONOrder(t, thinking.ProviderState, []byte(details))

	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
		Messages: content.AgenticMessages{
			response.Message,
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call-1"},
		},
	}
	encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("OpenRouter EncodeRequest() error = %v", err)
	}
	openRouterBody, _ := io.ReadAll(encoded.Body)
	var envelope struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(openRouterBody, &envelope); err != nil {
		t.Fatalf("OpenRouter body JSON = %v", err)
	}
	assertJSONEqual(t, envelope.Messages[0]["reasoning_details"], []byte(details))
	assertRawJSONOrder(t, envelope.Messages[0]["reasoning_details"], []byte(details))

	shared, err := (openaiapi.Codec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("OpenAI EncodeRequest() error = %v", err)
	}
	sharedBody, _ := io.ReadAll(shared.Body)
	if strings.Contains(string(sharedBody), "reasoning_details") || strings.Contains(string(sharedBody), "opaque") {
		t.Fatalf("shared OpenAI encoding leaked OpenRouter state: %s", sharedBody)
	}
}

func TestFragmentedStreamReasoningDetailsRoundTrip(t *testing.T) {
	t.Parallel()

	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"first","signature":"sig-1"}]}}]}` + "\n\n",
		`data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","index":1,"format":"anthropic-claude-v1","text":" second","signature":"sig-2"}]}}]}` + "\n\n",
		`data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","index":2,"format":"anthropic-claude-v1","data":"opaque","signature":"sig-3"}]}}]}` + "\n\n",
		`data: {"model":"anthropic/claude-sonnet-4","choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	reader, err := (requestCodec{}).DecodeStream(&http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       &chunkedReadCloser{chunks: sse},
	})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	var thinkingText string
	var finalState json.RawMessage
	var finalFormat string
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("Next() error = %v", nextErr)
		}
		if thinking, ok := chunk.(*content.ThinkingChunk); ok {
			thinkingText += thinking.Thinking
			if len(thinking.ProviderState) > 0 {
				finalState = append(finalState[:0], thinking.ProviderState...)
				finalFormat = thinking.ProviderStateFormat
			}
		}
	}
	block := content.NewThinkingBlock(thinkingText, "", finalState, finalFormat)
	if block.Thinking != "first second" {
		t.Errorf("stream thinking = %q, want fragmented text", block.Thinking)
	}
	if !block.ReplayableAs(openRouterReasoningDetailsFormat) {
		t.Fatalf("stream reasoning state format = %q, state = %s", block.ProviderStateFormat, block.ProviderState)
	}
	wantDetails := []byte(`[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"first","signature":"sig-1"},{"type":"reasoning.text","index":1,"format":"anthropic-claude-v1","text":" second","signature":"sig-2"},{"type":"reasoning.encrypted","index":2,"format":"anthropic-claude-v1","data":"opaque","signature":"sig-3"}]`)
	assertJSONEqual(t, block.ProviderState, wantDetails)
	assertRawJSONOrder(t, block.ProviderState, wantDetails)
	result, ok := reader.Result()
	if !ok || result.Model != "anthropic/claude-sonnet-4" || result.FinishReason != "tool_use" {
		t.Fatalf("Result() = %+v, %v; want preserved model/tool_use metadata", result, ok)
	}

	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{block, &content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)}}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call-1"},
		},
	}
	encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body, _ := io.ReadAll(encoded.Body)
	if !strings.Contains(string(body), `"data":"opaque"`) {
		t.Fatalf("re-encoded request lost accumulated reasoning details: %s", body)
	}
}

func TestStreamReasoningContentCoexistsWithReasoningDetails(t *testing.T) {
	t.Parallel()

	wantDetails := []byte(`[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"readable","signature":"sig"}]`)
	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_content":"readable","reasoning_details":` + string(wantDetails) + `}}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	reader, err := (requestCodec{}).DecodeStream(&http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       &chunkedReadCloser{chunks: sse},
	})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	thinking, ok := chunk.(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("chunk = %T, want *content.ThinkingChunk", chunk)
	}
	if thinking.Thinking != "readable" {
		t.Fatalf("Thinking = %q, want readable", thinking.Thinking)
	}
	if thinking.ProviderStateFormat != openRouterReasoningDetailsFormat {
		t.Fatalf("ProviderStateFormat = %q", thinking.ProviderStateFormat)
	}
	assertRawJSONOrder(t, thinking.ProviderState, wantDetails)

	block := content.NewThinkingBlock(thinking.Thinking, "", thinking.ProviderState, thinking.ProviderStateFormat)
	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{block, &content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)}},
		}}},
	}
	encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body, _ := io.ReadAll(encoded.Body)
	var envelope struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("request JSON = %v", err)
	}
	assertRawJSONOrder(t, envelope.Messages[0]["reasoning_details"], wantDetails)
}

// TestStreamEncryptedReasoningDetailsWithToolCallKeepsBoth pins the Claude
// tool-loop shape: one delta carrying encrypted (unreadable) reasoning details
// beside the tool call it decided on. Both the reasoning state and the tool
// call must survive, with the reasoning ahead of the tool use.
func TestStreamEncryptedReasoningDetailsWithToolCallKeepsBoth(t *testing.T) {
	t.Parallel()

	wantDetails := []byte(`[{"type":"reasoning.encrypted","index":0,"format":"anthropic-claude-v1","data":"opaque","signature":"sig"}]`)
	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_details":` + string(wantDetails) + `,"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}` + "\n\n",
		`data: {"model":"anthropic/claude-sonnet-4","choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	chunks := collectChunks(t, sse)
	if len(chunks) != 2 {
		t.Fatalf("chunks = %#v, want a thinking chunk and a tool-use chunk", chunks)
	}
	thinking, ok := chunks[0].(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("first chunk = %T, want *content.ThinkingChunk", chunks[0])
	}
	if thinking.Thinking != "" {
		t.Errorf("Thinking = %q, want no synthetic text on an encrypted-only detail", thinking.Thinking)
	}
	if thinking.ProviderStateFormat != openRouterReasoningDetailsFormat {
		t.Errorf("ProviderStateFormat = %q, want %q", thinking.ProviderStateFormat, openRouterReasoningDetailsFormat)
	}
	assertRawJSONOrder(t, thinking.ProviderState, wantDetails)
	toolUse, ok := chunks[1].(*content.ToolUseChunk)
	if !ok {
		t.Fatalf("second chunk = %T, want *content.ToolUseChunk", chunks[1])
	}
	if toolUse.ID != "call-1" || toolUse.Name != "lookup" {
		t.Errorf("tool-use chunk = %+v, want the lookup call", toolUse)
	}
}

// TestStreamEncryptedReasoningDetailsWithContentKeepsBoth is the assistant-text
// twin of the tool-call case: the visible text must not be displaced by the
// reasoning state travelling in the same delta.
func TestStreamEncryptedReasoningDetailsWithContentKeepsBoth(t *testing.T) {
	t.Parallel()

	wantDetails := []byte(`[{"type":"reasoning.encrypted","index":0,"format":"anthropic-claude-v1","data":"opaque","signature":"sig"}]`)
	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_details":` + string(wantDetails) + `,"content":"visible"}}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	chunks := collectChunks(t, sse)
	if len(chunks) != 2 {
		t.Fatalf("chunks = %#v, want a thinking chunk and a text chunk", chunks)
	}
	thinking, ok := chunks[0].(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("first chunk = %T, want *content.ThinkingChunk", chunks[0])
	}
	assertRawJSONOrder(t, thinking.ProviderState, wantDetails)
	text, ok := chunks[1].(*content.TextChunk)
	if !ok {
		t.Fatalf("second chunk = %T, want *content.TextChunk", chunks[1])
	}
	if text.Text != "visible" {
		t.Errorf("Text = %q, want visible", text.Text)
	}
}

// TestTransformedStreamCarriesNoInBandSentinel pins that reasoning state never
// rides inside a content field. The transformed SSE is what any wrapper-less
// reader of the body sees, so a synthetic marker there is observable corruption.
func TestTransformedStreamCarriesNoInBandSentinel(t *testing.T) {
	t.Parallel()

	source := `data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","index":0,"format":"anthropic-claude-v1","data":"opaque","signature":"sig"}]}}]}` + "\n\n"
	transformed, err := io.ReadAll(&reasoningResponseBody{source: &chunkedReadCloser{chunks: []string{source}}})
	if err != nil {
		t.Fatalf("read transformed body: %v", err)
	}
	for _, marker := range []string{"openrouter-reasoning-state", "\x00", "\\u0000"} {
		if bytes.Contains(transformed, []byte(marker)) {
			t.Fatalf("transformed SSE carries in-band marker %q: %q", marker, transformed)
		}
	}
	if string(transformed) != source {
		t.Fatalf("transformed SSE = %q, want the upstream line unmodified", transformed)
	}
}

// TestUpstreamReasoningTextIsNeverStripped pins that no reasoning text is
// deleted by exact-string match: a model emitting the bytes a former in-band
// marker happened to use must still see them decoded as its own reasoning.
func TestUpstreamReasoningTextIsNeverStripped(t *testing.T) {
	t.Parallel()

	upstream := "\x00openrouter-reasoning-state"
	encoded, err := json.Marshal(upstream)
	if err != nil {
		t.Fatalf("marshal upstream reasoning: %v", err)
	}
	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_content":` + string(encoded) + `}}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	chunks := collectChunks(t, sse)
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v, want a single thinking chunk", chunks)
	}
	thinking, ok := chunks[0].(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("chunk = %T, want *content.ThinkingChunk", chunks[0])
	}
	if thinking.Thinking != upstream {
		t.Fatalf("Thinking = %q, want the upstream text preserved verbatim", thinking.Thinking)
	}
}

func collectChunks(t *testing.T, sse []string) []content.Chunk {
	t.Helper()
	reader, err := (requestCodec{}).DecodeStream(&http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       &chunkedReadCloser{chunks: sse},
	})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	defer reader.Close()
	var chunks []content.Chunk
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			return chunks
		}
		if nextErr != nil {
			t.Fatalf("Next() error = %v", nextErr)
		}
		chunks = append(chunks, chunk)
	}
}

type chunkedReadCloser struct {
	chunks []string
}

func (r *chunkedReadCloser) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.chunks[0])
	r.chunks[0] = r.chunks[0][n:]
	if r.chunks[0] == "" {
		r.chunks = r.chunks[1:]
	}
	return n, nil
}

func (*chunkedReadCloser) Close() error { return nil }

func TestReasoningDetailsWrongFormatIsNotReplayed(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "m"),
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{content.NewThinkingBlock("thinking", "", json.RawMessage(`[{"data":"foreign"}]`), "another-dialect")},
		}}},
	}
	encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body, _ := io.ReadAll(encoded.Body)
	if strings.Contains(string(body), "reasoning_details") || strings.Contains(string(body), "foreign") {
		t.Fatalf("wrong-format provider state leaked: %s", body)
	}
}

// TestMalformedReasoningStateIsNotReplayed pins the egress shape check.
// OpenRouter documents reasoning_details as an array of records and rejects
// anything else with a 400, and a state can reach the encoder from a store
// round-trip, a compaction, or a hand-built message — so a payload that only
// carries the right format tag must degrade to absent, never reach the wire.
func TestMalformedReasoningStateIsNotReplayed(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		state json.RawMessage
	}{
		{name: "object", state: json.RawMessage(`{"not":"an array"}`)},
		{name: "scalar string", state: json.RawMessage(`"scalar"`)},
		{name: "scalar number", state: json.RawMessage(`7`)},
		{name: "null", state: json.RawMessage(`null`)},
		{name: "empty array", state: json.RawMessage(`[]`)},
		{name: "array of scalars", state: json.RawMessage(`["scalar"]`)},
		{name: "array with null record", state: json.RawMessage(`[{"type":"reasoning.text"},null]`)},
		{name: "empty record", state: json.RawMessage(`[{}]`)},
		{name: "text record missing text", state: json.RawMessage(`[{"type":"reasoning.text","id":null,"format":"unknown"}]`)},
		{name: "summary record missing summary", state: json.RawMessage(`[{"type":"reasoning.summary","id":null,"format":"unknown"}]`)},
		{name: "encrypted record missing data", state: json.RawMessage(`[{"type":"reasoning.encrypted","id":null,"format":"unknown"}]`)},
		{name: "record format has wrong type", state: json.RawMessage(`[{"type":"reasoning.text","id":null,"format":7,"text":"thinking"}]`)},
		{name: "truncated", state: json.RawMessage(`[`)},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			req := inference.Request{
				Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
				Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
					Role: content.RoleAssistant,
					Blocks: []content.Block{
						content.NewThinkingBlock("thinking", "", testCase.state, openRouterReasoningDetailsFormat),
						&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)},
					},
				}}},
			}
			encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v, want a request with the unusable state omitted", err)
			}
			body, err := io.ReadAll(encoded.Body)
			if err != nil {
				t.Fatalf("read encoded body: %v", err)
			}
			if strings.Contains(string(body), "reasoning_details") {
				t.Fatalf("malformed provider state reached the wire: %s", body)
			}
			// Degrading the state must not cost the turn itself.
			if !strings.Contains(string(body), `"tool_calls"`) {
				t.Fatalf("degraded request lost the assistant turn: %s", body)
			}
		})
	}
}

// TestWellFormedReasoningStateStillReplays guards the validator against
// over-rejection: the documented shape must pass through untouched.
func TestWellFormedReasoningStateStillReplays(t *testing.T) {
	t.Parallel()

	details := json.RawMessage(`[{"type":"reasoning.encrypted","index":0,"format":"anthropic-claude-v1","data":"opaque","signature":"sig"}]`)
	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
		Messages: content.AgenticMessages{&content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{content.NewThinkingBlock("", "", details, openRouterReasoningDetailsFormat)},
		}}},
	}
	encoded, err := (requestCodec{}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body, _ := io.ReadAll(encoded.Body)
	var envelope struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("request JSON = %v", err)
	}
	assertRawJSONOrder(t, envelope.Messages[0]["reasoning_details"], details)
}

// TestSharedEncoderEmitsOneWireMessagePerNeutralMessage pins the
// cross-repository coupling replayReasoningDetails depends on: the shared
// OpenAI encoder emits exactly one wire message per neutral message, plus one
// leading system message when Request.System is set. Reasoning state is
// positional, and OpenRouter treats a reasoning sequence attached to the wrong
// assistant turn as an unrecoverable ordering violation, so a change to that
// cardinality must break here rather than in production.
func TestSharedEncoderEmitsOneWireMessagePerNeutralMessage(t *testing.T) {
	t.Parallel()

	messages := content.AgenticMessages{
		&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{&content.TextBlock{Text: "rules"}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
			&content.TextBlock{Text: "look"},
			&content.ImageBlock{MediaType: "image/png", Source: content.ImageSource{URL: "https://example.test/i.png"}},
		}}},
		&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
			content.NewThinkingBlock("thinking", "", json.RawMessage(`[{"type":"reasoning.encrypted","data":"opaque"}]`), openRouterReasoningDetailsFormat),
			&content.TextBlock{Text: "calling"},
			&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)},
		}}},
		&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call-1"},
		&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "done"}}}},
	}
	for _, system := range []string{"", "you are a test"} {
		req := inference.Request{
			Model:    model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4", model.WithImages(), model.WithTools()),
			System:   system,
			Messages: messages,
		}
		built, err := openaiapi.BuildChatRequest(req, false)
		if err != nil {
			t.Fatalf("BuildChatRequest(System=%q) error = %v", system, err)
		}
		want := len(messages)
		if system != "" {
			want++
		}
		if len(built.Messages) != want {
			t.Fatalf("BuildChatRequest(System=%q) emitted %d wire messages, want %d: the positional reasoning replay in replayReasoningDetails no longer holds", system, len(built.Messages), want)
		}
	}
}

// TestReasoningReplayRejectsWireCardinalityMismatch pins the runtime half of
// the same guard: if the encoded body ever stops lining up one-to-one, the
// replay fails loudly instead of pinning encrypted signatures to the wrong turn.
func TestReasoningReplayRejectsWireCardinalityMismatch(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: model.CustomModel("openrouter", model.APIFormatOpenAI, "https://example.test", "anthropic/claude-sonnet-4"),
		Messages: content.AgenticMessages{
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewThinkingBlock("thinking", "", json.RawMessage(`[{"type":"reasoning.encrypted","data":"opaque"}]`), openRouterReasoningDetailsFormat),
				&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)},
			}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "result"}}}, ToolUseID: "call-1"},
		},
	}
	cases := []struct {
		name     string
		system   string
		messages string
	}{
		{name: "extra wire message", messages: `[{"role":"assistant"},{"role":"assistant"},{"role":"tool"}]`},
		{name: "missing wire message", messages: `[{"role":"assistant"}]`},
		{name: "unexpected system message", system: "rules", messages: `[{"role":"assistant"},{"role":"tool"}]`},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			mismatched := req
			mismatched.System = testCase.system
			body := map[string]json.RawMessage{"messages": json.RawMessage(testCase.messages)}
			err := replayReasoningDetails(body, mismatched)
			if err == nil {
				t.Fatalf("replayReasoningDetails() = nil, want a cardinality error; body = %s", body["messages"])
			}
			if !strings.Contains(err.Error(), "openrouter:") {
				t.Errorf("error = %v, want an openrouter-tagged error", err)
			}
		})
	}
}

func assertJSONEqual(t *testing.T, got, want []byte) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("got JSON = %s: %v", got, err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("want JSON = %s: %v", want, err)
	}
	gotCanonical, _ := json.Marshal(gotValue)
	wantCanonical, _ := json.Marshal(wantValue)
	if string(gotCanonical) != string(wantCanonical) {
		t.Fatalf("JSON = %s, want %s", gotCanonical, wantCanonical)
	}
}

func assertRawJSONOrder(t *testing.T, got, want []byte) {
	t.Helper()
	var gotCompact, wantCompact bytes.Buffer
	if err := json.Compact(&gotCompact, got); err != nil {
		t.Fatalf("compact got JSON: %v", err)
	}
	if err := json.Compact(&wantCompact, want); err != nil {
		t.Fatalf("compact want JSON: %v", err)
	}
	if gotCompact.String() != wantCompact.String() {
		t.Fatalf("raw JSON order = %s, want %s", gotCompact.String(), wantCompact.String())
	}
}
