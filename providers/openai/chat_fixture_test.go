package openai_test

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
	chat "github.com/looprig/inference/codec/openaiapi"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openai"
)

// Chat Completions fixture suite. Every test here follows the same three
// steps, in this order and no other: gate-validate the fixture against
// OpenAI's published schema, feed the validated bytes to the real
// providers/openai client over an httptest server, then assert the neutral
// result. The gate call is inside chatFixture/chatStreamFixture, so it is not
// possible to reach the bytes without it.

// serveBody starts a server that answers every request with body and
// contentType. The client under test is the real one; only the wire is faked.
func serveBody(t *testing.T, contentType string, body []byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func chatModel(t *testing.T, baseURL string) model.Model {
	t.Helper()
	return model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAI,
		baseURL+"/v1",
		"gpt-4.1",
	)
}

// invokeChatFixture gate-validates the named fixture and decodes it through
// openai.New(...).Invoke.
func invokeChatFixture(t *testing.T, name string) (*inference.Response, error) {
	t.Helper()
	body := chatFixture(t, name)
	srv := serveBody(t, "application/json", body)
	selected := chatModel(t, srv.URL)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	return client.Invoke(context.Background(), inference.Request{Model: selected})
}

func mustInvokeChatFixture(t *testing.T, name string) *inference.Response {
	t.Helper()
	resp, err := invokeChatFixture(t, name)
	if err != nil {
		t.Fatalf("Invoke(%s) error = %v", name, err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke(%s) returned %+v, want a decoded message", name, resp)
	}
	return resp
}

// streamOutcome is everything a stream fixture produces: the chunks in order,
// the terminal result, and the error (if any) that ended it.
type streamOutcome struct {
	chunks []content.Chunk
	result stream.StreamResult
	ok     bool
	err    error
}

func drain(t *testing.T, reader *stream.StreamReader[content.Chunk]) streamOutcome {
	t.Helper()
	var out streamOutcome
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			out.err = err
			break
		}
		out.chunks = append(out.chunks, chunk)
	}
	out.result, out.ok = reader.Result()
	return out
}

// streamChatFixture gate-validates every frame of the named SSE fixture and
// decodes the stream through the real client.
func streamChatFixture(t *testing.T, name string) streamOutcome {
	t.Helper()
	body := chatStreamFixture(t, name)
	srv := serveBody(t, "text/event-stream", body)
	selected := chatModel(t, srv.URL)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream(%s) error = %v", name, err)
	}
	defer func() { _ = reader.Close() }()
	return drain(t, reader)
}

// --- block/chunk assertion helpers ---------------------------------------

func wantText(t *testing.T, block content.Block, text string) {
	t.Helper()
	b, ok := block.(*content.TextBlock)
	if !ok {
		t.Fatalf("block = %#v, want *content.TextBlock", block)
	}
	if b.Text != text {
		t.Errorf("text = %q, want %q", b.Text, text)
	}
}

func wantThinking(t *testing.T, block content.Block, thinking string) *content.ThinkingBlock {
	t.Helper()
	b, ok := block.(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("block = %#v, want *content.ThinkingBlock", block)
	}
	if b.Thinking != thinking {
		t.Errorf("thinking = %q, want %q", b.Thinking, thinking)
	}
	return b
}

func wantToolUse(t *testing.T, block content.Block, id, name string) *content.ToolUseBlock {
	t.Helper()
	b, ok := block.(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("block = %#v, want *content.ToolUseBlock", block)
	}
	if b.ID != id || b.Name != name {
		t.Errorf("tool use = id %q name %q, want id %q name %q", b.ID, b.Name, id, name)
	}
	return b
}

func wantTextChunk(t *testing.T, chunk content.Chunk, text string) {
	t.Helper()
	c, ok := chunk.(*content.TextChunk)
	if !ok {
		t.Fatalf("chunk = %#v, want *content.TextChunk", chunk)
	}
	if c.Text != text {
		t.Errorf("text chunk = %q, want %q", c.Text, text)
	}
}

func wantRefusalChunk(t *testing.T, chunk content.Chunk, text string) {
	t.Helper()
	c, ok := chunk.(*content.RefusalChunk)
	if !ok {
		t.Fatalf("chunk = %#v, want *content.RefusalChunk", chunk)
	}
	if c.Text != text {
		t.Errorf("refusal chunk = %q, want %q", c.Text, text)
	}
}

func wantThinkingChunk(t *testing.T, chunk content.Chunk, thinking string) *content.ThinkingChunk {
	t.Helper()
	c, ok := chunk.(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("chunk = %#v, want *content.ThinkingChunk", chunk)
	}
	if c.Thinking != thinking {
		t.Errorf("thinking chunk = %q, want %q", c.Thinking, thinking)
	}
	return c
}

func wantToolChunk(t *testing.T, chunk content.Chunk, index int, id, name, input string) {
	t.Helper()
	c, ok := chunk.(*content.ToolUseChunk)
	if !ok {
		t.Fatalf("chunk = %#v, want *content.ToolUseChunk", chunk)
	}
	if c.Index != index || c.ID != id || c.Name != name || c.InputJSON != input {
		t.Errorf("tool chunk = %+v, want index %d id %q name %q input %q", c, index, id, name, input)
	}
}

// --- non-streaming --------------------------------------------------------

func TestChatPlainText(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "plain_text.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one text block", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "The capital of France is Paris.")
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}
	if resp.Model != "gpt-4.1" {
		t.Errorf("model = %q, want gpt-4.1", resp.Model)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 19 || resp.Usage.OutputTokens != 8 {
		t.Errorf("usage = %+v, want input 19 output 8", resp.Usage)
	}
}

// TestChatMultiChoice pins that only choice 0 reaches the neutral response.
// The neutral contract has one assistant turn, and the encoder never asks for
// n>1; a decoder that started concatenating choices would silently invent
// content the caller never requested.
func TestChatMultiChoice(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "multi_choice.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want only choice 0's text", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "First candidate answer.")
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("finish reason = %v, want choice 0's stop", resp.FinishReason)
	}
}

func TestChatFinishReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture string
		want    stream.FinishReason
	}{
		{"plain_text.json", stream.FinishReasonStop},
		{"finish_length.json", stream.FinishReasonLength},
		{"tool_call_single.json", stream.FinishReasonToolUse},
		{"finish_content_filter.json", stream.FinishReasonContentFilter},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()
			resp := mustInvokeChatFixture(t, tc.fixture)
			if resp.FinishReason != tc.want {
				t.Errorf("finish reason = %v, want %v", resp.FinishReason, tc.want)
			}
		})
	}
}

// TestChatContentFilterYieldsNoBlocks: a filtered completion carries
// "content": null, which is a legal ChatCompletionResponseMessage. The neutral
// signal is the finish reason, not an empty text block.
func TestChatContentFilterYieldsNoBlocks(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "finish_content_filter.json")

	if len(resp.Message.Blocks) != 0 {
		t.Errorf("blocks = %#v, want none for a null-content filtered choice", resp.Message.Blocks)
	}
	if resp.FinishReason != stream.FinishReasonContentFilter {
		t.Errorf("finish reason = %v, want content filter", resp.FinishReason)
	}
}

func TestChatSingleToolCall(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "tool_call_single.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one tool-use block", resp.Message.Blocks)
	}
	wantToolUse(t, resp.Message.Blocks[0], "call_weather_1", "get_weather")
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %v, want tool use", resp.FinishReason)
	}
}

// TestChatToolCallArgumentsDecodeToTheObject pins the fix for a shipped
// data-corruption defect, and REPLACES a test that asserted it.
//
// OpenAI's schema declares function.arguments as a STRING carrying JSON
// ("{\"city\":\"Paris\"}"). openaiapi used to store that literal in
// ToolUseBlock.Input unchanged, and its encoder quotes Input again on replay,
// so the follow-up request carried "\"{\\\"city\\\":\\\"Paris\\\"}\"" — two
// gateways rejected it by name ("Assistant tool call function.arguments must
// be a JSON object"), and the rest accepted the corrupted arguments silently.
// Input is the arguments OBJECT, as in every sibling codec.
func TestChatToolCallArgumentsDecodeToTheObject(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "tool_call_single.json")

	block := wantToolUse(t, resp.Message.Blocks[0], "call_weather_1", "get_weather")
	if string(block.Input) != `{"city":"Paris"}` {
		t.Errorf("tool input = %s, want the arguments object", block.Input)
	}
}

// TestChatToolCallReplayMatchesTheServersBytes runs the two-request cycle a
// real agent runs — decode a tool call, then send the decoded assistant turn
// back — and asserts the `arguments` we replay are byte-identical to the ones
// the server sent. That property is what was missing: each half looked right
// on its own, and only the round trip shows the extra layer of quoting.
//
// The captured body also goes through MustValidateRequest, which now carries a
// semantic check the schema cannot express (arguments must contain an object);
// `arguments` is spec-typed `string`, so the schema alone accepted the
// corrupted body.
func TestChatToolCallReplayMatchesTheServersBytes(t *testing.T) {
	t.Parallel()

	fixture := chatFixture(t, "tool_call_single.json")
	sent := toolArgumentsOf(t, fixture, "choices", 0)

	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		captured = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	t.Cleanup(srv.Close)

	selected := chatModel(t, srv.URL)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	first, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	if _, err := client.Invoke(context.Background(), inference.Request{
		Model:    selected,
		Messages: content.AgenticMessages{first.Message},
	}); err != nil {
		t.Fatalf("replay Invoke() error = %v", err)
	}

	conformance.MustValidateRequest(t, "openai", "chat_completion_request", captured)
	if replayed := toolArgumentsOf(t, captured, "messages", 0); replayed != sent {
		t.Errorf("replayed arguments = %s, want the server's own bytes %s", replayed, sent)
	}
}

// toolArgumentsOf reads the raw `arguments` member of the first tool call in
// the first element of the named array — "choices" for a response body (where
// the calls hang off `message`), "messages" for a request body — without
// interpreting it.
func toolArgumentsOf(t *testing.T, body []byte, array string, index int) string {
	t.Helper()
	type wireToolCall struct {
		Function struct {
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	type entry struct {
		ToolCalls []wireToolCall `json:"tool_calls"`
		Message   *struct {
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"message"`
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	raw, ok := envelope[array]
	if !ok {
		t.Fatalf("body has no %s: %s", array, body)
	}
	var entries []entry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatalf("unmarshal %s: %v", array, err)
	}
	if len(entries) <= index {
		t.Fatalf("body has no %s[%d]: %s", array, index, body)
	}
	calls := entries[index].ToolCalls
	if entries[index].Message != nil {
		calls = entries[index].Message.ToolCalls
	}
	if len(calls) == 0 {
		t.Fatalf("%s[%d] has no tool call: %s", array, index, body)
	}
	return string(calls[0].Function.Arguments)
}

func TestChatParallelToolCallsPreserveOrder(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "tool_calls_parallel.json")

	if len(resp.Message.Blocks) != 3 {
		t.Fatalf("blocks = %#v, want three tool-use blocks", resp.Message.Blocks)
	}
	wantToolUse(t, resp.Message.Blocks[0], "call_a", "get_weather")
	wantToolUse(t, resp.Message.Blocks[1], "call_b", "get_weather")
	wantToolUse(t, resp.Message.Blocks[2], "call_c", "get_time")
}

func TestChatToolCallWithEmptyArguments(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "tool_call_empty_arguments.json")

	block := wantToolUse(t, resp.Message.Blocks[0], "call_noargs", "list_files")
	// A no-argument call arrives as `"arguments":""`. Input is an arguments
	// object, and "" is not one, so it normalizes to the empty object — the
	// same value the encoder replays. This previously asserted `""`, the
	// second half of the double-encoding defect.
	if string(block.Input) != `{}` {
		t.Errorf("tool input = %s, want the empty object", block.Input)
	}
}

// TestChatRefusalSurfacesAsARefusalBlock pins the message-level `refusal`
// string, which ChatCompletionResponseMessage marks required and which once
// decoded as an empty assistant turn with a clean "stop".
//
// This test previously asserted the interim mapping — a *content.TextBlock plus
// a content_filter finish reason — which predated content.RefusalBlock. Both
// halves were defects: the block type now carries the "declined" signal per
// block, so the finish reason reports the wire's own value, and a genuine
// content_filter response stays distinguishable from a model refusal.
func TestChatRefusalSurfacesAsARefusalBlock(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "refusal.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want the refusal", resp.Message.Blocks)
	}
	refusal, ok := resp.Message.Blocks[0].(*content.RefusalBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want *content.RefusalBlock", resp.Message.Blocks[0])
	}
	if refusal.Text != "I'm sorry, I can't help with that request." {
		t.Errorf("refusal text = %q, want it verbatim", refusal.Text)
	}
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("finish reason = %v, want the wire's own stop", resp.FinishReason)
	}
}

// TestChatLogprobsAreIgnoredWithoutDisturbingContent proves the presence of a
// fully-populated choice-level logprobs object (required members `content` and
// `refusal`, each token requiring token/logprob/bytes/top_logprobs) does not
// perturb the decode.
func TestChatLogprobsAreIgnoredWithoutDisturbingContent(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "logprobs.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one text block", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "Paris")
}

// TestChatUsageDetails pins the cached/reasoning normalization: OpenAI reports
// prompt_tokens as the GROSS total including cached tokens, so the neutral
// InputTokens is prompt_tokens minus the cached breakdown.
func TestChatUsageDetails(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "usage_details.json")

	u := resp.Usage
	if u == nil {
		t.Fatal("usage = nil, want normalized usage")
	}
	if u.InputTokens != 128 || u.CacheReadTokens != 896 || u.OutputTokens != 96 || u.ReasoningTokens != 64 {
		t.Errorf("usage = %+v, want input 128 (1024-896) cacheRead 896 output 96 reasoning 64", u)
	}
	if resp.Message.Usage == nil || *resp.Message.Usage != *u {
		t.Errorf("message usage = %+v, want a copy of %+v", resp.Message.Usage, u)
	}
}

// TestChatReasoningContentThenText pins the block ORDER for a message that
// carries both the compatible-gateway `reasoning_content` extension and text.
func TestChatReasoningContentThenText(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "reasoning_content.json")

	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want thinking then text", resp.Message.Blocks)
	}
	wantThinking(t, resp.Message.Blocks[0], "Let me add six and seven groups.")
	wantText(t, resp.Message.Blocks[1], "42.")
}

func TestChatTextAndToolCallsCoexist(t *testing.T) {
	t.Parallel()
	resp := mustInvokeChatFixture(t, "text_and_tool_calls.json")

	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want text then tool use", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "Let me look that up.")
	wantToolUse(t, resp.Message.Blocks[1], "call_lookup", "search")
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %v, want tool use", resp.FinishReason)
	}
}

// TestChatErrorEnvelopeOnHTTP200 pins the repair for the OpenRouter-style
// failure shape: an OpenAI-compatible gateway reporting an upstream error
// inside a 200 body. Before the fix the caller received a successful, empty
// assistant turn; a regression would look exactly like that again.
//
// Fixture note: the bare {"error":{...}} body those gateways actually send is
// NOT a legal CreateChatCompletionResponse (choices/created/id/model/object are
// required), so the gated fixture carries the full envelope with an empty
// choices array — which is spec-legal and reaches the same decoder branch.
func TestChatErrorEnvelopeOnHTTP200(t *testing.T) {
	t.Parallel()

	t.Run("string code", func(t *testing.T) {
		t.Parallel()
		resp, err := invokeChatFixture(t, "error_envelope.json")
		if resp != nil {
			t.Fatalf("Invoke() = %+v, want no response", resp)
		}
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %T %v, want *failure.APIError", err, err)
		}
		if apiErr.Code != "rate_limit_exceeded" {
			t.Errorf("code = %q, want rate_limit_exceeded", apiErr.Code)
		}
	})

	t.Run("numeric code carries the smuggled HTTP status", func(t *testing.T) {
		t.Parallel()
		_, err := invokeChatFixture(t, "error_envelope_numeric_code.json")
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("error = %T %v, want *failure.APIError", err, err)
		}
		if apiErr.Status != http.StatusTooManyRequests {
			t.Errorf("status = %d, want 429 recovered from the numeric error code", apiErr.Status)
		}
	})
}

// TestChatEmptyChoicesWithoutErrorIsStillAFailure keeps the no-choices guard
// distinct from the error-envelope branch above.
func TestChatEmptyChoicesWithoutErrorIsStillAFailure(t *testing.T) {
	t.Parallel()

	_, err := invokeChatFixture(t, "empty_choices.json")
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %T %v, want *failure.APIError", err, err)
	}
}

// --- streaming ------------------------------------------------------------

func TestChatStreamText(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_text.sse")

	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want three text deltas", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Hello")
	wantTextChunk(t, out.chunks[1], ", world")
	wantTextChunk(t, out.chunks[2], "!")
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want stop", out.result, out.ok)
	}
}

// TestChatStreamRoleOnlyFirstChunkYieldsNothing pins that the mandatory
// role-only opener (and the bare finish chunk) produce no content.
func TestChatStreamRoleOnlyFirstChunkYieldsNothing(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_role_only.sse")

	if len(out.chunks) != 0 {
		t.Fatalf("chunks = %#v, want none", out.chunks)
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want stop", out.result, out.ok)
	}
}

func TestChatStreamToolCallAccumulatesByIndex(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_tool_call_single.sse")

	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want a seed plus two argument fragments", out.chunks)
	}
	wantToolChunk(t, out.chunks[0], 0, "call_stream_1", "get_weather", "")
	wantToolChunk(t, out.chunks[1], 0, "", "", `{"city":`)
	wantToolChunk(t, out.chunks[2], 0, "", "", `"Paris"}`)
	if out.result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %v, want tool use", out.result.FinishReason)
	}
}

func TestChatStreamParallelToolCallsKeepTheirIndexes(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_tool_calls_parallel.sse")

	if len(out.chunks) != 4 {
		t.Fatalf("chunks = %#v, want two seeds and two argument fragments", out.chunks)
	}
	// A single delta carrying two tool-call entries yields both, in wire order.
	wantToolChunk(t, out.chunks[0], 0, "call_p0", "get_weather", "")
	wantToolChunk(t, out.chunks[1], 1, "call_p1", "get_time", "")
	wantToolChunk(t, out.chunks[2], 0, "", "", `{"city":"Paris"}`)
	wantToolChunk(t, out.chunks[3], 1, "", "", `{"zone":"UTC"}`)
}

// TestChatStreamDeltaCarryingContentAndToolCalls pins one half of the
// precedence repair: content and tool_calls are independent members of a
// delta, so both must be emitted from the same event.
func TestChatStreamDeltaCarryingContentAndToolCalls(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_content_and_tool_calls.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want text AND the tool call from one delta", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Looking it up.")
	wantToolChunk(t, out.chunks[1], 0, "call_both", "search", `{"q":"x"}`)
}

// TestChatStreamDeltaCarryingReasoningAndToolCalls pins the DEFECT THAT WAS
// JUST REPAIRED: decodeEvent used to `return` on the first populated member,
// so a delta carrying reasoning_content plus tool_calls emitted the thinking
// chunk and silently DISCARDED the tool call — the model's decision to call a
// tool vanished. Reasoning/text/tool-calls is an emission order, never a
// precedence. If this test starts reporting one chunk, that regression is back.
func TestChatStreamDeltaCarryingReasoningAndToolCalls(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_reasoning_and_tool_calls.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want thinking AND the tool call from one delta", out.chunks)
	}
	wantThinkingChunk(t, out.chunks[0], "I should call the weather tool.")
	wantToolChunk(t, out.chunks[1], 0, "call_after_reasoning", "get_weather", `{"city":"Oslo"}`)
}

func TestChatStreamReasoningContentDeltas(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_reasoning_content.sse")

	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want two thinking deltas and one text delta", out.chunks)
	}
	wantThinkingChunk(t, out.chunks[0], "First, ")
	wantThinkingChunk(t, out.chunks[1], "then.")
	wantTextChunk(t, out.chunks[2], "Done.")
}

// TestChatStreamRefusalDeltasSurfaceAsRefusalChunks pins the streaming half:
// ChatCompletionStreamResponseDelta declares its own `refusal` delta channel,
// and the stream must reconstruct the same block — and the same finish reason —
// the non-streaming decoder produces for the same response. The fixture's
// first delta carries an empty refusal, which yields no chunk: the Chat delta's
// `refusal` is a plain string, so an empty one is indistinguishable from the
// absent member every non-refusal delta carries.
func TestChatStreamRefusalDeltasSurfaceAsRefusalChunks(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_refusal.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want the two non-empty refusal deltas", out.chunks)
	}
	wantRefusalChunk(t, out.chunks[0], "I'm sorry, ")
	wantRefusalChunk(t, out.chunks[1], "I can't help with that.")
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want the wire's own stop", out.result, out.ok)
	}
}

// TestChatStreamUsageOnlyFinalChunk pins the terminal chunk whose `choices` is
// an EMPTY array and whose only payload is usage.
func TestChatStreamUsageOnlyFinalChunk(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_usage_final_chunk.sse")

	if len(out.chunks) != 1 {
		t.Fatalf("chunks = %#v, want one text delta", out.chunks)
	}
	if !out.ok {
		t.Fatal("stream result not available; the [DONE] sentinel was not observed")
	}
	u := out.result.Usage
	if u == nil || u.InputTokens != 128 || u.CacheReadTokens != 384 || u.OutputTokens != 40 || u.ReasoningTokens != 24 {
		t.Errorf("usage = %+v, want input 128 (512-384) cacheRead 384 output 40 reasoning 24", u)
	}
	if out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("finish reason = %v, want the stop carried by the earlier chunk", out.result.FinishReason)
	}
	if out.result.Model != "gpt-5" {
		t.Errorf("model = %q, want gpt-5", out.result.Model)
	}
}

func TestChatStreamFinishReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture string
		want    stream.FinishReason
	}{
		{"stream_finish_length.sse", stream.FinishReasonLength},
		{"stream_finish_content_filter.sse", stream.FinishReasonContentFilter},
		{"stream_tool_call_single.sse", stream.FinishReasonToolUse},
		{"stream_text.sse", stream.FinishReasonStop},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()
			out := streamChatFixture(t, tc.fixture)
			if !out.ok {
				t.Fatalf("stream result unavailable for %s", tc.fixture)
			}
			if out.result.FinishReason != tc.want {
				t.Errorf("finish reason = %v, want %v", out.result.FinishReason, tc.want)
			}
		})
	}
}

// TestChatStreamErrorEnvelopeOnHTTP200 pins the streaming half of the
// OpenRouter-style repair. Before the fix the error frame was swallowed and
// the stream reported a clean, truncated success; now it terminates with a
// typed *openaiapi.StreamAPIError.
func TestChatStreamErrorEnvelopeOnHTTP200(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_error_envelope.sse")

	var apiErr *chat.StreamAPIError
	if !errors.As(out.err, &apiErr) {
		t.Fatalf("stream error = %T %v, want *openaiapi.StreamAPIError", out.err, out.err)
	}
	// The numeric `code` a compatible gateway smuggles the upstream status into
	// is rendered as its decimal digits rather than discarded.
	if apiErr.Code != "502" {
		t.Errorf("code = %q, want 502", apiErr.Code)
	}
	if apiErr.Message != "Upstream provider returned 502" {
		t.Errorf("message = %q, want the provider's message", apiErr.Message)
	}
	if out.ok {
		t.Error("stream reported a terminal result despite failing")
	}
}

func TestChatStreamLogprobsDoNotDisturbDeltas(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_logprobs.sse")

	if len(out.chunks) != 1 {
		t.Fatalf("chunks = %#v, want one text delta", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Paris")
}

// TestChatStreamEmptyToolCallEntryIsDropped pins that a tool_calls entry with
// no id, no name and no argument fragment yields nothing, while the real entry
// in the next delta still arrives.
func TestChatStreamEmptyToolCallEntryIsDropped(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_empty_tool_call_entry.sse")

	if len(out.chunks) != 1 {
		t.Fatalf("chunks = %#v, want only the populated tool-call entry", out.chunks)
	}
	wantToolChunk(t, out.chunks[0], 0, "call_real", "ping", "{}")
}

func TestChatStreamFullTurn(t *testing.T) {
	t.Parallel()
	out := streamChatFixture(t, "stream_full_turn.sse")

	if len(out.chunks) != 4 {
		t.Fatalf("chunks = %#v, want two text deltas, a tool seed and its arguments", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Checking")
	wantTextChunk(t, out.chunks[1], " the weather.")
	wantToolChunk(t, out.chunks[2], 0, "call_full", "get_weather", "")
	wantToolChunk(t, out.chunks[3], 0, "", "", `{"city":"Rome"}`)
	if !out.ok || out.result.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("result = %+v ok=%v, want tool use", out.result, out.ok)
	}
	if out.result.Usage == nil || out.result.Usage.InputTokens != 24 || out.result.Usage.CacheReadTokens != 64 {
		t.Errorf("usage = %+v, want input 24 (88-64) cacheRead 64", out.result.Usage)
	}
}
