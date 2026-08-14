package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	responses "github.com/looprig/inference/codec/openairesponses"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openai"
)

// Responses fixture suite. Same three steps as the Chat suite: gate-validate
// against OpenAI's published Responses schema, decode through the real
// providers/openai client, assert the neutral result.

func responsesModel(t *testing.T, baseURL string) model.Model {
	t.Helper()
	return model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAIResponses,
		baseURL+"/v1",
		"gpt-5",
		model.WithThinking(),
	)
}

func invokeResponsesFixture(t *testing.T, name string) (*inference.Response, error) {
	t.Helper()
	body := responsesFixture(t, name)
	srv := serveBody(t, "application/json", body)
	selected := responsesModel(t, srv.URL)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	return client.Invoke(context.Background(), inference.Request{Model: selected})
}

func mustInvokeResponsesFixture(t *testing.T, name string) *inference.Response {
	t.Helper()
	resp, err := invokeResponsesFixture(t, name)
	if err != nil {
		t.Fatalf("Invoke(%s) error = %v", name, err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke(%s) returned %+v, want a decoded message", name, resp)
	}
	return resp
}

func streamResponsesFixture(t *testing.T, name string) streamOutcome {
	t.Helper()
	body := responsesStreamFixture(t, name)
	srv := serveBody(t, "text/event-stream", body)
	selected := responsesModel(t, srv.URL)
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

// reasoningState mirrors the JSON this codec stores in
// ThinkingBlock.ProviderState. Decoding it in the test rather than comparing
// raw bytes keeps the assertion about the two members that matter for replay.
type reasoningState struct {
	ID               string `json:"id"`
	EncryptedContent string `json:"encrypted_content"`
}

func providerState(t *testing.T, raw json.RawMessage) reasoningState {
	t.Helper()
	var state reasoningState
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatalf("decode provider state %s: %v", raw, err)
	}
	return state
}

// --- non-streaming --------------------------------------------------------

func TestResponsesCompletedText(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "completed_text.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one text block", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "The capital of France is Paris.")
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("finish reason = %v, want stop", resp.FinishReason)
	}
	if resp.Model != "gpt-5" {
		t.Errorf("model = %q, want gpt-5", resp.Model)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 19 || resp.Usage.OutputTokens != 8 {
		t.Errorf("usage = %+v, want input 19 output 8", resp.Usage)
	}
}

// TestResponsesIncompleteReasons covers both spec-declared
// incomplete_details.reason values. Only max_output_tokens has a neutral
// counterpart; content_filter has none in the Responses mapping, so it lands
// on Unknown. That asymmetry is deliberate on the Chat side (which does map
// content_filter) and is recorded here as a FINDING for the Responses side.
func TestResponsesIncompleteReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture string
		want    stream.FinishReason
	}{
		{"incomplete_max_output_tokens.json", stream.FinishReasonLength},
		{"incomplete_content_filter.json", stream.FinishReasonUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()
			resp := mustInvokeResponsesFixture(t, tc.fixture)
			if resp.FinishReason != tc.want {
				t.Errorf("finish reason = %v, want %v", resp.FinishReason, tc.want)
			}
		})
	}
}

func TestResponsesFailedStatusIsAnError(t *testing.T) {
	t.Parallel()

	resp, err := invokeResponsesFixture(t, "failed.json")
	if resp != nil {
		t.Fatalf("Invoke() = %+v, want no response", resp)
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %T %v, want *failure.APIError", err, err)
	}
}

func TestResponsesReasoningSummary(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "reasoning_summary.json")

	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want thinking then text", resp.Message.Blocks)
	}
	// Multiple summary parts are joined with a blank line.
	thinking := wantThinking(t, resp.Message.Blocks[0],
		"First I identify the city.\n\nThen I recall its capital status.")
	if thinking.ProviderStateFormat != "openai-responses" {
		t.Errorf("provider state format = %q, want openai-responses", thinking.ProviderStateFormat)
	}
	// The reasoning item's id is required to replay the item at all, so it is
	// carried even when there is no encrypted content.
	if got := providerState(t, thinking.ProviderState); got.ID != "rs_1" {
		t.Errorf("provider state = %+v, want id rs_1", got)
	}
	wantText(t, resp.Message.Blocks[1], "Paris.")
}

func TestResponsesReasoningEncryptedContent(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "reasoning_encrypted.json")

	thinking := wantThinking(t, resp.Message.Blocks[0], "Considering the question.")
	state := providerState(t, thinking.ProviderState)
	if state.ID != "rs_2" || state.EncryptedContent != "gAAAAABn-opaque-reasoning-blob" {
		t.Errorf("provider state = %+v, want id rs_2 and the encrypted blob", state)
	}
}

// TestResponsesMultipleReasoningItemsDoNotCollapse pins the DEFECT THAT WAS
// JUST REPAIRED. A response may carry several reasoning items, each with its
// own required id. Provider state used to be the bare encrypted string with no
// id, so two items were indistinguishable and neither could be replayed —
// effectively collapsing them. Each item must now survive as its own
// ThinkingBlock carrying its own id and blob.
func TestResponsesMultipleReasoningItemsDoNotCollapse(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "reasoning_multiple.json")

	if len(resp.Message.Blocks) != 3 {
		t.Fatalf("blocks = %#v, want two thinking blocks and one text block", resp.Message.Blocks)
	}
	first := wantThinking(t, resp.Message.Blocks[0], "Step one.")
	second := wantThinking(t, resp.Message.Blocks[1], "Step two.")

	firstState := providerState(t, first.ProviderState)
	secondState := providerState(t, second.ProviderState)
	if firstState.ID != "rs_first" || firstState.EncryptedContent != "blob-one" {
		t.Errorf("first provider state = %+v, want rs_first/blob-one", firstState)
	}
	if secondState.ID != "rs_second" || secondState.EncryptedContent != "blob-two" {
		t.Errorf("second provider state = %+v, want rs_second/blob-two", secondState)
	}
	if firstState == secondState {
		t.Error("the two reasoning items collapsed into identical provider state")
	}
	wantText(t, resp.Message.Blocks[2], "Both steps done.")
}

func TestResponsesFunctionCall(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "function_call.json")

	if len(resp.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one tool-use block", resp.Message.Blocks)
	}
	// call_id (not the item id) is the identifier a follow-up tool result must
	// quote, so it wins.
	tool := wantToolUse(t, resp.Message.Blocks[0], "call_weather_1", "get_weather")
	// Unlike Chat Completions, the Responses decoder unwraps `arguments` into
	// the arguments OBJECT, which is what ToolUseBlock.Input is meant to hold.
	if string(tool.Input) != `{"city":"Paris"}` {
		t.Errorf("tool input = %s, want the arguments object", tool.Input)
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("finish reason = %v, want tool use", resp.FinishReason)
	}
}

// TestResponsesFunctionCallOutputItems covers the required-`output` member on
// FunctionToolCallOutput for both a populated and an EMPTY result. `output` is
// required by the spec, so an empty tool result still carries "output":"" —
// omitting it would be an illegal payload, which is exactly what the gate on
// these two fixtures proves.
//
// The neutral decoder skips function_call_output items (a tool result is the
// caller's own prior input, not model output), so what is asserted is that
// their presence does not disturb the surrounding blocks.
func TestResponsesFunctionCallOutputItems(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture  string
		callID   string
		toolName string
		text     string
	}{
		{"function_call_output_nonempty.json", "call_weather_2", "get_weather", "It is 21C in Paris."},
		{"function_call_output_empty.json", "call_noop", "noop", "Nothing came back."},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()
			resp := mustInvokeResponsesFixture(t, tc.fixture)
			if len(resp.Message.Blocks) != 2 {
				t.Fatalf("blocks = %#v, want the call and the final text", resp.Message.Blocks)
			}
			wantToolUse(t, resp.Message.Blocks[0], tc.callID, tc.toolName)
			wantText(t, resp.Message.Blocks[1], tc.text)
			if resp.FinishReason != stream.FinishReasonToolUse {
				t.Errorf("finish reason = %v, want tool use", resp.FinishReason)
			}
		})
	}
}

// TestResponsesRefusalContentPartSurfacesAsARefusalBlock pins the `refusal`
// member of the OutputContent union, which once was skipped along with every
// other non-output_text part — decoding a refused turn as an empty assistant
// message with a clean "stop".
//
// This test previously asserted the interim mapping — a *content.TextBlock plus
// a content_filter finish reason — which predated content.RefusalBlock. The
// block now carries the "declined" signal per block, so the finish reason
// reports the wire's own status:"completed".
func TestResponsesRefusalContentPartSurfacesAsARefusalBlock(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "refusal.json")

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
		t.Errorf("finish reason = %v, want stop (status:\"completed\")", resp.FinishReason)
	}
}

// TestResponsesReasoningTextContentIsDropped records a coverage FINDING: a
// reasoning item may carry a `content` array of reasoning_text parts (the full
// chain of thought) in addition to its `summary`. The decoder reads only the
// summary, so the reasoning_text is discarded.
func TestResponsesReasoningTextContentIsDropped(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "reasoning_text.json")

	thinking := wantThinking(t, resp.Message.Blocks[0], "Short summary.")
	if thinking.Thinking == "Full chain of thought, verbatim." {
		t.Fatal("reasoning_text decoding may have been added — update this finding")
	}
}

func TestResponsesMessageWithIDAndStatus(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "message_id_and_status.json")

	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want one block per output_text part", resp.Message.Blocks)
	}
	wantText(t, resp.Message.Blocks[0], "Part one. ")
	wantText(t, resp.Message.Blocks[1], "Part two.")
}

func TestResponsesParallelFunctionCallsPreserveOrder(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "parallel_function_calls.json")

	if len(resp.Message.Blocks) != 3 {
		t.Fatalf("blocks = %#v, want three tool-use blocks", resp.Message.Blocks)
	}
	wantToolUse(t, resp.Message.Blocks[0], "call_p0", "get_weather")
	wantToolUse(t, resp.Message.Blocks[1], "call_p1", "get_weather")
	wantToolUse(t, resp.Message.Blocks[2], "call_p2", "get_time")
}

// TestResponsesUsageDetails pins that Responses' input_tokens is the gross
// prompt total, so cached tokens are subtracted out into the neutral split —
// and that Responses has no cache-creation concept.
func TestResponsesUsageDetails(t *testing.T) {
	t.Parallel()
	resp := mustInvokeResponsesFixture(t, "usage_details.json")

	u := resp.Usage
	if u == nil {
		t.Fatal("usage = nil, want normalized usage")
	}
	if u.InputTokens != 128 || u.CacheReadTokens != 896 || u.OutputTokens != 96 || u.ReasoningTokens != 64 {
		t.Errorf("usage = %+v, want input 128 (1024-896) cacheRead 896 output 96 reasoning 64", u)
	}
	if u.CacheCreationTokens != 0 {
		t.Errorf("cache creation = %d, want 0; Responses has no cache-write concept", u.CacheCreationTokens)
	}
}

// --- streaming ------------------------------------------------------------

func TestResponsesStreamText(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_text.sse")

	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want two text deltas", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Hello")
	wantTextChunk(t, out.chunks[1], ", world!")
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop || out.result.Model != "gpt-5" {
		t.Errorf("result = %+v ok=%v, want stop/gpt-5", out.result, out.ok)
	}
}

func TestResponsesStreamFunctionCall(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_function_call.sse")

	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want a seed plus two argument fragments", out.chunks)
	}
	// output_item.added seeds id+name; the arguments arrive as fragments the
	// accumulator concatenates. output_index is the chunk Index.
	wantToolChunk(t, out.chunks[0], 0, "call_s2", "get_weather", "")
	wantToolChunk(t, out.chunks[1], 0, "", "", `{"city":`)
	wantToolChunk(t, out.chunks[2], 0, "", "", `"Paris"}`)
	if !out.ok || out.result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("result = %+v ok=%v, want tool use", out.result, out.ok)
	}
}

func TestResponsesStreamParallelFunctionCalls(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_parallel_function_calls.sse")

	if len(out.chunks) != 4 {
		t.Fatalf("chunks = %#v, want two seeds and two argument fragments", out.chunks)
	}
	wantToolChunk(t, out.chunks[0], 0, "call_s3a", "get_weather", "")
	wantToolChunk(t, out.chunks[1], 1, "call_s3b", "get_time", "")
	wantToolChunk(t, out.chunks[2], 0, "", "", `{"city":"Paris"}`)
	wantToolChunk(t, out.chunks[3], 1, "", "", `{"zone":"UTC"}`)
}

func TestResponsesStreamReasoningSummary(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_reasoning_summary.sse")

	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want two summary deltas and the item's replay state", out.chunks)
	}
	wantThinkingChunk(t, out.chunks[0], "Thinking ")
	wantThinkingChunk(t, out.chunks[1], "about it.")
	final := wantThinkingChunk(t, out.chunks[2], "")
	if got := providerState(t, final.ProviderState); got.ID != "rs_s4" {
		t.Errorf("provider state = %+v, want id rs_s4", got)
	}
	if final.ProviderStateFormat != "openai-responses" {
		t.Errorf("provider state format = %q, want openai-responses", final.ProviderStateFormat)
	}
}

// TestResponsesStreamReasoningTextDeltasAreDropped records a coverage FINDING:
// response.reasoning_text.delta is a spec-declared event carrying the model's
// verbatim reasoning, and the codec handles only
// response.reasoning_summary_text.delta, so the whole stream yields no chunks.
func TestResponsesStreamReasoningTextDeltasAreDropped(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_reasoning_text.sse")

	if len(out.chunks) != 0 {
		t.Fatalf("chunks = %#v; reasoning_text streaming may have been added — update this finding",
			out.chunks)
	}
	if !out.ok {
		t.Error("stream result unavailable despite response.completed")
	}
}

// TestResponsesStreamReasoningItemDoneCarriesReplayState pins the streaming
// half of the reasoning-id repair: an output_item.done for a reasoning item
// yields a ThinkingChunk whose provider state holds BOTH the required item id
// and the encrypted blob.
func TestResponsesStreamReasoningItemDoneCarriesReplayState(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_reasoning_item_done_encrypted.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want the summary delta and the replay state", out.chunks)
	}
	wantThinkingChunk(t, out.chunks[0], "Deliberating.")
	final := wantThinkingChunk(t, out.chunks[1], "")
	state := providerState(t, final.ProviderState)
	if state.ID != "rs_s6" || state.EncryptedContent != "gAAAAABn-encrypted-reasoning" {
		t.Errorf("provider state = %+v, want rs_s6 and the encrypted blob", state)
	}
}

// TestResponsesStreamMultipleReasoningItemsStayDistinct is the streaming twin
// of the collapse repair: two reasoning items in one stream must produce two
// distinct replay states.
//
// Distinct CHUNKS are necessary but not sufficient, and asserting only on them
// is what let the collapse survive: the chunks were always distinct, and the
// two states were fused afterwards by the accumulator that turns chunks into
// blocks. The block-level half below is the assertion that actually holds the
// continuation state together, so it runs the same real fixture through
// streamaccumulator exactly as the loop runtime does.
func TestResponsesStreamMultipleReasoningItemsStayDistinct(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_multiple_reasoning_items.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want one replay state per reasoning item", out.chunks)
	}
	first := providerState(t, wantThinkingChunk(t, out.chunks[0], "").ProviderState)
	second := providerState(t, wantThinkingChunk(t, out.chunks[1], "").ProviderState)
	if first.ID != "rs_s13a" || first.EncryptedContent != "blob-one" {
		t.Errorf("first provider state = %+v, want rs_s13a/blob-one", first)
	}
	if second.ID != "rs_s13b" || second.EncryptedContent != "blob-two" {
		t.Errorf("second provider state = %+v, want rs_s13b/blob-two", second)
	}
	if first == second {
		t.Error("the two streamed reasoning items collapsed into identical provider state")
	}

	var acc streamaccumulator.Thinking
	for _, chunk := range out.chunks {
		if tc, ok := chunk.(*content.ThinkingChunk); ok {
			acc.Add(tc)
		}
	}
	blocks := acc.Blocks()
	if len(blocks) != 2 {
		t.Fatalf("accumulated blocks = %d %#v, want one reasoning block per item", len(blocks), blocks)
	}
	if got := providerState(t, blocks[0].ProviderState); got != first {
		t.Errorf("block 0 provider state = %+v, want %+v", got, first)
	}
	if got := providerState(t, blocks[1].ProviderState); got != second {
		t.Errorf("block 1 provider state = %+v, want %+v", got, second)
	}
}

// TestResponsesStreamRefusalDeltasSurfaceAsRefusalChunks pins the streaming
// half: response.refusal.delta yields the same RefusalChunks that fold into the
// block the non-streaming decoder builds from the refusal content part, and
// response.refusal.done — which repeats the whole refusal — must not duplicate
// it.
func TestResponsesStreamRefusalDeltasSurfaceAsRefusalChunks(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_refusal.sse")

	if len(out.chunks) != 2 {
		t.Fatalf("chunks = %#v, want the two refusal deltas and nothing from .done", out.chunks)
	}
	wantRefusalChunk(t, out.chunks[0], "I'm sorry, ")
	wantRefusalChunk(t, out.chunks[1], "I can't help with that.")
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want stop (status:\"completed\")", out.result, out.ok)
	}
}

// TestResponsesStreamIncompleteIsTerminal pins the repair that added
// response.incomplete alongside response.completed as a terminal event. Before
// it, a truncated stream produced no terminal result at all: the caller could
// not tell that the answer stopped at the token limit.
func TestResponsesStreamIncompleteIsTerminal(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_incomplete.sse")

	if len(out.chunks) != 1 {
		t.Fatalf("chunks = %#v, want the single text delta", out.chunks)
	}
	if !out.ok {
		t.Fatal("stream result unavailable; response.incomplete must be terminal")
	}
	if out.result.FinishReason != stream.FinishReasonLength {
		t.Errorf("finish reason = %v, want length", out.result.FinishReason)
	}
}

func TestResponsesStreamFailedEvent(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_failed.sse")

	var apiErr *responses.StreamAPIError
	if !errors.As(out.err, &apiErr) {
		t.Fatalf("stream error = %T %v, want *openairesponses.StreamAPIError", out.err, out.err)
	}
	if apiErr.Code != "server_error" {
		t.Errorf("code = %q, want server_error", apiErr.Code)
	}
	if out.ok {
		t.Error("stream reported a terminal result despite response.failed")
	}
}

// TestResponsesStreamErrorEvent pins the DEFECT THAT WAS JUST REPAIRED: the
// spec's top-level `error` event carries code/message directly, with no
// enclosing `response` object, so it never reached the response.failed arm.
// Skipping it as an unknown type ended the stream at natural EOF and reported
// a truncated answer as a success.
func TestResponsesStreamErrorEvent(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_error_event.sse")

	var apiErr *responses.StreamAPIError
	if !errors.As(out.err, &apiErr) {
		t.Fatalf("stream error = %T %v, want *openairesponses.StreamAPIError", out.err, out.err)
	}
	if apiErr.Code != "rate_limit_exceeded" || apiErr.Message != "Rate limit reached." {
		t.Errorf("stream error = %+v, want the event's own code and message", apiErr)
	}
	if out.ok {
		t.Error("stream reported a terminal result despite the error event")
	}
}

func TestResponsesStreamEmptyTextDeltaIsSkipped(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_empty_text_delta.sse")

	if len(out.chunks) != 1 {
		t.Fatalf("chunks = %#v, want only the non-empty delta", out.chunks)
	}
	wantTextChunk(t, out.chunks[0], "Real text.")
}

func TestResponsesStreamCompletedUsage(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_completed_usage.sse")

	if !out.ok {
		t.Fatal("stream result unavailable")
	}
	u := out.result.Usage
	if u == nil || u.InputTokens != 128 || u.CacheReadTokens != 896 || u.OutputTokens != 96 || u.ReasoningTokens != 64 {
		t.Errorf("usage = %+v, want input 128 (1024-896) cacheRead 896 output 96 reasoning 64", u)
	}
}

// TestResponsesStreamEmptyArgumentsDeltaIsEmittedVerbatim pins that an empty
// function_call_arguments delta still yields a chunk. Unlike a text delta it is
// NOT skipped: the accumulator concatenates argument fragments, and dropping an
// empty one silently would hide an index the provider did open.
func TestResponsesStreamEmptyArgumentsDeltaIsEmittedVerbatim(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_function_call_empty_args_delta.sse")

	if len(out.chunks) != 3 {
		t.Fatalf("chunks = %#v, want the seed and both argument deltas", out.chunks)
	}
	wantToolChunk(t, out.chunks[0], 0, "call_s14", "noop", "")
	wantToolChunk(t, out.chunks[1], 0, "", "", "")
	wantToolChunk(t, out.chunks[2], 0, "", "", "{}")
}

func TestResponsesStreamFullTurn(t *testing.T) {
	t.Parallel()
	out := streamResponsesFixture(t, "stream_full_turn.sse")

	if len(out.chunks) != 5 {
		t.Fatalf("chunks = %#v, want reasoning, replay state, text, tool seed and arguments",
			out.chunks)
	}
	wantThinkingChunk(t, out.chunks[0], "Plan the call.")
	state := providerState(t, wantThinkingChunk(t, out.chunks[1], "").ProviderState)
	if state.ID != "rs_s15" || state.EncryptedContent != "gAAAAABn-full-turn" {
		t.Errorf("provider state = %+v, want rs_s15 and the encrypted blob", state)
	}
	wantTextChunk(t, out.chunks[2], "Checking the weather.")
	wantToolChunk(t, out.chunks[3], 2, "call_s15", "get_weather", "")
	wantToolChunk(t, out.chunks[4], 2, "", "", `{"city":"Rome"}`)

	if !out.ok || out.result.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("result = %+v ok=%v, want tool use", out.result, out.ok)
	}
	if out.result.Usage == nil || out.result.Usage.InputTokens != 24 || out.result.Usage.CacheReadTokens != 64 {
		t.Errorf("usage = %+v, want input 24 (88-64) cacheRead 64", out.result.Usage)
	}
}
