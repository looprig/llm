package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/anthropic"
)

// Every test in this file follows the same three steps, in this order:
//
//	conformance.MustValidate  →  the real provider client decodes it  →  assert
//
// The gate runs first on every run. A fixture that stopped being a legal
// Anthropic payload would fail before the decoder could form an opinion about
// it, so a green assertion here is always an assertion about a payload
// Anthropic could really have sent.

// invokeFixture serves one gate-validated message fixture from a local server
// and returns what the real Anthropic client decoded from it.
func invokeFixture(t *testing.T, name string) *inference.Response {
	t.Helper()
	raw := fixture(t, name)
	conformance.MustValidate(t, "anthropic", kindMessage, raw)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(raw)
	}))
	t.Cleanup(srv.Close)

	client := newFixtureClient(t, srv.URL)
	resp, err := client.Invoke(context.Background(), inference.Request{Model: fixtureModel(srv.URL)})
	if err != nil {
		t.Fatalf("Invoke(%s) error = %v", name, err)
	}
	if resp == nil || resp.Message == nil {
		t.Fatalf("Invoke(%s) returned no message", name)
	}
	return resp
}

func fixtureModel(baseURL string) model.Model {
	return model.CustomModel(
		model.ProviderName(llm.ProviderAnthropic),
		model.APIFormatAnthropic,
		baseURL+"/v1",
		"claude-sonnet-4-5",
		model.WithTools(),
		model.WithThinkingDialect(model.ThinkingDialectAdaptive),
	)
}

func newFixtureClient(t *testing.T, baseURL string) inference.Client {
	t.Helper()
	client, err := anthropic.New(fixtureModel(baseURL), auth.APIKey("sk-ant-test"))
	if err != nil {
		t.Fatalf("anthropic.New() error = %v", err)
	}
	return client
}

func textAt(t *testing.T, blocks []content.Block, i int) *content.TextBlock {
	t.Helper()
	if i >= len(blocks) {
		t.Fatalf("block %d missing; got %d blocks", i, len(blocks))
	}
	block, ok := blocks[i].(*content.TextBlock)
	if !ok {
		t.Fatalf("block %d = %T, want *content.TextBlock", i, blocks[i])
	}
	return block
}

func thinkingAt(t *testing.T, blocks []content.Block, i int) *content.ThinkingBlock {
	t.Helper()
	if i >= len(blocks) {
		t.Fatalf("block %d missing; got %d blocks", i, len(blocks))
	}
	block, ok := blocks[i].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("block %d = %T, want *content.ThinkingBlock", i, blocks[i])
	}
	return block
}

func toolUseAt(t *testing.T, blocks []content.Block, i int) *content.ToolUseBlock {
	t.Helper()
	if i >= len(blocks) {
		t.Fatalf("block %d missing; got %d blocks", i, len(blocks))
	}
	block, ok := blocks[i].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("block %d = %T, want *content.ToolUseBlock", i, blocks[i])
	}
	return block
}

func TestDecodeTextResponses(t *testing.T) {
	t.Parallel()

	t.Run("plain text", func(t *testing.T) {
		t.Parallel()
		resp := invokeFixture(t, "message_text_plain.json")
		blocks := resp.Message.Blocks
		if len(blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(blocks))
		}
		if got := textAt(t, blocks, 0).Text; got != "The capital of France is Paris." {
			t.Errorf("text = %q", got)
		}
		if resp.FinishReason != stream.FinishReasonStop {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonStop)
		}
		if resp.Model != "claude-sonnet-4-5-20250929" {
			t.Errorf("Model = %q, want the model the response names", resp.Model)
		}
		if resp.Usage == nil || resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 7 {
			t.Errorf("Usage = %+v, want the fixture's counts", resp.Usage)
		}
	})

	t.Run("multiple text blocks keep their order", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_multi_block.json").Message.Blocks
		want := []string{"First paragraph.", "Second paragraph.", "Third paragraph."}
		if len(blocks) != len(want) {
			t.Fatalf("blocks = %d, want %d", len(blocks), len(want))
		}
		for i, text := range want {
			if got := textAt(t, blocks, i).Text; got != text {
				t.Errorf("block %d text = %q, want %q", i, got, text)
			}
		}
	})
}

// TestDecodeStopReasons pins the whole StopReason → FinishReason mapping against
// one gate-validated fixture per wire value.
func TestDecodeStopReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture string
		want    stream.FinishReason
	}{
		{"message_stop_reason_end_turn.json", stream.FinishReasonStop},
		{"message_stop_reason_max_tokens.json", stream.FinishReasonLength},
		{"message_stop_reason_stop_sequence.json", stream.FinishReasonStop},
		{"message_stop_reason_tool_use.json", stream.FinishReasonToolUse},
		{"message_stop_reason_refusal.json", stream.FinishReasonContentFilter},
	}

	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()
			if got := invokeFixture(t, tc.fixture).FinishReason; got != tc.want {
				t.Errorf("FinishReason = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDecodeToolUse(t *testing.T) {
	t.Parallel()

	t.Run("single call", func(t *testing.T) {
		t.Parallel()
		resp := invokeFixture(t, "message_tool_use_single.json")
		blocks := resp.Message.Blocks
		if len(blocks) != 1 {
			t.Fatalf("blocks = %d, want 1", len(blocks))
		}
		call := toolUseAt(t, blocks, 0)
		if call.ID != "toolu_01SingleCall" || call.Name != "search" {
			t.Errorf("tool call = %+v, want the fixture's id and name", call)
		}
		assertJSONEqual(t, "Input", call.Input, `{"query":"conformance gate"}`)
	})

	// Parallel tool use is ordered: the neutral blocks must come back in the
	// order Anthropic emitted them, with each id bound to its own call.
	t.Run("parallel calls keep id-to-position binding", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_tool_use_parallel.json").Message.Blocks
		if len(blocks) != 3 {
			t.Fatalf("blocks = %d, want text + two tool calls", len(blocks))
		}
		if got := textAt(t, blocks, 0).Text; got != "Checking both cities." {
			t.Errorf("leading text = %q", got)
		}
		first, second := toolUseAt(t, blocks, 1), toolUseAt(t, blocks, 2)
		if first.ID != "toolu_01ParisCall" {
			t.Errorf("first call id = %q, want toolu_01ParisCall", first.ID)
		}
		assertJSONEqual(t, "first call input", first.Input, `{"city":"Paris"}`)
		if second.ID != "toolu_02TokyoCall" {
			t.Errorf("second call id = %q, want toolu_02TokyoCall", second.ID)
		}
		assertJSONEqual(t, "second call input", second.Input, `{"city":"Tokyo"}`)
	})
}

// assertJSONEqual compares two JSON documents by value rather than by bytes.
// Tool-call arguments are carried as raw wire bytes on purpose — they must
// round-trip verbatim — so a fixture formatted for readability decodes to
// whitespace the wire had, which is correct behaviour and not something a test
// should pin. Comparing semantically lets fixtures stay legible.
func assertJSONEqual(t *testing.T, label string, got json.RawMessage, want string) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Errorf("%s: got is not valid JSON (%v): %s", label, err, got)
		return
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("%s: want is not valid JSON: %v", label, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("%s = %s, want %s", label, got, want)
	}
}

func TestDecodeReasoningBlocks(t *testing.T) {
	t.Parallel()

	t.Run("thinking carries its signature", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_thinking_signature.json").Message.Blocks
		if len(blocks) != 2 {
			t.Fatalf("blocks = %d, want thinking + text", len(blocks))
		}
		reasoning := thinkingAt(t, blocks, 0)
		if reasoning.Thinking == "" || reasoning.Signature == "" {
			t.Errorf("thinking block = %+v, want both text and signature", reasoning)
		}
	})

	// DEFECT PIN — required-field defect, repaired. Current-generation models
	// default to display:"omitted" and return {"thinking":"","signature":"..."}.
	// The signature is the whole payload: it is what makes the block replayable,
	// and the neutral block must keep it even though the text is empty.
	t.Run("empty thinking still carries a signature", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_thinking_empty_with_signature.json").Message.Blocks
		reasoning := thinkingAt(t, blocks, 0)
		if reasoning.Thinking != "" {
			t.Errorf("Thinking = %q, want the empty string the wire carried", reasoning.Thinking)
		}
		if reasoning.Signature == "" {
			t.Fatal("Signature was dropped; the block is no longer replayable, which is the HTTP 400 the encoder repair exists to prevent")
		}
	})

	t.Run("redacted thinking becomes replayable provider state", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_redacted_thinking.json").Message.Blocks
		if len(blocks) != 2 {
			t.Fatalf("blocks = %d, want redacted thinking + text", len(blocks))
		}
		reasoning := thinkingAt(t, blocks, 0)
		if reasoning.Thinking != "" || reasoning.Signature != "" {
			t.Errorf("redacted block = %+v, want no plaintext reasoning", reasoning)
		}
		if !reasoning.ReplayableAs("anthropic-redacted-thinking") {
			t.Errorf("ProviderState = %s / %q, want Anthropic-scoped replayable state",
				reasoning.ProviderState, reasoning.ProviderStateFormat)
		}
	})

	// DEFECT PIN — the collapse defect, on the NON-streaming path. Interleaved
	// thinking opens a fresh block around each step, and every block carries its
	// own signature. Folding them into one block would rebind the last signature
	// to the concatenated text, which the provider rejects on replay.
	t.Run("two thinking blocks keep distinct signatures", func(t *testing.T) {
		t.Parallel()
		blocks := invokeFixture(t, "message_thinking_multiple_signatures.json").Message.Blocks
		if len(blocks) != 3 {
			t.Fatalf("blocks = %d, want two thinking blocks + text (a collapse would give 2)", len(blocks))
		}
		first, second := thinkingAt(t, blocks, 0), thinkingAt(t, blocks, 1)
		if first.Thinking == second.Thinking {
			t.Errorf("both thinking blocks carry %q; the reasoning text was merged", first.Thinking)
		}
		if first.Signature == "" || second.Signature == "" {
			t.Fatalf("signatures = %q / %q, want one per block", first.Signature, second.Signature)
		}
		if first.Signature == second.Signature {
			t.Errorf("both blocks carry signature %q; the per-block binding collapsed", first.Signature)
		}
	})
}

func TestDecodeUsage(t *testing.T) {
	t.Parallel()

	t.Run("cache read and cache creation totals", func(t *testing.T) {
		t.Parallel()
		usage := invokeFixture(t, "message_usage_cache_tokens.json").Usage
		if usage == nil {
			t.Fatal("Usage is nil")
		}
		if usage.CacheReadTokens != 8192 || usage.CacheCreationTokens != 1024 {
			t.Errorf("cache usage = read %d / creation %d, want 8192 / 1024",
				usage.CacheReadTokens, usage.CacheCreationTokens)
		}
		if usage.InputTokens != 6 || usage.OutputTokens != 21 {
			t.Errorf("usage = %+v, want the fixture's counts", usage)
		}
	})

	// The per-TTL breakdown under `cache_creation` is a real wire field the gate
	// enforces the shape of, but the neutral vocabulary has one
	// CacheCreationTokens counter, so only the top-level total survives. This
	// asserts the total is right and records the narrowing as intentional rather
	// than accidental.
	t.Run("per-TTL cache_creation breakdown narrows to one total", func(t *testing.T) {
		t.Parallel()
		usage := invokeFixture(t, "message_usage_cache_creation_breakdown.json").Usage
		if usage == nil {
			t.Fatal("Usage is nil")
		}
		if usage.CacheCreationTokens != 3072 {
			t.Errorf("CacheCreationTokens = %d, want the 3072 total that 1024(5m) + 2048(1h) sums to",
				usage.CacheCreationTokens)
		}
		if usage.CacheReadTokens != 4096 {
			t.Errorf("CacheReadTokens = %d, want 4096", usage.CacheReadTokens)
		}
	})
}

// --------------------------------------------------------------------------
// Streaming
// --------------------------------------------------------------------------

// streamFixture serves one gate-validated SSE fixture and returns the chunks the
// real client decoded plus its terminal result.
func streamFixture(t *testing.T, name string, validate bool) ([]content.Chunk, stream.StreamResult, bool) {
	t.Helper()
	raw := fixture(t, name)
	if validate {
		conformance.MustValidateStream(t, "anthropic", kindEvent, raw)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(raw)
	}))
	t.Cleanup(srv.Close)

	reader, err := newFixtureClient(t, srv.URL).Stream(context.Background(), inference.Request{Model: fixtureModel(srv.URL)})
	if err != nil {
		t.Fatalf("Stream(%s) error = %v", name, err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	var chunks []content.Chunk
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("Stream(%s).Next() error = %v", name, nextErr)
		}
		chunks = append(chunks, chunk)
	}
	result, ok := reader.Result()
	return chunks, result, ok
}

func TestDecodeStreamLifecycle(t *testing.T) {
	t.Parallel()

	chunks, result, ok := streamFixture(t, "stream_text_thinking_tool.sse", true)
	if !ok {
		t.Fatal("stream produced no terminal result; message_stop was not honoured")
	}

	var (
		text     streamaccumulator.Text
		thinking streamaccumulator.Thinking
		tools    streamaccumulator.ToolUses
	)
	for _, chunk := range chunks {
		switch c := chunk.(type) {
		case *content.TextChunk:
			text.Add(c)
		case *content.ThinkingChunk:
			thinking.Add(c)
		case *content.ToolUseChunk:
			tools.Add(c)
		default:
			t.Fatalf("unexpected chunk %T", chunk)
		}
	}

	if got := text.Block(); got == nil || got.Text != "Checking Paris now." {
		t.Errorf("text = %+v, want the concatenated deltas", got)
	}
	reasoning := thinking.Blocks()
	if len(reasoning) != 1 {
		t.Fatalf("thinking blocks = %d, want 1", len(reasoning))
	}
	if reasoning[0].Thinking != "Weather needs the tool." {
		t.Errorf("thinking text = %q", reasoning[0].Thinking)
	}
	// signature_delta is the terminal delta of a thinking block; without it the
	// assembled block cannot be replayed.
	if reasoning[0].Signature == "" {
		t.Error("signature_delta did not reach the assembled thinking block")
	}

	calls := tools.Blocks()
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].ID != "toolu_01StreamCall" || calls[0].Name != "get_weather" {
		t.Errorf("tool call = %+v, want the id and name from content_block_start", calls[0])
	}
	if string(calls[0].Input) != `{"city":"Paris"}` {
		t.Errorf("tool input = %s, want the concatenated input_json_delta fragments", calls[0].Input)
	}

	if result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, stream.FinishReasonToolUse)
	}
	if result.Usage == nil || result.Usage.InputTokens != 27 || result.Usage.OutputTokens != 84 {
		t.Errorf("stream usage = %+v, want message_start input merged with message_delta output", result.Usage)
	}
}

// TestDecodeStreamParallelToolUse pins the index binding that makes parallel
// tool calls recoverable: each input_json_delta carries the content block index,
// so the accumulator keys the fragments to the right call.
func TestDecodeStreamParallelToolUse(t *testing.T) {
	t.Parallel()

	chunks, result, ok := streamFixture(t, "stream_parallel_tool_use.sse", true)
	if !ok {
		t.Fatal("stream produced no terminal result")
	}
	var tools streamaccumulator.ToolUses
	for _, chunk := range chunks {
		if c, isTool := chunk.(*content.ToolUseChunk); isTool {
			tools.Add(c)
		}
	}
	calls := tools.Blocks()
	if len(calls) != 2 {
		t.Fatalf("tool calls = %d, want 2", len(calls))
	}
	if calls[0].ID != "toolu_01ParisCall" || string(calls[0].Input) != `{"city":"Paris"}` {
		t.Errorf("call 0 = %+v", calls[0])
	}
	if calls[1].ID != "toolu_02TokyoCall" || string(calls[1].Input) != `{"city":"Tokyo"}` {
		t.Errorf("call 1 = %+v", calls[1])
	}
	if result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

// TestDecodeStreamPingAndError uses the one fixture the gate cannot bless (see
// gateGapFixtures): "ping" is a keep-alive the decoder must ignore, and "error"
// must abort the stream with a typed API error rather than being tolerated as an
// unknown event. Both frames are individually asserted to be gate-rejected in
// TestGateGapEventsAreRejected, so this test's use of an unvalidated fixture is
// explicit rather than an escape from the gate.
func TestDecodeStreamPingAndError(t *testing.T) {
	t.Parallel()

	raw := fixture(t, "stream_ping_and_error.sse")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(raw)
	}))
	t.Cleanup(srv.Close)

	reader, err := newFixtureClient(t, srv.URL).Stream(context.Background(), inference.Request{Model: fixtureModel(srv.URL)})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	var (
		chunks   []content.Chunk
		streamed error
	)
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			streamed = nextErr
			break
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want only the text delta; ping must not produce one", len(chunks))
	}
	if got, isText := chunks[0].(*content.TextChunk); !isText || got.Text != "Partial" {
		t.Errorf("chunk = %#v, want the text delta", chunks[0])
	}
	if streamed == nil {
		t.Fatal("the error event did not abort the stream")
	}
	if _, resultOK := reader.Result(); resultOK {
		t.Error("an aborted stream produced a terminal result")
	}
}
