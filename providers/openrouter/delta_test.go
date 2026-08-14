package openrouter_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openrouter"
)

// OpenRouter PROVIDER-DELTA suite.
//
// OpenRouter speaks OpenAI Chat Completions as an aggregator, and every fixture
// here is a shape OpenAI itself never sends. The four divergences the audit
// proved:
//
//  1. `reasoning_details` — an array of records that carries an upstream
//     model's reasoning across the aggregator. Its variants differ in which
//     member holds the payload (`.text` → text, `.summary` → summary,
//     `.encrypted` → data+signature with NO readable text), and OpenRouter adds
//     new variants over time, so an unknown one must round-trip untouched
//     rather than be dropped or rejected.
//  2. An HTTP **200** whose body is a bare top-level `error` — upstream
//     failures are relayed inside a success status.
//  3. `: OPENROUTER PROCESSING` SSE keepalive comments, emitted while a slow
//     upstream is queued.
//  4. Usage extended with `cost` / `cost_details` alongside standard
//     `prompt_tokens_details.cached_tokens`.
//
// WHAT IS SCHEMA-BACKED, precisely:
//   - the BASE ENVELOPE of nine of the eleven fixtures. Each .json is held
//     against OpenAI's CreateChatCompletion and each .sse frame against
//     CreateChatCompletionStreamResponse before any decoder sees it. That is
//     what proves the keepalive comments do not break SSE framing (the frames
//     around them still parse and validate) and that a usage object carrying
//     `cost` is still a well-formed completion.
//
// WHAT IS NOT SCHEMA-BACKED:
//   - `reasoning_details` ITSELF, in every variant. It is absent from OpenAI's
//     spec and survives the gate only because OpenAI closes
//     `additionalProperties` on 3 of its 54 Chat Completions object shapes,
//     none of them the message or delta these records hang off. The gate is
//     indifferent to the member's spelling, its variant tags and its nesting;
//     `reasoning_detials` would pass just as happily. The same applies to
//     `usage.cost` and `usage.cost_details`.
//   - the two error fixtures, which are DECODE-ONLY by necessity: a body with
//     a top-level `error` and no `choices` cannot be a legal chat_completion,
//     since `choices` is required. See openrouterUngated.

const openrouterKey = "sk-or-test"

// openrouterUngated names the fixtures deliberately held out of the gate, with
// the reason each cannot be validated. Asserted against the corpus below so
// the list cannot drift.
var openrouterUngated = map[string]string{
	"error_on_http_200.json": "a top-level `error` body has no `choices`, which " +
		"CreateChatCompletion requires — the divergence IS the illegality",
	"error_on_http_200.sse": "same, for the frame that replaces the final chunk",
}

func orFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

func gateORObject(t testing.TB, name string) []byte {
	t.Helper()
	raw := orFixture(t, name)
	conformance.MustValidateResponse(t, "openai", "chat_completion", raw)
	return raw
}

func gateORStream(t testing.TB, name string) []byte {
	t.Helper()
	raw := orFixture(t, name)
	if n := conformance.MustValidateStream(t, "openai", "chat_completion_chunk", raw); n == 0 {
		t.Fatalf("%s validated no frames", name)
	}
	return raw
}

func orServe(t *testing.T, contentType string, body []byte) model.Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return model.CustomModel(
		model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI,
		srv.URL, "anthropic/claude-sonnet-4.5", model.WithThinking(),
	)
}

func orPrompt(selected model.Model) inference.Request {
	return inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "capital of France?"}},
		}}},
	}
}

func orInvokeBody(t *testing.T, body []byte) (*inference.Response, error) {
	t.Helper()
	selected := orServe(t, "application/json", body)
	client, err := openrouter.New(selected, openrouterKey)
	if err != nil {
		t.Fatalf("openrouter.New() error = %v", err)
	}
	return client.Invoke(context.Background(), orPrompt(selected))
}

func orInvoke(t *testing.T, name string) (*inference.Response, error) {
	t.Helper()
	return orInvokeBody(t, gateORObject(t, name))
}

type orStreamOutcome struct {
	chunks []content.Chunk
	result stream.StreamResult
	ok     bool
	err    error
}

func orStreamBody(t *testing.T, body []byte) orStreamOutcome {
	t.Helper()
	selected := orServe(t, "text/event-stream", body)
	client, err := openrouter.New(selected, openrouterKey)
	if err != nil {
		t.Fatalf("openrouter.New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), orPrompt(selected))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	var out orStreamOutcome
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			out.err = nextErr
			break
		}
		out.chunks = append(out.chunks, chunk)
	}
	out.result, out.ok = reader.Result()
	return out
}

func orStream(t *testing.T, name string) orStreamOutcome {
	t.Helper()
	return orStreamBody(t, gateORStream(t, name))
}

func orStreamText(out orStreamOutcome) string {
	var sb strings.Builder
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.TextChunk); ok {
			sb.WriteString(c.Text)
		}
	}
	return sb.String()
}

func orThinkingBlock(t *testing.T, resp *inference.Response) *content.ThinkingBlock {
	t.Helper()
	if resp == nil || resp.Message == nil {
		t.Fatalf("response = %+v, want a decoded message", resp)
	}
	for _, block := range resp.Message.Blocks {
		if thinking, ok := block.(*content.ThinkingBlock); ok {
			return thinking
		}
	}
	t.Fatalf("blocks = %#v, want a thinking block carrying the reasoning state", resp.Message.Blocks)
	return nil
}

// TestOpenRouterCorpusIsLegalOpenAI walks the corpus itself so a fixture added
// to testdata/ can never sit there ungated, pins the corpus size, and pins the
// two deliberate exclusions.
func TestOpenRouterCorpusIsLegalOpenAI(t *testing.T) {
	t.Parallel()

	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("ReadDir(testdata) error = %v", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) != 11 {
		t.Errorf("testdata holds %d fixtures, want 11", len(names))
	}

	var gated int
	for _, name := range names {
		switch {
		case openrouterUngated[name] != "":
			// Deliberately ungated; see the file header.
		case strings.HasSuffix(name, ".sse"):
			gateORStream(t, name)
			gated++
		case strings.HasSuffix(name, ".json"):
			gateORObject(t, name)
			gated++
		default:
			t.Errorf("testdata/%s is neither a .json nor a .sse fixture", name)
		}
	}
	if gated != 9 {
		t.Errorf("gated %d fixtures, want 9", gated)
	}
	for name := range openrouterUngated {
		if _, err := os.Stat(filepath.Join("testdata", name)); err != nil {
			t.Errorf("openrouterUngated names %s, which does not exist: %v", name, err)
		}
	}
}

// TestOpenRouterReasoningDetailVariants is divergence 1. The three documented
// variants put the payload in three different members, and only two of them
// have any readable text at all — `.encrypted` carries opaque data plus a
// signature, so its ThinkingBlock is legitimately text-less and exists purely
// to carry replay state. Schema-backed: the surrounding completion. Everything
// about reasoning_details itself is assertion-only.
func TestOpenRouterReasoningDetailVariants(t *testing.T) {
	t.Parallel()

	tests := []struct {
		fixture      string
		wantThinking string
	}{
		{"reasoning_details_text.json", "Let me recall European capitals."},
		{"reasoning_details_summary.json", "Recalled the capital directly."},
		{"reasoning_details_encrypted.json", ""},
	}

	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			t.Parallel()

			resp, err := orInvoke(t, tt.fixture)
			if err != nil {
				t.Fatalf("Invoke(%s) error = %v", tt.fixture, err)
			}
			thinking := orThinkingBlock(t, resp)
			if thinking.Thinking != tt.wantThinking {
				t.Errorf("thinking = %q, want %q", thinking.Thinking, tt.wantThinking)
			}
			// Whatever the variant, the raw records are preserved verbatim for
			// replay: OpenRouter rejects a mangled reasoning_details with a 400,
			// so re-encoding them is never safe.
			if len(thinking.ProviderState) == 0 {
				t.Fatal("ProviderState is empty; the reasoning_details were not captured for replay")
			}
			if thinking.ProviderStateFormat == "" {
				t.Error("ProviderStateFormat is empty; untagged state would be replayed into the wrong dialect")
			}
			var records []map[string]json.RawMessage
			if err := json.Unmarshal(thinking.ProviderState, &records); err != nil {
				t.Fatalf("ProviderState is not the documented array of records: %v", err)
			}
			if len(records) != 1 {
				t.Fatalf("records = %d, want 1", len(records))
			}
			// The answer text is untouched by the reasoning handling.
			var answer string
			for _, block := range resp.Message.Blocks {
				if text, ok := block.(*content.TextBlock); ok {
					answer += text.Text
				}
			}
			if answer != "Paris." {
				t.Errorf("answer = %q, want Paris.", answer)
			}
		})
	}
}

// TestOpenRouterUnknownReasoningVariantRoundTrips is the forward-compatibility
// half of divergence 1, and the reason the records are stored raw. OpenRouter
// adds variants without notice; one this build has never heard of must survive
// byte-for-byte into ProviderState, because the next request replays it to an
// upstream that DOES understand it.
func TestOpenRouterUnknownReasoningVariantRoundTrips(t *testing.T) {
	t.Parallel()

	resp, err := orInvoke(t, "reasoning_details_unknown_variant.json")
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	thinking := orThinkingBlock(t, resp)

	// The known variant still yields readable text.
	if thinking.Thinking != "visible part" {
		t.Errorf("thinking = %q, want the readable record's text", thinking.Thinking)
	}

	var records []map[string]json.RawMessage
	if err := json.Unmarshal(thinking.ProviderState, &records); err != nil {
		t.Fatalf("ProviderState is not an array of records: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("records = %d, want both the known and the unknown variant preserved", len(records))
	}
	var kind string
	if err := json.Unmarshal(records[1]["type"], &kind); err != nil || kind != "reasoning.quantum_trace" {
		t.Errorf("record 1 type = %q (err %v), want the unknown variant kept as-is", kind, err)
	}
	// Its members must survive untouched, including ones this build cannot
	// interpret. Compared structurally: the state is stored as the provider's
	// own bytes, whitespace and all, so a literal-string compare would be
	// asserting the fixture's formatting rather than the round trip.
	var payload struct {
		Nested []int `json:"nested"`
	}
	if err := json.Unmarshal(records[1]["payload"], &payload); err != nil {
		t.Fatalf("unknown record payload did not survive: %v (%s)", err, records[1]["payload"])
	}
	if len(payload.Nested) != 3 || payload.Nested[0] != 1 || payload.Nested[2] != 3 {
		t.Errorf("unknown record payload = %s, want its nested array intact", records[1]["payload"])
	}
	if _, ok := records[1]["trace_id"]; !ok {
		t.Error("unknown record lost trace_id; unrecognised members must not be stripped")
	}
}

// TestOpenRouterStreamedReasoningDetails is divergence 1 on the streaming path,
// where the records arrive on `delta` and must be accumulated rather than
// overwritten. The encrypted case is the sharp one: it produces no text at all,
// so a decoder that only creates a thinking carrier when it has text to put in
// it would silently drop the replay state.
func TestOpenRouterStreamedReasoningDetails(t *testing.T) {
	t.Parallel()

	tests := []struct {
		fixture      string
		wantThinking string
	}{
		{"reasoning_details_stream.sse", "weighing"},
		{"reasoning_details_encrypted_stream.sse", ""},
	}

	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			t.Parallel()

			out := orStream(t, tt.fixture)
			if out.err != nil {
				t.Fatalf("stream error = %v", out.err)
			}
			if got := orStreamText(out); got != "Paris." {
				t.Errorf("streamed text = %q, want Paris.", got)
			}

			var thinking strings.Builder
			var state json.RawMessage
			var format string
			for _, chunk := range out.chunks {
				c, ok := chunk.(*content.ThinkingChunk)
				if !ok {
					continue
				}
				thinking.WriteString(c.Thinking)
				if len(c.ProviderState) > 0 {
					state = c.ProviderState
					format = c.ProviderStateFormat
				}
			}
			if thinking.String() != tt.wantThinking {
				t.Errorf("thinking = %q, want %q", thinking.String(), tt.wantThinking)
			}
			if len(state) == 0 {
				t.Fatalf("no chunk carried reasoning state; chunks = %#v", out.chunks)
			}
			if format == "" {
				t.Error("state carries no format tag")
			}
			var records []map[string]json.RawMessage
			if err := json.Unmarshal(state, &records); err != nil || len(records) != 1 {
				t.Errorf("streamed state = %s (err %v), want one record", state, err)
			}
		})
	}
}

// TestOpenRouterKeepaliveCommentsAreInert is divergence 3. A `:` line is an SSE
// comment, not a frame; OpenRouter emits it while an upstream is queued, and a
// reader that treats it as data — or that stops at it — loses the turn.
// Schema-backed: the real frames on either side still validate, which is what
// proves the comments did not corrupt the framing.
func TestOpenRouterKeepaliveCommentsAreInert(t *testing.T) {
	t.Parallel()

	// The fixture really does contain the keepalives, so a silent edit cannot
	// turn this into a test of nothing.
	raw := orFixture(t, "keepalive_processing.sse")
	if n := strings.Count(string(raw), ": OPENROUTER PROCESSING"); n != 3 {
		t.Fatalf("fixture holds %d keepalive comments, want 3", n)
	}

	out := orStream(t, "keepalive_processing.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v, want the keepalives ignored", out.err)
	}
	if got := orStreamText(out); got != "Paris." {
		t.Errorf("streamed text = %q, want Paris.", got)
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want a stop finish", out.result, out.ok)
	}
}

// TestOpenRouterUsageWithCostAndCachedTokens is divergence 4. `cost` and
// `cost_details` are OpenRouter's own; `prompt_tokens_details.cached_tokens` is
// standard. The requirement is that the extra members neither break the usage
// decode nor leak into the neutral Usage, and that the standard cached count
// still lands.
func TestOpenRouterUsageWithCostAndCachedTokens(t *testing.T) {
	t.Parallel()

	t.Run("non-streaming", func(t *testing.T) {
		t.Parallel()
		resp, err := orInvoke(t, "usage_cost_and_cached.json")
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("usage = nil, want the decoded counts")
		}
		// prompt_tokens is the TOTAL including cached; the neutral InputTokens
		// is the uncached remainder.
		if resp.Usage.CacheReadTokens != 24 {
			t.Errorf("cache read = %d, want 24", resp.Usage.CacheReadTokens)
		}
		if resp.Usage.OutputTokens != 9 {
			t.Errorf("output = %d, want 9", resp.Usage.OutputTokens)
		}
		if resp.Usage.ReasoningTokens != 5 {
			t.Errorf("reasoning = %d, want 5", resp.Usage.ReasoningTokens)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		t.Parallel()
		out := orStream(t, "usage_cost_stream.sse")
		if out.err != nil {
			t.Fatalf("stream error = %v", out.err)
		}
		if !out.ok || out.result.Usage == nil {
			t.Fatalf("result = %+v ok=%v, want terminal usage", out.result, out.ok)
		}
		if out.result.Usage.CacheReadTokens != 24 || out.result.Usage.OutputTokens != 9 {
			t.Errorf("usage = %+v, want cache=24 output=9", out.result.Usage)
		}
	})
}

// TestOpenRouterErrorInsideHTTP200 is divergence 2, and DECODE-ONLY: a body
// carrying a top-level `error` has no `choices`, which CreateChatCompletion
// requires, so the payload cannot be gated — the illegality IS the divergence.
// An aggregator relaying an upstream failure inside a 200 is the single most
// dangerous shape here: read as a success it yields an empty assistant turn
// with no indication anything went wrong.
func TestOpenRouterErrorInsideHTTP200(t *testing.T) {
	t.Parallel()

	t.Run("non-streaming", func(t *testing.T) {
		t.Parallel()

		const name = "error_on_http_200.json"
		if openrouterUngated[name] == "" {
			t.Fatalf("%s must stay ungated; see the file header", name)
		}
		resp, err := orInvokeBody(t, orFixture(t, name))
		if err == nil {
			t.Fatalf("Invoke() = %+v, nil error; a 200 carrying an error body must not read as success", resp)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		t.Parallel()

		const name = "error_on_http_200.sse"
		if openrouterUngated[name] == "" {
			t.Fatalf("%s must stay ungated; see the file header", name)
		}
		out := orStreamBody(t, orFixture(t, name))
		if out.err == nil {
			t.Fatalf("stream ended cleanly; chunks = %#v result = %+v", out.chunks, out.result)
		}
		// A stream that died must not also report a clean stop.
		if out.ok && out.result.FinishReason == stream.FinishReasonStop {
			t.Errorf("result = %+v, want no clean stop after an in-band error", out.result)
		}
	})
}
