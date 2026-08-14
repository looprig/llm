package azure_test

import (
	"context"
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
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	responses "github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/azure"
)

// Azure OpenAI RESPONSES PROVIDER-DELTA suite.
//
// Azure speaks the Responses dialect with three divergences the shared
// openairesponses codec deliberately does not model, each compensated for in
// providers/azure/codec.go. Every fixture here is one of them; nothing
// duplicates what providers/openai already covers.
//
//  1. `incomplete_details.reason: "content_filter"`. Azure terminates a
//     filtered turn as status "incomplete" with that reason. The shared codec
//     maps "incomplete" to a length-style finish, which would report a
//     suppressed answer as a merely truncated one.
//  2. `reasoning_text` content. Azure puts reasoning directly in an item's
//     `content` array instead of the `summary` array the shared codec reads.
//  3. `response.failed` AND the top-level `error` event. Azure reports
//     content-management rejections as a stream event rather than an HTTP
//     error, and the envelope is not always readable. BOTH the well-formed and
//     the unreadable case are pinned, because both were swallow defects: a
//     failure whose envelope could not be parsed must still be a failure, or a
//     truncated turn is reported as a clean success. The spec's separate
//     ResponseErrorEvent (`type:"error"`, no enclosing `response`) is pinned
//     alongside them because this codec forks the shared decode half, and the
//     shared codec routes that event through its COLLECTOR — a half the fork
//     does not inherit.
//
// WHAT IS SCHEMA-BACKED, precisely:
//   - the BASE ENVELOPE of every fixture except one. The .json fixtures are
//     held against OpenAI's Response schema and every .sse frame against the
//     Responses stream_event union — which also cross-checks each frame's SSE
//     `event:` name against its payload's own `type`. That is what proves
//     `incomplete_details.reason: "content_filter"` is a real enum member,
//     that `reasoning_text` is a real ReasoningTextContent variant rather than
//     an Azure invention, and that `response.failed` is a real event type.
//
// WHAT IS NOT — three fixtures are DECODE-ONLY, each for a stated reason (see
// azureUngated below, which is asserted against the corpus so the list cannot
// drift):
//   - failed_unreadable_envelope.sse and failed_null_response.sse carry a
//     `response` member that is NOT a legal Response object. That is their
//     whole point; gating them would be a contradiction.
//   - failed_azure_content_filter.sse carries Azure's real
//     `error.code: "content_filter"`, which OpenAI's ResponseErrorCode enum
//     does not contain. The gate found that, and it is a genuine provider
//     divergence rather than a fixture bug — so the legal-enum case is gated
//     separately in failed_well_formed.sse and this one is asserted on
//     behaviour alone.
//   - `additionalProperties` is closed on only 5 of the Responses spec's 147
//     object shapes, so an extra Azure member almost anywhere would pass
//     unremarked. The gate proves these payloads are well-typed, not that they
//     are exhaustive.

const azureFixtureKey = "azure-test-key"

// azureUngated names the fixtures deliberately held OUT of the gate, with the
// reason each one cannot be validated. They are DECODE-ONLY, and the reason is
// recorded here rather than left implicit so nobody later "fixes" them into
// the gate and loses the case they exist to cover.
var azureUngated = map[string]string{
	"failed_unreadable_envelope.sse": "`response` is a bare string, deliberately not a legal Response object",
	"failed_null_response.sse":       "`response` is null, which ResponseFailedEvent does not permit",
	"failed_azure_content_filter.sse": "Azure's `error.code: \"content_filter\"` is NOT a member of " +
		"OpenAI's ResponseErrorCode enum (server_error, rate_limit_exceeded, invalid_prompt, " +
		"the image_* family, …) — a real, gate-detected provider divergence, not a fixture bug",
}

func azureFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

func gateAzureResponse(t testing.TB, name string) []byte {
	t.Helper()
	raw := azureFixture(t, name)
	conformance.MustValidateResponse(t, "openai-responses", "response", raw)
	return raw
}

func gateAzureStream(t testing.TB, name string) []byte {
	t.Helper()
	raw := azureFixture(t, name)
	if n := conformance.MustValidateStream(t, "openai-responses", "stream_event", raw); n == 0 {
		t.Fatalf("%s validated no frames", name)
	}
	return raw
}

func azureServe(t *testing.T, contentType string, body []byte) model.Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return model.CustomModel(
		model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses,
		srv.URL+"/openai/v1", "gpt-4.1", model.WithThinking(),
	)
}

func azureClient(t *testing.T, selected model.Model) inference.Client {
	t.Helper()
	client, err := azure.New(selected, auth.APIKey(azureFixtureKey))
	if err != nil {
		t.Fatalf("azure.New() error = %v", err)
	}
	return client
}

func azurePrompt(selected model.Model) inference.Request {
	return inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	}
}

func azureInvoke(t *testing.T, name string) (*inference.Response, error) {
	t.Helper()
	selected := azureServe(t, "application/json", gateAzureResponse(t, name))
	return azureClient(t, selected).Invoke(context.Background(), azurePrompt(selected))
}

type azureStreamOutcome struct {
	chunks []content.Chunk
	result stream.StreamResult
	ok     bool
	err    error
}

// azureStreamBody drives a stream from raw bytes, so the one deliberately
// ungated fixture can share the driver without pretending it was validated.
func azureStreamBody(t *testing.T, body []byte) azureStreamOutcome {
	t.Helper()
	selected := azureServe(t, "text/event-stream", body)
	reader, err := azureClient(t, selected).Stream(context.Background(), azurePrompt(selected))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()

	var out azureStreamOutcome
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

func azureStream(t *testing.T, name string) azureStreamOutcome {
	t.Helper()
	return azureStreamBody(t, gateAzureStream(t, name))
}

func azureStreamText(out azureStreamOutcome) string {
	var sb strings.Builder
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.TextChunk); ok {
			sb.WriteString(c.Text)
		}
	}
	return sb.String()
}

// TestAzureCorpusIsLegalResponses walks the corpus itself so a fixture added to
// testdata/ can never sit there ungated, and pins both the corpus size and the
// single deliberate exclusion.
func TestAzureCorpusIsLegalResponses(t *testing.T) {
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
	if len(names) != 13 {
		t.Errorf("testdata holds %d fixtures, want 13", len(names))
	}

	var gated int
	for _, name := range names {
		switch {
		case azureUngated[name] != "":
			// Deliberately ungated; see the file header.
		case strings.HasSuffix(name, ".sse"):
			gateAzureStream(t, name)
			gated++
		case strings.HasSuffix(name, ".json"):
			gateAzureResponse(t, name)
			gated++
		default:
			t.Errorf("testdata/%s is neither a .json nor a .sse fixture", name)
		}
	}
	if gated != 10 {
		t.Errorf("gated %d fixtures, want 10", gated)
	}
	for name := range azureUngated {
		if _, err := os.Stat(filepath.Join("testdata", name)); err != nil {
			t.Errorf("azureUngated names %s, which does not exist: %v", name, err)
		}
	}
}

// TestAzureIncompleteContentFilterIsNotJustTruncation is divergence 1. Two
// fixtures, identical but for `incomplete_details.reason`, so the test proves
// the reason is what drives the mapping rather than the status alone.
func TestAzureIncompleteContentFilterIsNotJustTruncation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		fixture string
		want    stream.FinishReason
	}{
		{"incomplete_content_filter.json", stream.FinishReasonContentFilter},
		{"incomplete_max_output_tokens.json", stream.FinishReasonLength},
	}

	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			t.Parallel()
			resp, err := azureInvoke(t, tt.fixture)
			if err != nil {
				t.Fatalf("Invoke(%s) error = %v", tt.fixture, err)
			}
			if resp.FinishReason != tt.want {
				t.Errorf("finish = %v, want %v", resp.FinishReason, tt.want)
			}
			// The partial answer produced before termination survives either way.
			if len(resp.Message.Blocks) == 0 {
				t.Error("blocks = none, want the partial answer preserved")
			}
		})
	}
}

// TestAzureIncompleteContentFilterInAStream is the same divergence on the
// streaming path, where it arrives as a response.incomplete terminal event.
func TestAzureIncompleteContentFilterInAStream(t *testing.T) {
	t.Parallel()

	out := azureStream(t, "incomplete_content_filter.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonContentFilter {
		t.Errorf("result = %+v ok=%v, want a content_filter finish", out.result, out.ok)
	}
	if got := azureStreamText(out); got != "Sure, here is" {
		t.Errorf("streamed text = %q, want the pre-filter partial", got)
	}
}

// TestAzureReasoningTextBecomesThinking is divergence 2: reasoning delivered as
// `reasoning_text` content rather than a `summary` array must still surface as
// a neutral ThinkingBlock, not be dropped on the floor by the shared codec's
// summary-only reader.
func TestAzureReasoningTextBecomesThinking(t *testing.T) {
	t.Parallel()

	resp, err := azureInvoke(t, "reasoning_text_item.json")
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(resp.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want thinking then text", resp.Message.Blocks)
	}
	thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
	if !ok || thinking.Thinking != "direct thought" {
		t.Errorf("block 0 = %#v, want the reasoning_text as thinking", resp.Message.Blocks[0])
	}
	text, ok := resp.Message.Blocks[1].(*content.TextBlock)
	if !ok || text.Text != "the answer" {
		t.Errorf("block 1 = %#v, want the answer", resp.Message.Blocks[1])
	}
}

// TestAzureReasoningTextDeltaBecomesThinkingChunks is the streaming form of the
// same divergence: response.reasoning_text.delta is an Azure event name the
// shared codec does not route.
func TestAzureReasoningTextDeltaBecomesThinkingChunks(t *testing.T) {
	t.Parallel()

	out := azureStream(t, "reasoning_text_delta.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	var thinking strings.Builder
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.ThinkingChunk); ok {
			thinking.WriteString(c.Thinking)
		}
	}
	if thinking.String() != "weighing options" {
		t.Errorf("thinking = %q, want the reasoning_text delta", thinking.String())
	}
	if got := azureStreamText(out); got != "done" {
		t.Errorf("text = %q, want done", got)
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want a stop finish", out.result, out.ok)
	}
}

// TestAzureMultipleReasoningItemsStayDistinct drives a real multi-item Azure
// stream through the accumulator that turns chunks into blocks. Two reasoning
// items must reconstruct as two blocks, each holding the item id OpenAI's
// ReasoningItem marks required for replay — and the block list must match what
// the non-streaming decoder builds from the same response, which is the
// streaming/non-streaming invariant inference/CLAUDE.md states.
//
// Azure's reasoning deltas are handled by this provider's private fork of the
// Responses collector, so the shared codec's fix does not reach them on its
// own; that is exactly what this fixture pins.
func TestAzureMultipleReasoningItemsStayDistinct(t *testing.T) {
	t.Parallel()

	out := azureStream(t, "reasoning_text_multiblock.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	var acc streamaccumulator.Thinking
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.ThinkingChunk); ok {
			acc.Add(c)
		}
	}
	streamed := acc.Blocks()
	if len(streamed) != 2 {
		t.Fatalf("accumulated blocks = %d %#v, want one per reasoning item", len(streamed), streamed)
	}
	if streamed[0].Thinking != "first" || string(streamed[0].ProviderState) != `{"id":"rs_az_a"}` {
		t.Errorf("block 0 = %q/%s, want \"first\" and rs_az_a", streamed[0].Thinking, streamed[0].ProviderState)
	}
	if streamed[1].Thinking != "second" || string(streamed[1].ProviderState) != `{"id":"rs_az_b"}` {
		t.Errorf("block 1 = %q/%s, want \"second\" and rs_az_b", streamed[1].Thinking, streamed[1].ProviderState)
	}

	resp, err := azureInvoke(t, "reasoning_text_multiblock.json")
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	var direct []content.ThinkingBlock
	for _, b := range resp.Message.Blocks {
		if tb, ok := b.(*content.ThinkingBlock); ok {
			direct = append(direct, *tb)
		}
	}
	if len(streamed) != len(direct) {
		t.Fatalf("streamed %d reasoning block(s), non-streaming %d: %#v vs %#v", len(streamed), len(direct), streamed, direct)
	}
	for i := range direct {
		if streamed[i].Thinking != direct[i].Thinking ||
			string(streamed[i].ProviderState) != string(direct[i].ProviderState) ||
			streamed[i].ProviderStateFormat != direct[i].ProviderStateFormat {
			t.Errorf("block %d: streamed %q/%s/%q, non-streaming %q/%s/%q", i,
				streamed[i].Thinking, streamed[i].ProviderState, streamed[i].ProviderStateFormat,
				direct[i].Thinking, direct[i].ProviderState, direct[i].ProviderStateFormat)
		}
	}
}

// TestAzureFailedEventWithAWellFormedEnvelope is divergence 3, readable half:
// the failure must terminate the stream as an error carrying the provider's
// own code, not end it quietly. Fully gated — the envelope is a legal
// ResponseFailedEvent, `error.code` included.
func TestAzureFailedEventWithAWellFormedEnvelope(t *testing.T) {
	t.Parallel()

	out := azureStream(t, "failed_well_formed.sse")
	assertAzureFailure(t, out, "server_error")
}

// TestAzureFailedEventWithAnOffEnumCode records a FINDING the gate itself
// surfaced. Azure reports a content-management rejection as
// `error.code: "content_filter"`, and that value is NOT in OpenAI's
// ResponseErrorCode enum — the spec admits only server_error,
// rate_limit_exceeded, invalid_prompt, data_residency_mismatch, bio_policy,
// vector_store_timeout and the image_* family. The payload is therefore one
// OpenAI's own schema rejects while Azure demonstrably sends it, so the
// fixture is DECODE-ONLY by necessity, not by preference.
//
// The behavioural consequence pinned here: Looprig passes the code through
// verbatim rather than normalising or dropping it, so a caller matching on
// "content_filter" keeps working even though no OpenAI-derived schema will
// ever bless the value.
func TestAzureFailedEventWithAnOffEnumCode(t *testing.T) {
	t.Parallel()

	const name = "failed_azure_content_filter.sse"
	if azureUngated[name] == "" {
		t.Fatalf("%s must stay ungated; see the file header", name)
	}
	assertAzureFailure(t, azureStreamBody(t, azureFixture(t, name)), "content_filter")
}

func assertAzureFailure(t *testing.T, out azureStreamOutcome, wantCode string) {
	t.Helper()

	if out.err == nil {
		t.Fatalf("stream ended with no error; chunks = %#v result = %+v", out.chunks, out.result)
	}
	var streamErr *responses.StreamAPIError
	if !errors.As(out.err, &streamErr) {
		t.Fatalf("error = %T %v, want *openairesponses.StreamAPIError", out.err, out.err)
	}
	if streamErr.Code != wantCode {
		t.Errorf("code = %q, want %q", streamErr.Code, wantCode)
	}
	if streamErr.Message == "" {
		t.Error("message is empty; the provider's own explanation was dropped")
	}
	// The partial text delivered before the failure is not retracted.
	if got := azureStreamText(out); got != "Sure, here is" {
		t.Errorf("streamed text = %q, want the pre-failure partial", got)
	}
	// A failed turn must not also report a successful terminal result.
	if out.ok && out.result.FinishReason == stream.FinishReasonStop {
		t.Errorf("result = %+v, want no clean stop after a failure", out.result)
	}
}

// TestAzureTopLevelErrorEventTerminatesTheStream is divergence 3's third
// channel, and the one this fork silently opted out of.
//
// The Responses spec defines TWO independent failure events: ResponseFailedEvent
// (`response.failed`, wrapping a Response object) and ResponseErrorEvent
// (`type:"error"`, carrying code/message/param at the top level with no
// enclosing `response`). The shared codec handles the second in its stream
// COLLECTOR, not in decodeEnvelope — decodeEnvelope deliberately returns
// (nil, nil) for it. providers/azure forked the decode half only, so its
// default branch delegates to the shared DecodeEvent, gets that (nil, nil), and
// a mid-generation failure ends the stream at natural EOF as a clean success.
//
// Fully gated: ResponseErrorEvent types `code` as a nullable free string rather
// than the ResponseErrorCode enum, so a real provider code is legal here.
func TestAzureTopLevelErrorEventTerminatesTheStream(t *testing.T) {
	t.Parallel()

	out := azureStream(t, "error_event.sse")
	assertAzureFailure(t, out, "server_error")
	// The stream must also not claim a terminal result: no response.completed
	// was ever seen, so there is nothing to report.
	if out.ok {
		t.Errorf("result = %+v ok=true, want no terminal result after an error event", out.result)
	}
}

// TestAzureFailedEventWithAnUnreadableEnvelope is divergence 3, unreadable
// half, and the one DECODE-ONLY case in this suite: the fixture is not gated
// because its `response` member is deliberately not a legal Response object.
//
// This was a swallow defect. An envelope the codec cannot parse must degrade
// only the DIAGNOSTICS — the failure itself is unconditional, or a stream that
// died mid-turn is reported to the caller as a complete one.
func TestAzureFailedEventWithAnUnreadableEnvelope(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"failed_unreadable_envelope.sse", "failed_null_response.sse"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if azureUngated[name] == "" {
				t.Fatalf("%s is gated; this test only covers illegal envelopes", name)
			}
			out := azureStreamBody(t, azureFixture(t, name))
			if out.err == nil {
				t.Fatalf("stream ended with no error; chunks = %#v result = %+v", out.chunks, out.result)
			}
			var streamErr *responses.StreamAPIError
			if !errors.As(out.err, &streamErr) {
				t.Fatalf("error = %T %v, want *openairesponses.StreamAPIError", out.err, out.err)
			}
			// Diagnostics are legitimately empty here; the failure is not.
			if streamErr.Code != "" {
				t.Errorf("code = %q, want empty for an unreadable envelope", streamErr.Code)
			}
			if out.ok && out.result.FinishReason == stream.FinishReasonStop {
				t.Errorf("result = %+v, want no clean stop after a failure", out.result)
			}
		})
	}
}

// TestAzureFailedStatusInANonStreamingResponse pins the non-streaming
// counterpart: a body whose status is "failed" is an error, never an empty
// success.
func TestAzureFailedStatusInANonStreamingResponse(t *testing.T) {
	t.Parallel()

	resp, err := azureInvoke(t, "failed_status.json")
	if err == nil {
		t.Fatalf("Invoke() = %+v, nil error; want a failure for status:failed", resp)
	}
}
