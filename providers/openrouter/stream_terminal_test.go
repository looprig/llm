package openrouter

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
)

// errUpstream is a terminal that is NOT io.EOF — a dropped connection, a
// mid-stream transport fault, anything the body reports other than a clean end.
var errUpstream = errors.New("openrouter test: upstream failed")

// terminalReadCloser serves chunks and then returns terminal, which may be any
// error including a WRAPPED io.EOF. chunkedReadCloser cannot express either
// case: it always ends with a bare io.EOF.
type terminalReadCloser struct {
	chunks   []string
	terminal error
}

func (r *terminalReadCloser) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, r.terminal
	}
	n := copy(p, r.chunks[0])
	r.chunks[0] = r.chunks[0][n:]
	if r.chunks[0] == "" {
		r.chunks = r.chunks[1:]
	}
	return n, nil
}

func (*terminalReadCloser) Close() error { return nil }

func decodeTerminalStream(t *testing.T, chunks []string, terminal error) ([]content.Chunk, error) {
	t.Helper()
	reader, err := (requestCodec{}).DecodeStream(&http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       &terminalReadCloser{chunks: chunks, terminal: terminal},
	})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	var chunksOut []content.Chunk
	for {
		chunk, nextErr := reader.Next()
		if nextErr != nil {
			return chunksOut, nextErr
		}
		chunksOut = append(chunksOut, chunk)
	}
}

// encryptedOnlyDelta carries reasoning_details with NO readable text, which is
// the Claude tool-loop shape: the shared OpenAI decoder produces no thinking
// chunk for it, so this wrapper owes the stream a synthetic carrier.
const encryptedOnlyDelta = `data: {"choices":[{"delta":{"reasoning_details":` +
	`[{"type":"reasoning.encrypted","index":0,"format":"anthropic-claude-v1","data":"opaque","signature":"sig"}]}}]}` + "\n\n"

// TestNonEOFTerminalStillDeliversUndeliveredReasoningState is the defect: the
// undelivered-state rescue was reachable only under errors.Is(err, io.EOF), so
// a stream that died any other way threw away encrypted reasoning state no
// chunk had carried. Encrypted state is the ONLY continuation the next request
// has — it cannot be reconstructed from the text — so losing it silently
// disables reasoning replay for the rest of the session.
//
// The fix returns the carrier with a nil error and stashes the terminal for the
// following Next(), so the terminal is delayed by exactly one chunk and never
// swallowed.
func TestNonEOFTerminalStillDeliversUndeliveredReasoningState(t *testing.T) {
	t.Parallel()

	chunks, err := decodeTerminalStream(t, []string{encryptedOnlyDelta}, errUpstream)
	if !errors.Is(err, errUpstream) {
		t.Fatalf("terminal error = %v, want errUpstream; the terminal must be delayed, never swallowed", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v, want exactly one synthesized carrier", chunks)
	}
	thinking, ok := chunks[0].(*content.ThinkingChunk)
	if !ok {
		t.Fatalf("chunk = %T, want *content.ThinkingChunk", chunks[0])
	}
	if thinking.Thinking != "" {
		t.Errorf("carrier Thinking = %q, want empty: the details carry no readable text", thinking.Thinking)
	}
	if thinking.ProviderStateFormat != openRouterReasoningDetailsFormat {
		t.Errorf("carrier ProviderStateFormat = %q, want %q", thinking.ProviderStateFormat, openRouterReasoningDetailsFormat)
	}
	var entries []map[string]any
	if err := json.Unmarshal(thinking.ProviderState, &entries); err != nil {
		t.Fatalf("carrier ProviderState is not the reasoning_details array: %v (%s)", err, thinking.ProviderState)
	}
	if len(entries) != 1 || entries[0]["data"] != "opaque" {
		t.Errorf("carrier ProviderState = %s, want the captured encrypted entry", thinking.ProviderState)
	}
}

// TestTheRescuedCarrierSurvivesHarnessTruncation checks the claim that made the
// fix worth making, rather than assuming it. harness's truncation policy keeps
// only thinking blocks it can prove are COMPLETE: sealedReasoning requires the
// block to have content (hasReasoning: readable text OR provider state) AND to
// be sealed (a signature OR provider state). A zero-text carrier holding
// encrypted state satisfies both through the same field, so it now survives a
// truncated turn instead of vanishing with the rest of the failed step.
//
// The predicate itself is unexported in harness/internal/loopruntime, and llm
// cannot import it (both are tier 3). What IS checked here is the input side of
// the claim, end to end through the real accumulator harness uses: the carrier
// folds into a ThinkingBlock whose Thinking is empty and whose ProviderState is
// non-empty. That block satisfies both halves of sealedReasoning by inspection
// — len(ProviderState) > 0 is literally the second disjunct of each.
func TestTheRescuedCarrierSurvivesHarnessTruncation(t *testing.T) {
	t.Parallel()

	chunks, err := decodeTerminalStream(t, []string{encryptedOnlyDelta}, errUpstream)
	if !errors.Is(err, errUpstream) {
		t.Fatalf("terminal error = %v, want errUpstream", err)
	}

	var accumulator streamaccumulator.Thinking
	for _, chunk := range chunks {
		if thinking, ok := chunk.(*content.ThinkingChunk); ok {
			accumulator.Add(thinking)
		}
	}
	blocks := accumulator.Blocks()
	if len(blocks) != 1 {
		t.Fatalf("accumulated blocks = %d, want 1; a dropped carrier means nothing to keep", len(blocks))
	}
	block := blocks[0]

	// hasReasoning: Thinking != "" || len(ProviderState) > 0.
	if block.Thinking == "" && len(block.ProviderState) == 0 {
		t.Fatal("block fails hasReasoning; truncation would drop it as content-free")
	}
	// sealedReasoning adds: Signature != "" || len(ProviderState) > 0.
	if block.Signature == "" && len(block.ProviderState) == 0 {
		t.Fatal("block fails sealedReasoning; truncation would drop it as unsealed")
	}
	if !block.ReplayableAs(openRouterReasoningDetailsFormat) {
		t.Errorf("kept block is not replayable to OpenRouter: format = %q", block.ProviderStateFormat)
	}
}

// TestCleanEOFStillDeliversUndeliveredReasoningState is the positive control on
// the change: the path that already worked must keep working, and the stream
// must still terminate with io.EOF rather than the deferred terminal leaking as
// something else.
func TestCleanEOFStillDeliversUndeliveredReasoningState(t *testing.T) {
	t.Parallel()

	sse := []string{
		encryptedOnlyDelta,
		`data: {"model":"anthropic/claude-sonnet-4","choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	chunks, err := decodeTerminalStream(t, sse, io.EOF)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", err)
	}
	var carriers int
	for _, chunk := range chunks {
		if thinking, ok := chunk.(*content.ThinkingChunk); ok && len(thinking.ProviderState) > 0 {
			carriers++
		}
	}
	if carriers != 1 {
		t.Errorf("state-carrying chunks = %d, want exactly 1; the state must be delivered once, not lost or duplicated", carriers)
	}
}

// TestDeliveredStateIsNotResentOnANonEOFTerminal is the other positive control.
// The rescue must fire only for state NOTHING has carried; a stream whose
// details were already attached to a real thinking chunk must not have them
// replayed as a second block when it later fails, because the ordered
// reasoning_details array is an indivisible unit and a duplicate would be
// replayed twice.
func TestDeliveredStateIsNotResentOnANonEOFTerminal(t *testing.T) {
	t.Parallel()

	sse := []string{
		`data: {"choices":[{"delta":{"reasoning":"readable","reasoning_details":` +
			`[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"readable","signature":"sig"}]}}]}` + "\n\n",
	}
	chunks, err := decodeTerminalStream(t, sse, errUpstream)
	if !errors.Is(err, errUpstream) {
		t.Fatalf("terminal error = %v, want errUpstream", err)
	}
	var carriers int
	for _, chunk := range chunks {
		if thinking, ok := chunk.(*content.ThinkingChunk); ok && len(thinking.ProviderState) > 0 {
			carriers++
		}
	}
	if carriers != 1 {
		t.Errorf("state-carrying chunks = %d, want exactly 1 (the real thinking chunk); "+
			"a rescued duplicate would replay the reasoning_details array twice", carriers)
	}
}

// TestWrappedEOFOnTheBodyIsACleanFinish is the io.EOF-comparison defect.
// reasoningResponseBody.Read compared err == io.EOF, so a body whose terminal
// EOF arrives WRAPPED — anything that adds framing context on the way out —
// was recorded as a failure rather than an end of input, and the atEOF flush
// (processLines(true)) never ran.
//
// The observable consequence is the last event. splitSSELine only yields a line
// that has no terminator when it is told the input has ended, so a final
// unterminated event is held in the pending buffer and silently discarded — and
// here that event is the one carrying finish_reason, so a complete response
// arrives as a truncated one. errors.Is is the only comparison that survives a
// wrapper.
//
// The bare-EOF case below is the control: identical bytes, unwrapped terminal.
// If it ever stops completing, this test is measuring the framer rather than
// the comparison and must be rewritten.
func TestWrappedEOFOnTheBodyIsACleanFinish(t *testing.T) {
	t.Parallel()

	// No trailing newline: the final event is terminated by end-of-input alone.
	unterminatedTail := `data: {"model":"anthropic/claude-sonnet-4","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`

	for _, tc := range []struct {
		name     string
		terminal error
	}{
		{"bare io.EOF (control)", io.EOF},
		{"wrapped io.EOF", fmt.Errorf("framed body: %w", io.EOF)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			chunks, err := decodeTerminalStream(t, []string{encryptedOnlyDelta, unterminatedTail}, tc.terminal)
			if !errors.Is(err, io.EOF) {
				t.Fatalf("terminal error = %v, want io.EOF; the final event was discarded", err)
			}
			var text string
			var carriers int
			for _, chunk := range chunks {
				switch c := chunk.(type) {
				case *content.TextChunk:
					text += c.Text
				case *content.ThinkingChunk:
					if len(c.ProviderState) > 0 {
						carriers++
					}
				}
			}
			if text != "hi" {
				t.Errorf("decoded text = %q, want %q: the unterminated final event was dropped", text, "hi")
			}
			if carriers != 1 {
				t.Errorf("state-carrying chunks = %d, want 1", carriers)
			}
		})
	}
}
