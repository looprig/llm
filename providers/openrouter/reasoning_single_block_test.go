package openrouter

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
)

// TestStreamReasoningAccumulatesIntoExactlyOneBlock pins this provider's
// reasoning INDEX SEMANTICS through the real decode path.
//
// OpenRouter speaks OpenAI Chat, which has one reasoning channel per choice and
// no block index. Its `reasoning_details` extension is an ARRAY carried whole:
// the non-streaming decoder puts the entire array on ONE ThinkingBlock, and the
// streaming wrapper re-encodes every entry seen so far onto every thinking
// chunk it touches. Both the readable deltas and the synthesized zero-text
// carriers therefore have to land on the SAME block — index 0 — or the last
// carrier would open a second block holding a duplicate of the whole array,
// and the replayed request would send the details twice.
//
// This is the assertion that keeps a future "index everything" change from
// splitting one carrier stream in two.
func TestStreamReasoningAccumulatesIntoExactlyOneBlock(t *testing.T) {
	t.Parallel()

	details := `[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"first","signature":"sig-1"},` +
		`{"type":"reasoning.encrypted","index":1,"format":"anthropic-claude-v1","data":"opaque","signature":"sig-2"}]`
	sse := []string{
		`data: {"choices":[{"delta":{"reasoning_content":"first","reasoning_details":[{"type":"reasoning.text","index":0,"format":"anthropic-claude-v1","text":"first","signature":"sig-1"}]}}]}` + "\n\n",
		// An encrypted-only detail: no readable text, so the shared decoder
		// produces no thinking chunk and the wrapper synthesizes a carrier.
		`data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","index":1,"format":"anthropic-claude-v1","data":"opaque","signature":"sig-2"}]}}]}` + "\n\n",
		`data: {"model":"anthropic/claude-sonnet-4","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n",
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

	var acc streamaccumulator.Thinking
	for {
		chunk, nextErr := reader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("Next() error = %v", nextErr)
		}
		if thinking, ok := chunk.(*content.ThinkingChunk); ok {
			acc.Add(thinking)
		}
	}
	streamed := acc.Blocks()
	if len(streamed) != 1 {
		t.Fatalf("accumulated blocks = %d %#v, want exactly one carrier block", len(streamed), streamed)
	}
	if streamed[0].Thinking != "first" {
		t.Errorf("block thinking = %q, want %q", streamed[0].Thinking, "first")
	}
	assertJSONEqual(t, streamed[0].ProviderState, []byte(details))
	assertRawJSONOrder(t, streamed[0].ProviderState, []byte(details))

	// The non-streaming decoder for the SAME response must reconstruct the same
	// single block with the same state (inference/CLAUDE.md's streaming
	// invariant).
	body := []byte(`{"id":"r","model":"anthropic/claude-sonnet-4","choices":[{"message":{"role":"assistant","content":"ok","reasoning":"first","reasoning_details":` + details + `},"finish_reason":"stop"}]}`)
	response, err := (requestCodec{}).DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	var direct []content.ThinkingBlock
	for _, block := range response.Message.Blocks {
		if thinking, ok := block.(*content.ThinkingBlock); ok {
			direct = append(direct, *thinking)
		}
	}
	if len(direct) != len(streamed) {
		t.Fatalf("streamed %d reasoning block(s), non-streaming %d: %#v vs %#v", len(streamed), len(direct), streamed, direct)
	}
	for i := range direct {
		if streamed[i].Thinking != direct[i].Thinking || streamed[i].ProviderStateFormat != direct[i].ProviderStateFormat {
			t.Errorf("block %d: streamed %q/%q, non-streaming %q/%q", i,
				streamed[i].Thinking, streamed[i].ProviderStateFormat, direct[i].Thinking, direct[i].ProviderStateFormat)
		}
		assertJSONEqual(t, streamed[i].ProviderState, direct[i].ProviderState)
	}
	var parsed []json.RawMessage
	if err := json.Unmarshal(streamed[0].ProviderState, &parsed); err != nil {
		t.Fatalf("provider state is not a reasoning_details array: %v", err)
	}
	if len(parsed) != 2 {
		t.Errorf("provider state holds %d detail(s), want both entries exactly once", len(parsed))
	}
}
