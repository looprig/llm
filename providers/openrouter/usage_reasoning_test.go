package openrouter_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openrouter"
)

// TestInvokeKeepsGenerationWhenReasoningExceedsCompletion pins a live OpenRouter
// HTTP 200 that Looprig used to destroy. Against
// nvidia/nemotron-3-ultra-550b-a55b:free OpenRouter returned a complete answer
// with completion_tokens=216 and completion_tokens_details.reasoning_tokens=226
// — reasoning larger than the completion count it is documented to be a
// breakdown of.
//
// OpenRouter documents the subset relationship rather than an additive one:
// "Reasoning tokens are considered output tokens and charged accordingly"
// (https://openrouter.ai/docs/use-cases/reasoning-tokens), and its ResponseUsage
// type calls completion_tokens_details a "Breakdown of completion tokens" while
// defining total_tokens as the "Sum of the above two fields", prompt and
// completion, with no reasoning addend
// (https://openrouter.ai/docs/api-reference/overview). So these numbers are
// OpenRouter contradicting its own contract, not a documented dialect delta —
// there is no published arithmetic that would repair them, and inventing one
// would be worse than carrying what was reported.
//
// What must not happen is the answer being thrown away over the discrepancy.
func TestInvokeKeepsGenerationWhenReasoningExceedsCompletion(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"gen-1","model":"nvidia/nemotron-3-ultra-550b-a55b:free",`+
			`"choices":[{"message":{"role":"assistant","content":"the answer"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":31,"completion_tokens":216,"completion_tokens_details":{"reasoning_tokens":226}}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenRouter),
		model.APIFormatOpenAI,
		srv.URL+"/api/v1/",
		"nvidia/nemotron-3-ultra-550b-a55b:free",
	)
	client, err := openrouter.New(selected, auth.APIKey("sk-or-test"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	resp, err := client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v; a completed generation must survive an accounting mismatch", err)
	}
	if resp == nil || resp.Message == nil || len(resp.Message.Blocks) == 0 {
		t.Fatalf("Invoke() response = %+v, want the assistant content", resp)
	}
	text, ok := resp.Message.Blocks[0].(*content.TextBlock)
	if !ok || text.Text != "the answer" {
		t.Fatalf("first block = %#v, want TextBlock %q", resp.Message.Blocks[0], "the answer")
	}

	// The reported counts are carried through exactly as OpenRouter sent them.
	// Reconciling them would require arithmetic no provider document authorizes.
	want := content.Usage{InputTokens: 31, OutputTokens: 216, ReasoningTokens: 226}
	if resp.Usage == nil || *resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.Usage.ReasoningWithinOutput() {
		t.Error("ReasoningWithinOutput() = true; this usage violates the documented convention and must report so")
	}
}
