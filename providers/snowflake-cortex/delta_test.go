package snowflake_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	snowflake "github.com/looprig/llm/providers/snowflake-cortex"
)

// Snowflake Cortex PROVIDER-DELTA suite.
//
// Cortex speaks OpenAI Chat Completions with exactly one request-side
// divergence this adapter compensates for: Cortex names the output limit
// `max_completion_tokens` and does not accept OpenAI's legacy `max_tokens`, so
// the client rewrites the member after the shared openaiapi encoder has run.
//
// WHAT IS SCHEMA-BACKED HERE, precisely:
//   - the body Cortex actually receives, AFTER the rewrite, is held against
//     OpenAI's own CreateChatCompletionRequest. That is what proves the rewrite
//     produced a still-legal Chat Completions body rather than a mangled one:
//     the right JSON type in the right place, `messages` intact, nothing
//     structurally broken by the map round-trip the patch performs.
//
// WHAT IS NOT:
//   - that `max_completion_tokens` is the member Cortex WANTS. Both spellings
//     are individually legal in OpenAI's schema, so the gate is indifferent
//     between them; the rename is a decode-only, assertion-only claim resting
//     on Snowflake's documentation.
//   - the ABSENCE of `max_tokens`. OpenAI's Chat Completions spec closes
//     `additionalProperties` on only 3 of its 54 object shapes, and the request
//     root is not one of them, so a body carrying BOTH members would still
//     validate. The explicit assertion below is the only thing catching that.

const cortexToken = "snowflake-token"

// cortexCapture drives the real Cortex client against a recording server,
// gate-validates the body Cortex would have received, and returns it.
func cortexCapture(t *testing.T, req func(model.Model) inference.Request, opts ...model.ModelOption) []byte {
	t.Helper()

	bodies := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodies <- raw
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderSnowflakeCortex), model.APIFormatOpenAI,
		srv.URL, "llama3.1-70b", opts...,
	)
	client, err := snowflake.New(selected, auth.APIKey(cortexToken), snowflake.WithAccount("account"))
	if err != nil {
		t.Fatalf("snowflake.New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), req(selected)); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	select {
	case body := <-bodies:
		conformance.MustValidateRequest(t, "openai", "chat_completion_request", body)
		return body
	default:
		t.Fatal("no request body captured")
		return nil
	}
}

func cortexPrompt(selected model.Model) inference.Request {
	return inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	}
}

func cortexBody(t *testing.T, raw []byte) map[string]json.RawMessage {
	t.Helper()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode body: %v\n%s", err, raw)
	}
	return body
}

// TestCortexRewritesMaxTokens is the delta itself. Schema-backed: the rewritten
// body is a legal Chat Completions request. Assertion-only: which member
// carries the limit, and that the legacy one is gone.
func TestCortexRewritesMaxTokens(t *testing.T) {
	t.Parallel()

	body := cortexBody(t, cortexCapture(t, cortexPrompt,
		model.WithSampling(model.Sampling{MaxTokens: intPtr(42)})))

	if got := string(body["max_completion_tokens"]); got != "42" {
		t.Errorf("max_completion_tokens = %s, want 42", got)
	}
	if _, ok := body["max_tokens"]; ok {
		t.Error("request still carries the legacy max_tokens Cortex does not accept")
	}
	// The rewrite runs over a decoded map and re-marshals: prove the rest of
	// the body survived it, not just the member it touched.
	if _, ok := body["messages"]; !ok {
		t.Error("the body patch lost messages")
	}
}

// TestCortexRewriteIsAbsentWhenNoLimitIsSet pins that the patch adds nothing on
// its own: a request with no token limit must not acquire an invented one.
func TestCortexRewriteIsAbsentWhenNoLimitIsSet(t *testing.T) {
	t.Parallel()

	body := cortexBody(t, cortexCapture(t, cortexPrompt))
	for _, member := range []string{"max_tokens", "max_completion_tokens"} {
		if _, ok := body[member]; ok {
			t.Errorf("request carries %q with no limit configured", member)
		}
	}
}

// TestCortexRewritePreservesAnExistingCompletionLimit pins the precedence rule
// in the patch: where the encoder already produced max_completion_tokens (the
// reasoning-capability path in openaiapi), that value wins and is not
// overwritten by a legacy max_tokens. Both members can never ship together.
func TestCortexRewritePreservesAnExistingCompletionLimit(t *testing.T) {
	t.Parallel()

	body := cortexBody(t, cortexCapture(t, cortexPrompt,
		model.WithThinking(),
		model.WithSampling(model.Sampling{MaxTokens: intPtr(99)})))

	if got := string(body["max_completion_tokens"]); got != "99" {
		t.Errorf("max_completion_tokens = %s, want 99", got)
	}
	if _, ok := body["max_tokens"]; ok {
		t.Error("request carries both token-limit spellings")
	}
}

// TestCortexRewriteSurvivesAToolCallingBody holds the patch against a body with
// real structure in it — tools, a tool call, a tool result — because the patch
// decodes the whole request into a map and re-encodes it. A rewrite that is
// correct on a two-member body and lossy on a real one is the failure mode
// worth pinning.
func TestCortexRewriteSurvivesAToolCallingBody(t *testing.T) {
	t.Parallel()

	raw := cortexCapture(t, func(selected model.Model) inference.Request {
		return inference.Request{
			Model:  selected,
			System: "be brief",
			Messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{
					Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "weather?"}},
				}},
				&content.AIMessage{Message: content.Message{
					Role: content.RoleAssistant,
					Blocks: []content.Block{&content.ToolUseBlock{
						ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"city":"NYC"}`),
					}},
				}},
				&content.ToolResultMessage{
					Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "sunny"}}},
					ToolUseID: "call_1",
				},
			},
			Tools: []inference.Tool{{
				Name:   "lookup",
				Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			}},
			ToolChoice: inference.ToolRequired(),
		}
	}, model.WithTools(), model.WithSampling(model.Sampling{MaxTokens: intPtr(7)}))

	body := cortexBody(t, raw)
	if got := string(body["max_completion_tokens"]); got != "7" {
		t.Errorf("max_completion_tokens = %s, want 7", got)
	}
	if _, ok := body["max_tokens"]; ok {
		t.Error("request still carries max_tokens")
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("decode messages: %v", err)
	}
	if len(messages) != 4 {
		t.Errorf("messages = %d, want system/user/assistant/tool after the patch", len(messages))
	}
	if _, ok := body["tools"]; !ok {
		t.Error("the body patch lost tools")
	}
	if got := string(body["tool_choice"]); got != `"required"` {
		t.Errorf("tool_choice = %s, want \"required\" to survive the patch", got)
	}
}
