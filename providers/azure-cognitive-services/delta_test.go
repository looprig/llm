package azurecognitive_test

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
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	azurecognitive "github.com/looprig/llm/providers/azure-cognitive-services"
)

// Azure Cognitive Services CHAT PROVIDER-DELTA suite.
//
// ACS speaks OpenAI Chat Completions, and every shape here is one OpenAI
// itself never sends. Azure's content-management layer bolts additional
// members onto the standard envelope and, in the streaming case, emits a
// leading frame with an EMPTY choices array carrying only prompt-level filter
// verdicts. Each fixture is a shape the audit proved diverges; none duplicates
// what providers/openai already covers.
//
// WHAT IS SCHEMA-BACKED, precisely:
//   - the BASE ENVELOPE. Every .json fixture is held against OpenAI's
//     CreateChatCompletion and every .sse frame against
//     CreateChatCompletionStreamResponse before any decoder sees it. That is
//     what proves an empty-choices chunk is legal (the chunk schema requires
//     `choices` to be present but sets no minItems), that `finish_reason:
//     "content_filter"` is a real enum member rather than an invention, and
//     that the Azure members sit alongside a still-well-formed OpenAI object.
//
// WHAT IS NOT SCHEMA-BACKED, and must not be read as such:
//   - the Azure EXTENSIONS THEMSELVES. `prompt_filter_results`,
//     `content_filter_results` and `innererror` are absent from OpenAI's spec.
//     They pass the gate only because OpenAI closes `additionalProperties` on
//     3 of its 54 Chat Completions object shapes, and none of the ones these
//     members hang off is among them. The gate is therefore INDIFFERENT to
//     their spelling, nesting and value types; it would accept
//     `prompt_filter_result` singular just as readily. Nothing below is
//     evidence that Azure's own schema is satisfied.
//   - the ERROR fixture. There is no error kind in the gate's index for this
//     format, so responsible_ai_policy_violation.json is DECODE-ONLY: it is
//     asserted purely through the client's observable behaviour.

const acsKey = "azure-acs-key"

func acsFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

// gateACSObject validates a non-streaming fixture's BASE ENVELOPE.
func gateACSObject(t testing.TB, name string) []byte {
	t.Helper()
	raw := acsFixture(t, name)
	conformance.MustValidateResponse(t, "openai", "chat_completion", raw)
	return raw
}

// gateACSStream validates every frame of an SSE fixture's BASE ENVELOPE.
func gateACSStream(t testing.TB, name string) []byte {
	t.Helper()
	raw := acsFixture(t, name)
	if n := conformance.MustValidateStream(t, "openai", "chat_completion_chunk", raw); n == 0 {
		t.Fatalf("%s validated no frames", name)
	}
	return raw
}

func acsServe(t *testing.T, status int, contentType string, body []byte) model.Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return model.CustomModel(
		model.ProviderName(llm.ProviderAzureCognitiveServices), model.APIFormatOpenAI,
		srv.URL, "gpt-4o", model.WithTools(),
	)
}

func acsClient(t *testing.T, selected model.Model) inference.Client {
	t.Helper()
	client, err := azurecognitive.New(selected, auth.APIKey(acsKey), azurecognitive.WithResourceName("resource"))
	if err != nil {
		t.Fatalf("azurecognitive.New() error = %v", err)
	}
	return client
}

func acsPrompt(selected model.Model) inference.Request {
	return inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	}
}

func acsInvoke(t *testing.T, name string) (*inference.Response, error) {
	t.Helper()
	selected := acsServe(t, http.StatusOK, "application/json", gateACSObject(t, name))
	return acsClient(t, selected).Invoke(context.Background(), acsPrompt(selected))
}

type acsStreamOutcome struct {
	chunks []content.Chunk
	result stream.StreamResult
	ok     bool
	err    error
}

func acsStream(t *testing.T, name string) acsStreamOutcome {
	t.Helper()
	selected := acsServe(t, http.StatusOK, "text/event-stream", gateACSStream(t, name))
	reader, err := acsClient(t, selected).Stream(context.Background(), acsPrompt(selected))
	if err != nil {
		t.Fatalf("Stream(%s) error = %v", name, err)
	}
	defer func() { _ = reader.Close() }()

	var out acsStreamOutcome
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

// TestACSCorpusIsLegalOpenAI walks the fixture corpus itself so a file added to
// testdata/ can never sit there ungated, and pins the corpus size. The error
// envelope is excluded by name because the gate has no error kind for this
// format — see the file header.
func TestACSCorpusIsLegalOpenAI(t *testing.T) {
	t.Parallel()

	const errorFixture = "responsible_ai_policy_violation.json"
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
	if len(names) != 8 {
		t.Errorf("testdata holds %d fixtures, want 8", len(names))
	}

	var gated int
	for _, name := range names {
		switch {
		case name == errorFixture:
			// decode-only, deliberately.
		case strings.HasSuffix(name, ".sse"):
			gateACSStream(t, name)
			gated++
		case strings.HasSuffix(name, ".json"):
			gateACSObject(t, name)
			gated++
		default:
			t.Errorf("testdata/%s is neither a .json nor a .sse fixture", name)
		}
	}
	if gated != 7 {
		t.Errorf("gated %d fixtures, want 7", gated)
	}
}

// TestACSEmptyFirstChunkYieldsNoContent is the streaming delta. Azure opens
// every filtered stream with a frame whose `choices` array is EMPTY, carrying
// only prompt_filter_results. Schema-backed: such a frame is a legal
// chat.completion.chunk. Assertion-only: that the decoder treats it as
// contentless rather than as a malformed frame or an end-of-stream.
func TestACSEmptyFirstChunkYieldsNoContent(t *testing.T) {
	t.Parallel()

	out := acsStream(t, "empty_first_chunk.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v, want the empty leading frame to be tolerated", out.err)
	}
	var text strings.Builder
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.TextChunk); ok {
			text.WriteString(c.Text)
		}
	}
	if text.String() != "Hello" {
		t.Errorf("streamed text = %q, want Hello", text.String())
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonStop {
		t.Errorf("result = %+v ok=%v, want a stop finish", out.result, out.ok)
	}
}

// TestACSEmptyFirstChunkAheadOfToolCalls repeats the delta on the tool-calling
// path, where the leading empty frame lands before the tool_calls deltas the
// runner accumulates by index — the place a mis-handled empty frame would
// corrupt state rather than merely be ignored.
func TestACSEmptyFirstChunkAheadOfToolCalls(t *testing.T) {
	t.Parallel()

	out := acsStream(t, "tool_call_with_filter_results.sse")
	if out.err != nil {
		t.Fatalf("stream error = %v", out.err)
	}
	var id, name, args string
	for _, chunk := range out.chunks {
		if c, ok := chunk.(*content.ToolUseChunk); ok {
			if c.ID != "" {
				id = c.ID
			}
			if c.Name != "" {
				name = c.Name
			}
			args += c.InputJSON
		}
	}
	if id != "call_acs_1" || name != "lookup" || args != `{"city":"NYC"}` {
		t.Errorf("tool call = id %q name %q args %q, want the accumulated lookup call", id, name, args)
	}
	if !out.ok || out.result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("result = %+v ok=%v, want a tool_calls finish", out.result, out.ok)
	}
}

// TestACSContentFilterResultsDoNotDisturbContent pins that the per-choice and
// prompt-level filter verdicts Azure attaches are inert: they must neither
// break the decode nor leak into the neutral message. Schema-backed: the
// envelope around them is a legal chat.completion. Assertion-only: everything
// about the extension members themselves.
func TestACSContentFilterResultsDoNotDisturbContent(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"content_filter_results.json", "prompt_filter_only.json"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			resp, err := acsInvoke(t, name)
			if err != nil {
				t.Fatalf("Invoke(%s) error = %v", name, err)
			}
			if resp == nil || resp.Message == nil {
				t.Fatalf("Invoke(%s) = %+v, want a decoded message", name, resp)
			}
			if len(resp.Message.Blocks) != 1 {
				t.Fatalf("blocks = %#v, want a single text block", resp.Message.Blocks)
			}
			text, ok := resp.Message.Blocks[0].(*content.TextBlock)
			if !ok || text.Text == "" {
				t.Errorf("block = %#v, want the assistant text", resp.Message.Blocks[0])
			}
			if resp.FinishReason != stream.FinishReasonStop {
				t.Errorf("finish = %v, want stop", resp.FinishReason)
			}
		})
	}
}

// TestACSFinishReasonContentFilter pins the terminal case: Azure truncates the
// completion and reports finish_reason "content_filter" with a null message
// content. The neutral finish reason must carry that through rather than
// flattening it to a normal stop, because a caller cannot otherwise tell a
// complete answer from a suppressed one.
func TestACSFinishReasonContentFilter(t *testing.T) {
	t.Parallel()

	t.Run("non-streaming", func(t *testing.T) {
		t.Parallel()
		resp, err := acsInvoke(t, "finish_reason_content_filter.json")
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if resp == nil {
			t.Fatal("Invoke() = nil")
		}
		if resp.FinishReason != stream.FinishReasonContentFilter {
			t.Errorf("finish = %v, want content_filter", resp.FinishReason)
		}
		if resp.Message != nil && len(resp.Message.Blocks) != 0 {
			t.Errorf("blocks = %#v, want none for a filtered completion", resp.Message.Blocks)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		t.Parallel()
		out := acsStream(t, "finish_reason_content_filter.sse")
		if out.err != nil {
			t.Fatalf("stream error = %v", out.err)
		}
		if !out.ok || out.result.FinishReason != stream.FinishReasonContentFilter {
			t.Errorf("result = %+v ok=%v, want a content_filter finish", out.result, out.ok)
		}
		// The partial text emitted before the filter fired is still delivered:
		// dropping it would hide that the model had begun to answer.
		var text strings.Builder
		for _, chunk := range out.chunks {
			if c, ok := chunk.(*content.TextChunk); ok {
				text.WriteString(c.Text)
			}
		}
		if text.String() != "Sure, here is" {
			t.Errorf("streamed text = %q, want the pre-filter partial", text.String())
		}
	})
}

// TestACSEmptyChoicesWithoutErrorIsAFailure pins the non-streaming counterpart
// of the empty-choices frame. In a STREAM it is a routine prompt-filter
// preamble; in a completed response it means no answer was produced at all,
// and returning an empty success would let a caller mistake suppression for an
// empty answer.
func TestACSEmptyChoicesWithoutErrorIsAFailure(t *testing.T) {
	t.Parallel()

	resp, err := acsInvoke(t, "empty_choices_only.json")
	if err == nil {
		t.Fatalf("Invoke() = %+v, nil error; want a failure for a choice-less response", resp)
	}
}

// TestACSResponsibleAIPolicyViolation is DECODE-ONLY: the gate has no error
// kind for this format, so the fixture is not schema-checked and the assertion
// rests entirely on the client's observable behaviour.
//
// It records a FINDING about observability, not a crash. Azure reports a
// content-management refusal as HTTP 400 with `error.code: "content_filter"`
// and the real reason nested in `error.innererror.code`. Neither reaches the
// caller today:
//
//   - `content_filter` is not a member of failure's closed providerCodeAllowlist
//     (which carries the adjacent-but-different `content_policy_violation`), so
//     failure.APIError.Code and .ProviderCode both come out EMPTY;
//   - `innererror` has no representation in APIError at all.
//
// A caller therefore sees an unlabelled 400 and cannot distinguish a policy
// violation from a malformed request. The allowlist lives in inference/failure,
// outside this suite's remit, so the behaviour is pinned rather than changed:
// widening the allowlist would make this test fail loudly and correctly.
func TestACSResponsibleAIPolicyViolation(t *testing.T) {
	t.Parallel()

	raw := acsFixture(t, "responsible_ai_policy_violation.json")
	// Assert the fixture really is the shape being claimed, since nothing else
	// here does: a typo in the fixture would otherwise quietly test nothing.
	var envelope struct {
		Error struct {
			Code       string `json:"code"`
			InnerError struct {
				Code string `json:"code"`
			} `json:"innererror"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("fixture is not JSON: %v", err)
	}
	if envelope.Error.Code != "content_filter" || envelope.Error.InnerError.Code != "ResponsibleAIPolicyViolation" {
		t.Fatalf("fixture = %+v, want the ResponsibleAIPolicyViolation envelope", envelope)
	}

	selected := acsServe(t, http.StatusBadRequest, "application/json", raw)
	_, err := acsClient(t, selected).Invoke(context.Background(), acsPrompt(selected))
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Invoke() error = %T %v, want *failure.APIError", err, err)
	}
	if apiErr.Status != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", apiErr.Status)
	}
	// Today's behaviour, pinned. Both are empty because "content_filter" is not
	// on failure's allowlist; if either becomes non-empty the allowlist has
	// been widened and this finding is resolved.
	if apiErr.Code != "" || apiErr.ProviderCode != "" {
		t.Errorf("code = %q / provider code = %q, want both empty until "+
			"content_filter joins failure's providerCodeAllowlist — if it has, "+
			"update this finding", apiErr.Code, apiErr.ProviderCode)
	}
	// The inner Azure code never reaches the caller in any form.
	if strings.Contains(apiErr.Error(), "ResponsibleAIPolicyViolation") {
		t.Error("the inner Azure policy code now reaches the caller — update this finding")
	}
	// What a caller can act on today is the status alone.
	if !strings.Contains(apiErr.Error(), "400") {
		t.Errorf("error = %q, want it to at least carry the status", apiErr.Error())
	}
}
