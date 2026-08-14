package gemini_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/geminiapi"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	streampkg "github.com/looprig/inference/stream"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	gemini "github.com/looprig/llm/providers/gemini"
)

// This file is the schema-gated Gemini generateContent fixture suite. Every
// fixture is validated against Google's own v1beta discovery document — via the
// conformance gate — BEFORE any Looprig decoder sees it, so a decoder assertion
// below is always an assertion about a payload Gemini could really have sent.
//
// A standing caveat, and the reason the decoder assertions here are unusually
// detailed: the Gemini document is the WEAKEST of the gate's five. Google's
// discovery format carries no response-side required list, so all 46 of the
// document's object shapes declare zero required properties (see
// schema/unenforced.json). The gate therefore enforces types, enums, array-ness
// and nesting, and cannot enforce presence: a candidate with no content, a
// functionCall with no name, a Part with two content members all validate. The
// gate proves a fixture is not ill-typed; only the assertions carry the rest.

const geminiFixtureKey = "AIza-conformance-key"

// geminiFixture reads a checked-in fixture. The bytes on disk are the bytes
// validated and the bytes decoded; nothing is templated at test time.
func geminiFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

// gateResponse runs the conformance gate over a non-streaming fixture. Calling
// it first, in every case, is the invariant this suite exists to hold.
func gateResponse(t testing.TB, name string) []byte {
	t.Helper()
	body := geminiFixture(t, name)
	conformance.MustValidateResponse(t, "gemini", "generate_content_response", body)
	return body
}

// invokeFixture serves one fixture from an httptest.Server and drives the real
// providers/gemini client through it, so the decode path under test is the one
// production uses (transport + geminiapi codec), not a codec call in isolation.
// The request the client sends on the way in is gated too: every fixture case
// therefore also proves the encoder's own body is a legal Gemini request.
func invokeFixture(t testing.TB, body []byte) (*inference.Response, error) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gateSentRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	client := gemini.NewWithEndpoint(geminiFixtureKey, srv.URL)
	return client.Invoke(context.Background(), inference.Request{
		Model: geminiFixtureModel("gemini-2.5-pro"),
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	})
}

func geminiFixtureModel(name string, options ...model.ModelOption) model.Model {
	return model.CustomModel(
		model.ProviderName(llm.ProviderGoogle),
		model.APIFormatGemini,
		"https://generativelanguage.googleapis.com/v1beta",
		name,
		options...,
	)
}

// TestGeminiResponseFixturesAreSpecLegal is the gate run on its own, over every
// non-streaming fixture in the corpus. It is deliberately separate from the
// decoding tests: if a fixture stops being a legal Gemini payload, that must
// fail as a fixture defect here rather than as a confusing decoder assertion.
func TestGeminiResponseFixturesAreSpecLegal(t *testing.T) {
	t.Parallel()

	for _, name := range []string{
		"text_plain.json",
		"finish_max_tokens.json",
		"finish_safety.json",
		"finish_recitation.json",
		"finish_malformed_function_call.json",
		"finish_unexpected_tool_call.json",
		"finish_prohibited_content.json",
		"finish_missing_thought_signature.json",
		"function_call_single.json",
		"function_calls_parallel_no_id.json",
		"function_calls_parallel_with_ids.json",
		"function_response_in_candidate.json",
		"function_response_inline_data.json",
		"thought_signature_on_function_call.json",
		"thought_signature_on_text.json",
		"thought_signature_only_empty_text.json",
		"thought_parts_multiple_signatures.json",
		"thought_parts_plain.json",
		"usage_metadata_full.json",
		"usage_metadata_total_counts_tool_use.json",
		"prompt_feedback_blocked.json",
		"error_envelope.json",
		"part_two_members.json",
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			conformance.MustValidateResponse(t, "gemini", "generate_content_response", geminiFixture(t, name))
		})
	}
}

// TestGeminiTextAndFinishReasons walks the finishReason enum. The enum values
// themselves ARE schema-backed — Candidate.finishReason is one of the few
// Gemini positions the discovery document constrains — so an invalid reason
// fails the gate; the neutral FinishReason each maps to is decode-only.
func TestGeminiTextAndFinishReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture string
		want    streampkg.FinishReason
		text    string
	}{
		{"text_plain.json", streampkg.FinishReasonStop, "The forecast for Oslo is 4 degrees celsius."},
		{"finish_max_tokens.json", streampkg.FinishReasonLength, "The forecast for Oslo is"},
		{"finish_safety.json", streampkg.FinishReasonContentFilter, ""},
		{"finish_recitation.json", streampkg.FinishReasonContentFilter, "In the beginning"},
		{"finish_prohibited_content.json", streampkg.FinishReasonContentFilter, ""},
		// Gemini added these three reasons after mapFinishReason was written, so
		// they land in the default branch. Unknown is the honest answer for a
		// reason with no neutral equivalent; it is pinned so a future mapping
		// change is a deliberate one.
		{"finish_malformed_function_call.json", streampkg.FinishReasonUnknown, ""},
		{"finish_unexpected_tool_call.json", streampkg.FinishReasonUnknown, ""},
		{"finish_missing_thought_signature.json", streampkg.FinishReasonUnknown, ""},
	}

	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			t.Parallel()

			resp, err := invokeFixture(t, gateResponse(t, tc.fixture))
			if err != nil {
				t.Fatalf("Invoke(%s) error = %v", tc.fixture, err)
			}
			if resp.FinishReason != tc.want {
				t.Errorf("FinishReason = %v, want %v", resp.FinishReason, tc.want)
			}
			if tc.text == "" {
				if got := len(resp.Message.Blocks); got != 0 {
					t.Fatalf("blocks = %d, want 0 for a content-free candidate", got)
				}
				return
			}
			if got := len(resp.Message.Blocks); got != 1 {
				t.Fatalf("blocks = %d, want 1", got)
			}
			text, ok := resp.Message.Blocks[0].(*content.TextBlock)
			if !ok || text.Text != tc.text {
				t.Errorf("block = %#v, want TextBlock(%q)", resp.Message.Blocks[0], tc.text)
			}
		})
	}
}

// TestGeminiFunctionCalls covers single and parallel tool calls.
//
// function_calls_parallel_no_id.json pins the CROSS-ATTRIBUTION defect: the
// Developer API routinely omits FunctionCall.id, and the neutral vocabulary
// addresses a tool result by id alone, so before the fix all three id-less
// calls decoded to id "" and one call's output could answer another. Each
// id-less call must now get its own synthetic per-turn ordinal.
func TestGeminiFunctionCalls(t *testing.T) {
	t.Parallel()

	t.Run("single call", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "function_call_single.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		// A functionCall part with finishReason STOP is a tool turn, not a stop.
		if resp.FinishReason != streampkg.FinishReasonToolUse {
			t.Errorf("FinishReason = %v, want tool_use", resp.FinishReason)
		}
		call := onlyToolUse(t, resp)
		if call.Name != "get_weather" {
			t.Errorf("Name = %q, want get_weather", call.Name)
		}
		if call.ID != "gemini-positional-call-0" {
			t.Errorf("ID = %q, want the synthetic ordinal for an id-less call", call.ID)
		}
		assertJSONEquivalent(t, call.Input, json.RawMessage(`{"city":"Oslo","unit":"celsius"}`))
	})

	t.Run("parallel calls without ids stay individually addressable", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "function_calls_parallel_no_id.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		ids := make([]string, 0, 3)
		names := make([]string, 0, 3)
		for _, block := range resp.Message.Blocks {
			call, ok := block.(*content.ToolUseBlock)
			if !ok {
				t.Fatalf("block = %T, want *content.ToolUseBlock", block)
			}
			ids = append(ids, call.ID)
			names = append(names, call.Name)
		}
		wantIDs := []string{"gemini-positional-call-0", "gemini-positional-call-1", "gemini-positional-call-2"}
		if !reflect.DeepEqual(ids, wantIDs) {
			t.Fatalf("tool-call ids = %v, want %v; two id-less calls sharing an id is the "+
				"cross-attribution defect this fixture pins", ids, wantIDs)
		}
		if want := []string{"get_weather", "get_weather", "get_time"}; !reflect.DeepEqual(names, want) {
			t.Errorf("tool-call names = %v, want %v (part order must be preserved)", names, want)
		}
	})

	t.Run("parallel calls with ids keep the model's own ids", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "function_calls_parallel_with_ids.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		var ids []string
		for _, block := range resp.Message.Blocks {
			ids = append(ids, block.(*content.ToolUseBlock).ID)
		}
		if want := []string{"call_oslo_01", "call_bergen_02"}; !reflect.DeepEqual(ids, want) {
			t.Errorf("tool-call ids = %v, want %v", ids, want)
		}
	})
}

// TestGeminiFunctionResponseParts covers functionResponse shapes appearing in a
// model turn. The Part union legally admits them there (and the gate accepts
// them, since Part declares no required or mutually-exclusive members), but the
// decoder has no neutral mapping for a functionResponse it did not send, so it
// skips the part. This is a deliberate tolerant skip, pinned so the skip cannot
// silently become a drop of something that DOES matter.
func TestGeminiFunctionResponseParts(t *testing.T) {
	t.Parallel()

	t.Run("functionResponse beside text", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "function_response_in_candidate.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if got := len(resp.Message.Blocks); got != 1 {
			t.Fatalf("blocks = %d, want 1 (the functionResponse part is skipped)", got)
		}
		if text := resp.Message.Blocks[0].(*content.TextBlock).Text; text != "Oslo is 4 degrees celsius." {
			t.Errorf("text = %q", text)
		}
		// No functionCall part, so the finishReason is taken at face value.
		if resp.FinishReason != streampkg.FinishReasonStop {
			t.Errorf("FinishReason = %v, want stop", resp.FinishReason)
		}
	})

	t.Run("functionResponse carrying inline binary parts", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "function_response_inline_data.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if got := len(resp.Message.Blocks); got != 0 {
			t.Fatalf("blocks = %d, want 0; FunctionResponse.parts has no neutral mapping", got)
		}
	})
}

// TestGeminiThoughtSignatures covers Gemini 2.5's opaque continuation token.
//
// thought_parts_multiple_signatures.json pins the COLLAPSE defect: three thought
// parts each carry a DIFFERENT signature, and a decoder that folds a turn's
// thoughts into one block keeps only one of them, so a same-dialect replay
// sends back state the model never issued. Each part must survive as its own
// ThinkingBlock with its own ProviderState.
func TestGeminiThoughtSignatures(t *testing.T) {
	t.Parallel()

	t.Run("signature on a functionCall part", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "thought_signature_on_function_call.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		call := onlyToolUse(t, resp)
		if !call.ReplayableAs("gemini") {
			t.Fatalf("ToolUseBlock.ReplayableAs(gemini) = false, want the thoughtSignature carried as provider state")
		}
		assertOpaqueState(t, call.ProviderState, "CtEBAVSoXO9dLJm1ZDBQ0Vd1s9Yb3Uu8w==")
	})

	t.Run("signature-only thought part with empty text", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "thought_signature_only_empty_text.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if got := len(resp.Message.Blocks); got != 2 {
			t.Fatalf("blocks = %d, want 2; a signature-only thought must not be dropped", got)
		}
		thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
		if !ok {
			t.Fatalf("block[0] = %T, want *content.ThinkingBlock", resp.Message.Blocks[0])
		}
		if thinking.Thinking != "" {
			t.Errorf("Thinking = %q, want empty", thinking.Thinking)
		}
		assertOpaqueState(t, thinking.ProviderState, "CpMBAVSoXO9zaWduYXR1cmUtb25seQ==")
	})

	t.Run("distinct signatures on distinct thought parts", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "thought_parts_multiple_signatures.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		want := []string{
			"CoEBAVSoXO9zaWduYXR1cmUtb25l",
			"CoEBAVSoXO9zaWduYXR1cmUtdHdv",
			"CoMBAVSoXO9zaWduYXR1cmUtdGhyZWU=",
		}
		if got := len(resp.Message.Blocks); got != len(want)+1 {
			t.Fatalf("blocks = %d, want %d thinking blocks plus one text block; collapsing "+
				"thought parts loses the per-part signature", got, len(want))
		}
		for i, signature := range want {
			thinking, ok := resp.Message.Blocks[i].(*content.ThinkingBlock)
			if !ok {
				t.Fatalf("block[%d] = %T, want *content.ThinkingBlock", i, resp.Message.Blocks[i])
			}
			assertOpaqueState(t, thinking.ProviderState, signature)
		}
	})

	t.Run("thought parts with no signature", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "thought_parts_plain.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		// Three parts in, two blocks out: the middle part is a thought with
		// neither text nor signature, which carries nothing to replay.
		if got := len(resp.Message.Blocks); got != 2 {
			t.Fatalf("blocks = %d, want 2", got)
		}
		thinking, ok := resp.Message.Blocks[0].(*content.ThinkingBlock)
		if !ok || thinking.Thinking != "The user wants a forecast." {
			t.Fatalf("block[0] = %#v, want the thought text", resp.Message.Blocks[0])
		}
		if len(thinking.ProviderState) != 0 || thinking.ProviderStateFormat != "" {
			t.Errorf("ProviderState = %s/%q, want empty for a signature-less thought",
				thinking.ProviderState, thinking.ProviderStateFormat)
		}
	})

	// A thoughtSignature on an ordinary (thought:false) text part is legal per
	// the discovery document and the gate accepts it, but a TextBlock has
	// nowhere to carry provider state, so the signature is dropped. Pinned as
	// the current, lossy behaviour rather than asserted as correct.
	t.Run("signature on a non-thought text part is dropped", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "thought_signature_on_text.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if got := len(resp.Message.Blocks); got != 1 {
			t.Fatalf("blocks = %d, want 1", got)
		}
		if _, ok := resp.Message.Blocks[0].(*content.TextBlock); !ok {
			t.Fatalf("block = %T, want *content.TextBlock", resp.Message.Blocks[0])
		}
	})
}

// TestGeminiUsageMetadata covers the normalization arithmetic.
//
// None of it is schema-backed: the discovery document types every count as an
// int32 and says nothing about how they relate, so both fixtures below pass the
// gate. They differ only in totalTokenCount, and the decoder deliberately no
// longer distinguishes them — see the second subtest.
func TestGeminiUsageMetadata(t *testing.T) {
	t.Parallel()

	t.Run("cached, thought and tool-use counts", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "usage_metadata_full.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage = nil, want normalized usage")
		}
		// promptTokenCount is inclusive of the cached prefix, so the uncached
		// input is the difference; output is candidates + thoughts.
		//
		// toolUsePromptTokenCount is NOT inside promptTokenCount — this very
		// fixture proves it, since its promptTokensDetails sum to exactly the 100
		// promptTokenCount while toolUsePromptTokensDetails carries the 7
		// separately. Those 7 are billable prompt tokens with no distinct bucket
		// in the neutral Usage, so the codec adds them to InputTokens rather than
		// dropping them: 100 - 40 + 7.
		if got := int(resp.Usage.InputTokens); got != 67 {
			t.Errorf("InputTokens = %d, want 67 (100 prompt - 40 cached + 7 tool-use prompt)", got)
		}
		if got := int(resp.Usage.CacheReadTokens); got != 40 {
			t.Errorf("CacheReadTokens = %d, want 40", got)
		}
		if got := int(resp.Usage.OutputTokens); got != 25 {
			t.Errorf("OutputTokens = %d, want 25 (20 candidates + 5 thoughts)", got)
		}
		if got := int(resp.Usage.ReasoningTokens); got != 5 {
			t.Errorf("ReasoningTokens = %d, want 5", got)
		}
	})

	// The fixture pair covers both readings of totalTokenCount: 125 in
	// usage_metadata_full.json (prompt + candidates + thoughts, the discovery
	// document's prose) and 132 here (the same plus the 7 tool-use prompt
	// tokens). The codec no longer has to choose, because it no longer
	// reconciles the reported total against its modelled components: the total
	// feeds no field of the neutral Usage, so failing on it could only ever
	// discard a completed generation over an accounting field. Both fixtures
	// must therefore decode identically, and both must keep the answer.
	t.Run("total that also counts the tool-use prompt is accepted", func(t *testing.T) {
		t.Parallel()

		resp, err := invokeFixture(t, gateResponse(t, "usage_metadata_total_counts_tool_use.json"))
		if err != nil {
			t.Fatalf("Invoke() error = %v, want the generation preserved", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage = nil, want normalized usage")
		}
		if got := int(resp.Usage.InputTokens); got != 67 {
			t.Errorf("InputTokens = %d, want 67 (100 prompt - 40 cached + 7 tool-use prompt)", got)
		}
		if got := int(resp.Usage.CacheReadTokens); got != 40 {
			t.Errorf("CacheReadTokens = %d, want 40", got)
		}
		if got := int(resp.Usage.OutputTokens); got != 25 {
			t.Errorf("OutputTokens = %d, want 25 (20 candidates + 5 thoughts)", got)
		}
	})
}

// TestGeminiCandidateLessResponses covers the two shapes Gemini returns with no
// candidates at all. Both are legal payloads and both are *failure.APIError to a
// caller that classifies on that type — but they no longer collapse into the
// same contentless one: a prompt the content filter refused now names its
// blockReason, its safety ratings and the prompt tokens it was charged for,
// which is the difference between "policy" and "unknown failure" at the call
// site. The error-envelope shape has nothing comparable to read.
func TestGeminiCandidateLessResponses(t *testing.T) {
	t.Parallel()

	t.Run("prompt_feedback_blocked.json", func(t *testing.T) {
		t.Parallel()

		_, err := invokeFixture(t, gateResponse(t, "prompt_feedback_blocked.json"))
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("Invoke() error = %v (%T), want *failure.APIError", err, err)
		}
		if apiErr.Code != "content_policy_violation" {
			t.Errorf("APIError.Code = %q, want content_policy_violation", apiErr.Code)
		}

		var blocked *geminiapi.PromptBlockedError
		if !errors.As(err, &blocked) {
			t.Fatalf("Invoke() error = %v (%T), want *geminiapi.PromptBlockedError", err, err)
		}
		if blocked.BlockReason != "PROHIBITED_CONTENT" {
			t.Errorf("BlockReason = %q, want PROHIBITED_CONTENT", blocked.BlockReason)
		}
		if len(blocked.SafetyRatings) != 2 || !blocked.SafetyRatings[0].Blocked {
			t.Errorf("SafetyRatings = %+v, want both ratings with the first blocking", blocked.SafetyRatings)
		}
		// The fixture reports 17 prompt tokens. They were billed even though
		// nothing was generated, so the failure must carry them.
		if blocked.Usage == nil {
			t.Fatal("Usage = nil, want the fixture's billed prompt tokens")
		}
		if got := int(blocked.Usage.InputTokens); got != 17 {
			t.Errorf("Usage.InputTokens = %d, want 17", got)
		}
	})

	t.Run("error_envelope.json", func(t *testing.T) {
		t.Parallel()

		_, err := invokeFixture(t, gateResponse(t, "error_envelope.json"))
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("Invoke() error = %v (%T), want *failure.APIError", err, err)
		}
		var blocked *geminiapi.PromptBlockedError
		if errors.As(err, &blocked) {
			t.Fatalf("error = %v, want no block reason invented for an error envelope", err)
		}
	})
}

// TestGeminiPartWithTwoContentMembers is a negative case the gate CANNOT catch.
// Gemini's Part is a union in which exactly one content member should be set,
// but the discovery document expresses it as an ordinary object with every
// member optional, so a part carrying both text and functionCall validates
// cleanly. The decoder resolves it by precedence, dropping the text.
func TestGeminiPartWithTwoContentMembers(t *testing.T) {
	t.Parallel()

	body := gateResponse(t, "part_two_members.json")
	if err := conformance.Validate("gemini", "generate_content_response", body); err != nil {
		t.Fatalf("gate rejected the two-member part after all: %v", err)
	}

	resp, err := invokeFixture(t, body)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := len(resp.Message.Blocks); got != 1 {
		t.Fatalf("blocks = %d, want 1", got)
	}
	call, ok := resp.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("block = %T, want the functionCall to win the precedence", resp.Message.Blocks[0])
	}
	if call.Name != "render_chart" {
		t.Errorf("Name = %q, want render_chart", call.Name)
	}
}

// --- streaming -------------------------------------------------------------

// streamFixture gate-validates every SSE data frame against the response
// schema, then drives the real client's Stream path over the same bytes.
func streamFixture(t testing.TB, name string) (*streampkg.StreamReader[content.Chunk], error) {
	t.Helper()
	body := geminiFixture(t, name)
	conformance.MustValidateStream(t, "gemini", "generate_content_response", body)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gateSentRequest(t, r)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	client := gemini.NewWithEndpoint(geminiFixtureKey, srv.URL)
	return client.Stream(context.Background(), inference.Request{
		Model: geminiFixtureModel("gemini-2.5-pro"),
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	})
}

// drainStream reads a reader to completion, returning the chunks and the
// terminal error (io.EOF on a clean stream).
func drainStream(t testing.TB, reader *streampkg.StreamReader[content.Chunk]) ([]content.Chunk, error) {
	t.Helper()
	var chunks []content.Chunk
	for {
		chunk, err := reader.Next()
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
}

func TestGeminiStreamFixtures(t *testing.T) {
	t.Parallel()

	t.Run("text frames and a terminal finishReason", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_text.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		chunks, err := drainStream(t, reader)
		if !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		var text string
		for _, chunk := range chunks {
			text += chunk.(*content.TextChunk).Text
		}
		if want := "The forecast for Oslo is 4 degrees celsius."; text != want {
			t.Errorf("streamed text = %q, want %q", text, want)
		}
		result, ok := reader.Result()
		if !ok {
			t.Fatal("Result() ok = false, want a terminal result")
		}
		if result.FinishReason != streampkg.FinishReasonStop {
			t.Errorf("FinishReason = %v, want stop", result.FinishReason)
		}
		if result.Model != "gemini-2.5-flash" {
			t.Errorf("Model = %q, want gemini-2.5-flash", result.Model)
		}
	})

	t.Run("a chunk carrying only usageMetadata", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_usage_only_chunk.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		if _, err := drainStream(t, reader); !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		result, ok := reader.Result()
		if !ok {
			t.Fatal("Result() ok = false")
		}
		if result.Usage == nil {
			t.Fatal("Usage = nil; a candidate-less usage frame must still be collected")
		}
		if got := int(result.Usage.InputTokens); got != 12 {
			t.Errorf("InputTokens = %d, want 12", got)
		}
	})

	t.Run("a chunk with an empty candidates array", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_empty_candidates.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		chunks, err := drainStream(t, reader)
		if !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		if got := len(chunks); got != 1 {
			t.Fatalf("chunks = %d, want 1; an empty candidates array yields nothing", got)
		}
	})

	t.Run("terminal frame carrying only a finishReason", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_terminal_finish_reason.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		if _, err := drainStream(t, reader); !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		result, ok := reader.Result()
		if !ok {
			t.Fatal("Result() ok = false")
		}
		if result.FinishReason != streampkg.FinishReasonLength {
			t.Errorf("FinishReason = %v, want length", result.FinishReason)
		}
	})

	// Pins the sawTerminal gate. Gemini has no [DONE] sentinel, so a body that
	// simply ends is indistinguishable from a truncated one except by the
	// absence of a finishReason. Reporting a clean, short answer here was the
	// defect; the stream must fail instead.
	t.Run("a stream that ends with no finishReason is a truncation", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_truncated_no_terminal.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		chunks, err := drainStream(t, reader)
		if errors.Is(err, io.EOF) {
			t.Fatal("terminal error = io.EOF, want a truncation failure")
		}
		var decodeErr *geminiapi.StreamDecodeError
		if !errors.As(err, &decodeErr) {
			t.Fatalf("terminal error = %v (%T), want *geminiapi.StreamDecodeError", err, err)
		}
		if got := len(chunks); got != 2 {
			t.Errorf("chunks before the failure = %d, want 2", got)
		}
		if _, ok := reader.Result(); ok {
			t.Error("Result() ok = true after a truncation, want false")
		}
	})

	t.Run("parallel function calls in one frame", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_parallel_function_calls.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		chunks, err := drainStream(t, reader)
		if !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		if got := len(chunks); got != 2 {
			t.Fatalf("chunks = %d, want 2 tool-use chunks", got)
		}
		for i, chunk := range chunks {
			call, ok := chunk.(*content.ToolUseChunk)
			if !ok {
				t.Fatalf("chunk[%d] = %T, want *content.ToolUseChunk", i, chunk)
			}
			if call.Index != i {
				t.Errorf("chunk[%d].Index = %d, want %d", i, call.Index, i)
			}
			// Same cross-attribution rule as the non-streaming path: two
			// id-less calls must not share an identity.
			if want := "gemini-positional-call-" + string(rune('0'+i)); call.ID != want {
				t.Errorf("chunk[%d].ID = %q, want %q", i, call.ID, want)
			}
		}
		result, _ := reader.Result()
		if result.FinishReason != streampkg.FinishReasonToolUse {
			t.Errorf("FinishReason = %v, want tool_use", result.FinishReason)
		}
	})

	t.Run("distinct thought signatures across frames", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_thought_signatures.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		chunks, err := drainStream(t, reader)
		if !errors.Is(err, io.EOF) {
			t.Fatalf("terminal error = %v, want io.EOF", err)
		}
		want := []string{
			"CoEBAVSoXO9zdHJlYW0tc2lnLW9uZQ==",
			"CoEBAVSoXO9zdHJlYW0tc2lnLXR3bw==",
			"CoMBAVSoXO9zdHJlYW0tc2lnLXRocmVl",
		}
		if got := len(chunks); got != len(want)+1 {
			t.Fatalf("chunks = %d, want %d thinking chunks plus one text chunk", got, len(want))
		}
		for i, signature := range want {
			thinking, ok := chunks[i].(*content.ThinkingChunk)
			if !ok {
				t.Fatalf("chunk[%d] = %T, want *content.ThinkingChunk", i, chunks[i])
			}
			assertOpaqueState(t, thinking.ProviderState, signature)
		}
	})

	// An {"error":{...}} frame arrives AFTER the successful HTTP status, so it
	// is otherwise indistinguishable from an uninteresting candidate-less
	// chunk. It must surface as a failure, not finish the turn cleanly.
	t.Run("an error envelope frame fails the stream", func(t *testing.T) {
		t.Parallel()

		reader, err := streamFixture(t, "stream_error_frame.sse")
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer reader.Close()

		_, err = drainStream(t, reader)
		var apiErr *geminiapi.StreamAPIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("terminal error = %v (%T), want *geminiapi.StreamAPIError", err, err)
		}
		if apiErr.Code != 500 || apiErr.Status != "INTERNAL" {
			t.Errorf("StreamAPIError = %+v, want code 500 / INTERNAL", apiErr)
		}
	})
}

// --- multimodal requests ---------------------------------------------------

// gateRequest validates an ENCODED request body against Google's request
// schema. This catches Looprig's own encoder rather than its tolerance for
// provider output, which is the stronger of the two checks.
//
// Be clear about its strength. The Gemini request document is weak in the same
// way the response one is: of its 49 object shapes, one declares required
// properties and none are closed, so a missing field or a stray extra field
// passes. What it DOES enforce is types, enums, array-ness and $ref shapes —
// which is exactly what caught the functionDeclarations dialect defect (an
// uppercase-only type enum) and the `"contents": null` defect (a declared
// array). Fields typed `any` in the discovery document — parametersJsonSchema
// and responseJsonSchema among them — are unconstrained: nothing inside them
// is checked at all, so the codec tests own those assertions.
//
// It does NOT enforce union arity, despite Part being a union: the derived
// request document contains no oneOf anywhere, so a part carrying two members
// of Part.data passes. Nor does it assert `format: byte`, so a `data` member
// that is not base64 passes too. Both are pinned by explicit assertions —
// assertOneDataMemberPerPart and assertInlineData.
func gateRequest(t testing.TB, body []byte) {
	t.Helper()
	conformance.MustValidateRequest(t, "gemini", "generate_content_request", body)
}

// gateSentRequest gates a request body as an httptest handler received it, so a
// test that exists for the response side still holds the encoder to the schema.
// A body-less request (there is none on this route, but a future one would be
// silently skipped otherwise) fails rather than passing vacuously.
//
// It reports through t.Errorf rather than the gate's Fatalf helper because it
// runs on the server's goroutine, where FailNow is not allowed.
func gateSentRequest(t testing.TB, r *http.Request) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read request body: %v", err)
		return
	}
	if len(body) == 0 {
		t.Error("request body is empty, want an encoded generateContent request")
		return
	}
	if err := conformance.Validate("gemini", "generate_content_request", body); err != nil {
		t.Errorf("the encoded request is not a legal Gemini payload: %v", err)
	}
}

func geminiMultimodalModel() model.Model {
	return geminiFixtureModel("gemini-2.5-flash", model.WithTools(), model.WithImages())
}

func userTurn(blocks ...content.Block) content.Conversation {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

var (
	pngBytes  = []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	jpegBytes = []byte{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10}
	webpBytes = []byte("RIFF\x00\x00\x00\x00WEBPVP8 ")
)

// TestGeminiMultimodalRequestEncoding encodes real multimodal requests and holds
// the bytes against Google's generateContent REQUEST schema before comparing
// them to a checked-in golden. Both halves are schema-backed as far as the
// discovery document allows: types, enums and nesting are enforced, presence is
// not (the request document has the same no-required-properties limitation as
// the response one).
func TestGeminiMultimodalRequestEncoding(t *testing.T) {
	t.Parallel()

	cases := []struct {
		golden   string
		request  inference.Request
		assertOn func(t *testing.T, body []byte)
	}{
		{
			golden: "request_image_png_inline.json",
			request: inference.Request{
				Model: geminiMultimodalModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.TextBlock{Text: "What is in this image?"},
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: pngBytes}},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				parts := requestParts(t, body, 0)
				if len(parts) != 2 {
					t.Fatalf("parts = %d, want 2", len(parts))
				}
				// Text first, image second: block order must survive encoding.
				if _, ok := parts[0]["text"]; !ok {
					t.Error("parts[0] is not the text part; block order was not preserved")
				}
				assertInlineData(t, parts[1], "image/png", pngBytes)
			},
		},
		{
			golden: "request_image_jpeg_inline.json",
			request: inference.Request{
				Model: geminiMultimodalModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImageJPEG, Source: content.ImageSource{Data: jpegBytes}},
					&content.TextBlock{Text: "Describe it."},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				parts := requestParts(t, body, 0)
				// Image first this time — the reverse of the PNG case, so the
				// assertion is about order rather than about a fixed layout.
				assertInlineData(t, parts[0], "image/jpeg", jpegBytes)
				if _, ok := parts[1]["text"]; !ok {
					t.Error("parts[1] is not the text part")
				}
			},
		},
		{
			golden: "request_image_multiple_inline.json",
			request: inference.Request{
				Model:  geminiMultimodalModel(),
				System: "You compare images.",
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: pngBytes}},
					&content.ImageBlock{MediaType: content.MediaTypeImageWebP, Source: content.ImageSource{Data: webpBytes}},
					&content.TextBlock{Text: "Which is sharper?"},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				parts := requestParts(t, body, 0)
				if len(parts) != 3 {
					t.Fatalf("parts = %d, want 3", len(parts))
				}
				assertInlineData(t, parts[0], "image/png", pngBytes)
				assertInlineData(t, parts[1], "image/webp", webpBytes)
			},
		},
		{
			golden: "request_image_url_file_data.json",
			request: inference.Request{
				Model: geminiMultimodalModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{
						MediaType: content.MediaTypeImagePNG,
						Source:    content.ImageSource{URL: "https://generativelanguage.googleapis.com/v1beta/files/abc123"},
					},
					&content.TextBlock{Text: "Caption it."},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				parts := requestParts(t, body, 0)
				// A URL-sourced image degrades to the Files API form. Gemini
				// accepts a fileUri only for File API / gs:// / YouTube URIs,
				// which the encoder cannot check, so this is a structural
				// mapping and not a guarantee the call will succeed.
				fileData, ok := parts[0]["fileData"]
				if !ok {
					t.Fatalf("parts[0] = %v, want a fileData part", parts[0])
				}
				var got struct {
					MimeType string `json:"mimeType"`
					FileURI  string `json:"fileUri"`
				}
				if err := json.Unmarshal(fileData, &got); err != nil {
					t.Fatalf("decode fileData: %v", err)
				}
				if got.MimeType != "image/png" {
					t.Errorf("fileData.mimeType = %q, want image/png", got.MimeType)
				}
				if got.FileURI == "" {
					t.Error("fileData.fileUri is empty")
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.golden, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(tc.request)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			gateRequest(t, body)
			assertJSONEquivalent(t, body, geminiFixture(t, tc.golden))
			tc.assertOn(t, body)
		})
	}
}

// TestGeminiToolResultRequestEncoding pins the request-side functionResponse
// shape, including the name resolution Gemini actually pairs on (FunctionResponse.name
// is required; both ids are optional) and the stripping of the synthetic
// per-turn ordinal, which must never be echoed back as if the model issued it.
func TestGeminiToolResultRequestEncoding(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model: geminiMultimodalModel(),
		Messages: content.AgenticMessages{
			userTurn(&content.TextBlock{Text: "Weather in Oslo and Bergen?"}),
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewToolUseBlock("gemini-positional-call-0", "get_weather", json.RawMessage(`{"city":"Oslo"}`), nil, ""),
				content.NewToolUseBlock("gemini-positional-call-1", "get_weather", json.RawMessage(`{"city":"Bergen"}`), nil, ""),
			}}},
			&content.ToolResultMessage{
				ToolUseID: "gemini-positional-call-0",
				Message: content.Message{Role: content.RoleTool,
					Blocks: []content.Block{&content.TextBlock{Text: "4 degrees celsius"}}},
			},
			&content.ToolResultMessage{
				ToolUseID: "gemini-positional-call-1",
				Message: content.Message{Role: content.RoleTool,
					Blocks: []content.Block{&content.TextBlock{Text: "6 degrees celsius"}}},
			},
		},
		// No declared tools: this fixture is about the functionResponse shape,
		// and the declaration dialect it used to have to avoid is covered on
		// its own by TestGeminiFunctionDeclarationSchemaDialect.
	}

	body, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateRequest(t, body)
	assertJSONEquivalent(t, body, geminiFixture(t, "request_tool_result_function_response.json"))

	for _, index := range []int{1, 2, 3} {
		for _, part := range requestParts(t, body, index) {
			for _, key := range []string{"functionCall", "functionResponse"} {
				raw, ok := part[key]
				if !ok {
					continue
				}
				var got struct {
					ID   string `json:"id"`
					Name string `json:"name"`
				}
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatalf("decode %s: %v", key, err)
				}
				if got.ID != "" {
					t.Errorf("%s.id = %q, want empty; a synthetic in-process id must never reach the wire", key, got.ID)
				}
				if got.Name != "get_weather" {
					t.Errorf("%s.name = %q, want get_weather; Gemini pairs a result on the name", key, got.Name)
				}
			}
		}
	}
}

// TestGeminiFunctionDeclarationSchemaDialect covers the ENCODER DEFECT the
// request gate found, and is the reason request-side validation is worth more
// than response-side validation.
//
// Gemini's FunctionDeclaration has two mutually exclusive parameter fields:
// `parameters`, typed as Gemini's own Schema (whose `type` is the uppercase
// OpenAPI-style enum STRING/OBJECT/…, and which has no way to spell
// additionalProperties/$ref/$defs/oneOf/const), and `parametersJsonSchema`,
// which is untyped and takes a standard JSON Schema verbatim. Looprig used to
// write a caller's standard JSON Schema — lowercase `"type":"object"` and all —
// straight into `parameters`, which the discovery document does not admit.
//
// The encoder now projects into Gemini's Schema dialect when the caller's
// schema fits it, and falls back to `parametersJsonSchema` when it does not, so
// a constraint is never dropped. Both halves are gated below.
func TestGeminiFunctionDeclarationSchemaDialect(t *testing.T) {
	t.Parallel()

	// The shape Gemini actually documents for `parameters`, for comparison.
	gateRequest(t, geminiFixture(t, "request_tool_declaration_gemini_schema.json"))

	t.Run("a schema Gemini's Schema dialect can express is projected into parameters", func(t *testing.T) {
		t.Parallel()

		body, err := geminiapi.EncodeRequest(inference.Request{
			Model:    geminiMultimodalModel(),
			Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "Weather in Oslo?"})},
			Tools: []inference.Tool{{
				Name:        "get_weather",
				Description: "Look up a forecast",
				Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}`),
			}},
		})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		gateRequest(t, body)
		// Byte-for-byte the documented shape: uppercase types, same keywords.
		assertJSONEquivalent(t, body, geminiFixture(t, "request_tool_declaration_gemini_schema.json"))

		declaration := onlyFunctionDeclaration(t, body)
		if got := declarationParameterType(t, declaration.Parameters); got != "OBJECT" {
			t.Errorf("parameters.type = %q, want the uppercase Gemini Schema value", got)
		}
		if len(declaration.ParametersJSONSchema) != 0 {
			t.Error("parametersJsonSchema is set alongside parameters; the two fields are mutually exclusive")
		}
	})

	// additionalProperties is the keyword every real Looprig tool carries and
	// the one Gemini's Schema has no member for. Dropping it silently was the
	// second half of the defect; it must move the whole schema to the
	// JSON-Schema field instead.
	t.Run("a schema it cannot express goes to parametersJsonSchema verbatim", func(t *testing.T) {
		t.Parallel()

		schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer","enum":[1,3,7]}},"required":["city"],"additionalProperties":false}`)
		body, err := geminiapi.EncodeRequest(inference.Request{
			Model:    geminiMultimodalModel(),
			Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "Weather in Oslo?"})},
			Tools: []inference.Tool{{
				Name:        "get_weather",
				Description: "Look up a forecast",
				Schema:      schema,
			}},
		})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		gateRequest(t, body)

		declaration := onlyFunctionDeclaration(t, body)
		if len(declaration.Parameters) != 0 {
			t.Errorf("parameters = %s, want the schema in parametersJsonSchema instead", declaration.Parameters)
		}
		assertJSONEquivalent(t, declaration.ParametersJSONSchema, schema)
	})
}

// functionDeclarationWire is the parameter half of an encoded FunctionDeclaration.
type functionDeclarationWire struct {
	Parameters           json.RawMessage `json:"parameters"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema"`
}

func onlyFunctionDeclaration(t testing.TB, body []byte) functionDeclarationWire {
	t.Helper()
	var declared struct {
		Tools []struct {
			FunctionDeclarations []functionDeclarationWire `json:"functionDeclarations"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &declared); err != nil {
		t.Fatalf("unmarshal encoded request: %v", err)
	}
	if len(declared.Tools) != 1 || len(declared.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %+v, want one declaration", declared.Tools)
	}
	return declared.Tools[0].FunctionDeclarations[0]
}

func declarationParameterType(t testing.TB, parameters json.RawMessage) string {
	t.Helper()
	var schema struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(parameters, &schema); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	return schema.Type
}

// TestGeminiDocumentAndAudioRequestEncoding holds the encodings of a
// DocumentBlock and an AudioBlock against Google's REQUEST schema and against
// goldens that were, until these blocks were implemented, examples of legal
// requests Looprig could not produce (request_document_pdf_inline.json and
// request_audio_inline.json, formerly the *_unreachable pair). Encoding to the
// exact bytes that corpus already vouched for is the point: the fixture was
// written from the discovery document, not from our encoder.
//
// What the gate proves about these two parts specifically: `inlineData` matches
// the Blob $ref — an object, not a string or an array — and both `mimeType` and
// `data` are strings. What it cannot prove is stated where it is checked
// instead: the mime is one Blob accepts (an encoder allowlist), `data` is really
// base64 (`format: byte` is an annotation the gate does not assert), and no part
// carries two members of the Part.data union (Part has no oneOf in the discovery
// document — see TestGeminiPartWithTwoContentMembers and the codec's own
// assertSingleDataMember).
func TestGeminiDocumentAndAudioRequestEncoding(t *testing.T) {
	t.Parallel()

	pdfBytes := fixtureInlineBytes(t, "request_document_pdf_inline.json")
	mp3Bytes := fixtureInlineBytes(t, "request_audio_inline.json")

	cases := []struct {
		golden   string
		request  inference.Request
		assertOn func(t *testing.T, body []byte)
	}{
		{
			golden: "request_document_pdf_inline.json",
			request: inference.Request{
				Model: geminiMultimodalModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "contract.pdf", Data: pdfBytes},
					&content.TextBlock{Text: "Summarise this contract."},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				parts := requestParts(t, body, 0)
				if len(parts) != 2 {
					t.Fatalf("parts = %d, want 2", len(parts))
				}
				assertInlineData(t, parts[0], "application/pdf", pdfBytes)
				if _, ok := parts[1]["text"]; !ok {
					t.Error("parts[1] is not the text part; block order was not preserved")
				}
			},
		},
		{
			golden: "request_audio_inline.json",
			request: inference.Request{
				Model: geminiMultimodalModel(),
				Messages: content.AgenticMessages{userTurn(
					// audio/mp3 rather than the audio/mpeg core/content names:
					// Blob's documented audio group is the wildcard `audio/*`,
					// so the encoder must carry whichever subtype the caller
					// holds instead of folding it to a known constant.
					&content.AudioBlock{MediaType: content.MediaType("audio/mp3"), Data: mp3Bytes},
					&content.TextBlock{Text: "Transcribe this clip."},
				)},
			},
			assertOn: func(t *testing.T, body []byte) {
				assertInlineData(t, requestParts(t, body, 0)[0], "audio/mp3", mp3Bytes)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.golden, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(tc.request)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			gateRequest(t, body)
			assertJSONEquivalent(t, body, geminiFixture(t, tc.golden))
			tc.assertOn(t, body)
			assertOneDataMemberPerPart(t, body)
		})
	}
}

// TestGeminiToolResultMediaRequestEncoding gates the path an MCP tool result
// takes when it carries media (mcp/pkg/harness/tools.go builds AudioBlock,
// ImageBlock and DocumentBlock from tool output, and the harness persists them).
// The classic functionResponse has no media member, so the bytes ride as
// inlineData parts of the same user turn — a shape the gate confirms is legal,
// and one the model reads in the right position in the thread.
func TestGeminiToolResultMediaRequestEncoding(t *testing.T) {
	t.Parallel()

	body, err := geminiapi.EncodeRequest(inference.Request{
		Model: geminiMultimodalModel(),
		Messages: content.AgenticMessages{
			userTurn(&content.TextBlock{Text: "read the memo aloud"}),
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewToolUseBlock("call_speak_01", "speak", json.RawMessage(`{"voice":"alto"}`), nil, ""),
			}}},
			&content.ToolResultMessage{
				ToolUseID: "call_speak_01",
				Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{
					&content.TextBlock{Text: "spoken"},
					&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{0x49, 0x44, 0x33}},
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "memo.pdf", Data: []byte("%PDF-1.4\n")},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateRequest(t, body)
	assertOneDataMemberPerPart(t, body)

	parts := requestParts(t, body, 2)
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want the functionResponse plus two media parts", len(parts))
	}
	if _, ok := parts[0]["functionResponse"]; !ok {
		t.Errorf("parts[0] = %v, want the functionResponse first", parts[0])
	}
	assertInlineData(t, parts[1], "audio/mpeg", []byte{0x49, 0x44, 0x33})
	assertInlineData(t, parts[2], "application/pdf", []byte("%PDF-1.4\n"))
}

// TestGeminiUnreachableMultimodalInputs records what remains out of reach after
// documents and audio were implemented. Each fixture is a legal generateContent
// REQUEST — proven so by the gate — that Looprig still has no way to express:
//
//   - a Files API URI for a document. Gemini's fileData carries any mime, but
//     the neutral AudioBlock and DocumentBlock hold bytes and have no URL
//     source, so only an image can reach that member.
//   - video, in either member: there is no video block in the vocabulary.
//   - FunctionResponse.parts, the native multimodal tool result. The discovery
//     document describes referencing such a part from the response object by a
//     `$ref` to its `inline_data.display_name`, but declares no display_name on
//     FunctionResponseBlob — the form cannot be built from the document that
//     defines it. Media in a tool result therefore travels as ordinary parts of
//     the same turn (TestGeminiToolResultMediaRequestEncoding).
//
// What the encoder still refuses outright is a media type Blob does not accept:
// a .docx or .xlsx blob is not in Google's list, so it fails closed with a typed
// error naming the type rather than being sent to be rejected.
func TestGeminiUnreachableMultimodalInputs(t *testing.T) {
	t.Parallel()

	t.Run("the shapes are legal Gemini requests", func(t *testing.T) {
		t.Parallel()

		for _, name := range []string{
			"request_file_data_pdf_unreachable.json",
			"request_video_file_data_unreachable.json",
			"request_function_response_inline_data_unreachable.json",
		} {
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				gateRequest(t, geminiFixture(t, name))
			})
		}
	})

	// Fail-secure, not silent: a media type the contract does not admit is
	// refused so nobody learns about it from a provider 400.
	t.Run("the encoder refuses a media type Blob does not accept", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			name  string
			block content.Block
		}{
			{"docx document", &content.DocumentBlock{
				MediaType: content.MediaTypeDocumentDOCX,
				Name:      "contract.docx",
				Data:      []byte{0x50, 0x4b, 0x03, 0x04},
			}},
			{"xlsx document", &content.DocumentBlock{
				MediaType: content.MediaTypeDocumentXLSX,
				Name:      "figures.xlsx",
				Data:      []byte{0x50, 0x4b, 0x03, 0x04},
			}},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				_, err := geminiapi.EncodeRequest(inference.Request{
					Model:    geminiMultimodalModel(),
					Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "look"}, tc.block)},
				})
				var unsupported *geminiapi.UnsupportedBlockError
				if !errors.As(err, &unsupported) {
					t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
				}
			})
		}
	})
}

// --- helpers ---------------------------------------------------------------

func onlyToolUse(t testing.TB, resp *inference.Response) *content.ToolUseBlock {
	t.Helper()
	if got := len(resp.Message.Blocks); got != 1 {
		t.Fatalf("blocks = %d, want 1", got)
	}
	call, ok := resp.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("block = %T, want *content.ToolUseBlock", resp.Message.Blocks[0])
	}
	return call
}

// assertOpaqueState checks the provider-private state a codec stores for a
// thoughtSignature: the JSON-encoded form of the exact wire string, tagged with
// this dialect so it can never be replayed toward another provider.
func assertOpaqueState(t testing.TB, state json.RawMessage, wantSignature string) {
	t.Helper()
	var got string
	if err := json.Unmarshal(state, &got); err != nil {
		t.Fatalf("ProviderState %s is not a JSON string: %v", state, err)
	}
	if got != wantSignature {
		t.Errorf("thoughtSignature = %q, want %q", got, wantSignature)
	}
}

func assertJSONEquivalent(t testing.TB, got, want []byte) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("unmarshal got: %v", err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("unmarshal want: %v", err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("JSON mismatch\n got: %s\nwant: %s", got, want)
	}
}

// requestParts projects contents[index].parts out of an encoded request body.
func requestParts(t testing.TB, body []byte, index int) []map[string]json.RawMessage {
	t.Helper()
	var envelope struct {
		Contents []struct {
			Parts []map[string]json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if index >= len(envelope.Contents) {
		t.Fatalf("contents has %d entries, want at least %d", len(envelope.Contents), index+1)
	}
	return envelope.Contents[index].Parts
}

// fixtureInlineBytes recovers the raw bytes behind the first part's inlineData
// in a request fixture, so a test can encode FROM the fixture's own payload and
// compare back to it. The blob is decoded rather than hand-copied because
// `format: byte` is an annotation the gate does not assert: if the fixture's
// data were not valid base64, only this decode would say so.
func fixtureInlineBytes(t testing.TB, name string) []byte {
	t.Helper()
	parts := requestParts(t, geminiFixture(t, name), 0)
	var blob struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(parts[0]["inlineData"], &blob); err != nil {
		t.Fatalf("fixture %s: decode inlineData: %v", name, err)
	}
	raw, err := base64.StdEncoding.DecodeString(blob.Data)
	if err != nil {
		t.Fatalf("fixture %s: inlineData.data is not base64: %v", name, err)
	}
	return raw
}

// partDataMembers is the Part.data union: "A `Part` can only contain one of the
// accepted types in `Part.data`". thought, thoughtSignature, videoMetadata and
// partMetadata are Part attributes that legally accompany a data member, not
// members of the union.
var partDataMembers = []string{
	"text", "inlineData", "fileData", "functionCall", "functionResponse",
	"executableCode", "codeExecutionResult", "toolCall", "toolResponse",
}

// assertOneDataMemberPerPart checks every part of every turn carries exactly one
// data member. This is the arity the gate cannot see: the discovery document
// expresses Part as an ordinary object with all members optional, so a part
// holding both inlineData and text validates cleanly
// (TestGeminiPartWithTwoContentMembers proves it on the response side).
func assertOneDataMemberPerPart(t testing.TB, body []byte) {
	t.Helper()
	var envelope struct {
		Contents []struct {
			Parts []map[string]json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	for turn, entry := range envelope.Contents {
		for index, part := range entry.Parts {
			var set []string
			for _, member := range partDataMembers {
				if _, ok := part[member]; ok {
					set = append(set, member)
				}
			}
			if len(set) != 1 {
				t.Errorf("contents[%d].parts[%d] carries %v, want exactly one Part.data member", turn, index, set)
			}
		}
	}
}

func assertInlineData(t testing.TB, part map[string]json.RawMessage, wantMIME string, wantBytes []byte) {
	t.Helper()
	raw, ok := part["inlineData"]
	if !ok {
		t.Fatalf("part = %v, want an inlineData part", part)
	}
	var got struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode inlineData: %v", err)
	}
	if got.MimeType != wantMIME {
		t.Errorf("inlineData.mimeType = %q, want %q", got.MimeType, wantMIME)
	}
	if want := base64.StdEncoding.EncodeToString(wantBytes); got.Data != want {
		t.Errorf("inlineData.data = %q, want %q", got.Data, want)
	}
}

// --- named tool choice -----------------------------------------------------

// TestGeminiNamedToolChoiceRequestEncoding gates the forced-tool body. Gemini
// has no dedicated single-tool mode: FunctionCallingConfig spells it as mode
// ANY plus a one-element allowedFunctionNames list.
func TestGeminiNamedToolChoiceRequestEncoding(t *testing.T) {
	t.Parallel()

	body, err := geminiapi.EncodeRequest(inference.Request{
		Model:    geminiFixtureModel("gemini-2.5-flash", model.WithTools()),
		Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "weather in NYC?"})},
		Tools: []inference.Tool{
			{Name: "get_weather", Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "get_time", Schema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: inference.ToolNamed("get_time"),
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateRequest(t, body)

	var envelope struct {
		ToolConfig struct {
			FunctionCallingConfig struct {
				Mode                 string   `json:"mode"`
				AllowedFunctionNames []string `json:"allowedFunctionNames"`
			} `json:"functionCallingConfig"`
		} `json:"toolConfig"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	config := envelope.ToolConfig.FunctionCallingConfig
	if config.Mode != "ANY" {
		t.Errorf("functionCallingConfig.mode = %q, want ANY", config.Mode)
	}
	if len(config.AllowedFunctionNames) != 1 || config.AllowedFunctionNames[0] != "get_time" {
		t.Errorf("allowedFunctionNames = %v, want [get_time]", config.AllowedFunctionNames)
	}
}

// TestToolChoiceGateStrength measures what Gemini's request gate really
// enforces on toolConfig, and it is the weakest of the four by a wide margin.
//
// The derived request schema declares required properties on 1 of 49 shapes
// and contains no oneOf at all, so FunctionCallingConfig accepts an empty
// object, an allowlist on a mode that forbids one, and a name that matches no
// declared function. The single thing it does catch is the `mode` enum, which
// the discovery document does spell out. Everything else about this dialect's
// tool choice is carried by the encoder and by
// inference.ValidateRequestFeatures, not by the gate.
func TestToolChoiceGateStrength(t *testing.T) {
	t.Parallel()

	body := func(toolConfig string) []byte {
		return []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],` +
			`"tools":[{"functionDeclarations":[{"name":"get_weather","parametersJsonSchema":{"type":"object"}}]}],` +
			`"toolConfig":` + toolConfig + `}`)
	}

	cases := []struct {
		name       string
		toolConfig string
		wantReject bool
		because    string
	}{
		{
			name:       "the shape the encoder emits",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["get_weather"]}}`,
			because:    "ANY is a declared enum member and allowedFunctionNames is a string array",
		},
		{
			name:       "unknown mode",
			toolConfig: `{"functionCallingConfig":{"mode":"FORCED"}}`,
			wantReject: true,
			because:    "mode carries the discovery document's enum; this is the only tool-choice constraint the gate enforces",
		},
		{
			name:       "allowed names that are not strings",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":[7]}}`,
			wantReject: true,
			because:    "allowedFunctionNames.items is typed string",
		},
		{
			name:       "no mode at all",
			toolConfig: `{"functionCallingConfig":{"allowedFunctionNames":["get_weather"]}}`,
			because:    "FunctionCallingConfig declares no required properties",
		},
		{
			name:       "allowlist on a mode that forbids one",
			toolConfig: `{"functionCallingConfig":{"mode":"NONE","allowedFunctionNames":["get_weather"]}}`,
			because:    "the `should only be set when Mode is ANY or VALIDATED` rule is prose, not schema",
		},
		{
			name:       "name that matches no declared function",
			toolConfig: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["undeclared"]}}`,
			because:    "no cross-field constraint exists; ValidateRequestFeatures carries it",
		},
		{
			name:       "empty function calling config",
			toolConfig: `{"functionCallingConfig":{}}`,
			because:    "nothing inside it is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate("gemini", "generate_content_request", body(tc.toolConfig))
			if tc.wantReject && err == nil {
				t.Fatalf("gate accepted %s, want rejection (%s)", tc.toolConfig, tc.because)
			}
			if !tc.wantReject && err != nil {
				t.Fatalf("gate rejected %s (%v), want acceptance (%s)", tc.toolConfig, err, tc.because)
			}
		})
	}
}
