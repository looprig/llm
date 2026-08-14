//go:build live

package livetest

import (
	"testing"

	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	openaiprovider "github.com/looprig/llm/providers/openai"
)

// TestLiveOpenAIResponsesFormat drives providers/openai in its Responses dialect
// against a live Responses-compatible gateway.
//
// SCOPE WARNING. This is OpenCode Zen's Responses surface in front of a
// GPT-5.6-class model, not api.openai.com/v1/responses. Treat every result as
// evidence about that gateway.
//
// Three encoder details this format carries are only checkable live:
//
//   - FunctionTool.strict. The spec lists it in FunctionTool.required, so a tool
//     object omitting it is not a legal request body; every request we ever sent
//     omitted it until it was fixed. Any subtest here that declares a tool sends
//     it, so a 200 on the tool request is that fix's first live evidence.
//   - reasoning item id replay. A reasoning item echoed back to the server must
//     carry the id the server issued; the continuation request in
//     tool_round_trip replays the decoded assistant turn verbatim, id included.
//   - function_call_output.output on an EMPTY result. `output` is required, so a
//     tool that returns nothing must still send `"output": ""` rather than
//     dropping the member — the exact shape omitempty used to erase. The
//     empty_tool_result subtest exists solely to send that body.
func TestLiveOpenAIResponsesFormat(t *testing.T) {
	const alias = "opencode-go-gpt-5.6-luna"
	row := entry(t, alias)
	if row.APIFormat != string(model.APIFormatOpenAIResponses) {
		t.Skipf("catalogue row %q is api_format %q, not openai-responses", alias, row.APIFormat)
	}
	rec := newRecorder(t, row.BaseURL)
	build := func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
		t.Helper()
		selected := row.selectedModel(rec.baseURL(), opts...)
		client, err := openaiprovider.New(selected, row.key())
		if err != nil {
			t.Fatalf("openai.New: %v", scrub(err.Error()))
		}
		return withRetries(client), selected
	}
	client, selected := build(t)

	plain := scenario{client: client, selected: selected, rec: rec, effort: model.EffortNone, maxTokens: 512, rebuild: build}
	t.Run("text", plain.textTurn)
	t.Run("stream", plain.streamTurn)
	t.Run("parallel_tool_calls", plain.parallelToolCalls)
	t.Run("named_tool_choice", plain.namedToolChoice)
	t.Run("required_tool_choice", plain.requiredToolChoice)
	t.Run("image_input", plain.imageInput)
	t.Run("document_pdf", func(t *testing.T) { plain.documentInput(t, pdfDocument()) })
	t.Run("document_text", func(t *testing.T) { plain.documentInput(t, textDocument()) })
	t.Run("structured_output", func(t *testing.T) { plain.structuredOutput(t, false) })
	t.Run("structured_output_with_tools", func(t *testing.T) { plain.structuredOutput(t, true) })
	t.Run("refusal", plain.refusal)

	effort := model.EffortNone
	switch {
	case row.supportsEffort("medium"):
		effort = model.EffortMedium
	case row.supportsEffort("low"):
		effort = model.EffortLow
	case row.supportsEffort("high"):
		effort = model.EffortHigh
	}
	reasoning := scenario{client: client, selected: selected, rec: rec, effort: effort, maxTokens: 4096, rebuild: build}
	if effort != model.EffortNone && row.Capabilities.Thinking {
		t.Run("reasoning", reasoning.thinkingTurn)
	}

	t.Run("tool_round_trip", func(t *testing.T) {
		reasoning.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
	})

	// An empty tool result is not a contrived case: a tool whose whole effect is
	// a side effect (a write, a delete) legitimately returns nothing, and that
	// is precisely the request the missing-`output` defect produced illegally.
	t.Run("empty_tool_result", func(t *testing.T) {
		reasoning.toolRoundTrip(t, "")
	})
}
