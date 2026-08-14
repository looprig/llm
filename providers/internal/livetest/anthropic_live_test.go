//go:build live

package livetest

import (
	"testing"

	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	anthropicprovider "github.com/looprig/llm/providers/anthropic"
)

// TestLiveAnthropicFormat drives providers/anthropic — the real client
// llm/auto.New dispatches to for an `anthropic`-format model — against a live
// Anthropic-Messages-compatible gateway.
//
// SCOPE WARNING. The endpoint behind these aliases is OpenCode Zen's
// Anthropic-compatible surface fronting MiniMax and Qwen, not api.anthropic.com.
// A pass proves our body is accepted by a real independent implementation of the
// Messages contract; it does NOT prove Anthropic's own server would accept it,
// and a rejection here may be a gateway limitation rather than an encoder bug.
// Every subtest reports the server's own error text so the two can be told apart.
//
// The highest-value assertion is the tool-result continuation: that second
// request carries our thinking-block encoding (the `thinking`/`signature`
// members the request schema marks required), the tool_use id we minted-and-
// echoed against Anthropic's ^[a-zA-Z0-9_-]+$ pattern, and a tool_result
// referencing it. A 200 there is live proof of all three at once.
func TestLiveAnthropicFormat(t *testing.T) {
	aliases := []string{"opencode-go-minimax-m3", "opencode-go-qwen3.8-max"}
	for _, alias := range aliases {
		t.Run(alias, func(t *testing.T) {
			row := entry(t, alias)
			if row.APIFormat != string(model.APIFormatAnthropic) {
				t.Skipf("catalogue row %q is api_format %q, not anthropic", alias, row.APIFormat)
			}
			rec := newRecorder(t, row.BaseURL)
			build := func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
				t.Helper()
				selected := row.selectedModel(rec.baseURL(), opts...)
				client, err := anthropicprovider.New(selected, row.key())
				if err != nil {
					t.Fatalf("anthropic.New: %v", scrub(err.Error()))
				}
				return withRetries(client), selected
			}
			client, selected := build(t)

			// Effort none keeps the baseline probes free of the thinking block,
			// so a text/stream failure cannot be blamed on reasoning.
			plain := scenario{client: client, selected: selected, rec: rec, effort: model.EffortNone, maxTokens: 512, rebuild: build}
			t.Run("text", plain.textTurn)
			t.Run("stream", plain.streamTurn)
			t.Run("empty_tool_result", func(t *testing.T) { plain.toolRoundTrip(t, "") })
			t.Run("parallel_tool_calls", plain.parallelToolCalls)
			t.Run("named_tool_choice", plain.namedToolChoice)
			t.Run("required_tool_choice", plain.requiredToolChoice)
			t.Run("image_input", plain.imageInput)
			t.Run("document_text", func(t *testing.T) { plain.documentInput(t, textDocument()) })
			t.Run("structured_output", func(t *testing.T) { plain.structuredOutput(t, false) })
			t.Run("refusal", plain.refusal)

			// The thinking probes need the catalogue to advertise the effort;
			// sending a level the gateway does not list would produce a
			// rejection that says nothing about our encoder.
			effort := model.EffortNone
			switch {
			case row.supportsEffort("high"):
				effort = model.EffortHigh
			case row.supportsEffort("max"):
				effort = model.EffortMax
			}
			if effort == model.EffortNone || !row.Capabilities.Thinking {
				t.Log("skipping thinking probes: catalogue advertises no thinking effort for this row")
				return
			}

			// A thinking turn needs headroom for the reasoning block plus an
			// answer; too small a budget truncates and looks like a decode bug.
			thinking := scenario{client: client, selected: selected, rec: rec, effort: effort, maxTokens: 4096, rebuild: build}
			t.Run("thinking", thinking.thinkingTurn)
			t.Run("tool_round_trip_with_thinking", func(t *testing.T) {
				thinking.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
			})
		})
	}
}
