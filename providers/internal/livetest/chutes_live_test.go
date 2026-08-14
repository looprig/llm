//go:build live

package livetest

import (
	"testing"

	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm/providers/chutes"
)

// TestLiveChutes probes the third independent OpenAI-Chat implementation in the
// catalogue. It is separated from TestLiveOpenAIChatFormat because the Chutes
// client is structurally different from every other provider here: it seals the
// request through an attested TEE tunnel (ML-KEM-768 + ChaCha20-Poly1305) and
// binds several gateway hosts of its own, so it cannot be routed through the
// loopback recorder and there is no plaintext body to capture.
//
// It still reuses inference/codec/openaiapi for both encoding and decoding
// (providers/chutes/encode.go embeds openaiapi.ChatRequest and decode.go calls
// openaiapi.DecodeResponse), so a pass here is genuine live evidence for the
// shared Chat encoder — reached over a completely different transport.
//
// A failure in NVIDIA attestation is an ENVIRONMENT result, not an encoder
// result: nothing was ever sent. The probe reports the error verbatim
// (scrubbed) so the distinction is visible in the log.
func TestLiveChutes(t *testing.T) {
	for _, alias := range []string{"chutes-kimi-k3", "chutes-glm-5.2"} {
		t.Run(alias, func(t *testing.T) {
			row := entry(t, alias)
			if row.APIFormat != string(model.APIFormatOpenAI) {
				t.Skipf("catalogue row %q is api_format %q, not openai", alias, row.APIFormat)
			}
			selected := row.selectedModel(row.BaseURL)
			client := chutes.New(row.BaseURL, row.APIKey)

			build := func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
				t.Helper()
				return withRetries(chutes.New(row.BaseURL, row.APIKey)), row.selectedModel(row.BaseURL, opts...)
			}
			probe := scenario{client: client, selected: selected, rec: nil, effort: model.EffortNone, maxTokens: 512, rebuild: build}
			t.Run("text", probe.textTurn)
			t.Run("stream", probe.streamTurn)
			t.Run("tool_round_trip", func(t *testing.T) {
				probe.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
			})
			t.Run("empty_tool_result", func(t *testing.T) { probe.toolRoundTrip(t, "") })
			t.Run("parallel_tool_calls", probe.parallelToolCalls)
			t.Run("named_tool_choice", probe.namedToolChoice)
			t.Run("required_tool_choice", probe.requiredToolChoice)
			t.Run("structured_output", func(t *testing.T) { probe.structuredOutput(t, false) })
		})
	}
}
