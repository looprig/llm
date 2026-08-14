//go:build live

package livetest

import (
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	openaiprovider "github.com/looprig/llm/providers/openai"
	openrouterprovider "github.com/looprig/llm/providers/openrouter"
	syntheticprovider "github.com/looprig/llm/providers/synthetic"
)

// chatTarget names one live OpenAI-Chat-compatible endpoint. Three INDEPENDENT
// implementations are covered on purpose: "an API format is not a guarantee of
// identical behaviour" is this module's founding rule, and a single gateway
// accepting a body proves nothing about the dialect, only about that gateway.
type chatTarget struct {
	alias string
	// construct builds the provider client the composition root would pick for
	// this row. Each provider package is exercised as itself, not through a
	// shared generic transport, so provider-specific routing and auth are live
	// too.
	construct func(t *testing.T, selected model.Model, row catalogEntry) inference.Client
}

func openAIChatTargets() []chatTarget {
	openAI := func(t *testing.T, selected model.Model, row catalogEntry) inference.Client {
		t.Helper()
		client, err := openaiprovider.New(selected, row.key())
		if err != nil {
			t.Fatalf("openai.New: %v", scrub(err.Error()))
		}
		return client
	}
	synthetic := func(t *testing.T, selected model.Model, row catalogEntry) inference.Client {
		t.Helper()
		client, err := syntheticprovider.New(selected, row.key())
		if err != nil {
			t.Fatalf("synthetic.New: %v", scrub(err.Error()))
		}
		return client
	}
	openRouter := func(t *testing.T, selected model.Model, row catalogEntry) inference.Client {
		t.Helper()
		client, err := openrouterprovider.New(selected, row.key())
		if err != nil {
			t.Fatalf("openrouter.New: %v", scrub(err.Error()))
		}
		return client
	}
	return []chatTarget{
		{alias: "opencode-go-glm-5.2", construct: openAI},
		{alias: "opencode-go-kimi-k3", construct: openAI},
		{alias: "opencode-go-grok-4.5", construct: openAI},
		{alias: "opencode-go-deepseek-v4-flash", construct: openAI},
		{alias: "synthetics-kimi-k3", construct: synthetic},
		{alias: "synthetics-glm-5.2", construct: synthetic},
		// OpenRouter is a fourth independent implementation of the dialect and
		// the only one here that is itself a router: the model id selects an
		// upstream whose own compatibility varies, so a rejection may come from
		// OpenRouter's parser or from whatever it forwarded to. The `:free`
		// tiers keep the volume free as well as modest.
		{alias: "openrouter-gemma-4-free", construct: openRouter},
		{alias: "openrouter-gpt-oss-20b-free", construct: openRouter},
		{alias: "openrouter-nemotron-free", construct: openRouter},
	}
}

// TestLiveOpenAIChatFormat runs the full use-case matrix against every
// configured Chat-compatible gateway: the baseline turns, then the shapes a
// real agent loop produces that no schema can adjudicate.
func TestLiveOpenAIChatFormat(t *testing.T) {
	for _, target := range openAIChatTargets() {
		t.Run(target.alias, func(t *testing.T) {
			row := entry(t, target.alias)
			if row.APIFormat != string(model.APIFormatOpenAI) {
				t.Skipf("catalogue row %q is api_format %q, not openai", target.alias, row.APIFormat)
			}
			rec := newRecorder(t, row.BaseURL)
			build := func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
				t.Helper()
				selected := row.selectedModel(rec.baseURL(), opts...)
				return withRetries(target.construct(t, selected, row)), selected
			}
			client, selected := build(t)

			effort := model.EffortNone
			for _, candidate := range []model.Effort{model.EffortMedium, model.EffortLow, model.EffortHigh} {
				if row.supportsEffort(string(candidate)) {
					effort = candidate
					break
				}
			}
			probe := scenario{client: client, selected: selected, rec: rec, effort: effort, maxTokens: 1024, rebuild: build}
			t.Run("text", probe.textTurn)
			t.Run("stream", probe.streamTurn)
			t.Run("tool_round_trip", func(t *testing.T) {
				probe.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
			})
			// An empty tool result is not contrived: a tool whose whole effect
			// is a side effect legitimately returns nothing, and the Chat
			// dialect's tool message has a required `content` member that an
			// omitempty tag would erase.
			t.Run("empty_tool_result", func(t *testing.T) { probe.toolRoundTrip(t, "") })
			t.Run("parallel_tool_calls", probe.parallelToolCalls)
			t.Run("named_tool_choice", probe.namedToolChoice)
			t.Run("required_tool_choice", probe.requiredToolChoice)
			t.Run("image_input", probe.imageInput)
			t.Run("document_text", func(t *testing.T) { probe.documentInput(t, textDocument()) })
			t.Run("document_pdf", func(t *testing.T) { probe.documentInput(t, pdfDocument()) })
			t.Run("structured_output", func(t *testing.T) { probe.structuredOutput(t, false) })
			t.Run("structured_output_with_tools", func(t *testing.T) { probe.structuredOutput(t, true) })
			t.Run("refusal", probe.refusal)
			if row.Capabilities.Thinking {
				t.Run("thinking", probe.thinkingTurn)
			}
		})
	}
}

// TestLiveOpenAIChatTokenLimitField probes the one encoder decision this dialect
// makes that no schema can adjudicate: WHICH token-limit member to send.
//
// openaiapi.BuildChatRequest gates the choice on Model.Caps.Thinking —
// max_completion_tokens for a reasoning model, the legacy max_tokens otherwise —
// because OpenAI's reasoning models reject max_tokens outright while many
// OpenAI-compatible servers understand only max_tokens. Both spellings are legal
// per the schema (they are two optional members), so the request gate passes
// either way and only a live server can say which one it honours.
//
// This sends the SAME prompt twice per gateway, differing only in the advertised
// Thinking capability, and reports each outcome separately. A rejection of
// either spelling is a gateway compatibility finding worth recording, not
// necessarily an encoder defect — that is exactly why the encoder branches.
func TestLiveOpenAIChatTokenLimitField(t *testing.T) {
	targets := []chatTarget{
		{alias: "opencode-go-glm-5.2"},
		{alias: "synthetics-glm-5.2"},
	}
	for _, target := range targets {
		t.Run(target.alias, func(t *testing.T) {
			row := entry(t, target.alias)
			if row.APIFormat != string(model.APIFormatOpenAI) {
				t.Skipf("catalogue row %q is api_format %q, not openai", target.alias, row.APIFormat)
			}
			cases := []struct {
				name     string
				thinking bool
				field    string
			}{
				{name: "reasoning_model_max_completion_tokens", thinking: true, field: "max_completion_tokens"},
				{name: "plain_model_max_tokens", thinking: false, field: "max_tokens"},
			}
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					rec := newRecorder(t, row.BaseURL)
					// selectedModel derives Caps from the catalogue; override
					// the one capability under test so the two runs differ in
					// nothing else.
					forced := row
					forced.Capabilities.Thinking = tc.thinking
					selected := forced.selectedModel(rec.baseURL())

					var client inference.Client
					var err error
					switch llm.Provider(row.Provider) {
					case llm.ProviderSynthetic:
						client, err = syntheticprovider.New(selected, row.key())
					default:
						client, err = openaiprovider.New(selected, row.key())
					}
					if err != nil {
						t.Fatalf("client construction: %v", scrub(err.Error()))
					}

					ctx := probeContext(t)
					_, invokeErr := client.Invoke(ctx, inference.Request{
						Model:    selected,
						System:   systemPrompt,
						Messages: content.AgenticMessages{userText("Reply with exactly the word: ready")},
						Override: &model.Sampling{MaxTokens: intPtr(64)},
					})

					// Confirm the encoder actually emitted the field under test
					// before reading anything into the server's answer; a probe
					// that asserts on a body it never sent proves nothing.
					sent := rec.snapshot()
					if len(sent) == 0 {
						t.Fatalf("no request reached the recorder")
					}
					body := string(sent[0].RequestBody)
					if !strings.Contains(body, `"`+tc.field+`"`) {
						t.Fatalf("encoder did not emit %s; body was %s", tc.field, scrubBytes(sent[0].RequestBody))
					}
					if invokeErr != nil {
						rejected(t, rec, tc.field, invokeErr)
						return
					}
					t.Logf("%s accepted by %s", tc.field, row.Provider)
				})
			}
		})
	}
}
