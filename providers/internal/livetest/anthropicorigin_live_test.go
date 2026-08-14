//go:build live

package livetest

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"

	anthropicprovider "github.com/looprig/llm/providers/anthropic"
)

// TestLiveAnthropicOrigin drives providers/anthropic against api.anthropic.com
// ITSELF, not a compatible gateway.
//
// This is the only target in the suite where a pass generalizes. Everywhere
// else the endpoint is a third party's reimplementation of somebody's contract,
// so a 200 says "this gateway's parser accepted our body" and nothing more.
// Here the origin defines the contract, so a 200 is the contract, and a 4xx is
// unambiguously ours to fix.
//
// It is also the only place two constructs can be tested at all:
//
//   - the reasoning SIGNATURE round trip (TestLiveAnthropicSignatureRoundTrip).
//     Anthropic cryptographically validates the signature on a replayed
//     thinking block; a compatible gateway that echoes reasoning without
//     validating it cannot distinguish a correct replay from a corrupt one.
//   - the `document` block. Anthropic's Base64PDFSource/PlainTextSource union
//     is an origin feature; most compatible gateways front models with no
//     document channel at all.
func TestLiveAnthropicOrigin(t *testing.T) {
	// Haiku is first and does the bulk of the work: it is the cheapest model
	// that exercises every path here. Sonnet runs only the cases where model
	// class plausibly changes the ANSWER rather than the encoding.
	for _, alias := range []string{"anthropic-haiku-4.5", "anthropic-sonnet-5"} {
		t.Run(alias, func(t *testing.T) {
			row := entry(t, alias)
			rec := newRecorder(t, row.BaseURL)
			build := anthropicBuilder(t, row, rec)
			client, selected := build(t)

			plain := scenario{
				client: client, selected: selected, rec: rec,
				effort: model.EffortNone, maxTokens: 512,
				rebuild: build,
			}
			t.Run("text", plain.textTurn)
			t.Run("stream", plain.streamTurn)
			t.Run("tool_round_trip", func(t *testing.T) {
				plain.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
			})
			t.Run("empty_tool_result", func(t *testing.T) { plain.toolRoundTrip(t, "") })
			t.Run("parallel_tool_calls", plain.parallelToolCalls)
			t.Run("named_tool_choice", plain.namedToolChoice)
			t.Run("required_tool_choice", plain.requiredToolChoice)
			t.Run("image_input", plain.imageInput)
			t.Run("document_pdf", func(t *testing.T) { plain.documentInput(t, pdfDocument()) })
			t.Run("document_text", func(t *testing.T) { plain.documentInput(t, textDocument()) })
			t.Run("structured_output", func(t *testing.T) { plain.structuredOutput(t, false) })
			t.Run("structured_output_with_tools", func(t *testing.T) { plain.structuredOutput(t, true) })
			t.Run("refusal", plain.refusal)

			// Thinking runs under whichever dialect this model actually
			// accepts, and the shared codec now emits it: the catalogue row
			// carries the model's ThinkingDialect and anthropicapi encodes the
			// matching `thinking` member. There is no per-model escape hatch
			// here any more — if this probe 400s on the thinking member, the
			// catalogue row is wrong or the codec is, and either is a finding
			// rather than something to route around.
			thinkingClient, thinkingModel := build(t)
			thinking := scenario{
				client: thinkingClient, selected: thinkingModel, rec: rec,
				effort: model.EffortMedium, maxTokens: 6000,
				rebuild: build,
			}
			t.Run("thinking", thinking.thinkingTurn)
			t.Run("thinking_tool_round_trip", func(t *testing.T) {
				thinking.toolRoundTrip(t, `{"temp_c": 17, "conditions": "overcast"}`)
			})
		})
	}
}

// TestLiveAnthropicThinkingDialect measures WHICH thinking request body each
// first-party model accepts, and holds the catalogue to the answer.
//
// The origin API answers this question per MODEL, and both wrong answers are
// hard 400s rather than degradations — claude-haiku-4-5 rejects
// `{"type":"adaptive"}` and claude-sonnet-5 rejects
// `{"type":"enabled","budget_tokens":N}`, each naming the other spelling. The
// codec used to choose from one boolean and always send the adaptive form,
// which left a budget-only model with no reasoning path through the shared
// encoder at all. It now emits the model's declared Caps.ThinkingDialect.
//
// Reported as a matrix rather than a pass/fail per mode: the useful output is
// which spelling each model takes. The assertion is that the spelling the
// CATALOGUE declares for this model is the one the server accepts — a
// catalogue row and a live endpoint disagreeing is the defect this probe now
// exists to catch, and the only thing that can reintroduce the original bug.
func TestLiveAnthropicThinkingDialect(t *testing.T) {
	for _, alias := range []string{"anthropic-haiku-4.5", "anthropic-sonnet-5"} {
		t.Run(alias, func(t *testing.T) {
			row := entry(t, alias)
			accepted := map[model.ThinkingDialect]bool{}
			for _, mode := range thinkingModes() {
				t.Run(mode.name, func(t *testing.T) {
					rec := newRecorder(t, row.BaseURL)
					selected := row.withDialect(mode.dialect).selectedModel(rec.baseURL())
					client, err := anthropicprovider.New(selected, row.key())
					if err != nil {
						t.Fatalf("anthropic.New: %v", scrub(err.Error()))
					}
					_, invokeErr := client.Invoke(probeContext(t), inference.Request{
						Model:    selected,
						System:   systemPrompt,
						Messages: content.AgenticMessages{userText("Reply with exactly the word: ready")},
						Override: &model.Sampling{MaxTokens: intPtr(4096), Effort: model.EffortMedium},
					})
					if invokeErr == nil {
						accepted[mode.dialect] = true
						t.Logf("%s accepts %s", row.Model, mode.name)
						return
					}
					body := rec.lastErrorBody()
					if modelRefusesFeature(body) {
						t.Logf("%s REJECTS %s as unsupported; server said: %s", row.Model, mode.name, body)
						return
					}
					rejected(t, rec, mode.name, invokeErr)
				})
			}
			declared := model.ThinkingDialect(row.Capabilities.ThinkingDialect)
			if declared == model.ThinkingDialectUnknown {
				t.Errorf("CATALOGUE FINDING: %s is advertised as thinking-capable but declares no thinking dialect, so the codec fails closed and reasoning is unreachable for it; the live answer is accepted=%v",
					row.Model, accepted)
				return
			}
			if !accepted[declared] {
				t.Errorf("CATALOGUE FINDING: %s declares the %q thinking dialect, but the origin API did not accept the body the codec emits for it (accepted=%v)",
					row.Model, declared, accepted)
			}
		})
	}
}

// anthropicBuilder returns a factory for a first-party client whose model
// descriptor can carry extra capabilities. Capabilities shape the request
// (Thinking selects the thinking block, AcceptsImages gates image input,
// StructuredOutput gates output_config.format), so a capability probe must
// build its own descriptor rather than mutate one shared with the baseline.
func anthropicBuilder(t *testing.T, row catalogEntry, rec *recorder) func(*testing.T, ...model.ModelOption) (inference.Client, model.Model) {
	t.Helper()
	if row.APIFormat != string(model.APIFormatAnthropic) {
		t.Skipf("catalogue row %q is api_format %q, not anthropic", row.Alias, row.APIFormat)
	}
	return func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
		t.Helper()
		selected := row.selectedModel(rec.baseURL(), opts...)
		client, err := anthropicprovider.New(selected, row.key())
		if err != nil {
			t.Fatalf("anthropic.New: %v", scrub(err.Error()))
		}
		return withRetries(client), selected
	}
}

// TestLiveAnthropicSignatureRoundTrip is the highest-value probe in this suite.
//
// Anthropic returns a `signature` on every thinking block and CRYPTOGRAPHICALLY
// VALIDATES it when the block is replayed: a corrupted signature is rejected
// with an HTTP 400 naming the field. That makes this the one endpoint where a
// successful reasoning replay proves something a schema never could — not just
// that the members are present and well-typed, but that the exact bytes the
// server issued survived decode, storage in the neutral ThinkingBlock, and
// re-encode without a single character changing.
//
// The shape under test is the one a real agent produces, and it is why a tool
// call sits in the middle: the assistant turn that carries the signature is the
// same turn that requests the tool, so the signature is replayed as part of a
// message that ALSO has to satisfy the tool_use/tool_result pairing rules. A
// probe that replayed a bare thinking turn would exercise neither interaction.
//
// The tamper control is not decoration. Without it, a 200 on the replay is
// consistent with a server that ignores the signature entirely, and the whole
// probe would prove nothing. Sending a deliberately corrupted signature and
// observing a 400 establishes that this endpoint really does check, which is
// what gives the passing case its meaning.
// It runs once per (model, thinking dialect) pair because which dialect a model
// accepts is a live question no schema answers, and a signature minted under
// either one must round-trip identically. Both dialects now go through the
// shared codec — the pair whose dialect the model refuses skips itself on the
// server's own "not supported" message.
func TestLiveAnthropicSignatureRoundTrip(t *testing.T) {
	for _, alias := range []string{"anthropic-haiku-4.5", "anthropic-sonnet-5"} {
		for _, mode := range thinkingModes() {
			t.Run(alias+"/"+mode.name, func(t *testing.T) {
				signatureRoundTrip(t, alias, mode)
			})
		}
	}
}

// thinkingMode is one way of asking Anthropic for reasoning. Both spellings are
// legal request bodies; only a server can say which a given model honours. Both
// are reachable through the shared codec now, selected by the dialect declared
// on the model descriptor rather than by a provider-level body override.
type thinkingMode struct {
	name    string
	dialect model.ThinkingDialect
}

func thinkingModes() []thinkingMode {
	return []thinkingMode{
		{name: "codec_adaptive", dialect: model.ThinkingDialectAdaptive},
		{name: "codec_budget", dialect: model.ThinkingDialectBudget},
	}
}

// withDialect returns a copy of the row whose declared thinking dialect is
// replaced, so a probe can sweep both spellings against one endpoint without
// mutating the shared catalogue row.
func (e catalogEntry) withDialect(d model.ThinkingDialect) catalogEntry {
	e.Capabilities.Thinking = true
	e.Capabilities.ThinkingDialect = string(d)
	return e
}

func signatureRoundTrip(t *testing.T, alias string, mode thinkingMode) {
	t.Helper()
	row := entry(t, alias)
	if row.APIFormat != string(model.APIFormatAnthropic) {
		t.Skipf("catalogue row %q is api_format %q, not anthropic", alias, row.APIFormat)
	}
	rec := newRecorder(t, row.BaseURL)

	// The dialect is declared on the model descriptor and the codec encodes the
	// matching `thinking` member; nothing rewrites the body afterwards, so a
	// rejection is attributable to exactly one spelling.
	selected := row.withDialect(mode.dialect).selectedModel(rec.baseURL())
	client, err := anthropicprovider.New(selected, row.key())
	if err != nil {
		t.Fatalf("anthropic.New: %v", scrub(err.Error()))
	}

	ctx := probeContext(t)
	tool := weatherTool()
	sampling := &model.Sampling{MaxTokens: intPtr(6000), Effort: model.EffortMedium}
	first := inference.Request{
		Model:      selected,
		System:     systemPrompt,
		Messages:   content.AgenticMessages{userText("What is the weather in Paris? Think it through, then call the get_weather tool.")},
		Tools:      []inference.Tool{tool},
		ToolChoice: inference.ToolAuto(),
		Override:   sampling,
	}

	resp, err := client.Invoke(ctx, first)
	if err != nil {
		// A model that does not speak this thinking dialect is an UNSUPPORTED
		// combination, not a defect in the replay path this test exists to
		// measure — and Anthropic says so in words. Recording it as a finding
		// keeps the suite's failures meaningful while still putting the
		// server's verbatim message in the log, which is the whole point of
		// running these live. The finding itself is reported by
		// TestLiveAnthropicThinkingDialect, whose entire job it is.
		if body := rec.lastErrorBody(); modelRefusesFeature(body) {
			t.Skipf("thinking mode %q is UNSUPPORTED on %s; server said: %s", mode.name, row.Model, body)
			return
		}
		rejected(t, rec, "signature mint", err)
		return
	}
	blocks := aiBlocks(resp)
	think := firstThinking(blocks)
	use := firstToolUse(blocks)
	if think == nil {
		t.Skipf("signature round trip: turn 1 accepted (200) but returned no thinking block; blocks=%v finish=%q — nothing to replay",
			blockKinds(blocks), resp.FinishReason)
		return
	}
	if think.Signature == "" {
		t.Errorf("signature round trip: turn 1 returned a thinking block with an EMPTY signature; Anthropic always signs a complete thinking block, so the decoder dropped it")
		rec.report(t)
		return
	}
	t.Logf("turn 1 OK: blocks=%v thinking_len=%d signature_len=%d tool_use=%t finish=%q",
		blockKinds(blocks), len(think.Thinking), len(think.Signature), use != nil, resp.FinishReason)
	if use == nil {
		t.Skipf("signature round trip: turn 1 emitted no tool_use, so the tool-in-the-middle shape is unreachable this run (model behaviour)")
		return
	}

	// Replay verbatim. resp.Message is passed through untouched on purpose: a
	// reconstructed message would test a hand-written body, and the whole point
	// is that the bytes the decoder produced are the bytes the encoder replays.
	replay := first
	replay.Messages = append(content.AgenticMessages{}, first.Messages...)
	replay.Messages = append(replay.Messages, resp.Message, toolResult(use.ID, `{"temp_c": 17, "conditions": "overcast"}`))

	second, err := client.Invoke(ctx, replay)
	if err != nil {
		t.Errorf("SIGNATURE ROUND TRIP FAILED: api.anthropic.com rejected a thinking block it had just issued, replayed verbatim")
		rejected(t, rec, "signature replay", err)
		return
	}
	t.Logf("SIGNATURE ROUND TRIP OK: api.anthropic.com accepted the replayed thinking block; blocks=%v finish=%q usage=%v",
		blockKinds(aiBlocks(second)), second.FinishReason, usageSummary(second.Usage))

	// Tamper control. The corrupted block keeps its SignatureFormat label,
	// because the label is what admits it to the wire at all: anthropicapi
	// refuses to replay a signature no dialect claims, so a relabelled — or
	// unlabelled — block is stopped by our own encoder and never reaches
	// Anthropic. That guard is correct and is asserted separately below; here
	// it would defeat the control, which needs the bad bytes to actually be
	// sent so the SERVER is the thing that rejects them.
	t.Run("tamper_control", func(t *testing.T) {
		bad := first
		bad.Messages = append(content.AgenticMessages{}, first.Messages...)
		bad.Messages = append(bad.Messages,
			tamperedAssistant(blocks, func(b *content.ThinkingBlock) *content.ThinkingBlock {
				return content.NewSignedThinkingBlock(
					b.Thinking,
					corruptSignature(b.Signature),
					b.SignatureFormat, // kept: the corruption is the payload, not the label
					b.ProviderState,
					b.ProviderStateFormat,
				)
			}),
			toolResult(use.ID, `{"temp_c": 17}`))

		_, err := client.Invoke(probeContext(t), bad)
		if err == nil {
			t.Errorf("tamper control: api.anthropic.com ACCEPTED a corrupted signature — the passing round trip above therefore does not prove byte-exact replay")
			return
		}
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			t.Errorf("tamper control INCONCLUSIVE: the corrupted block never reached the server, so the round trip above is unvalidated. Local error: %v", scrub(err.Error()))
			return
		}
		body := rec.lastErrorBody()
		t.Logf("tamper control OK: corrupted signature rejected with status=%d code=%q; server said: %s",
			apiErr.Status, apiErr.Code, body)
		if apiErr.Status != 400 {
			t.Errorf("tamper control: expected a 400 for a corrupted signature, got %d", apiErr.Status)
		}
		if !strings.Contains(strings.ToLower(body), "signature") {
			t.Errorf("tamper control: the rejection does not name the signature field, so it may have been rejected for an unrelated reason: %s", body)
		}
	})

	// Label guard. A signature is provider-private state, and this dialect
	// replays only the signatures it minted — so an UNLABELLED one (a block
	// rebuilt without its format tag, or restored from a store that predates
	// the tag) must be refused by us, before any I/O, rather than sent to
	// Anthropic to be rejected there.
	//
	// It belongs in the LIVE suite and not only in a unit test because of what
	// it is asserted against: the exact block api.anthropic.com just issued and
	// this process just decoded, whose signature is real and whose replay is
	// known — one subtest above — to be accepted when it is labelled. That
	// isolates the label as the only difference between acceptance and refusal.
	// A hand-built block could not distinguish "refused for its missing label"
	// from "refused because it was never valid".
	t.Run("unlabelled_signature_guard", func(t *testing.T) {
		unlabelled := first
		unlabelled.Messages = append(content.AgenticMessages{}, first.Messages...)
		unlabelled.Messages = append(unlabelled.Messages,
			tamperedAssistant(blocks, func(b *content.ThinkingBlock) *content.ThinkingBlock {
				// Same signature bytes, same reasoning text; only the dialect
				// label is dropped.
				return content.NewThinkingBlock(b.Thinking, b.Signature, b.ProviderState, b.ProviderStateFormat)
			}),
			toolResult(use.ID, `{"temp_c": 17}`))

		// "Refused before I/O" is a claim about the WIRE, so it is checked on
		// the wire: the recorder must see no new exchange. A local error alone
		// would not distinguish a guard that fired from one that fired after
		// the request had already been sent.
		before := len(rec.snapshot())
		_, err := client.Invoke(probeContext(t), unlabelled)
		if err == nil {
			t.Errorf("unlabelled signature guard: an unlabelled signature was SENT and accepted; provider-private state must not leave this process without a dialect label")
			return
		}
		var apiErr *failure.APIError
		if errors.As(err, &apiErr) {
			t.Errorf("unlabelled signature guard: the block reached the server (status=%d) instead of being refused locally; the guard did not fire", apiErr.Status)
			return
		}
		if after := len(rec.snapshot()); after != before {
			t.Errorf("unlabelled signature guard: %d request(s) crossed the wire before the refusal; the unlabelled signature was transmitted", after-before)
			return
		}
		t.Logf("unlabelled signature guard OK: refused with a typed error and zero requests emitted: %v", scrub(err.Error()))
	})
}

// tamperedAssistant rebuilds the assistant turn, passing every thinking block
// through rewrite and cloning everything else. The tool_use block in particular
// must survive untouched: the request stays a valid tool turn, so the only
// thing the server (or our encoder) can object to is the reasoning block.
func tamperedAssistant(blocks []content.Block, rewrite func(*content.ThinkingBlock) *content.ThinkingBlock) *content.AIMessage {
	out := &content.AIMessage{Message: content.Message{Role: content.RoleAssistant}}
	for _, b := range blocks {
		if original, ok := b.(*content.ThinkingBlock); ok {
			out.Blocks = append(out.Blocks, rewrite(original))
			continue
		}
		out.Blocks = append(out.Blocks, content.CloneBlock(b))
	}
	return out
}

// corruptSignature changes exactly one character of a signature, keeping its
// length and alphabet so the request stays well-formed and only the
// cryptographic check can reject it. A truncated or empty signature would be
// caught by a mere shape check, which would prove nothing about validation.
func corruptSignature(signature string) string {
	if signature == "" {
		return "A"
	}
	runes := []rune(signature)
	middle := len(runes) / 2
	if runes[middle] == 'A' {
		runes[middle] = 'B'
	} else {
		runes[middle] = 'A'
	}
	return string(runes)
}
