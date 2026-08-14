//go:build live

package livetest

import (
	"bytes"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

// This file holds the USE-CASE probes: the request shapes a real agent loop
// produces that the baseline text/stream/single-tool probes in probe.go never
// reach. Each one exists because a schema gate cannot adjudicate it — either
// the construct is legal-but-unsupported (a gateway that parses tool_choice and
// ignores it), or the shape is only reachable after a server has answered once
// (a reasoning signature, a parallel call's ids).

// parallelToolCalls drives two tool calls in ONE assistant turn and returns both
// results.
//
// Three separate things can only fail here. (1) Identity: every call must
// survive decoding with its own id — a dialect that decoded two calls into one
// block, or gave both the same id, would answer one call's request with the
// other's output. (2) Order: the results are replayed in the order the calls
// were decoded, so a codec that reorders either side silently swaps them.
// (3) Grouping: Anthropic delivers parallel results as tool_result blocks and
// the neutral vocabulary has one ToolResultMessage per result, so the
// continuation request carries N consecutive tool-result messages — a shape no
// single-call probe ever produces and no request schema constrains.
//
// The assertion of record is the CONTENT of the final answer, not just a 200.
// Each call gets a distinguishable result, and the model is asked to report
// both; if the two are swapped in the answer, the pairing broke somewhere
// between decode and replay, and that is exactly the cross-attribution defect
// an id-less dialect risks.
func (s scenario) parallelToolCalls(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	tool := weatherTool()
	first := inference.Request{
		Model:  s.selected,
		System: systemPrompt,
		Messages: content.AgenticMessages{userText(
			"Call get_weather twice in this one turn: first for Paris, then for Tokyo. " +
				"When you have both results, reply with exactly: Paris=<paris temp_c>, Tokyo=<tokyo temp_c>")},
		Tools:      []inference.Tool{tool},
		ToolChoice: inference.ToolAuto(),
		Override:   s.sampling(),
	}
	resp, err := s.client.Invoke(ctx, first)
	if err != nil {
		rejected(t, s.rec, "parallel tool request", err)
		return
	}
	uses := allToolUses(aiBlocks(resp))
	if len(uses) < 2 {
		// One call is a MODEL choice — several of these models emit calls
		// sequentially by design. The request was accepted, so nothing about
		// our encoding is in question; say so instead of failing.
		t.Skipf("parallel tool request accepted (200) but model emitted %d tool call(s); blocks=%v finish=%q — model behaviour, not an encoding result",
			len(uses), blockKinds(aiBlocks(resp)), resp.FinishReason)
		return
	}

	seen := make(map[string]struct{}, len(uses))
	for i, use := range uses {
		t.Logf("parallel call %d: name=%q id=%q args=%s", i+1, use.Name, use.ID, string(use.Input))
		if use.ID == "" {
			t.Errorf("parallel call %d decoded with an empty id, so its result cannot be addressed", i+1)
			continue
		}
		if _, duplicate := seen[use.ID]; duplicate {
			t.Errorf("parallel call %d reuses id %q; two calls sharing an id cannot be answered independently", i+1, use.ID)
		}
		seen[use.ID] = struct{}{}
		if trimmed := bytes.TrimSpace(use.Input); len(trimmed) == 0 || trimmed[0] != '{' {
			t.Errorf("parallel call %d decoded arguments that are not a JSON object, so replay double-encodes them: %s", i+1, string(use.Input))
		}
	}

	// Order check. The prompt names Paris first; a codec that preserves the
	// server's order therefore decodes Paris first. A model that ignores the
	// order is possible, so this is reported rather than failed when the two
	// cities are simply not the ones asked for.
	firstCity := strings.ToLower(toolArg(uses[0], "city"))
	secondCity := strings.ToLower(toolArg(uses[1], "city"))
	switch {
	case strings.Contains(firstCity, "paris") && strings.Contains(secondCity, "tokyo"):
		t.Logf("parallel ordering preserved: call 1 = Paris, call 2 = Tokyo")
	case strings.Contains(firstCity, "tokyo") && strings.Contains(secondCity, "paris"):
		t.Logf("parallel ordering NOTE: model emitted Tokyo before Paris (model behaviour); attribution is still checked below")
	default:
		t.Logf("parallel ordering NOTE: cities were %q and %q, not the pair requested", firstCity, secondCity)
	}

	// Distinguishable results, paired by the city each call actually asked for
	// so a model that reversed the order is still answered correctly and the
	// attribution check below stays meaningful.
	temps := map[string]string{"paris": "17", "tokyo": "31"}
	continued := first
	continued.Messages = append(content.AgenticMessages{}, first.Messages...)
	continued.Messages = append(continued.Messages, resp.Message)
	expected := make([]string, 0, len(uses))
	for _, use := range uses {
		city := strings.ToLower(toolArg(use, "city"))
		temp := "17"
		for name, value := range temps {
			if strings.Contains(city, name) {
				temp = value
			}
		}
		expected = append(expected, temp)
		continued.Messages = append(continued.Messages, toolResult(use.ID, `{"temp_c": `+temp+`}`))
	}

	second, err := s.client.Invoke(ctx, continued)
	if err != nil {
		rejected(t, s.rec, "parallel tool result continuation", err)
		return
	}
	answer := allText(aiBlocks(second))
	t.Logf("parallel continuation OK: blocks=%v finish=%q answer=%q", blockKinds(aiBlocks(second)), second.FinishReason, answer)
	if len(aiBlocks(second)) == 0 {
		t.Errorf("parallel continuation: server accepted %d tool results but decoded to zero blocks", len(uses))
		s.rec.report(t)
		return
	}

	// Cross-attribution. Only an exact swap is failed: that is a pairing
	// defect, not a model mistake. Anything else is reported, because a model
	// can legitimately paraphrase or decline to restate the numbers.
	lowered := strings.ToLower(answer)
	paris := strings.Contains(lowered, "paris=17") || strings.Contains(lowered, "paris = 17") || strings.Contains(lowered, "paris: 17")
	tokyo := strings.Contains(lowered, "tokyo=31") || strings.Contains(lowered, "tokyo = 31") || strings.Contains(lowered, "tokyo: 31")
	swappedParis := strings.Contains(lowered, "paris=31") || strings.Contains(lowered, "paris = 31") || strings.Contains(lowered, "paris: 31")
	swappedTokyo := strings.Contains(lowered, "tokyo=17") || strings.Contains(lowered, "tokyo = 17") || strings.Contains(lowered, "tokyo: 17")
	switch {
	case swappedParis && swappedTokyo:
		t.Errorf("parallel attribution BROKEN: results were swapped between calls; expected Paris=17, Tokyo=31 but answer was %q (results sent, in call order: %v)", answer, expected)
	case paris && tokyo:
		t.Logf("parallel attribution verified: each call's own result reached the model")
	default:
		t.Logf("parallel attribution NOTE: answer %q does not restate both values verbatim, so attribution is unproven from the text", answer)
	}
	s.rec.dump(t)
}

// namedToolChoice sends tool_choice in its NAMED form — the wire shape this
// session added and never sent to a server.
//
// It declares two tools and asks a question that points squarely at the tool it
// does NOT force. That asymmetry is what makes the result readable: with one
// declared tool, or with a prompt that matches the forced tool, "the server
// honoured the choice" and "the model would have called it anyway" produce
// identical transcripts. Here only an honoured choice can produce the observed
// call, so a server that parses the field and ignores it is visible.
func (s scenario) namedToolChoice(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	forced := timeTool()
	req := inference.Request{
		Model:      s.selected,
		System:     systemPrompt,
		Messages:   content.AgenticMessages{userText("What is the weather in Paris?")},
		Tools:      []inference.Tool{weatherTool(), forced},
		ToolChoice: inference.ToolNamed(forced.Name),
		Override:   s.sampling(),
	}
	resp, err := s.client.Invoke(ctx, req)
	if err != nil {
		rejected(t, s.rec, "named tool choice", err)
		return
	}
	uses := allToolUses(aiBlocks(resp))
	if len(uses) == 0 {
		finding(t, "named tool choice ACCEPTED BUT NOT HONOURED: the request returned 200 and the model called no tool at all; blocks=%v finish=%q. The encoder emitted the field; this endpoint parses and ignores it.",
			blockKinds(aiBlocks(resp)), resp.FinishReason)
		return
	}
	if uses[0].Name != forced.Name {
		finding(t, "named tool choice ACCEPTED BUT NOT HONOURED: forced %q, model called %q. The encoder emitted the field; this endpoint parses and ignores it.",
			forced.Name, uses[0].Name)
		return
	}
	t.Logf("named tool choice OK: forced %q and got it (id=%q args=%s)", forced.Name, uses[0].ID, string(uses[0].Input))
	s.rec.dump(t)
}

// requiredToolChoice sends tool_choice in its "some tool" form against a prompt
// that invites a plain text answer. A text-only response means the field was
// ignored.
func (s scenario) requiredToolChoice(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	req := inference.Request{
		Model:      s.selected,
		System:     systemPrompt,
		Messages:   content.AgenticMessages{userText("Say hello.")},
		Tools:      []inference.Tool{weatherTool()},
		ToolChoice: inference.ToolRequired(),
		Override:   s.sampling(),
	}
	resp, err := s.client.Invoke(ctx, req)
	if err != nil {
		rejected(t, s.rec, "required tool choice", err)
		return
	}
	uses := allToolUses(aiBlocks(resp))
	if len(uses) == 0 {
		finding(t, "required tool choice ACCEPTED BUT NOT HONOURED: the request returned 200 and the model answered in text; blocks=%v finish=%q. The encoder emitted the field; this endpoint parses and ignores it.",
			blockKinds(aiBlocks(resp)), resp.FinishReason)
		return
	}
	t.Logf("required tool choice OK: model called %q", uses[0].Name)
	s.rec.dump(t)
}

// imageInput sends inline base64 image bytes.
//
// The image is a solid colour with one correct answer, so the check is about
// DELIVERY, not vision: a model that names the colour necessarily received and
// decoded the bytes. A 200 alone would not prove that — several gateways accept
// a multimodal body and quietly drop the media part.
func (s scenario) imageInput(t *testing.T) {
	t.Helper()
	probe, ok := s.variant(t, model.WithImages())
	if !ok {
		t.Skip("image input: this target has no model rebuilder, so AcceptsImages cannot be advertised")
		return
	}
	ctx := probeContext(t)
	resp, err := probe.client.Invoke(ctx, inference.Request{
		Model:    probe.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{imageMessage("What is the dominant colour of this image? Answer with one word.")},
		Override: probe.sampling(),
	})
	if err != nil {
		rejected(t, probe.rec, "image input", err)
		return
	}
	answer := allText(aiBlocks(resp))
	t.Logf("image input accepted: blocks=%v finish=%q answer=%q usage=%v",
		blockKinds(aiBlocks(resp)), resp.FinishReason, answer, usageSummary(resp.Usage))
	if !strings.Contains(strings.ToLower(answer), imageColourName) {
		finding(t, "image input ACCEPTED BUT NOT DELIVERED: the body returned 200 and the model did not name the colour %q (answer %q), so the image part was accepted and dropped rather than read.",
			imageColourName, answer)
		return
	}
	probe.rec.dump(t)
}

// documentInput sends one document block and checks the model can read a nonce
// out of it. Same reasoning as imageInput: the nonce is unguessable, so echoing
// it is proof the document arrived, and a 200 without the nonce means it was
// accepted and dropped.
func (s scenario) documentInput(t *testing.T, doc *content.DocumentBlock) {
	t.Helper()
	ctx := probeContext(t)
	resp, err := s.client.Invoke(ctx, inference.Request{
		Model:    s.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{documentMessage(doc, "What is the verification code in the attached document? Reply with the code only.")},
		Override: s.sampling(),
	})
	if err != nil {
		// A document is carried by a content part that many endpoints simply do
		// not model — the OpenAI-Chat `file` part is in OpenAI's own spec and is
		// absent from several compatible servers' request schemas, so their
		// validator rejects the MESSAGE as unmatched rather than naming the
		// part. That is a coverage gap in the endpoint, not a malformed body,
		// and it is checked only here because the phrase is too generic to
		// trust anywhere else: a schema-union rejection of some other request
		// really would be our defect.
		if body := s.rec.lastErrorBody(); rejectsContentPart(body) {
			finding(t, "document input (%s) UNSUPPORTED: this endpoint's request schema does not model the document content part, so the message matched no message shape. Verdict is about the endpoint's coverage, not our encoding — and it is NOT evidence about the origin API, which was not reachable for this format. Server said: %s",
				doc.MediaType, body)
			return
		}
		rejected(t, s.rec, "document input ("+string(doc.MediaType)+")", err)
		return
	}
	answer := allText(aiBlocks(resp))
	t.Logf("document input (%s) accepted: blocks=%v finish=%q answer=%q",
		doc.MediaType, blockKinds(aiBlocks(resp)), resp.FinishReason, answer)
	if !strings.Contains(strings.ToUpper(answer), documentNonce) {
		finding(t, "document input (%s) ACCEPTED BUT NOT DELIVERED: the body returned 200 and the model did not echo the nonce %q (answer %q), so the document part was accepted and dropped rather than read.",
			doc.MediaType, documentNonce, answer)
		return
	}
	s.rec.dump(t)
}

// structuredOutput sends a schema-constrained Output, optionally alongside
// tools.
//
// The with-tools variant is a separate capability in the neutral model
// (Capabilities.StructuredOutputWithTools) because several dialects treat the
// combination as mutually exclusive, and it is the combination an agent loop
// actually produces: a loop does not stop declaring its tools just because this
// turn wants a typed answer.
//
// Decoding into a typed struct rather than a map is deliberate:
// inference.DecodeOutput is strict, so a response that is JSON but not THIS
// schema fails here rather than passing as a map with unexpected keys.
func (s scenario) structuredOutput(t *testing.T, withTools bool) {
	t.Helper()
	opts := []model.ModelOption{model.WithStructuredOutput()}
	if withTools {
		opts = append(opts, model.WithTools(), model.WithStructuredOutputWithTools())
	}
	probe, ok := s.variant(t, opts...)
	if !ok {
		t.Skip("structured output: this target has no model rebuilder, so the capability cannot be advertised")
		return
	}
	req := inference.Request{
		Model:    probe.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{userText("It is 17 degrees Celsius in Paris. Report that as the structured result.")},
		Output:   weatherOutputSchema(),
		Override: probe.sampling(),
	}
	if withTools {
		req.Tools = []inference.Tool{weatherTool()}
		req.ToolChoice = inference.ToolAuto()
	}
	ctx := probeContext(t)
	resp, err := probe.client.Invoke(ctx, req)
	if err != nil {
		rejected(t, probe.rec, structuredStage(withTools), err)
		return
	}
	// With tools declared, a model may legitimately call one BEFORE producing
	// the structured result, and the finish reason then says tool_use rather
	// than stop. That is the combination working, not failing: the request
	// carrying both an output schema and a tools array was accepted, and the
	// loop is expected to continue. Answering the call and asking again is what
	// a real caller does, and it is the only way to reach the structured result
	// on this path — treating the tool call as the end would report a
	// model-behaviour outcome as an encoder defect.
	if use := firstToolUse(aiBlocks(resp)); use != nil {
		t.Logf("%s: model called %q first (finish=%q); answering it and continuing to the structured turn",
			structuredStage(withTools), use.Name, resp.FinishReason)
		continued := req
		continued.Messages = append(content.AgenticMessages{}, req.Messages...)
		continued.Messages = append(continued.Messages, resp.Message, toolResult(use.ID, `{"temp_c": 17, "conditions": "overcast"}`))
		resp, err = probe.client.Invoke(ctx, continued)
		if err != nil {
			rejected(t, probe.rec, structuredStage(withTools)+" (after tool call)", err)
			return
		}
	}
	var report cityReport
	if err := inference.DecodeOutput(resp, &report); err != nil {
		// The request carrying the schema was ACCEPTED; what came back does
		// not obey it. That is the endpoint declining to enforce a field it
		// took, not an encoding defect — and it is worth recording precisely
		// because a caller cannot tell the two apart from a 200.
		finding(t, "%s ACCEPTED BUT NOT ENFORCED: the schema-carrying request returned 200 and the response does not satisfy it: %v (blocks=%v finish=%q text=%q)",
			structuredStage(withTools), scrub(err.Error()), blockKinds(aiBlocks(resp)), resp.FinishReason, truncate(allText(aiBlocks(resp)), 200))
		return
	}
	t.Logf("%s OK: decoded city=%q temperature_c=%d finish=%q", structuredStage(withTools), report.City, report.TemperatureC, resp.FinishReason)
	probe.rec.dump(t)
}

// rejectsContentPart reports whether an error body describes an endpoint whose
// request schema has no shape for the content part we sent. Deliberately narrow
// in use (documentInput only) and deliberately broad in match: these validators
// report a union failure, not the offending member.
func rejectsContentPart(body string) bool {
	lowered := strings.ToLower(body)
	for _, phrase := range []string{
		"invalid part type",
		"did not match any option in the schema",
		"unsupported content",
		"content[0].type",
		// A Pydantic message union that fell through every arm reports only
		// the LAST arm's complaint, so the visible error is "this user message
		// should have been a tool message" — a message-shape union failure
		// wearing a misleading label, not a statement about roles.
		"function-after[",
		"chatmessagerole",
	} {
		if strings.Contains(lowered, phrase) {
			return true
		}
	}
	return false
}

func structuredStage(withTools bool) string {
	if withTools {
		return "structured output with tools"
	}
	return "structured output"
}

// refusal tries to elicit a REFUSAL as a wire construct — a typed refusal block
// or a refusal finish reason — rather than a model that merely declines in
// prose.
//
// The distinction matters and is easy to fake: any prompt a model dislikes
// produces prose, and calling that "refusal coverage" would assert nothing
// about the decoder. The prompt used is a benign, reliably-declined request
// (verbatim reproduction of copyrighted song lyrics). The outcome is REPORTED,
// never failed: a provider that has no refusal construct is not defective, and
// this probe exists to record which providers actually emit one.
func (s scenario) refusal(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	resp, err := s.client.Invoke(ctx, inference.Request{
		Model:  s.selected,
		System: systemPrompt,
		Messages: content.AgenticMessages{userText(
			"Reproduce the complete lyrics of the song \"Bohemian Rhapsody\" by Queen, word for word.")},
		Override: s.sampling(),
	})
	if err != nil {
		// A rejection is a legitimate outcome here and is still a finding: some
		// gateways refuse at the API layer rather than in the response.
		rejected(t, s.rec, "refusal", err)
		return
	}
	blocks := aiBlocks(resp)
	if block := firstRefusal(blocks); block != nil {
		t.Logf("refusal OK: decoded a typed refusal block (text=%q finish=%q)", block.Text, resp.FinishReason)
		s.rec.dump(t)
		return
	}
	answer := allText(blocks)
	t.Logf("refusal UNPROVEN: no typed refusal block; blocks=%v finish=%q answer=%q — this provider expressed the refusal (if any) as ordinary text, so the refusal decode path remains schema-only here",
		blockKinds(blocks), resp.FinishReason, truncate(answer, 200))
	s.rec.dump(t)
}

func truncate(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}
