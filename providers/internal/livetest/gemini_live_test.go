//go:build live

package livetest

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	geminiapi "github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"

	geminiprovider "github.com/looprig/llm/providers/gemini"
)

// TestLiveGemini drives providers/gemini against Google's generativelanguage
// endpoint — the origin, not a gateway.
//
// Gemini has never been live-tested end to end in this suite. Declaration
// acceptance was probed (does the server take our functionDeclarations?), but a
// declaration is the easy half: the hard half is the RESULT, because Gemini
// pairs a functionResponse to its call by the Required `name` field while
// FunctionCall.id is Optional and the Developer API routinely omits it. The
// neutral vocabulary addresses a result by id alone, so an id-less pair of
// parallel calls is the exact shape in which one call's output can be delivered
// as the other's — the cross-attribution defect this session fixed with the
// synthetic-ordinal + positional-queue scheme in geminiapi/toolcallid.go.
//
// ROUTING NOTE. providers/gemini fixes its endpoint at construction (New takes
// only a key; the base URL override is test-only and package-private), so
// unlike every other target here its traffic cannot cross the loopback
// recorder. A 4xx therefore arrives as a sanitized failure.APIError with no
// body, which is precisely the blindness the recorder exists to remove. The
// probes recover it out of band: geminiErrorBody re-encodes the identical
// request with the same shared codec the client uses and posts it directly, so
// the server's own words still reach the log.
func TestLiveGemini(t *testing.T) {
	// Two models, because the free quota is 20 requests PER DAY PER MODEL and a
	// full sweep costs more than that. The lite tier carries the matrix; the
	// flash tier is the same code path against a different model, and whichever
	// has quota left on a given day produces results while the other reports
	// ENVIRONMENT rather than a false verdict.
	for _, alias := range []string{"gemini-flash-lite", "gemini-flash"} {
		t.Run(alias, func(t *testing.T) { geminiMatrix(t, alias) })
	}
}

func geminiMatrix(t *testing.T, alias string) {
	t.Helper()
	row := entry(t, alias)
	if row.APIFormat != string(model.APIFormatGemini) {
		t.Skipf("catalogue row %q is api_format %q, not gemini", alias, row.APIFormat)
	}
	build := geminiBuilder(t, row)
	client, selected := build(t)

	probe := scenario{
		client: client, selected: selected, rec: nil,
		effort: model.EffortNone, maxTokens: 1024,
		rebuild: build,
	}
	t.Run("text", probe.textTurn)
	t.Run("stream", probe.streamTurn)
	t.Run("tool_round_trip", func(t *testing.T) { geminiToolRoundTrip(t, probe, row, `{"temp_c": 17}`) })
	t.Run("empty_tool_result", func(t *testing.T) { geminiToolRoundTrip(t, probe, row, "") })
	t.Run("parallel_tool_calls", probe.parallelToolCalls)
	t.Run("named_tool_choice", probe.namedToolChoice)
	t.Run("required_tool_choice", probe.requiredToolChoice)
	t.Run("image_input", probe.imageInput)
	t.Run("document_pdf", func(t *testing.T) { probe.documentInput(t, pdfDocument()) })
	t.Run("document_text", func(t *testing.T) { probe.documentInput(t, textDocument()) })
	t.Run("structured_output", func(t *testing.T) { probe.structuredOutput(t, false) })
	t.Run("structured_output_with_tools", func(t *testing.T) { probe.structuredOutput(t, true) })
	t.Run("refusal", probe.refusal)
	t.Run("id_less_parallel_results", func(t *testing.T) { geminiIDLessParallel(t, probe, row) })

	thinking := probe
	thinking.effort = model.EffortLow
	thinking.maxTokens = 4096
	t.Run("thinking", thinking.thinkingTurn)
}

func geminiBuilder(t *testing.T, row catalogEntry) func(*testing.T, ...model.ModelOption) (inference.Client, model.Model) {
	t.Helper()
	return func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model) {
		t.Helper()
		selected := row.selectedModel(row.BaseURL, opts...)
		client, err := geminiprovider.New(row.key())
		if err != nil {
			t.Fatalf("gemini.New: %v", scrub(err.Error()))
		}
		// The free tier rate-limits aggressively and the endpoint sheds load
		// with 503s; neither is a statement about our request body.
		return withRetries(client), selected
	}
}

// geminiToolRoundTrip is the single-call round trip plus the two assertions
// that are specific to this dialect and invisible everywhere else.
//
// First: the wire must carry NO fabricated id. When Gemini omits
// FunctionCall.id the decoder mints a synthetic per-turn ordinal so the call
// stays addressable in process, and that ordinal is a LOCAL fiction — echoing
// it back would hand Gemini an identifier it never issued. The check is on the
// encoded continuation body, because that is the only place the fiction could
// escape.
//
// Second: the functionResponse must be named. Gemini matches on `name`, so a
// result encoded without one is silently unpaired rather than rejected, and a
// 200 would hide it.
func geminiToolRoundTrip(t *testing.T, s scenario, row catalogEntry, resultText string) {
	t.Helper()
	ctx := probeContext(t)
	tool := weatherTool()
	first := inference.Request{
		Model:      s.selected,
		System:     systemPrompt,
		Messages:   content.AgenticMessages{userText("What is the weather in Paris? Call the get_weather tool.")},
		Tools:      []inference.Tool{tool},
		ToolChoice: inference.ToolAuto(),
		Override:   s.sampling(),
	}
	resp, err := s.client.Invoke(ctx, first)
	if err != nil {
		geminiRejected(t, row, first, "gemini tool call request", err)
		return
	}
	use := firstToolUse(aiBlocks(resp))
	if use == nil {
		t.Skipf("gemini tool request accepted (200) but returned no functionCall; blocks=%v finish=%q — model behaviour",
			blockKinds(aiBlocks(resp)), resp.FinishReason)
		return
	}
	synthetic := strings.HasPrefix(use.ID, "gemini-positional-call-")
	t.Logf("gemini tool call OK: name=%q id=%q synthetic_id=%t args=%s finish=%q",
		use.Name, use.ID, synthetic, string(use.Input), resp.FinishReason)
	if use.ID == "" {
		t.Errorf("gemini tool call decoded with an empty id; the neutral vocabulary addresses a result by id alone, so this call is unanswerable")
	}

	continued := first
	continued.Messages = append(content.AgenticMessages{}, first.Messages...)
	continued.Messages = append(continued.Messages, resp.Message, toolResult(use.ID, resultText))

	assertGeminiWire(t, continued, 1)

	second, err := s.client.Invoke(ctx, continued)
	if err != nil {
		geminiRejected(t, row, continued, "gemini tool result continuation", err)
		return
	}
	t.Logf("gemini tool result continuation OK: blocks=%v finish=%q answer=%q",
		blockKinds(aiBlocks(second)), second.FinishReason, truncate(allText(aiBlocks(second)), 160))
	if len(aiBlocks(second)) == 0 {
		t.Errorf("gemini tool result continuation: accepted but decoded to zero blocks")
	}
}

// geminiIDLessParallel is the probe the cross-attribution fix exists for.
//
// It cannot be reached by asking the model nicely. Gemini's FunctionCall.id is
// Optional, and whether a given model version populates it is the server's
// choice, not ours — the live model currently DOES emit ids, so waiting for an
// id-less turn would leave the fix untested forever. The thread is therefore
// constructed: two calls with no id at all and two results with no ToolUseID,
// which is exactly what reaches the encoder from a transcript recorded before
// ids existed, or one carried over from a dialect that never had them.
//
// Two DIFFERENT tools, not the same tool twice. Gemini pairs a result to its
// call on the Required `name`, so two same-named calls are indistinguishable on
// the wire by construction — a limitation of the protocol, not of this codec.
// With distinct names the pairing is observable: if the positional queue
// mis-assigns, get_weather's output goes out labelled get_local_time, and the
// model answers with the time when asked for the temperature.
func geminiIDLessParallel(t *testing.T, s scenario, row catalogEntry) {
	t.Helper()
	weather := weatherTool()
	clock := timeTool()

	// Turn one is REAL. The assistant turn is then rebuilt with its ids removed
	// and everything else — names, arguments, and crucially each call's
	// ProviderState — left intact.
	//
	// Stripping a real turn rather than fabricating one is not a detail. Gemini
	// 3.x rejects a functionCall part that carries no `thoughtSignature`
	// ("Function call is missing a thought_signature in functionCall parts",
	// HTTP 400), and that signature lives in ToolUseBlock.ProviderState, which
	// only the server can mint. A hand-built thread is therefore rejected
	// before the pairing logic is ever reached, and the probe would measure the
	// signature requirement instead of the thing it is for.
	opening := inference.Request{
		Model:  s.selected,
		System: systemPrompt,
		Messages: content.AgenticMessages{userText(
			"Call get_weather and get_local_time, both for Paris, in this one turn. " +
				"When you have both results reply with exactly: temp=<temp_c>, time=<local_time>")},
		Tools:      []inference.Tool{weather, clock},
		ToolChoice: inference.ToolAuto(),
		Override:   s.sampling(),
	}
	opened, err := s.client.Invoke(probeContext(t), opening)
	if err != nil {
		geminiRejected(t, row, opening, "id-less parallel setup", err)
		return
	}
	uses := allToolUses(aiBlocks(opened))
	if len(uses) < 2 {
		t.Skipf("id-less parallel: model emitted %d call(s) in the opening turn, so there is no parallel turn to strip; blocks=%v — model behaviour",
			len(uses), blockKinds(aiBlocks(opened)))
		return
	}

	stripped := &content.AIMessage{Message: content.Message{Role: content.RoleAssistant}}
	for _, b := range aiBlocks(opened) {
		use, ok := b.(*content.ToolUseBlock)
		if !ok {
			stripped.Blocks = append(stripped.Blocks, content.CloneBlock(b))
			continue
		}
		stripped.Blocks = append(stripped.Blocks, content.NewToolUseBlock(
			"", // the id, and only the id, is removed
			use.Name, use.Input, use.ProviderState, use.ProviderStateFormat,
		))
	}

	// Results in call order, each carrying an unmistakable value and NO id, so
	// only the positional queue can pair them.
	payloads := map[string]string{weather.Name: `{"temp_c": 17}`, clock.Name: `{"local_time": "09:45"}`}
	req := opening
	req.Messages = append(content.AgenticMessages{}, opening.Messages...)
	req.Messages = append(req.Messages, stripped)
	for _, use := range uses {
		payload, ok := payloads[use.Name]
		if !ok {
			payload = `{"result": "unknown"}`
		}
		req.Messages = append(req.Messages, toolResult("", payload))
	}

	// Wire check first: the pairing is decided at encode time, so a mis-pairing
	// is visible in the body before the server ever sees it, and the live call
	// then confirms the server agrees.
	body, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("gemini encode: %v", scrub(err.Error()))
	}
	// Expected order is the order the model actually called in, not a guess:
	// the positional queue's contract is "results answer calls head-first in
	// wire order", so the check has to be against the observed calls.
	firstAt := bytes.Index(body, []byte(`"functionResponse":{"name":"`+uses[0].Name+`"`))
	secondAt := bytes.LastIndex(body, []byte(`"functionResponse":{"name":"`+uses[1].Name+`"`))
	switch {
	case firstAt < 0 || secondAt < 0:
		t.Errorf("id-less pairing BROKEN: the wire does not carry one named functionResponse per call (wanted %q then %q); body was %s",
			uses[0].Name, uses[1].Name, scrubBytes(body))
		return
	case firstAt > secondAt:
		t.Errorf("id-less pairing BROKEN: results were paired in reverse; %q must precede %q because the calls were emitted in that order. Body: %s",
			uses[0].Name, uses[1].Name, scrubBytes(body))
		return
	}
	if bytes.Contains(body, []byte(`"functionCall":{"id"`)) {
		t.Errorf("id-less pairing: the wire carries an `id` on a functionCall that never had one; Gemini never issued it. Body: %s", scrubBytes(body))
	}
	// Each result must sit under its OWN call's name: the temperature under
	// get_weather, the clock time under get_local_time. A queue that swapped
	// them would put the clock's marker in the weather call's response. The
	// check is on a MARKER rather than the payload text because the payload is
	// JSON-escaped inside the wire's `response` member.
	markers := map[string]string{weather.Name: "temp_c", clock.Name: "local_time"}
	segment := body[firstAt:secondAt]
	if mine, ok := markers[uses[0].Name]; ok {
		if !bytes.Contains(segment, []byte(mine)) {
			t.Errorf("id-less pairing BROKEN: the first call (%q) got a functionResponse that does not carry its own result marker %q. Body: %s",
				uses[0].Name, mine, scrubBytes(body))
		}
		if theirs, ok := markers[uses[1].Name]; ok && bytes.Contains(segment, []byte(theirs)) {
			t.Errorf("id-less pairing BROKEN: the first call (%q) got the OTHER call's result (marker %q). Body: %s",
				uses[0].Name, theirs, scrubBytes(body))
		}
	}
	t.Logf("id-less wire pairing OK: %q then %q, paired positionally onto their own names, with no fabricated id and each call's own thoughtSignature preserved",
		uses[0].Name, uses[1].Name)

	resp, err := s.client.Invoke(probeContext(t), req)
	if err != nil {
		geminiRejected(t, row, req, "id-less parallel results", err)
		return
	}
	answer := allText(aiBlocks(resp))
	t.Logf("id-less parallel accepted: blocks=%v finish=%q answer=%q", blockKinds(aiBlocks(resp)), resp.FinishReason, answer)
	switch {
	case strings.Contains(answer, "17") && strings.Contains(answer, "09:45"):
		t.Logf("id-less attribution verified end to end: the server read each result under its own call's name")
	case strings.Contains(answer, "09:45") && strings.Contains(answer, "temp=09:45"):
		t.Errorf("id-less attribution BROKEN: the model read the clock result as the temperature; answer was %q", answer)
	default:
		t.Logf("id-less attribution NOTE: answer %q does not restate both values, so end-to-end attribution is unproven from the text (the wire pairing above is still verified)", answer)
	}
}

// TestLiveGeminiReconstructedToolThread records a constraint this suite
// discovered and no schema states: Gemini 3.x REJECTS a functionCall part that
// carries no `thoughtSignature`.
//
// It matters well beyond this probe. The signature is minted by the server and
// lives in ToolUseBlock.ProviderState, so it exists only for a turn this
// process decoded from Gemini itself. Every other way of arriving at a tool
// thread — a transcript compacted and rebuilt, a session that switched models
// and is replaying a thread another dialect produced, a synthesized thread —
// has no signature to carry, and is therefore not replayable to Gemini 3.x at
// all. That is exactly the population the codec's positional id-less queue was
// written to serve, so the queue is correct (see
// TestLiveGemini/.../id_less_parallel_results, which verifies it) and
// simultaneously unreachable for its original motivating case on this model.
//
// Recorded as a REPORT, not a failure: nothing here is our encoder's error, and
// the useful output is the server's own words.
func TestLiveGeminiReconstructedToolThread(t *testing.T) {
	const alias = "gemini-flash-lite"
	row := entry(t, alias)
	build := geminiBuilder(t, row)
	client, selected := build(t)

	weather := weatherTool()
	req := inference.Request{
		Model:  selected,
		System: systemPrompt,
		Messages: content.AgenticMessages{
			userText("What is the weather in Paris?"),
			// A rebuilt assistant turn: correct in every neutral respect and
			// carrying no provider-private state, because it never came from
			// this provider.
			&content.AIMessage{Message: content.Message{
				Role:   content.RoleAssistant,
				Blocks: []content.Block{content.NewToolUseBlock("call_1", weather.Name, []byte(`{"city":"Paris"}`), nil, "")},
			}},
			toolResult("call_1", `{"temp_c": 17}`),
		},
		Tools:      []inference.Tool{weather},
		ToolChoice: inference.ToolAuto(),
		Override:   &model.Sampling{MaxTokens: intPtr(256)},
	}

	_, err := client.Invoke(probeContext(t), req)
	if err == nil {
		t.Logf("reconstructed tool thread ACCEPTED by %s: a signature-less functionCall is replayable on this model", row.Model)
		return
	}
	if isTransient(err) {
		t.Skipf("reconstructed tool thread: ENVIRONMENT, endpoint returned %d", transientStatus(err))
		return
	}
	body := geminiErrorBody(t, row, req)
	if strings.Contains(body, "thought_signature") {
		t.Logf("FINDING (provider constraint, not an encoder defect): %s REJECTS a reconstructed tool thread. Any thread not decoded from Gemini itself — compacted, cross-dialect, or synthesized — cannot be replayed. Server said: %s",
			row.Model, body)
		return
	}
	t.Errorf("reconstructed tool thread rejected for an unexpected reason: %v; server said: %s", scrub(err.Error()), body)
}

// assertGeminiWire encodes the request with the same codec the client uses and
// checks the two dialect invariants that no response can reveal. It is a local
// assertion deliberately placed in a LIVE probe: it inspects the exact body
// that is about to be sent on the very next line, so a pass here and a 200
// below are evidence about the same bytes.
func assertGeminiWire(t *testing.T, req inference.Request, wantResponses int) {
	t.Helper()
	body, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("gemini encode: %v", scrub(err.Error()))
	}
	if bytes.Contains(body, []byte("gemini-positional-call-")) {
		t.Errorf("gemini wire carries a SYNTHETIC tool-call id; that identifier is a local fiction Gemini never issued and must be stripped before sending: %s", scrubBytes(body))
	}
	if got := bytes.Count(body, []byte(`"functionResponse"`)); got != wantResponses {
		t.Errorf("gemini wire carries %d functionResponse parts, expected %d: %s", got, wantResponses, scrubBytes(body))
	}
	// Gemini pairs on the Required `name`, so an unnamed response is delivered
	// to nothing at all — and the server accepts it, which is what makes the
	// omission dangerous rather than merely wrong.
	if bytes.Contains(body, []byte(`"functionResponse":{"name":""`)) || bytes.Contains(body, []byte(`"functionResponse":{"response"`)) {
		t.Errorf("gemini wire carries a functionResponse with no name; Gemini matches a result to its call by name, so this result would be silently unpaired: %s", scrubBytes(body))
	}
}

// geminiRejected reports a Gemini failure and recovers the server's message.
// failure.APIError is sanitized by construction and this client cannot be
// routed through the loopback recorder, so without this the log would carry a
// status code and nothing that names the offending field.
func geminiRejected(t *testing.T, row catalogEntry, req inference.Request, stage string, err error) {
	t.Helper()
	rejected(t, nil, stage, err)
	if body := geminiErrorBody(t, row, req); body != "" {
		t.Errorf("%s: server error body (recovered by re-sending the identical encoded body): %s", stage, body)
	}
}

// geminiErrorBody re-encodes req with the shared codec and posts it directly,
// returning the scrubbed error body. It sends the SAME bytes the client sent,
// so the recovered message describes the same request; on a 2xx (a transient
// failure that has since cleared) it returns nothing rather than pretending to
// have reproduced the fault.
func geminiErrorBody(t *testing.T, row catalogEntry, req inference.Request) string {
	t.Helper()
	body, err := geminiapi.EncodeRequest(req)
	if err != nil {
		return "<could not re-encode request: " + scrub(err.Error()) + ">"
	}
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	url := strings.TrimRight(row.BaseURL, "/") + "/models/" + req.Model.Name + ":generateContent"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "<could not build diagnostic request: " + scrub(err.Error()) + ">"
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", row.APIKey)
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "<diagnostic request failed: " + scrub(err.Error()) + ">"
	}
	defer func() { _ = resp.Body.Close() }()
	raw, readErr := io.ReadAll(io.LimitReader(resp.Body, maxCapturedBody))
	if readErr != nil {
		return "<diagnostic response unreadable: " + scrub(readErr.Error()) + ">"
	}
	if resp.StatusCode/100 == 2 {
		return "<re-sent request returned " + http.StatusText(resp.StatusCode) + "; the original failure did not reproduce>"
	}
	return scrubBytes(raw)
}
