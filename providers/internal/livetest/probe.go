//go:build live

package livetest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// probeTimeout bounds one live call. These probes send tiny prompts; anything
// slower than this is an endpoint problem, not a slow generation.
const probeTimeout = 120 * time.Second

// systemPrompt keeps every probe's output short. Live endpoints are paid, and a
// conformance probe cares about the SHAPE of the response, never its length.
const systemPrompt = "You are a terse test fixture. Answer in at most one short sentence."

func probeContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	t.Cleanup(cancel)
	return ctx
}

func intPtr(v int) *int { return &v }

func userText(text string) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{
		Role:   content.RoleUser,
		Blocks: []content.Block{&content.TextBlock{Text: text}},
	}}
}

func toolResult(id, text string) *content.ToolResultMessage {
	blocks := []content.Block{}
	if text != "" {
		blocks = append(blocks, &content.TextBlock{Text: text})
	}
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleTool, Blocks: blocks},
		ToolUseID: id,
	}
}

// weatherTool is the single tool every tool probe exposes. The schema is
// deliberately ordinary — one required string property, additionalProperties
// left unset — because the probe is testing OUR tool envelope (the Responses
// FunctionTool.strict member, Anthropic's input_schema, Chat's function object),
// not the gateway's JSON Schema support.
func weatherTool() inference.Tool {
	return inference.Tool{
		Name:        "get_weather",
		Description: "Return the current weather for one city.",
		Schema: json.RawMessage(`{
			"type": "object",
			"properties": {"city": {"type": "string", "description": "City name"}},
			"required": ["city"]
		}`),
	}
}

// aiBlocks returns the response's blocks, tolerating a nil message so a probe
// reports "no message" rather than panicking.
func aiBlocks(resp *inference.Response) []content.Block {
	if resp == nil || resp.Message == nil {
		return nil
	}
	return resp.Message.Blocks
}

func firstText(blocks []content.Block) *content.TextBlock {
	for _, b := range blocks {
		if text, ok := b.(*content.TextBlock); ok {
			return text
		}
	}
	return nil
}

func firstToolUse(blocks []content.Block) *content.ToolUseBlock {
	for _, b := range blocks {
		if use, ok := b.(*content.ToolUseBlock); ok {
			return use
		}
	}
	return nil
}

// allToolUses returns every tool_use block in order. Order is load-bearing for
// the parallel probe: a dialect that reorders calls relative to the assistant
// turn it decoded would pair each result with the wrong call.
func allToolUses(blocks []content.Block) []*content.ToolUseBlock {
	var uses []*content.ToolUseBlock
	for _, b := range blocks {
		if use, ok := b.(*content.ToolUseBlock); ok {
			uses = append(uses, use)
		}
	}
	return uses
}

func firstRefusal(blocks []content.Block) *content.RefusalBlock {
	for _, b := range blocks {
		if refusal, ok := b.(*content.RefusalBlock); ok {
			return refusal
		}
	}
	return nil
}

// allText concatenates every text block, which is what a probe checking for a
// substring in the model's answer needs; several providers split one answer
// across multiple blocks.
func allText(blocks []content.Block) string {
	var out strings.Builder
	for _, b := range blocks {
		if text, ok := b.(*content.TextBlock); ok {
			out.WriteString(text.Text)
		}
	}
	return out.String()
}

// toolArg reads one string member out of a decoded tool call's arguments. A
// call whose arguments will not parse as an object is reported by the caller;
// here a miss is simply the empty string.
func toolArg(use *content.ToolUseBlock, key string) string {
	if use == nil {
		return ""
	}
	var args map[string]any
	if err := json.Unmarshal(use.Input, &args); err != nil {
		return ""
	}
	value, _ := args[key].(string)
	return value
}

func firstThinking(blocks []content.Block) *content.ThinkingBlock {
	for _, b := range blocks {
		if think, ok := b.(*content.ThinkingBlock); ok {
			return think
		}
	}
	return nil
}

func blockKinds(blocks []content.Block) []string {
	kinds := make([]string, 0, len(blocks))
	for _, b := range blocks {
		switch b.(type) {
		case *content.TextBlock:
			kinds = append(kinds, "text")
		case *content.ThinkingBlock:
			kinds = append(kinds, "thinking")
		case *content.ToolUseBlock:
			kinds = append(kinds, "tool_use")
		case *content.ToolResultBlock:
			kinds = append(kinds, "tool_result")
		case *content.RefusalBlock:
			kinds = append(kinds, "refusal")
		case *content.ImageBlock:
			kinds = append(kinds, "image")
		default:
			kinds = append(kinds, "other")
		}
	}
	return kinds
}

// rejected reports a live rejection with the server's own words. A 4xx here is a
// FINDING, not noise to swallow: the status alone never names the field the
// server objected to, and failure.APIError is sanitized by construction, so the
// recorder's captured body is the only place that information exists.
func rejected(t *testing.T, rec *recorder, stage string, err error) {
	t.Helper()
	// A capacity failure that survived the retry decorator is an ENVIRONMENT
	// result: the server never read our body, so nothing about the request is
	// in question. Skipping keeps it visible in the run summary without
	// asserting a conformance verdict it cannot support.
	if isTransient(err) {
		if status := transientStatus(err); status != 0 {
			t.Skipf("%s: ENVIRONMENT (not a conformance result): endpoint returned %d after retries; the request body was never evaluated",
				stage, status)
			return
		}
		t.Skipf("%s: ENVIRONMENT (not a conformance result): transport failed after retries, so no server ever evaluated the body: %v",
			stage, scrub(err.Error()))
		return
	}
	body := rec.lastErrorBody()
	switch {
	case endpointRefusesUs(body):
		t.Skipf("%s: ENVIRONMENT (not a conformance result): this account/region cannot reach the model, so the body was never evaluated. Server said: %s", stage, body)
		return
	case modelRefusesFeature(body):
		finding(t, "%s: UNSUPPORTED by this model/gateway (our encoder is not implicated). Server said: %s", stage, body)
		return
	}

	var apiErr *failure.APIError
	if errors.As(err, &apiErr) {
		t.Errorf("%s: server rejected our request: status=%d code=%q request_id=%q",
			stage, apiErr.Status, apiErr.Code, apiErr.RequestID)
	} else {
		t.Errorf("%s: call failed: %v", stage, scrub(err.Error()))
	}
	switch {
	case body == "":
		t.Errorf("%s: the endpoint returned no error body, so the rejection cannot be attributed from the response alone", stage)
	case !strings.Contains(strings.ToLower(body), "message") && len(body) < 64:
		t.Errorf("%s: server error body names no reason, so this rejection is UNCLASSIFIED — compare it against the same construct on an origin API before treating it as an encoder defect: %s",
			stage, body)
	default:
		t.Errorf("%s: server error body: %s", stage, body)
	}
	rec.report(t)
}

// Classifying a 4xx is the substance of this suite, not bookkeeping around it.
// A status code cannot distinguish the three outcomes that matter — our encoder
// is wrong / this endpoint will not serve us / this model has no such feature —
// and all three arrive as a 400 or a 404. Only the server's prose separates
// them, which is the reason the recorder retains a body that failure.APIError
// deliberately drops. These two predicates are where that prose is read, and
// they are conservative: anything they do not recognize stays a FAILURE, so an
// unclassified encoder defect can never be quietly downgraded to a note.

// modelRefusesFeature reports whether the server said the MODEL (or the
// upstream behind a router) does not offer the feature the request used.
func modelRefusesFeature(body string) bool {
	lowered := strings.ToLower(body)
	for _, phrase := range []string{
		"is not supported on this model",
		"is not supported for this model",
		"is not supported for model",
		"does not support",
		"does not appear to support",
		"not supported by this model",
		"no endpoints found that support",
		"unsupported model",
		"is incompatible with thinking",
		"invalid part type",
	} {
		if strings.Contains(lowered, phrase) {
			return true
		}
	}
	return false
}

// endpointRefusesUs reports whether the request was turned away for reasons of
// ACCOUNT rather than content: an unentitled region, an exhausted balance, a
// key without access to the model. Nothing about the body was evaluated, so
// like a 503 this is an environment result, not a conformance one.
func endpointRefusesUs(body string) bool {
	lowered := strings.ToLower(body)
	for _, phrase := range []string{
		"regionerror",
		"requires explicit opt in",
		"in balance",
		"add credits",
		"insufficient",
		"quota",
		"not authorized",
		"unauthorized",
		"permission",
	} {
		if strings.Contains(lowered, phrase) {
			return true
		}
	}
	return false
}

// finding records an outcome that is REAL and worth reporting but is not this
// module's defect: a gateway that ignores a field it accepted, a model with no
// such capability. It terminates the subtest as skipped so the run's failures
// stay a list of things to fix, while the message keeps the result visible and
// quotes the server verbatim wherever there is something to quote.
func finding(t *testing.T, format string, args ...any) {
	t.Helper()
	t.Skipf("FINDING: "+format, args...)
}

// scenario is one live probe against one client. Effort drives the dialect's
// reasoning control (Anthropic's thinking block, Chat's reasoning_effort,
// Responses' reasoning object), so a thinking probe is the same scenario with a
// non-none effort.
type scenario struct {
	client   inference.Client
	selected model.Model
	rec      *recorder
	effort   model.Effort
	// maxTokens must be generous enough for a reasoning model to emit a
	// thinking block AND an answer; a too-small budget produces a truncated
	// response that looks like a decode failure but is not one.
	maxTokens int
	// rebuild constructs a client for the SAME endpoint with extra model
	// capabilities enabled. Capabilities are request-shaping input, not just
	// documentation — inference.ValidateRequestFeatures refuses a structured
	// Output on a model that does not advertise StructuredOutput, and refuses
	// image blocks without AcceptsImages — so a probe for a capability the
	// catalogue does not list has to build its own descriptor rather than
	// mutate a shared one. nil means the probes that need it skip.
	rebuild func(t *testing.T, opts ...model.ModelOption) (inference.Client, model.Model)
}

// variant returns this scenario re-pointed at a model descriptor carrying opts.
// It exists so a capability probe cannot accidentally leak its capability into
// the baseline probes that share the scenario value.
func (s scenario) variant(t *testing.T, opts ...model.ModelOption) (scenario, bool) {
	t.Helper()
	if s.rebuild == nil {
		return scenario{}, false
	}
	client, selected := s.rebuild(t, opts...)
	s.client = client
	s.selected = selected
	return s, true
}

func (s scenario) sampling() *model.Sampling {
	return &model.Sampling{MaxTokens: intPtr(s.maxTokens), Effort: s.effort}
}

// textTurn: the baseline. One user message, no tools, expect at least one
// decoded TextBlock and a terminal finish reason.
func (s scenario) textTurn(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	resp, err := s.client.Invoke(ctx, inference.Request{
		Model:    s.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{userText("Reply with exactly the word: ready")},
		Override: s.sampling(),
	})
	if err != nil {
		rejected(t, s.rec, "text turn", err)
		return
	}
	blocks := aiBlocks(resp)
	text := firstText(blocks)
	if text == nil || text.Text == "" {
		t.Errorf("text turn: decoded blocks %v carry no non-empty text", blockKinds(blocks))
		s.rec.report(t)
		return
	}
	t.Logf("text turn OK: blocks=%v finish=%q model=%q usage=%v",
		blockKinds(blocks), resp.FinishReason, resp.Model, usageSummary(resp.Usage))
	if resp.FinishReason == stream.FinishReasonUnknown {
		// Not a failure: the neutral zero value means the provider reported no
		// recognized reason, which several gateways genuinely do.
		t.Logf("text turn: NOTE gateway reported no recognized finish reason")
	}
}

// toolRoundTrip is the probe that matters most. It runs the two-request cycle a
// real agent runs:
//
//  1. request with one tool declared -> the model answers with a tool_use;
//  2. the SAME assistant message is replayed verbatim (thinking blocks, tool ids
//     and provider state included) followed by a tool result.
//
// The second request returning 200 is the real proof that our replay encoding is
// valid, because that body contains every construct a schema gate can only check
// structurally: the tool_use id we echo back, the thinking block with its
// required `thinking`/`signature` members, the Responses reasoning item id, and
// the function_call_output whose `output` member is required.
func (s scenario) toolRoundTrip(t *testing.T, resultText string) {
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
		rejected(t, s.rec, "tool call request", err)
		return
	}
	blocks := aiBlocks(resp)
	use := firstToolUse(blocks)
	if use == nil {
		// The model declining to call a tool is a MODEL behaviour, not an
		// encoder defect: the request was accepted. Say so and stop, rather
		// than failing and implying our tool envelope was rejected.
		t.Skipf("tool call request accepted (200) but model returned no tool_use; blocks=%v finish=%q — model behaviour, not an encoding result",
			blockKinds(blocks), resp.FinishReason)
		return
	}
	t.Logf("tool call OK: blocks=%v tool=%q id=%q args=%s finish=%q",
		blockKinds(blocks), use.Name, use.ID, string(use.Input), resp.FinishReason)
	if use.Name != tool.Name {
		t.Errorf("tool call: model called %q, expected %q", use.Name, tool.Name)
	}
	// ToolUseBlock.Input is the tool's ARGUMENTS OBJECT, not a transport
	// encoding of it. Both other decode paths agree — anthropicapi decodes
	// Anthropic's native object, and openaiapi's own server_decode unwraps the
	// Chat wire's JSON-string form with decodeToolCallArguments — and the Chat
	// ENCODER depends on it: encodeAIMessage quotes b.Input to build
	// function.arguments, so an Input that is already a JSON string is emitted
	// double-encoded on the very next turn. json.Valid alone cannot see this,
	// because a quoted string is perfectly valid JSON; only the leading byte
	// distinguishes the two.
	if trimmed := bytes.TrimSpace(use.Input); len(trimmed) == 0 || trimmed[0] != '{' {
		t.Errorf("tool call: decoded arguments are not a JSON object, so replay will double-encode them: %s", string(use.Input))
	}

	// Replay the assistant turn EXACTLY as decoded. Rebuilding it would test a
	// hand-written message rather than the round trip a real loop performs.
	continued := first
	continued.Messages = append(content.AgenticMessages{}, first.Messages...)
	continued.Messages = append(continued.Messages, resp.Message, toolResult(use.ID, resultText))

	second, err := s.client.Invoke(ctx, continued)
	if err != nil {
		rejected(t, s.rec, "tool result continuation", err)
		return
	}
	t.Logf("tool result continuation OK: blocks=%v finish=%q usage=%v",
		blockKinds(aiBlocks(second)), second.FinishReason, usageSummary(second.Usage))
	if len(aiBlocks(second)) == 0 {
		t.Errorf("tool result continuation: server accepted the replay but decoded to zero blocks")
		s.rec.report(t)
		return
	}

	// Third turn, only when the continuation actually produced reasoning. This
	// is the one request that carries a REPLAYED reasoning item — the Responses
	// reasoning item's required `id`, or an Anthropic thinking block reached by
	// a second hop — and it exists because turn two cannot: a reasoning block
	// has to come back from the server before it can be sent to one.
	echo := firstThinking(aiBlocks(second))
	if echo == nil {
		t.Logf("no reasoning replay probe: continuation returned no thinking block (blocks=%v)", blockKinds(aiBlocks(second)))
		s.rec.dump(t)
		return
	}
	t.Logf("reasoning replay: echoing thinking block signature_present=%t provider_state=%d bytes format=%q",
		echo.Signature != "", len(echo.ProviderState), echo.ProviderStateFormat)

	replay := continued
	replay.Messages = append(content.AgenticMessages{}, continued.Messages...)
	replay.Messages = append(replay.Messages, second.Message, userText("Thanks. Reply with exactly the word: done"))
	third, err := s.client.Invoke(ctx, replay)
	if err != nil {
		rejected(t, s.rec, "reasoning replay", err)
		return
	}
	t.Logf("reasoning replay OK: blocks=%v finish=%q", blockKinds(aiBlocks(third)), third.FinishReason)
	s.rec.dump(t)
}

// streamTurn drives Stream to clean EOF and asserts the neutral chunk stream and
// its terminal StreamResult. A stream that ends without its terminal event is a
// truncation the reader must report as an error, so reaching io.EOF is itself an
// assertion.
func (s scenario) streamTurn(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	reader, err := s.client.Stream(ctx, inference.Request{
		Model:    s.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{userText("Count from one to three, words only.")},
		Override: s.sampling(),
	})
	if err != nil {
		rejected(t, s.rec, "stream open", err)
		return
	}
	defer func() { _ = reader.Close() }()

	var (
		text     string
		thinking int
		tools    int
		chunks   int
	)
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			rejected(t, s.rec, "stream read", err)
			return
		}
		chunks++
		switch c := chunk.(type) {
		case *content.TextChunk:
			text += c.Text
		case *content.ThinkingChunk:
			thinking++
		case *content.ToolUseChunk:
			tools++
		}
	}
	if chunks == 0 {
		t.Errorf("stream: reached clean EOF with zero chunks")
		s.rec.report(t)
		return
	}
	if text == "" {
		t.Errorf("stream: %d chunks but no text accumulated (thinking=%d tool=%d)", chunks, thinking, tools)
		s.rec.report(t)
		return
	}
	result, ok := reader.Result()
	if !ok {
		t.Errorf("stream: clean EOF produced no terminal StreamResult")
		s.rec.report(t)
		return
	}
	t.Logf("stream OK: chunks=%d thinking_deltas=%d text=%q finish=%q model=%q usage=%v",
		chunks, thinking, text, result.FinishReason, result.Model, usageSummary(result.Usage))
}

// thinkingTurn asserts the reasoning path decodes into a ThinkingBlock. A
// gateway that silently drops reasoning is a gateway finding, not a decoder
// defect, so the absence of a thinking block is logged and skipped rather than
// failed — the load-bearing thinking assertion is the REPLAY in toolRoundTrip,
// which is what a server can actually reject.
func (s scenario) thinkingTurn(t *testing.T) {
	t.Helper()
	ctx := probeContext(t)
	resp, err := s.client.Invoke(ctx, inference.Request{
		Model:    s.selected,
		System:   systemPrompt,
		Messages: content.AgenticMessages{userText("Two apples cost 6 euros. What do five cost? Give only the number.")},
		Override: s.sampling(),
	})
	if err != nil {
		rejected(t, s.rec, "thinking turn", err)
		return
	}
	blocks := aiBlocks(resp)
	think := firstThinking(blocks)
	if think == nil {
		t.Skipf("thinking turn accepted (200) but gateway returned no reasoning block; blocks=%v — gateway/model behaviour, not a decode result",
			blockKinds(blocks))
		return
	}
	t.Logf("thinking turn OK: blocks=%v thinking_len=%d signature_present=%t provider_state=%d bytes format=%q",
		blockKinds(blocks), len(think.Thinking), think.Signature != "", len(think.ProviderState), think.ProviderStateFormat)
}

func usageSummary(u *content.Usage) string {
	if u == nil {
		return "none"
	}
	out, err := json.Marshal(struct {
		In        content.TokenCount `json:"in"`
		Out       content.TokenCount `json:"out"`
		CacheRead content.TokenCount `json:"cache_read"`
		Reasoning content.TokenCount `json:"reasoning"`
	}{u.InputTokens, u.OutputTokens, u.CacheReadTokens, u.ReasoningTokens})
	if err != nil {
		return "unserializable"
	}
	return string(out)
}
