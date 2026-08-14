package bedrock

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/bedrockconverse"
	model "github.com/looprig/inference/model"
)

func TestConfigCloneCopiesMutableOptions(t *testing.T) {
	t.Parallel()

	budget := 100
	config := config{}
	WithReasoning(ReasoningOptions{Type: "enabled", BudgetTokens: &budget})(&config)
	WithAdditionalModelRequestFields(json.RawMessage(`{"custom":true}`))(&config)
	WithAdditionalModelResponseFieldPaths("/trace")(&config)
	WithRequestMetadata(map[string]string{"tenant": "one"})(&config)
	WithPromptCachePoint(CachePointOptions{Type: "default", TTL: CachePointTTL1h})(&config)
	clone := config.clone()

	budget = 200
	config.additionalModelRequestFields[2] = 'X'
	config.additionalResponseFieldPaths[0] = "/changed"
	config.requestMetadata["tenant"] = "changed"
	config.cachePoint.Type = "changed"
	if *clone.reasoning.BudgetTokens != 100 || string(clone.additionalModelRequestFields) != `{"custom":true}` || clone.additionalResponseFieldPaths[0] != "/trace" || clone.requestMetadata["tenant"] != "one" || clone.cachePoint.Type != "default" || clone.cachePoint.TTL != CachePointTTL1h {
		t.Fatalf("clone shares mutable option state: %#v", clone)
	}
}

func TestConfigApplyConverseOptionsRejectsNonObjectAdditionalFields(t *testing.T) {
	t.Parallel()

	config := config{}
	WithAdditionalModelRequestFields(json.RawMessage(`[]`))(&config)
	_, err := config.applyConverse([]byte(`{"messages":[]}`), false, nil)
	var optionErr *OptionError
	if !errors.As(err, &optionErr) {
		t.Fatalf("error = %T (%v), want *OptionError", err, err)
	}
}

func TestApplyCachePointFallsBackToFinalMessage(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	encoded, err := config.applyConverse([]byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`), false, &projectedCacheBoundary{messageIndex: 0, contentIndex: 1})
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
	gateConverseBody(t, encoded)
	var body map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages JSON = %v", err)
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(messages[0]["content"], &blocks); err != nil {
		t.Fatalf("content JSON = %v", err)
	}
	if _, ok := blocks[1]["cachePoint"]; !ok {
		t.Fatalf("final content block = %#v, want cachePoint", blocks[1])
	}
}

func TestApplyConverseGuardrailStreamProcessingModeOnlyStreams(t *testing.T) {
	t.Parallel()

	config := config{}
	WithGuardrail(GuardrailOptions{Identifier: "guard", Version: "1", StreamProcessingMode: "async"})(&config)
	invoke, err := config.applyConverse([]byte(`{"messages":[]}`), false, nil)
	if err != nil {
		t.Fatalf("invoke applyConverse() error = %v", err)
	}
	stream, err := config.applyConverse([]byte(`{"messages":[]}`), true, nil)
	if err != nil {
		t.Fatalf("stream applyConverse() error = %v", err)
	}
	// guardrailConfig carries the two patterned Smithy strings the gate can
	// actually check: GuardrailIdentifier and GuardrailVersion.
	gateConverseBody(t, invoke)
	gateConverseBody(t, stream)
	var invokeBody, streamBody map[string]json.RawMessage
	if err := json.Unmarshal(invoke, &invokeBody); err != nil {
		t.Fatalf("invoke body = %v", err)
	}
	if err := json.Unmarshal(stream, &streamBody); err != nil {
		t.Fatalf("stream body = %v", err)
	}
	var invokeGuardrail, streamGuardrail map[string]json.RawMessage
	if err := json.Unmarshal(invokeBody["guardrailConfig"], &invokeGuardrail); err != nil {
		t.Fatalf("invoke guardrail = %v", err)
	}
	if err := json.Unmarshal(streamBody["guardrailConfig"], &streamGuardrail); err != nil {
		t.Fatalf("stream guardrail = %v", err)
	}
	if _, ok := invokeGuardrail["streamProcessingMode"]; ok {
		t.Fatal("invoke guardrail unexpectedly contains streamProcessingMode")
	}
	if got := string(streamGuardrail["streamProcessingMode"]); got != `"async"` {
		t.Errorf("streamProcessingMode = %s, want async", got)
	}
}

func TestApplyCountTokensIncludesOnlyAdditionalModelRequestFields(t *testing.T) {
	t.Parallel()

	budget := 256
	config := config{}
	WithReasoning(ReasoningOptions{BudgetTokens: &budget})(&config)
	WithAdditionalModelRequestFields(json.RawMessage(`{"temperature_top_k":50}`))(&config)
	body, err := config.applyConverseCountTokens([]byte(`{"messages":[]}`), nil)
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
	gateConverseBody(t, body)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("body = %v", err)
	}
	var additional map[string]json.RawMessage
	if err := json.Unmarshal(fields["additionalModelRequestFields"], &additional); err != nil {
		t.Fatalf("additional fields = %v", err)
	}
	if string(additional["temperature_top_k"]) != "50" || additional["thinking"] == nil {
		t.Fatalf("additional fields = %#v, want custom + thinking", additional)
	}
}

func TestApplyCountTokensIncludesPromptCachePoint(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{TTL: CachePointTTL1h})(&config)
	body, err := config.applyConverseCountTokens([]byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`), &projectedCacheBoundary{messageIndex: 0, contentIndex: 1})
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
	gateConverseBody(t, body)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("body = %v", err)
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(fields["messages"], &messages); err != nil {
		t.Fatalf("messages = %v", err)
	}
	var contentBlocks []map[string]json.RawMessage
	if err := json.Unmarshal(messages[0]["content"], &contentBlocks); err != nil {
		t.Fatalf("content = %v", err)
	}
	var cachePoint map[string]json.RawMessage
	if err := json.Unmarshal(contentBlocks[1]["cachePoint"], &cachePoint); err != nil {
		t.Fatalf("cachePoint = %v", err)
	}
	if got := string(cachePoint["ttl"]); got != `"1h"` {
		t.Errorf("cachePoint.ttl = %s, want 1h", got)
	}
}

// TestApplyCachePointNeverSynthesisesNullMessages pins a defect the request
// gate surfaced. applyCachePoint decoded fields["messages"] into a slice and
// wrote it back unconditionally, so a body carrying no "messages" key at all
// came out carrying "messages":null — and Converse models messages as a list,
// which null is not. The rewrite must now leave an absent key absent and
// normalise a present-but-null one to [].
func TestApplyCachePointNeverSynthesisesNullMessages(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		body        string
		wantPresent bool
		want        string
	}{
		{name: "absent messages stays absent", body: `{"system":[{"text":"prefix"}]}`},
		{name: "null messages normalises to an array", body: `{"system":[{"text":"prefix"}],"messages":null}`, wantPresent: true, want: "[]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			config := config{}
			WithPromptCachePoint(CachePointOptions{})(&config)
			encoded, err := config.applyConverse([]byte(tt.body), false, nil)
			if err != nil {
				t.Fatalf("applyConverse() error = %v", err)
			}
			gateConverseBody(t, encoded)
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &fields); err != nil {
				t.Fatalf("body JSON = %v", err)
			}
			raw, present := fields["messages"]
			if present != tt.wantPresent {
				t.Fatalf("messages present = %v (%s), want %v", present, raw, tt.wantPresent)
			}
			if tt.wantPresent && string(raw) != tt.want {
				t.Errorf("messages = %s, want %s", raw, tt.want)
			}
		})
	}
}

func TestApplyCachePointIncludesTTL(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{TTL: CachePointTTL5m})(&config)
	encoded, err := config.applyConverse([]byte(`{"system":[{"text":"prefix"}]}`), false, nil)
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
	gateConverseBody(t, encoded)
	var body map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("body = %v", err)
	}
	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system = %v", err)
	}
	var cachePoint map[string]json.RawMessage
	if err := json.Unmarshal(system[1]["cachePoint"], &cachePoint); err != nil {
		t.Fatalf("cachePoint = %v", err)
	}
	if got := string(cachePoint["ttl"]); got != `"5m"` {
		t.Errorf("cachePoint.ttl = %s, want 5m", got)
	}
}

func TestApplyCachePointSupportsStringSystem(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	encoded, err := config.applyConverse([]byte(`{"system":"prefix"}`), false, nil)
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
	gateConverseBody(t, encoded)
	var body map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON = %v", err)
	}
	if len(system) != 2 || string(system[0]["text"]) != `"prefix"` || len(system[1]["cachePoint"]) == 0 {
		t.Fatalf("system = %#v, want text then cachePoint", system)
	}
}

func TestApplyCachePointUsesLastCommittedMessage(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{Type: "default", TTL: CachePointTTL1h})(&config)
	body := []byte(`{"system":[{"text":"stable system"}],"messages":[{"role":"user","content":[{"text":"stable"}]},{"role":"user","content":[{"text":"runtime"}]}]}`)
	encoded, err := config.applyConverse(body, false, &projectedCacheBoundary{messageIndex: 0, contentIndex: 1})
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
	assertCachePointBoundary(t, encoded, 0)
}

func TestApplyCountTokensCachePointUsesLastCommittedMessage(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	body := []byte(`{"system":[{"text":"stable system"}],"messages":[{"role":"user","content":[{"text":"stable"}]},{"role":"user","content":[{"text":"runtime"}]}]}`)
	encoded, err := config.applyConverseCountTokens(body, &projectedCacheBoundary{messageIndex: 0, contentIndex: 1})
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
	assertCachePointBoundary(t, encoded, 0)
}

func TestApplyCachePointRejectsInvalidOptions(t *testing.T) {
	t.Parallel()

	for _, options := range []CachePointOptions{{Type: "ephemeral"}, {Type: "default", TTL: "30m"}} {
		config := config{}
		WithPromptCachePoint(options)(&config)
		_, err := config.applyConverse([]byte(`{"system":[{"text":"stable"}]}`), false, nil)
		var optionErr *OptionError
		if !errors.As(err, &optionErr) {
			t.Errorf("applyConverse(%+v) error = %T (%v), want *OptionError", options, err, err)
		}
	}
}

func TestCommittedConverseBoundaryRejectsTransientSystemContext(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:             model.CustomModel("bedrock", model.APIFormatBedrockConverse, "https://example.test", "model"),
		TransientMessages: 1,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable"}}}},
			&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
		},
	}
	body, encodeErr := bedrockconverse.EncodeRequest(req)
	if encodeErr != nil {
		t.Fatalf("EncodeRequest() error = %v", encodeErr)
	}
	gateConverseBody(t, body)
	_, err := cacheBoundaryForRequest(req, body)
	var optionErr *OptionError
	if !errors.As(err, &optionErr) {
		t.Fatalf("cacheBoundaryForRequest() error = %T (%v), want *OptionError", err, err)
	}
}

func TestProjectedCacheBoundarySplitsMergedCommittedAndTransientUserContent(t *testing.T) {
	t.Parallel()

	req := bedrockCacheTestRequest(
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable"}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
	)
	for _, countTokens := range []bool{false, true} {
		encoded := encodeProjectedCacheRequest(t, req, countTokens)
		messages := decodeProjectedMessages(t, encoded)
		assertTopLevelContentKinds(t, messages[0], []string{"text:stable", "cachePoint", "text:runtime"})
	}
}

func TestProjectedCacheBoundaryPrecedesVolatileToolResultTurn(t *testing.T) {
	t.Parallel()

	req := bedrockCacheTestRequest(
		&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{}`)}}}},
		&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "stable result"}}}, ToolUseID: "call-1"},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
	)
	for _, countTokens := range []bool{false, true} {
		encoded := encodeProjectedCacheRequest(t, req, countTokens)
		messages := decodeProjectedMessages(t, encoded)
		assertTopLevelContentKinds(t, messages[0], []string{"toolUse", "cachePoint"})
		assertTopLevelContentKinds(t, messages[1], []string{"toolResult"})
		if !bytesContain(messages[1]["content"], []byte("runtime")) {
			t.Fatalf("tool-result turn dropped runtime content: %s", messages[1]["content"])
		}
	}
}

func TestProjectedCacheBoundaryHandlesCompactedAdjacentCommittedUsers(t *testing.T) {
	t.Parallel()

	req := bedrockCacheTestRequest(
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable one"}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable two"}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
	)
	for _, countTokens := range []bool{false, true} {
		encoded := encodeProjectedCacheRequest(t, req, countTokens)
		messages := decodeProjectedMessages(t, encoded)
		assertTopLevelContentKinds(t, messages[0], []string{"text:stable one", "text:stable two", "cachePoint", "text:runtime"})
	}
}

func bedrockCacheTestRequest(messages ...content.Conversation) inference.Request {
	return inference.Request{
		Model:             model.CustomModel("bedrock", model.APIFormatBedrockConverse, "https://example.test", "model"),
		Messages:          messages,
		TransientMessages: 1,
	}
}

func encodeProjectedCacheRequest(t *testing.T, req inference.Request, countTokens bool) []byte {
	t.Helper()
	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	if !countTokens {
		encoded, err := (&Client{options: config}).encodeConverse(req, false)
		if err != nil {
			t.Fatalf("encodeConverse() error = %v", err)
		}
		return encoded
	}
	body, err := bedrockconverse.EncodeCountTokensInput(req)
	if err != nil {
		t.Fatalf("EncodeCountTokensInput() error = %v", err)
	}
	boundary, err := cacheBoundaryForRequest(req, body)
	if err != nil {
		t.Fatalf("cacheBoundaryForRequest() error = %v", err)
	}
	encoded, err := config.applyConverseCountTokens(body, boundary)
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
	return encoded
}

func decodeProjectedMessages(t *testing.T, encoded []byte) []map[string]json.RawMessage {
	t.Helper()
	gateConverseBody(t, encoded)
	var body map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages JSON = %v", err)
	}
	return messages
}

func assertTopLevelContentKinds(t *testing.T, message map[string]json.RawMessage, want []string) {
	t.Helper()
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(message["content"], &blocks); err != nil {
		t.Fatalf("content JSON = %v", err)
	}
	got := make([]string, 0, len(blocks))
	for _, block := range blocks {
		switch {
		case block["cachePoint"] != nil:
			got = append(got, "cachePoint")
		case block["toolUse"] != nil:
			got = append(got, "toolUse")
		case block["toolResult"] != nil:
			got = append(got, "toolResult")
		default:
			var text string
			_ = json.Unmarshal(block["text"], &text)
			got = append(got, "text:"+text)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("content kinds = %v, want %v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("content kinds = %v, want %v", got, want)
		}
	}
}

func bytesContain(haystack, needle []byte) bool {
	for index := 0; index+len(needle) <= len(haystack); index++ {
		if string(haystack[index:index+len(needle)]) == string(needle) {
			return true
		}
	}
	return false
}

func assertCachePointBoundary(t *testing.T, encoded []byte, stableMessage int) {
	t.Helper()
	gateConverseBody(t, encoded)
	var body map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON = %v", err)
	}
	if len(system) != 1 {
		t.Fatalf("system blocks = %d, want no cachePoint duplicate", len(system))
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages JSON = %v", err)
	}
	for index, message := range messages {
		var blocks []map[string]json.RawMessage
		if err := json.Unmarshal(message["content"], &blocks); err != nil {
			t.Fatalf("message %d content = %v", index, err)
		}
		wantBlocks := 1
		if index == stableMessage {
			wantBlocks = 2
		}
		if len(blocks) != wantBlocks {
			t.Errorf("message %d blocks = %d, want %d", index, len(blocks), wantBlocks)
		}
	}
}
