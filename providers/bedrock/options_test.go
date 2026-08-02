package bedrock

import (
	"encoding/json"
	"errors"
	"testing"
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
	_, err := config.applyConverse([]byte(`{"messages":[]}`), false)
	var optionErr *OptionError
	if !errors.As(err, &optionErr) {
		t.Fatalf("error = %T (%v), want *OptionError", err, err)
	}
}

func TestApplyCachePointFallsBackToFinalMessage(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	encoded, err := config.applyConverse([]byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`), false)
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
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
	invoke, err := config.applyConverse([]byte(`{"messages":[]}`), false)
	if err != nil {
		t.Fatalf("invoke applyConverse() error = %v", err)
	}
	stream, err := config.applyConverse([]byte(`{"messages":[]}`), true)
	if err != nil {
		t.Fatalf("stream applyConverse() error = %v", err)
	}
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
	body, err := config.applyConverseCountTokens([]byte(`{"messages":[]}`))
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
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
	body, err := config.applyConverseCountTokens([]byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`))
	if err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
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

func TestApplyCachePointIncludesTTL(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{TTL: CachePointTTL5m})(&config)
	encoded, err := config.applyConverse([]byte(`{"system":[{"text":"prefix"}]}`), false)
	if err != nil {
		t.Fatalf("applyConverse() error = %v", err)
	}
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
