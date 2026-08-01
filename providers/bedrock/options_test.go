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
	WithPromptCachePoint(CachePointOptions{Type: "default"})(&config)
	clone := config.clone()

	budget = 200
	config.additionalModelRequestFields[2] = 'X'
	config.additionalResponseFieldPaths[0] = "/changed"
	config.requestMetadata["tenant"] = "changed"
	config.cachePoint.Type = "changed"
	if *clone.reasoning.BudgetTokens != 100 || string(clone.additionalModelRequestFields) != `{"custom":true}` || clone.additionalResponseFieldPaths[0] != "/trace" || clone.requestMetadata["tenant"] != "one" || clone.cachePoint.Type != "default" {
		t.Fatalf("clone shares mutable option state: %#v", clone)
	}
}

func TestConfigApplyConverseOptionsRejectsNonObjectAdditionalFields(t *testing.T) {
	t.Parallel()

	config := config{}
	WithAdditionalModelRequestFields(json.RawMessage(`[]`))(&config)
	_, err := config.applyConverse([]byte(`{"messages":[]}`))
	var optionErr *OptionError
	if !errors.As(err, &optionErr) {
		t.Fatalf("error = %T (%v), want *OptionError", err, err)
	}
}

func TestApplyCachePointFallsBackToFinalMessage(t *testing.T) {
	t.Parallel()

	config := config{}
	WithPromptCachePoint(CachePointOptions{})(&config)
	encoded, err := config.applyConverse([]byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`))
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
