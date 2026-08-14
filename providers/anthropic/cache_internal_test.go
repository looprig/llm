package anthropic

import (
	"encoding/json"
	"errors"
	"io"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	model "github.com/looprig/inference/model"
)

func TestExplicitCacheControlUsesLastCommittedMessage(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel("anthropic", model.APIFormatAnthropic, "https://example.test", "claude-test", model.WithPromptCaching())
	req := inference.Request{
		Model:             selected,
		System:            "stable system",
		TransientMessages: 1,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable"}}}},
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
		},
	}
	encoded, err := (requestCodec{config: config{cacheControl: &CacheControlOptions{Type: "ephemeral", TTL: "1h"}}}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	bodyBytes, err := io.ReadAll(encoded.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	// A cache_control breakpoint is written into an already-encoded body, so the
	// gate runs on the rewritten bytes: an explicit boundary that lands on a
	// block which may not carry cache_control is a 400, not a style problem.
	gateRequestBody(t, bodyBytes)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		t.Fatalf("body JSON: %v", err)
	}

	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON: %v", err)
	}
	if _, exists := system[0]["cache_control"]; exists {
		t.Fatal("automatic system cache_control remains alongside explicit stable boundary")
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages JSON: %v", err)
	}
	for index, message := range messages {
		var blocks []map[string]json.RawMessage
		if err := json.Unmarshal(message["content"], &blocks); err != nil {
			t.Fatalf("message %d content: %v", index, err)
		}
		cache, exists := blocks[0]["cache_control"]
		if index == 0 {
			if !exists || string(cache) != `{"type":"ephemeral","ttl":"1h"}` {
				t.Fatalf("stable cache_control = %s, want explicit ephemeral/1h", cache)
			}
		} else if exists {
			t.Fatalf("transient message carries cache_control: %s", cache)
		}
	}
}

// TestExplicitCacheControlAtTheZeroTransientBoundaryCachesThePrefix pins the
// DEFAULT, which is the case every caller that does not opt into a transient
// window gets.
//
// TransientMessages is 0 here, so committedMessages == len(messages) and the
// live user turn counts as committed. Marking it would place the breakpoint on
// content that differs on every request: each turn writes a fresh cache entry
// and reads none, and the codec's stable system/tools breakpoint gets stripped
// on the way out — a net LOSS against not enabling the option at all. Nothing
// is transient means nothing to exclude, so the boundary belongs on the stable
// system prefix.
func TestExplicitCacheControlAtTheZeroTransientBoundaryCachesThePrefix(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel("anthropic", model.APIFormatAnthropic, "https://example.test", "claude-test", model.WithPromptCaching())
	req := inference.Request{
		Model:  selected,
		System: "stable system",
		// TransientMessages deliberately left at its zero value.
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "earlier"}}}},
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "live turn"}}}},
		},
	}
	encoded, err := (requestCodec{config: config{cacheControl: &CacheControlOptions{Type: "ephemeral", TTL: "5m"}}}).EncodeRequest(req, codec.RequestModeInvoke)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	bodyBytes, err := io.ReadAll(encoded.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	gateRequestBody(t, bodyBytes)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		t.Fatalf("body JSON: %v", err)
	}

	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON: %v", err)
	}
	if len(system) != 1 {
		t.Fatalf("system blocks = %d, want one cached text block", len(system))
	}
	if got := string(system[0]["cache_control"]); got != `{"type":"ephemeral","ttl":"5m"}` {
		t.Fatalf("system cache_control = %s, want the stable prefix to carry the breakpoint", got)
	}

	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages JSON: %v", err)
	}
	for index, message := range messages {
		var blocks []map[string]json.RawMessage
		if err := json.Unmarshal(message["content"], &blocks); err != nil {
			t.Fatalf("message %d content: %v", index, err)
		}
		for blockIndex, block := range blocks {
			if cache, exists := block["cache_control"]; exists {
				t.Fatalf("messages[%d].content[%d] carries cache_control %s; with nothing transient the "+
					"breakpoint would move every turn, writing a new entry and reading none",
					index, blockIndex, cache)
			}
		}
	}
}

func TestExplicitCacheControlFallsBackToStringSystem(t *testing.T) {
	t.Parallel()

	// The input is a complete CreateMessageParams rather than the two fields
	// the rewrite happens to touch, so the result can be held to the request
	// schema. A partial body would validate nothing.
	body := map[string]json.RawMessage{
		"model":      json.RawMessage(`"claude-sonnet-4-5"`),
		"max_tokens": json.RawMessage(`1024`),
		"system":     json.RawMessage(`"stable system"`),
		"messages":   json.RawMessage(`[{"role":"user","content":[{"type":"text","text":"runtime"}]}]`),
	}
	if err := applyPromptCacheControl(body, CacheControlOptions{Type: "ephemeral", TTL: "5m"}, false); err != nil {
		t.Fatalf("applyPromptCacheControl() error = %v", err)
	}
	gateRequestFields(t, body)

	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON: %v", err)
	}
	if got := string(system[0]["cache_control"]); got != `{"type":"ephemeral","ttl":"5m"}` {
		t.Fatalf("system cache_control = %s, want explicit ephemeral/5m", got)
	}
}

func TestExplicitCacheControlFallsBackToBlockSystem(t *testing.T) {
	t.Parallel()

	body := map[string]json.RawMessage{
		"model":      json.RawMessage(`"claude-sonnet-4-5"`),
		"max_tokens": json.RawMessage(`1024`),
		"messages":   json.RawMessage(`[{"role":"user","content":[{"type":"text","text":"runtime"}]}]`),
		"system":     json.RawMessage(`[{"type":"text","text":"one"},{"type":"text","text":"two","cache_control":{"type":"ephemeral"}}]`),
	}
	if err := applyPromptCacheControl(body, CacheControlOptions{Type: "ephemeral", TTL: "1h"}, false); err != nil {
		t.Fatalf("applyPromptCacheControl() error = %v", err)
	}
	gateRequestFields(t, body)

	var system []map[string]json.RawMessage
	if err := json.Unmarshal(body["system"], &system); err != nil {
		t.Fatalf("system JSON: %v", err)
	}
	if _, exists := system[0]["cache_control"]; exists {
		t.Fatal("earlier system block retained a cache boundary")
	}
	if got := string(system[1]["cache_control"]); got != `{"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("last system cache_control = %s", got)
	}
}

func TestExplicitCacheControlRejectsInvalidOptions(t *testing.T) {
	t.Parallel()

	for _, options := range []CacheControlOptions{{Type: "forever"}, {Type: "ephemeral", TTL: "30m"}} {
		body := map[string]json.RawMessage{"system": json.RawMessage(`"stable"`)}
		err := applyPromptCacheControl(body, options, false)
		var optionErr *OptionError
		if !errors.As(err, &optionErr) {
			t.Errorf("applyPromptCacheControl(%+v) error = %T (%v), want *OptionError", options, err, err)
		}
	}
}

func TestExplicitCacheControlRejectsTransientSystemContext(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:             model.CustomModel("anthropic", model.APIFormatAnthropic, "https://example.test", "claude-test"),
		TransientMessages: 1,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "stable"}}}},
			&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{&content.TextBlock{Text: "runtime"}}}},
		},
	}
	_, err := (requestCodec{config: config{cacheControl: &CacheControlOptions{Type: "ephemeral"}}}).EncodeRequest(req, codec.RequestModeInvoke)
	var optionErr *OptionError
	if !errors.As(err, &optionErr) {
		t.Fatalf("EncodeRequest() error = %T (%v), want *OptionError", err, err)
	}
}
