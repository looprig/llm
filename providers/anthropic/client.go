// Package anthropic provides a native Anthropic Messages API client. It keeps
// Anthropic's top-level system prompt, typed content blocks, tool_result/tool_use
// turns, thinking controls, and SSE lifecycle intact by delegating wire semantics
// to inference/codec/anthropicapi.
package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	codec "github.com/looprig/inference/codec"
	anthropicapi "github.com/looprig/inference/codec/anthropicapi"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
)

const (
	defaultBaseURL   = "https://api.anthropic.com/v1"
	anthropicVersion = "2023-06-01"
)

// New constructs an Anthropic native Messages API client. The selected model
// must identify Anthropic and model.APIFormatAnthropic. An empty model base URL
// uses Anthropic's canonical API root.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderAnthropic) {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: fmt.Sprintf("Anthropic constructor requires provider %q", llm.ProviderAnthropic),
		}
	}
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderAnthropic, Kind: auth.AuthAPIKey}
	}

	cfg := config{}
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	cfg = cfg.clone()
	baseURL := selected.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return transport.New(
		transport.Endpoint{
			BaseURL:   baseURL,
			Provider:  selected.Provider,
			APIFormat: selected.APIFormat,
		},
		messagesRouter{betaHeaders: cfg.headers()},
		requestCodec{config: cfg},
		auth.Header(key, "x-api-key"),
	), nil
}

type messagesRouter struct {
	betaHeaders []string
}

func (r messagesRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	built, err := route.StaticChat("/messages").BuildRoute(baseURL, req, mode)
	if err != nil {
		return route.Route{}, err
	}
	built.Header = make(http.Header)
	built.Header.Set("anthropic-version", anthropicVersion)
	if len(r.betaHeaders) > 0 {
		built.Header.Set("anthropic-beta", strings.Join(r.betaHeaders, ","))
	}
	return built, nil
}

type requestCodec struct {
	config config
}

var _ codec.StreamingCodec = requestCodec{}

func (c requestCodec) EncodeRequest(req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, error) {
	encodedReq := req
	if c.config.cacheControl != nil {
		// Let the shared codec compute the block-accurate committed boundary.
		// The provider option replaces the marker's value below, but must not
		// re-derive a projected boundary from neutral message cardinality.
		encodedReq.Model.Caps.PromptCaching = true
	}
	encoded, err := (anthropicapi.Codec{}).EncodeRequest(encodedReq, mode)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	if !c.hasBodyOptions() {
		return encoded, nil
	}
	raw, err := io.ReadAll(encoded.Body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("anthropic: read encoded request: %w", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("anthropic: decode encoded request: %w", err)
	}
	if c.config.thinking != nil {
		thinking := *c.config.thinking
		if thinking.Type == "" {
			thinking.Type = "adaptive"
		}
		body["thinking"], err = json.Marshal(makeThinkingRequest(thinking))
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("anthropic: encode thinking option: %w", err)
		}
		if thinking.Effort != "" {
			var outputConfig map[string]json.RawMessage
			if rawConfig, ok := body["output_config"]; ok {
				_ = json.Unmarshal(rawConfig, &outputConfig)
			}
			if outputConfig == nil {
				outputConfig = make(map[string]json.RawMessage)
			}
			outputConfig["effort"], err = json.Marshal(thinking.Effort)
			if err != nil {
				return codec.EncodedRequest{}, fmt.Errorf("anthropic: encode thinking effort: %w", err)
			}
			body["output_config"], err = json.Marshal(outputConfig)
			if err != nil {
				return codec.EncodedRequest{}, fmt.Errorf("anthropic: encode output config: %w", err)
			}
		}
	}
	if c.config.metadataUserID != "" {
		body["metadata"], err = json.Marshal(map[string]string{"user_id": c.config.metadataUserID})
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("anthropic: encode metadata: %w", err)
		}
	}
	if c.config.cacheControl != nil {
		committedSource := len(req.Messages) - req.TransientMessages
		for index, message := range req.Messages {
			if index >= committedSource {
				if _, system := message.(*content.SystemMessage); system {
					return codec.EncodedRequest{}, &OptionError{Reason: "prompt cache control cannot precede transient system context"}
				}
				continue
			}
		}
		if err := applyPromptCacheControl(body, *c.config.cacheControl, req.TransientMessages > 0); err != nil {
			return codec.EncodedRequest{}, err
		}
	}
	patched, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("anthropic: encode extended request: %w", err)
	}
	return codec.EncodedRequest{Header: encoded.Header.Clone(), Body: bytes.NewReader(patched)}, nil
}

type thinkingRequest struct {
	Type         string `json:"type"`
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

func makeThinkingRequest(options ThinkingOptions) thinkingRequest {
	return thinkingRequest{Type: options.Type, BudgetTokens: options.BudgetTokens}
}

func (c requestCodec) hasBodyOptions() bool {
	return c.config.thinking != nil || c.config.metadataUserID != "" || c.config.cacheControl != nil
}

func applyPromptCacheControl(body map[string]json.RawMessage, control CacheControlOptions, preferMessageBoundary bool) error {
	if control.Type == "" {
		control.Type = "ephemeral"
	}
	if control.Type != "ephemeral" {
		return &OptionError{Reason: "cache control type must be ephemeral"}
	}
	if control.TTL != "" && control.TTL != "5m" && control.TTL != "1h" {
		return &OptionError{Reason: "cache control TTL must be 5m or 1h"}
	}
	cacheRaw, err := json.Marshal(control)
	if err != nil {
		return &OptionError{Reason: "encode cache control", Err: err}
	}

	// Explicit policy replaces any codec capability breakpoints, so a request
	// never carries contradictory automatic and explicit boundaries.
	if rawMessages, ok := body["messages"]; ok {
		var messages []map[string]json.RawMessage
		if err := json.Unmarshal(rawMessages, &messages); err != nil {
			return &OptionError{Reason: "decode messages for cache control", Err: err}
		}
		boundaryMessage, boundaryBlock := -1, -1
		for index := range messages {
			var blocks []map[string]json.RawMessage
			if err := json.Unmarshal(messages[index]["content"], &blocks); err != nil {
				return &OptionError{Reason: "decode message content for cache control", Err: err}
			}
			for blockIndex := range blocks {
				if _, marked := blocks[blockIndex]["cache_control"]; marked {
					boundaryMessage, boundaryBlock = index, blockIndex
				}
				delete(blocks[blockIndex], "cache_control")
			}
			messages[index]["content"], _ = json.Marshal(blocks)
		}
		// A message breakpoint is only worth writing when it sits BEFORE
		// something transient. When committedMessages == len(messages) —
		// which is the zero value of Request.TransientMessages, and therefore
		// the default — the "last committed message" IS the live turn: its
		// content differs on every request, so the breakpoint moves every
		// turn, writing a fresh cache entry and reading none, and the codec's
		// stable system/tools breakpoint would be cleared on the way out. That
		// is strictly worse than not caching at all, so fall through to the
		// system branch, where the prefix is genuinely stable.
		if preferMessageBoundary && boundaryMessage >= 0 {
			var blocks []map[string]json.RawMessage
			_ = json.Unmarshal(messages[boundaryMessage]["content"], &blocks)
			blocks[boundaryBlock]["cache_control"] = cacheRaw
			messages[boundaryMessage]["content"], _ = json.Marshal(blocks)
			body["messages"], err = json.Marshal(messages)
			if err != nil {
				return &OptionError{Reason: "encode cached messages", Err: err}
			}
			return clearSystemCacheControls(body)
		}
		body["messages"], _ = json.Marshal(messages)
	}
	if err := clearSystemCacheControls(body); err != nil {
		return err
	}
	rawSystem, ok := body["system"]
	if !ok {
		return &OptionError{Reason: "prompt cache control requires a committed message or non-empty system prompt"}
	}
	var text string
	if json.Unmarshal(rawSystem, &text) == nil {
		if text == "" {
			return &OptionError{Reason: "prompt cache control requires a non-empty system prompt"}
		}
		block := map[string]json.RawMessage{"cache_control": cacheRaw}
		block["type"], _ = json.Marshal("text")
		block["text"], _ = json.Marshal(text)
		body["system"], err = json.Marshal([]map[string]json.RawMessage{block})
		return err
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(rawSystem, &blocks); err != nil || len(blocks) == 0 {
		return &OptionError{Reason: "decode system for cache control", Err: err}
	}
	blocks[len(blocks)-1]["cache_control"] = cacheRaw
	body["system"], err = json.Marshal(blocks)
	if err != nil {
		return &OptionError{Reason: "encode cached system", Err: err}
	}
	return nil
}

func clearSystemCacheControls(body map[string]json.RawMessage) error {
	rawSystem, ok := body["system"]
	if !ok {
		return nil
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(rawSystem, &blocks); err != nil {
		// A string system has no block-level cache_control to remove.
		return nil
	}
	for index := range blocks {
		delete(blocks[index], "cache_control")
	}
	encoded, err := json.Marshal(blocks)
	if err != nil {
		return &OptionError{Reason: "encode system cache controls", Err: err}
	}
	body["system"] = encoded
	return nil
}

func (requestCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	return (anthropicapi.Codec{}).DecodeResponse(body)
}

func (requestCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	return (anthropicapi.Codec{}).DecodeStream(resp)
}
