// Package openrouter provides the OpenRouter-specific construction and request
// options for the OpenAI-compatible Chat Completions API.
package openrouter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// ReasoningOptions controls OpenRouter's provider-specific reasoning object.
// Pointer fields preserve the difference between an omitted value and an
// explicitly supplied false or zero.
type ReasoningOptions struct {
	Effort    string `json:"effort,omitempty"`
	MaxTokens *int   `json:"max_tokens,omitempty"`
	Exclude   *bool  `json:"exclude,omitempty"`
	Enabled   *bool  `json:"enabled,omitempty"`
}

// ProviderRoutingOptions controls OpenRouter's provider-selection policy.
// Pointer fields preserve the difference between an omitted value and an
// explicitly supplied false.
type ProviderRoutingOptions struct {
	Order             []string `json:"order,omitempty"`
	AllowFallbacks    *bool    `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool    `json:"require_parameters,omitempty"`
	DataCollection    string   `json:"data_collection,omitempty"`
	ZDR               *bool    `json:"zdr,omitempty"`
}

type config struct {
	headers         http.Header
	usage           *bool
	reasoning       *ReasoningOptions
	promptCacheKey  string
	providerRouting *ProviderRoutingOptions
}

// Option customizes an OpenRouter client at construction time.
type Option func(*config)

// WithHTTPReferer adds OpenRouter's optional HTTP-Referer attribution header.
func WithHTTPReferer(value string) Option {
	return func(c *config) {
		if c.headers == nil {
			c.headers = make(http.Header)
		}
		if value == "" {
			c.headers.Del("HTTP-Referer")
			return
		}
		c.headers.Set("HTTP-Referer", value)
	}
}

// WithTitle adds OpenRouter's optional X-OpenRouter-Title attribution header.
func WithTitle(value string) Option {
	return func(c *config) {
		if c.headers == nil {
			c.headers = make(http.Header)
		}
		if value == "" {
			c.headers.Del("X-OpenRouter-Title")
			return
		}
		c.headers.Set("X-OpenRouter-Title", value)
	}
}

// WithUsage requests OpenRouter to include its usage metadata in the response.
func WithUsage(include bool) Option {
	return func(c *config) {
		c.usage = boolPtr(include)
	}
}

// WithReasoning adds OpenRouter's reasoning request object.
func WithReasoning(options ReasoningOptions) Option {
	return func(c *config) {
		c.reasoning = cloneReasoningOptions(options)
	}
}

// WithPromptCacheKey sets OpenRouter's stable prompt-cache key.
func WithPromptCacheKey(value string) Option {
	return func(c *config) {
		c.promptCacheKey = value
	}
}

// WithProviderRouting adds OpenRouter provider-routing preferences.
func WithProviderRouting(options ProviderRoutingOptions) Option {
	return func(c *config) {
		c.providerRouting = cloneProviderRoutingOptions(options)
	}
}

// New builds an OpenRouter inference client. The selected model must identify
// OpenRouter and the OpenAI API format; an empty API key is rejected before a
// client is constructed. An empty model base URL uses OpenRouter's canonical
// API root.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderOpenRouter) {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: fmt.Sprintf("OpenRouter constructor requires provider %q", llm.ProviderOpenRouter),
		}
	}
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderOpenRouter, Kind: auth.AuthAPIKey}
	}

	cfg := config{}
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	cfg = cloneConfig(cfg)

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
		chatRouter{headers: cfg.headers},
		requestCodec{config: cfg},
		auth.Key(key),
	), nil
}

type chatRouter struct {
	headers http.Header
}

func (r chatRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	built, err := route.StaticChat("/chat/completions").BuildRoute(baseURL, req, mode)
	if err != nil {
		return route.Route{}, err
	}
	built.Header = r.headers.Clone()
	return built, nil
}

type requestCodec struct {
	config config
}

var _ codec.StreamingCodec = requestCodec{}

func (c requestCodec) EncodeRequest(req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (openaiapi.Codec{}).EncodeRequest(req, mode)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	if !c.config.hasBodyOptions() {
		return encoded, nil
	}

	raw, err := io.ReadAll(encoded.Body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: read encoded request: %w", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: decode encoded request: %w", err)
	}
	if c.config.usage != nil {
		body["usage"], err = json.Marshal(struct {
			Include bool `json:"include"`
		}{Include: *c.config.usage})
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode usage option: %w", err)
		}
	}
	if c.config.reasoning != nil {
		body["reasoning"], err = json.Marshal(c.config.reasoning)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode reasoning option: %w", err)
		}
	}
	if c.config.promptCacheKey != "" {
		body["prompt_cache_key"], err = json.Marshal(c.config.promptCacheKey)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode prompt cache key: %w", err)
		}
	}
	if c.config.providerRouting != nil {
		body["provider"], err = json.Marshal(c.config.providerRouting)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode provider routing option: %w", err)
		}
	}

	patched, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode extended request: %w", err)
	}
	return codec.EncodedRequest{
		Header: encoded.Header.Clone(),
		Body:   bytes.NewReader(patched),
	}, nil
}

func (requestCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	return (openaiapi.Codec{}).DecodeResponse(body)
}

func (requestCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	return (openaiapi.Codec{}).DecodeStream(resp)
}

func (c config) hasBodyOptions() bool {
	return c.usage != nil || c.reasoning != nil || c.promptCacheKey != "" || c.providerRouting != nil
}

func cloneConfig(in config) config {
	out := in
	out.headers = in.headers.Clone()
	if in.usage != nil {
		out.usage = boolPtr(*in.usage)
	}
	if in.reasoning != nil {
		out.reasoning = cloneReasoningOptions(*in.reasoning)
	}
	if in.providerRouting != nil {
		out.providerRouting = cloneProviderRoutingOptions(*in.providerRouting)
	}
	return out
}

func cloneReasoningOptions(in ReasoningOptions) *ReasoningOptions {
	out := in
	out.MaxTokens = intPtrValue(in.MaxTokens)
	out.Exclude = boolPtrValue(in.Exclude)
	out.Enabled = boolPtrValue(in.Enabled)
	return &out
}

func cloneProviderRoutingOptions(in ProviderRoutingOptions) *ProviderRoutingOptions {
	out := in
	out.Order = append([]string(nil), in.Order...)
	out.AllowFallbacks = boolPtrValue(in.AllowFallbacks)
	out.RequireParameters = boolPtrValue(in.RequireParameters)
	out.ZDR = boolPtrValue(in.ZDR)
	return &out
}

func boolPtr(value bool) *bool { return &value }

func boolPtrValue(value *bool) *bool {
	if value == nil {
		return nil
	}
	return boolPtr(*value)
}

func intPtrValue(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
