// Package openai provides the OpenAI Responses API client. OpenAI Chat
// Completions remains available through inference/codec/openaiapi; this
// provider deliberately binds the newer item-based Responses wire format.
package openai

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
	responses "github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
)

const defaultBaseURL = "https://api.openai.com/v1"

// New constructs an OpenAI Responses API client. The selected model must
// identify OpenAI and model.APIFormatOpenAIResponses. An empty model base URL
// uses OpenAI's canonical Responses API root.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderOpenAI) {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: fmt.Sprintf("OpenAI constructor requires provider %q", llm.ProviderOpenAI),
		}
	}
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderOpenAI, Kind: auth.AuthAPIKey}
	}

	var cfg config
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
		responsesRouter{},
		requestCodec{config: cfg},
		auth.Key(key),
	), nil
}

type responsesRouter struct{}

func (responsesRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	return route.StaticChat("/responses").BuildRoute(baseURL, req, mode)
}

type requestCodec struct {
	config config
}

var _ codec.StreamingCodec = requestCodec{}

func (c requestCodec) EncodeRequest(req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (responses.Codec{}).EncodeRequest(req, mode)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	if !c.config.hasBodyOptions() {
		return encoded, nil
	}

	raw, err := io.ReadAll(encoded.Body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openai: read encoded request: %w", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openai: decode encoded request: %w", err)
	}
	if c.config.reasoning != nil {
		body["reasoning"], err = json.Marshal(c.config.reasoning)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openai: encode reasoning option: %w", err)
		}
		delete(body, "reasoning_effort")
	}
	if c.config.serviceTier != "" {
		body["service_tier"], err = json.Marshal(c.config.serviceTier)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openai: encode service tier option: %w", err)
		}
	}
	if c.config.metadata != nil {
		body["metadata"], err = json.Marshal(c.config.metadata)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openai: encode metadata option: %w", err)
		}
	}
	if c.config.promptCacheKey != "" {
		body["prompt_cache_key"], err = json.Marshal(c.config.promptCacheKey)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openai: encode prompt cache key: %w", err)
		}
	}

	patched, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openai: encode extended request: %w", err)
	}
	return codec.EncodedRequest{Header: encoded.Header.Clone(), Body: bytes.NewReader(patched)}, nil
}

func (requestCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	return (responses.Codec{}).DecodeResponse(body)
}

func (requestCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	return (responses.Codec{}).DecodeStream(resp)
}
