// Package azure provides an Azure OpenAI Responses API client. It keeps
// Azure's resource endpoint and api-key authentication separate, delegates
// request encoding, non-streaming response decoding, tool and usage semantics
// to inference's shared OpenAI Responses codec, and normalizes Azure-specific
// reasoning and termination variants locally.
//
// SSE SEMANTICS ARE NOT DELEGATED. streamCollector.mapFrame in codec.go is a
// private fork of the shared codec's stream collector: it owns frame decoding,
// both failure channels, and the terminal envelope, and delegates only unknown
// frames back to the shared per-event decoder. See that function's doc comment
// for what Azure does differently, the primary source for each divergence, and
// what would let the fork be deleted. A fix landed in
// inference/codec/openairesponses' collector does NOT reach this package.
package azure

import (
	"fmt"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	codec "github.com/looprig/inference/codec"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
)

// New constructs an Azure OpenAI Responses API client. An explicit model base
// URL wins; otherwise WithResourceName or AZURE_RESOURCE_NAME supplies the
// Azure resource used to build the modern /openai/v1 endpoint.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderAzure) {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: fmt.Sprintf("Azure constructor requires provider %q", llm.ProviderAzure),
		}
	}
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderAzure, Kind: auth.AuthAPIKey}
	}

	var cfg config
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	cfg = cfg.clone()
	baseURL, err := resolveBaseURL(selected.BaseURL, cfg.resourceName)
	if err != nil {
		return nil, err
	}
	return transport.New(
		transport.Endpoint{
			BaseURL:   baseURL,
			Provider:  selected.Provider,
			APIFormat: selected.APIFormat,
		},
		responsesRouter{},
		requestCodec{config: cfg},
		auth.Header(key, "api-key"),
	), nil
}

type responsesRouter struct{}

func (responsesRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	return route.StaticChat("/responses").BuildRoute(baseURL, req, mode)
}
