// Package openaisubscription provides inference authorized against a user's
// ChatGPT plan through the public Sign in with ChatGPT contract.
package openaisubscription

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/streamaccumulator"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"
	"github.com/looprig/llm"
)

const BaseURL = "https://api.openai.com/v1"

type Option func(*config)
type config struct{ transport []transport.Option }

func WithRoundTripper(rt http.RoundTripper) Option {
	return func(c *config) { c.transport = append(c.transport, transport.WithRoundTripper(rt)) }
}
func WithTLSRootCAs(roots *x509.CertPool) Option {
	return func(c *config) { c.transport = append(c.transport, transport.WithTLSRootCAs(roots)) }
}

// Client supports call-scoped authorization. Construct through auto.NewWithAuth
// to bind an OAuth subscription Source to it; no credential is read implicitly.
type Client struct{ *transport.Client }

var _ inference.Client = (*Client)(nil)

func New(selected model.Model, options ...Option) (*Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderOpenAISubscription) {
		return nil, &model.ValidationError{Field: "Provider", Reason: "subscription constructor requires openai-subscription"}
	}
	var cfg config
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	args := make([]any, len(cfg.transport))
	for i, option := range cfg.transport {
		args[i] = option
	}
	inner := transport.NewWithAuth(transport.Endpoint{BaseURL: BaseURL, Provider: selected.Provider, APIFormat: selected.APIFormat}, route.StaticChat("/responses"), subscriptionCodec{}, args...)
	return &Client{Client: inner}, nil
}

// Invoke collects the same mandatory streaming wire flow used by Stream.
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	reader, err := c.Client.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	return collect(reader)
}
func (c *Client) InvokeWithAuth(ctx context.Context, req inference.Request, auth httpauth.Authorizer) (*inference.Response, error) {
	reader, err := c.StreamWithAuth(ctx, req, auth)
	if err != nil {
		return nil, err
	}
	return collect(reader)
}

func collect(reader *stream.StreamReader[content.Chunk]) (*inference.Response, error) {
	defer reader.Close()
	var thinking streamaccumulator.Thinking
	var text streamaccumulator.Text
	var refusal streamaccumulator.Refusal
	var tools streamaccumulator.ToolUses
	var images streamaccumulator.Images
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		switch chunk := chunk.(type) {
		case *content.ThinkingChunk:
			thinking.Add(chunk)
		case *content.TextChunk:
			text.Add(chunk)
		case *content.RefusalChunk:
			refusal.Add(chunk)
		case *content.ToolUseChunk:
			tools.Add(chunk)
		case *content.ImageChunk:
			images.Add(chunk)
		}
	}
	result, ok := reader.Result()
	if !ok {
		return nil, errors.New("openai-subscription: stream has no completed response")
	}
	message := &content.AIMessage{Message: content.Message{Role: content.RoleAssistant}}
	if result.Usage != nil {
		usage := *result.Usage
		message.Usage = &usage
	}
	for _, block := range thinking.Blocks() {
		message.Blocks = append(message.Blocks, &block)
	}
	if block := text.Block(); block != nil {
		message.Blocks = append(message.Blocks, block)
	}
	if block := refusal.Block(); block != nil {
		message.Blocks = append(message.Blocks, block)
	}
	for _, block := range tools.Blocks() {
		message.Blocks = append(message.Blocks, &block)
	}
	for _, block := range images.Blocks() {
		message.Blocks = append(message.Blocks, &block)
	}
	return &inference.Response{Message: message, Usage: result.Usage, Model: result.Model, FinishReason: result.FinishReason, Attempts: result.Attempts}, nil
}

type subscriptionCodec struct{ openairesponses.Codec }

func (subscriptionCodec) EncodeRequest(req inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (openairesponses.Codec{}).EncodeRequest(req, codec.RequestModeStream)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	var body map[string]json.RawMessage
	if err := json.NewDecoder(encoded.Body).Decode(&body); err != nil {
		return codec.EncodedRequest{}, err
	}
	body["store"] = json.RawMessage("false")
	body["stream"] = json.RawMessage("true")
	raw, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	return codec.EncodedRequest{Header: encoded.Header, Body: bytes.NewReader(raw)}, nil
}
