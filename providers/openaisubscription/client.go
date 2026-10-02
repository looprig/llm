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
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"
	"github.com/looprig/inference/wire/sse"
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
	reader, err := c.Stream(ctx, req)
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

// InvokeWithAuthorizer is a descriptive alias for InvokeWithAuth. It is
// declared here so the promoted transport alias cannot bypass the mandatory
// streaming flow.
func (c *Client) InvokeWithAuthorizer(ctx context.Context, req inference.Request, auth httpauth.Authorizer) (*inference.Response, error) {
	return c.InvokeWithAuth(ctx, req, auth)
}

// Stream sends one streaming request with the client's default authorizer.
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	reader, err := c.Client.Stream(ctx, req)
	return reader, classifyAdmission(err)
}

// StreamWithAuth sends one streaming request with a call-scoped authorizer.
func (c *Client) StreamWithAuth(ctx context.Context, req inference.Request, auth httpauth.Authorizer) (*stream.StreamReader[content.Chunk], error) {
	reader, err := c.Client.StreamWithAuth(ctx, req, auth)
	return reader, classifyAdmission(err)
}

// StreamWithAuthorizer is a descriptive alias for StreamWithAuth.
func (c *Client) StreamWithAuthorizer(ctx context.Context, req inference.Request, auth httpauth.Authorizer) (*stream.StreamReader[content.Chunk], error) {
	return c.StreamWithAuth(ctx, req, auth)
}

// classifyAdmission replaces the one documented pre-stream failure whose
// recovery contradicts its HTTP status class. Every other error, including a
// retryable 503 usage_unavailable, is returned unchanged.
func classifyAdmission(err error) error {
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) && apiErr.Status == http.StatusTooManyRequests && apiErr.Code == codeUsageLimitExceeded {
		return &UsageLimitError{RequestID: apiErr.RequestID}
	}
	return err
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

// unsupportedRequestFields are the CreateResponse members the generic
// Responses codec can emit but the ChatGPT-plan route refuses. They are
// omitted rather than failing the request, matching how the generic codec
// already omits dialect-unsupported knobs (e.g. Sampling.Stop). Source:
// https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
// ("Unsupported fields: Omit ... max_output_tokens, ... temperature, ...
// top_p ...").
var unsupportedRequestFields = []string{"max_output_tokens", "temperature", "top_p"}

// EncodeRequest adapts the generic Responses body to the ChatGPT-plan route
// (https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations):
//
//   - store:false and stream:true on every request;
//   - unsupported fields omitted (unsupportedRequestFields);
//   - function tools are not accepted as flat top-level tools; the route
//     requires them grouped in namespaces or supplied through additional_tools
//     input items. They are moved, unchanged, into one leading
//     {"type":"additional_tools","role":"developer","tools":[...]} input
//     item, the shape documented at
//     https://developers.openai.com/api/docs/guides/tools-tool-search ("Add
//     tools at a specific point in the input"). Function calls from tools
//     supplied this way carry no namespace, so history round-trips through
//     the generic codec unchanged. openai/codex sends the same leading item
//     (codex-rs/core/src/client.rs build_responses_request, use_responses_lite
//     with flat tools when namespace_tools is off).
func (subscriptionCodec) EncodeRequest(req inference.Request, _ codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (openairesponses.Codec{}).EncodeRequest(req, codec.RequestModeStream)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	var body map[string]json.RawMessage
	if err := json.NewDecoder(encoded.Body).Decode(&body); err != nil {
		return codec.EncodedRequest{}, err
	}
	for _, field := range unsupportedRequestFields {
		delete(body, field)
	}
	if tools, ok := body["tools"]; ok {
		delete(body, "tools")
		var input []json.RawMessage
		if raw, ok := body["input"]; ok {
			if err := json.Unmarshal(raw, &input); err != nil {
				return codec.EncodedRequest{}, err
			}
		}
		item, err := json.Marshal(struct {
			Type  string          `json:"type"`
			Role  string          `json:"role"`
			Tools json.RawMessage `json:"tools"`
		}{Type: "additional_tools", Role: "developer", Tools: tools})
		if err != nil {
			return codec.EncodedRequest{}, err
		}
		input = append([]json.RawMessage{item}, input...)
		if body["input"], err = json.Marshal(input); err != nil {
			return codec.EncodedRequest{}, err
		}
	}
	body["store"] = json.RawMessage("false")
	body["stream"] = json.RawMessage("true")
	raw, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	return codec.EncodedRequest{Header: encoded.Header, Body: bytes.NewReader(raw)}, nil
}

// DecodeStream replaces the generic Responses terminal handling for this
// route only. Inference is successful only after response.completed;
// response.incomplete, response.failed, a top-level error event, and a stream
// ending without a terminal event all fail
// (https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).
// Content chunks are decoded by the generic codec unchanged. Provider error
// messages are never retained: only bounded code tokens are.
//
// This is a deliberate, narrow fork of openairesponses' terminal collector:
// the generic Responses contract accepts response.incomplete as a terminal
// result, while this route does not. Content decoding (DecodeEvent) and the
// completed envelope (DecodeResponse: model, usage, finish reason) still come
// from the shared codec, so its fixes reach this provider. The fork can be
// deleted once the shared codec offers a "completed-only" terminal option.
func (subscriptionCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &terminalCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.result), nil
}

type terminalCollector struct {
	completed bool
	terminal  stream.StreamResult
}

type terminalEvent struct {
	Type     string          `json:"type"`
	Code     string          `json:"code"`
	Param    string          `json:"param"`
	Response json.RawMessage `json:"response"`
}

type terminalResponse struct {
	Error *struct {
		Code  string `json:"code"`
		Param string `json:"param"`
	} `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

func (c *terminalCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	var event terminalEvent
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return nil, &openairesponses.StreamEventDecodeError{Err: err}
	}
	switch event.Type {
	case "error":
		return nil, responseFailure(event.Code, event.Param)
	case "response.failed", "response.incomplete":
		var response terminalResponse
		if len(event.Response) > 0 {
			if err := json.Unmarshal(event.Response, &response); err != nil {
				return nil, &openairesponses.StreamEventDecodeError{Err: err}
			}
		}
		if event.Type == "response.incomplete" {
			reason := ""
			if response.IncompleteDetails != nil {
				reason = safeToken(response.IncompleteDetails.Reason)
			}
			return nil, &IncompleteError{Reason: reason}
		}
		if response.Error == nil {
			return nil, responseFailure("", "")
		}
		return nil, responseFailure(response.Error.Code, response.Error.Param)
	case "response.completed":
		decoded, err := openairesponses.DecodeResponse(event.Response)
		if err != nil {
			return nil, err
		}
		c.completed = true
		c.terminal = stream.StreamResult{Model: decoded.Model, Usage: decoded.Usage, FinishReason: decoded.FinishReason}
	}
	return (openairesponses.Codec{}).DecodeEvent(frame.Data)
}

func (c *terminalCollector) result() (stream.StreamResult, bool, error) {
	if !c.completed {
		return stream.StreamResult{}, false, &openairesponses.StreamDecodeError{Reason: "ended before response.completed"}
	}
	return c.terminal, true, nil
}
