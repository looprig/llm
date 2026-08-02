// Package bedrock is an AWS Bedrock Runtime client for Bedrock InvokeModel and
// native Converse/ConverseStream. It routes the selected native
// dialect to the corresponding model path and signs every request with AWS
// Signature Version 4.
//
// Credentials are AWS SigV4, not a bearer key, so a Bedrock client is constructed
// directly via New (auto.New cannot supply SigV4 credentials and errors to here).
package bedrock

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	inferauth "github.com/looprig/inference/auth"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auth"
)

// Compile-time proof that Client honors the inference.Client contract.
var _ inference.Client = (*Client)(nil)

const (
	// bedrockService is the SigV4 service name; the signer keys its non-s3
	// canonical-URI double-encoding on it (the model path's ":" -> "%3A").
	bedrockService = "bedrock"
	// endpointScheme and hostFormat build the region-routed Bedrock Runtime endpoint.
	endpointScheme = "https"
	hostFormat     = "bedrock-runtime.%s.amazonaws.com"
	// path fragments: /model/<model-id>/invoke.
	pathModelPrefix          = "/model/"
	pathInvokeSuffix         = "/invoke"
	pathConverseSuffix       = "/converse"
	pathConverseStreamSuffix = "/converse-stream"
	pathCountSuffix          = "/count-tokens"

	contentTypeJSON = "application/json"
	maxRegionBytes  = 64
)

// Timeout budget for the connect/TLS/header phases, mirroring the generic
// transport client's hygiene. There is deliberately no whole-request
// http.Client.Timeout: the per-request deadline is the caller's context (Invoke
// takes ctx), and omitting it keeps the client forward-compatible with a future
// long-lived streaming body.
const (
	dialTimeout           = 10 * time.Second
	tlsHandshakeTimeout   = 10 * time.Second
	responseHeaderTimeout = 60 * time.Second
	expectContinueTimeout = 1 * time.Second
	idleConnTimeout       = 90 * time.Second
)

// Client is a region-bound Bedrock inference client. It owns one
// SigV4 signer (built from the caller's credentials) and one http.Client, and is
// safe for concurrent use (both are immutable after construction). Connection
// binding is by provider+region: a request whose Model.Provider is not
// ProviderBedrock is rejected pre-I/O with *failure.ModelMismatchError.
type Client struct {
	region   string
	endpoint string // scheme://host base, e.g. https://bedrock-runtime.us-east-1.amazonaws.com
	signer   inferauth.Authenticator
	codec    anthropicapi.Codec
	options  config
	hc       *http.Client
}

// New constructs a Bedrock client bound to region, signing with creds. It fails
// closed with *ConfigError when the region or either mandatory credential field
// (AccessKeyID, SecretAccessKey) is empty — no Client and no network object are
// created. The session token is optional (used for temporary credentials).
func New(creds auth.SigV4Credentials, region string, options ...Option) (inference.Client, error) {
	if err := validateConfig(creds, region); err != nil {
		return nil, err
	}
	return newClient(creds, region, defaultEndpoint(region), options...), nil
}

func validateConfig(creds auth.SigV4Credentials, region string) error {
	if region == "" {
		return &ConfigError{Field: "region", Reason: "AWS region must not be empty"}
	}
	if !validRegion(region) {
		return &ConfigError{Field: "region", Reason: "AWS region must contain only lowercase ASCII letters, digits, and interior hyphens"}
	}
	if creds.AccessKeyID == "" {
		return &ConfigError{Field: "AccessKeyID", Reason: "SigV4 AccessKeyID must not be empty"}
	}
	if creds.SecretAccessKey == "" {
		return &ConfigError{Field: "SecretAccessKey", Reason: "SigV4 SecretAccessKey must not be empty"}
	}
	return nil
}

func validRegion(region string) bool {
	if len(region) == 0 || len(region) > maxRegionBytes || region[0] == '-' || region[len(region)-1] == '-' {
		return false
	}
	for _, char := range region {
		if char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '-' {
			continue
		}
		return false
	}
	return true
}

// newClient wires a Client for a validated region + endpoint. endpoint is the
// scheme://host base; New derives it from the region, tests override it to reach
// an httptest.Server. It builds the SigV4 signer once and the phase-bounded,
// TLS>=1.2 http.Client.

func newClient(creds auth.SigV4Credentials, region, endpoint string, options ...Option) *Client {
	cfg := config{}
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	return &Client{
		region:   region,
		endpoint: endpoint,
		signer:   auth.SigV4(creds, region, bedrockService),
		options:  cfg.clone(),
		hc:       newHTTPClient(),
	}
}

func newHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   dialTimeout,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12},
			TLSHandshakeTimeout:   tlsHandshakeTimeout,
			ResponseHeaderTimeout: responseHeaderTimeout,
			ExpectContinueTimeout: expectContinueTimeout,
			IdleConnTimeout:       idleConnTimeout,
			ForceAttemptHTTP2:     true,
		},
	}
}

// defaultEndpoint returns the region-routed Bedrock Runtime endpoint base.
func defaultEndpoint(region string) string {
	return endpointScheme + "://" + fmt.Sprintf(hostFormat, region)
}

// Invoke sends a non-streaming request in the selected Bedrock dialect. Ordered,
// all pre-I/O guards first: provider binding, model validation, API-format
// selection, local encoding, request construction, SigV4 signing, HTTP error
// mapping, and response decoding.
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	if err := c.checkBinding(req.Model); err != nil {
		return nil, err
	}
	if err := llm.ValidateModel(req.Model); err != nil {
		return nil, err
	}
	switch req.Model.APIFormat {
	case model.APIFormatAnthropic:
		return c.invokeAnthropic(ctx, req)
	case model.APIFormatBedrockConverse:
		return c.invokeConverse(ctx, req)
	default:
		return nil, &UnsupportedAPIFormatError{APIFormat: req.Model.APIFormat}
	}
}

func (c *Client) invokeAnthropic(ctx context.Context, req inference.Request) (*inference.Response, error) {
	anthropicBody, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		return nil, err
	}
	body, err := toBedrockBody(anthropicBody)
	if err != nil {
		return nil, err
	}
	httpReq, err := c.buildRequest(ctx, req.Model.Name, body)
	if err != nil {
		return nil, err
	}
	httpResp, err := c.doSigned(ctx, httpReq)
	if err != nil {
		return nil, err
	}
	respBody, err := readResponseBody(httpResp)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode/100 != 2 {
		return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(respBody), Body: respBody}
	}
	return c.codec.DecodeResponse(respBody)
}

func (c *Client) invokeConverse(ctx context.Context, req inference.Request) (*inference.Response, error) {
	body, err := c.encodeConverse(req, false)
	if err != nil {
		return nil, err
	}
	httpReq, err := buildRuntimeRequestWithAccept(ctx, c.endpoint, req.Model.Name, pathConverseSuffix, body, contentTypeJSON)
	if err != nil {
		return nil, err
	}
	httpResp, err := c.doSigned(ctx, httpReq)
	if err != nil {
		return nil, err
	}
	respBody, err := readResponseBody(httpResp)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode/100 != 2 {
		return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(respBody), Body: respBody}
	}
	response, err := (bedrockconverse.Codec{}).DecodeResponse(respBody)
	if err != nil {
		return nil, err
	}
	if response.Model == "" {
		response.Model = req.Model.Name
	}
	return response, nil
}

// Stream sends a native ConverseStream request. Anthropic InvokeModel streaming
// remains intentionally unsupported because it uses a different Bedrock wire
// contract and must not be silently switched to Converse.
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	if err := c.checkBinding(req.Model); err != nil {
		return nil, err
	}
	if err := llm.ValidateModel(req.Model); err != nil {
		return nil, err
	}
	if req.Model.APIFormat == model.APIFormatAnthropic {
		return nil, &StreamingNotSupportedError{}
	}
	if req.Model.APIFormat != model.APIFormatBedrockConverse {
		return nil, &UnsupportedAPIFormatError{APIFormat: req.Model.APIFormat}
	}
	body, err := c.encodeConverse(req, true)
	if err != nil {
		return nil, err
	}
	httpReq, err := buildRuntimeRequestWithAccept(ctx, c.endpoint, req.Model.Name, pathConverseStreamSuffix, body, "application/vnd.amazon.eventstream")
	if err != nil {
		return nil, err
	}
	httpResp, err := c.doSigned(ctx, httpReq)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode/100 != 2 {
		respBody, readErr := readResponseBody(httpResp)
		if readErr != nil {
			return nil, readErr
		}
		return nil, &failure.APIError{Status: httpResp.StatusCode, Message: string(respBody), Body: respBody}
	}
	reader, err := (bedrockconverse.Codec{}).DecodeStream(httpResp)
	if err != nil {
		_ = httpResp.Body.Close()
		return nil, err
	}
	return withModel(reader, req.Model.Name), nil
}

func (c *Client) encodeConverse(req inference.Request, streaming bool) ([]byte, error) {
	body, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		return nil, err
	}
	return c.options.applyConverse(body, streaming)
}

func (c *Client) doSigned(ctx context.Context, request *http.Request) (*http.Response, error) {
	if err := c.signer.Authorize(ctx, request); err != nil {
		return nil, err
	}
	response, err := c.hc.Do(request)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	if response == nil || response.Body == nil {
		return nil, &failure.NetworkError{Err: fmt.Errorf("bedrock: empty HTTP response")}
	}
	return response, nil
}

func readResponseBody(response *http.Response) ([]byte, error) {
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	return body, nil
}

func withModel(reader *stream.StreamReader[content.Chunk], modelID string) *stream.StreamReader[content.Chunk] {
	return stream.NewStreamReaderWithResult(reader.Next, reader.Close, func() (stream.StreamResult, bool, error) {
		result, ok := reader.Result()
		if !ok {
			return stream.StreamResult{}, false, nil
		}
		if result.Model == "" {
			result.Model = modelID
		}
		return result, true, nil
	})
}

// checkBinding fails closed when the request's Model names a provider other than
// Bedrock, before any I/O. Bedrock is region-bound (the Model carries no region
// and, by convention, an empty BaseURL), so the enforceable binding is the
// provider; the region is fixed at construction.
func (c *Client) checkBinding(m model.Model) error {
	if llm.Provider(m.Provider) != llm.ProviderBedrock {
		return &failure.ModelMismatchError{
			BoundProvider:   model.ProviderName(llm.ProviderBedrock),
			RequestProvider: m.Provider,
			BoundEndpoint:   c.endpoint,
			RequestEndpoint: m.BaseURL,
		}
	}
	return nil
}

// buildRequest constructs the ctx-bound POST to
// <endpoint>/model/<escaped model id>/invoke with the JSON content/accept headers.
// The model id is path-escaped so it is a single URL path segment; its ":" stays
// literal on the wire (url.PathEscape does not escape ":"), and the SigV4 signer
// then double-encodes it into the canonical URI ("%3A"). Headers are set before
// signing so they are covered by the signature.
func (c *Client) buildRequest(ctx context.Context, modelID string, body []byte) (*http.Request, error) {
	return buildRuntimeRequest(ctx, c.endpoint, modelID, pathInvokeSuffix, body)
}

func buildRuntimeRequest(ctx context.Context, endpoint, modelID, suffix string, body []byte) (*http.Request, error) {
	return buildRuntimeRequestWithAccept(ctx, endpoint, modelID, suffix, body, contentTypeJSON)
}

func buildRuntimeRequestWithAccept(ctx context.Context, endpoint, modelID, suffix string, body []byte, accept string) (*http.Request, error) {
	rawURL := endpoint + pathModelPrefix + url.PathEscape(modelID) + suffix
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, rawURL, bytes.NewReader(body))
	if err != nil {
		return nil, &RequestBuildError{Err: err}
	}
	httpReq.Header.Set("Content-Type", contentTypeJSON)
	httpReq.Header.Set("Accept", accept)
	return httpReq, nil
}
