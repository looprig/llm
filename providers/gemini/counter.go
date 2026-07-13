package gemini

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	geminicodec "github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/llm"
)

const (
	// counterTokenizerRevision pins both the provider count method and the API
	// revision whose request encoding is used.
	counterTokenizerRevision inference.TokenizerRevision = "google-gemini-countTokens-v1beta"
	// googleSecurityPolicyRevision identifies the provider endpoint's
	// secret-free transport/auth policy. It deliberately excludes the RPC method,
	// tokenizer, credential, and retention metadata so inference and counting on
	// the same endpoint can share one security identity.
	googleSecurityPolicyRevision = "google-generative-language-api-key-tls-v1"
	maxCountResponseBodyBytes    = 64 << 10
)

// Counter is a separately constructed Gemini countTokens client. It is not
// embedded in Client so an inference client can never acquire the optional
// ContextCounter capability accidentally.
type Counter struct {
	endpoint    string
	endpointErr *CounterEndpointError
	auth        inference.Authenticator
	hc          requestDoer
}

var _ inference.ContextCounter = (*Counter)(nil)

type requestDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// NewCounter constructs an exact provider counter authenticated with key.
func NewCounter(key auth.APIKey) (inference.ContextCounter, error) {
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderGoogle, Kind: inference.AuthAPIKey}
	}
	counter := newCounter(key, defaultBaseURL)
	if counter.endpointErr != nil {
		return nil, counter.endpointErr
	}
	return counter, nil
}

func newCounter(key auth.APIKey, endpoint string) *Counter {
	canonical, endpointErr := canonicalCounterEndpoint(endpoint)
	return &Counter{
		endpoint:    canonical,
		endpointErr: endpointErr,
		auth:        auth.Header(key, apiKeyHeader),
		hc:          newHTTPClient(),
	}
}

// CountContext sends the complete encoded inference request to countTokens.
func (c *Counter) CountContext(ctx context.Context, req inference.Request) (inference.ContextCount, error) {
	if c.endpointErr != nil {
		return inference.ContextCount{}, c.endpointErr
	}
	body, err := c.preflight(req)
	if err != nil {
		return inference.ContextCount{}, err
	}
	httpReq, err := buildRequest(ctx, c.endpoint, req.Model.Name, methodCountTokens, "", body)
	if err != nil {
		return inference.ContextCount{}, err
	}
	httpReq.Header.Set("Accept", contentTypeJSON)
	if err := c.auth.Authorize(ctx, httpReq); err != nil {
		return inference.ContextCount{}, err
	}
	count, err := c.do(httpReq)
	if err != nil {
		return inference.ContextCount{}, err
	}
	return inference.ContextCount{Model: req.Model.Key(), InputTokens: count, Quality: inference.CountQualityExactProvider}, nil
}

func (c *Counter) preflight(req inference.Request) ([]byte, error) {
	if llm.Provider(req.Model.Provider) != llm.ProviderGoogle {
		return nil, &inference.ModelMismatchError{
			BoundProvider: inference.ProviderName(llm.ProviderGoogle), RequestProvider: req.Model.Provider,
			BoundEndpoint: c.endpoint, RequestEndpoint: req.Model.BaseURL,
		}
	}
	if err := llm.ValidateModel(req.Model); err != nil {
		return nil, err
	}
	if req.Model.APIFormat != inference.APIFormatGemini {
		return nil, &UnsupportedAPIFormatError{APIFormat: req.Model.APIFormat}
	}
	generateBody, err := geminicodec.EncodeRequest(req)
	if err != nil {
		return nil, err
	}
	return wrapGenerateContentRequest(req.Model.Name, generateBody)
}

func wrapGenerateContentRequest(modelName string, generateBody []byte) ([]byte, error) {
	object := bytes.TrimSpace(generateBody)
	if !json.Valid(object) || len(object) < 2 || object[0] != '{' || object[len(object)-1] != '}' {
		return nil, &CounterRequestError{Reason: CounterRequestGenerateBodyInvalid}
	}
	model, err := json.Marshal("models/" + modelName)
	if err != nil {
		return nil, &CounterRequestError{Reason: CounterRequestModelEncodingFailed, Err: err}
	}
	const prefix = `{"generateContentRequest":{"model":`
	body := make([]byte, 0, len(prefix)+len(model)+len(object)+2)
	body = append(body, prefix...)
	body = append(body, model...)
	if len(bytes.TrimSpace(object[1:len(object)-1])) > 0 {
		body = append(body, ',')
	}
	body = append(body, object[1:]...)
	body = append(body, '}')
	return body, nil
}

func (c *Counter) do(req *http.Request) (content.TokenCount, error) {
	response, err := c.hc.Do(req)
	if err != nil {
		return 0, &inference.NetworkError{Err: err}
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return 0, providerAPIError(response)
	}
	body, tooLarge, err := readCountResponseBody(response.Body)
	if err != nil {
		return 0, &inference.NetworkError{Err: err}
	}
	if tooLarge {
		return 0, &CounterResponseError{Reason: CounterResponseBodyTooLarge}
	}
	return decodeCountResponse(body)
}

func readCountResponseBody(reader io.Reader) ([]byte, bool, error) {
	body, err := io.ReadAll(io.LimitReader(reader, maxCountResponseBodyBytes+1))
	if err != nil {
		return nil, false, err
	}
	if len(body) > maxCountResponseBodyBytes {
		return nil, true, nil
	}
	return body, false, nil
}

type countTokensResponse struct {
	TotalTokens countScalar `json:"totalTokens"`
}

type countScalar struct {
	present bool
	raw     []byte
}

func (c *countScalar) UnmarshalJSON(data []byte) error {
	c.present = true
	c.raw = append(c.raw[:0], data...)
	return nil
}

func decodeCountResponse(body []byte) (content.TokenCount, error) {
	var response countTokensResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return 0, &CounterResponseError{Reason: CounterResponseMalformed, Err: err}
	}
	if !response.TotalTokens.present {
		return 0, &CounterResponseError{Reason: CounterResponseMissingCount}
	}
	count, err := response.TotalTokens.tokenCount()
	if err != nil {
		return 0, &CounterResponseError{Reason: CounterResponseInvalidCount, Err: err}
	}
	return count, nil
}

func (c countScalar) tokenCount() (content.TokenCount, error) {
	raw := bytes.TrimSpace(c.raw)
	if bytes.Equal(raw, []byte("null")) {
		return 0, countNormalizationError(inference.UsageNormalizationReasonNull)
	}
	if !countNumber(raw) {
		return 0, countNormalizationError(inference.UsageNormalizationReasonInvalidType)
	}
	if bytes.ContainsAny(raw, ".eE") {
		return 0, countNormalizationError(inference.UsageNormalizationReasonFractional)
	}
	value, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		if errors.Is(err, strconv.ErrRange) {
			return 0, countNormalizationError(inference.UsageNormalizationReasonOutOfRange)
		}
		return 0, countNormalizationError(inference.UsageNormalizationReasonInvalidType)
	}
	if value < 0 {
		return 0, &inference.UsageNormalizationError{Field: inference.UsageNormalizationFieldInputTokens, Reason: inference.UsageNormalizationReasonNegative, Value: value}
	}
	return content.TokenCount(value), nil
}

func countNumber(raw []byte) bool {
	return len(raw) > 0 && (raw[0] == '-' || raw[0] >= '0' && raw[0] <= '9')
}

func countNormalizationError(reason inference.UsageNormalizationReason) error {
	return &inference.UsageNormalizationError{Field: inference.UsageNormalizationFieldInputTokens, Reason: reason}
}

// CounterCapability declares countTokens as exact provider counting over the
// same Google API endpoint, conservatively allowing provider logging.
func (c *Counter) CounterCapability() inference.CounterCapability {
	if c == nil || c.endpointErr != nil {
		return inference.CounterCapability{}
	}
	return inference.CounterCapability{
		Provider:         inference.ProviderID(llm.ProviderGoogle),
		Transport:        inference.CounterTransportSameEndpoint,
		SecurityIdentity: counterSecurityIdentity(c.endpoint),
		Retention:        inference.RetentionLogged,
		TokenizerRev:     counterTokenizerRevision,
		Quality:          inference.CountQualityExactProvider,
	}
}

func counterSecurityIdentity(endpoint string) inference.SecurityIdentity {
	material := "provider=google\nendpoint=" + endpoint + "\nauth=x-goog-api-key\ntransport=tls\npolicy=" + googleSecurityPolicyRevision
	return inference.SecurityIdentity(sha256.Sum256([]byte(material)))
}

func canonicalCounterEndpoint(endpoint string) (string, *CounterEndpointError) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", counterEndpointError(CounterEndpointMalformed)
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.Scheme == "" {
		return "", counterEndpointError(CounterEndpointMalformed)
	}
	if parsed.User != nil {
		return "", counterEndpointError(CounterEndpointCredentials)
	}
	if parsed.Host == "" || parsed.Hostname() == "" {
		return "", counterEndpointError(CounterEndpointMissingHost)
	}
	switch parsed.Scheme {
	case "https":
	case "http":
		if !isLoopbackHost(parsed.Hostname()) {
			return "", counterEndpointError(CounterEndpointInsecureTransport)
		}
	default:
		return "", counterEndpointError(CounterEndpointUnsupportedScheme)
	}
	if !validEndpointPort(parsed.Port()) {
		return "", counterEndpointError(CounterEndpointMalformed)
	}
	parsed.Host = canonicalHost(parsed)
	parsed.User = nil
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	return parsed.String(), nil
}

func canonicalHost(parsed *url.URL) string {
	host := strings.ToLower(parsed.Hostname())
	port := parsed.Port()
	if (parsed.Scheme == "https" && port == "443") || (parsed.Scheme == "http" && port == "80") {
		port = ""
	}
	if port != "" {
		return net.JoinHostPort(host, port)
	}
	if strings.Contains(host, ":") {
		return "[" + host + "]"
	}
	return host
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func validEndpointPort(port string) bool {
	if port == "" {
		return true
	}
	value, err := strconv.ParseUint(port, 10, 16)
	return err == nil && value != 0
}

func counterEndpointError(reason CounterEndpointReason) *CounterEndpointError {
	return &CounterEndpointError{Reason: reason}
}
