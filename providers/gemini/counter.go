package gemini

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
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
	endpoint string
	auth     inference.Authenticator
	hc       requestDoer
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
	return newCounter(key, defaultBaseURL), nil
}

func newCounter(key auth.APIKey, endpoint string) *Counter {
	return &Counter{
		endpoint: endpoint,
		auth:     auth.Header(key, apiKeyHeader),
		hc:       newHTTPClient(),
	}
}

// CountContext sends the complete encoded inference request to countTokens.
func (c *Counter) CountContext(ctx context.Context, req inference.Request) (inference.ContextCount, error) {
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
	return geminicodec.EncodeRequest(req)
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
	if c == nil {
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
	canonical := canonicalCounterEndpoint(endpoint)
	material := "provider=google\nendpoint=" + canonical + "\nauth=x-goog-api-key\ntransport=tls\npolicy=" + googleSecurityPolicyRevision
	return inference.SecurityIdentity(sha256.Sum256([]byte(material)))
}

func canonicalCounterEndpoint(endpoint string) string {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return strings.TrimRight(endpoint, "/")
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	parsed.Host = strings.ToLower(parsed.Host)
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}
