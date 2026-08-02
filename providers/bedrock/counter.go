package bedrock

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
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	inferauth "github.com/looprig/inference/auth"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	contextcount "github.com/looprig/inference/contextcount"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	usage "github.com/looprig/inference/usage"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auth"
)

const (
	counterTokenizerRevision      contextcount.TokenizerRevision = "aws-bedrock-count-tokens-2023-09-30-invoke-model-v1" // #nosec G101 -- public tokenizer revision identifier, not a credential
	bedrockSecurityPolicyRevision                                = "aws-bedrock-runtime-sigv4-tls-v1"
	maxCountResponseBodyBytes                                    = 64 << 10
	maxBedrockModelIDBytes                                       = 256
	maxInvokeModelTokensBodyBytes int                            = 25_000_000
	defaultCounterTimeout                                        = 60 * time.Second
)

// Counter is a separately constructed exact Bedrock CountTokens client. Keeping
// it separate from Client prevents ordinary inference composition from
// accidentally acquiring the optional ContextCounter capability.
type Counter struct {
	region      string
	endpoint    string
	endpointErr *CounterEndpointError
	signer      inferauth.Authenticator
	hc          requestDoer
	timeout     time.Duration
	options     config
}

var _ contextcount.ContextCounter = (*Counter)(nil)

type requestDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// NewCounter constructs an exact provider counter bound to one AWS region.
func NewCounter(creds auth.SigV4Credentials, region string, options ...Option) (contextcount.ContextCounter, error) {
	if err := validateConfig(creds, region); err != nil {
		return nil, err
	}
	counter := newCounter(creds, region, defaultEndpoint(region), options...)
	if counter.endpointErr != nil {
		return nil, counter.endpointErr
	}
	return counter, nil
}

func newCounter(creds auth.SigV4Credentials, region, endpoint string, options ...Option) *Counter {
	canonical, endpointErr := canonicalCounterEndpoint(endpoint)
	cfg := config{}
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	return &Counter{
		region:      region,
		endpoint:    canonical,
		endpointErr: endpointErr,
		signer:      auth.SigV4(creds, region, bedrockService),
		hc:          newHTTPClient(),
		timeout:     defaultCounterTimeout,
		options:     cfg.clone(),
	}
}

// CountContext sends the byte-identical InvokeModel body in AWS's base64 binary
// CountTokens envelope. Unsupported models remain provider API errors; this
// exact counter never falls back to an estimate.
func (c *Counter) CountContext(ctx context.Context, req inference.Request) (contextcount.ContextCount, error) {
	if err := c.validateState(ctx); err != nil {
		return contextcount.ContextCount{}, err
	}
	counterCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, err := c.preflight(req)
	if err != nil {
		return contextcount.ContextCount{}, err
	}
	httpReq, err := buildRuntimeRequest(counterCtx, c.endpoint, req.Model.Name, pathCountSuffix, body)
	if err != nil {
		return contextcount.ContextCount{}, err
	}
	if err := c.signer.Authorize(counterCtx, httpReq); err != nil {
		return contextcount.ContextCount{}, err
	}
	count, err := c.do(httpReq)
	if err != nil {
		return contextcount.ContextCount{}, err
	}
	return contextcount.ContextCount{
		Model:       req.Model.Key(),
		InputTokens: count,
		Quality:     contextcount.CountQualityExactProvider,
	}, nil
}

func (c *Counter) validateState(ctx context.Context) error {
	if c == nil {
		return &CounterStateError{Reason: CounterStateNilReceiver}
	}
	if ctx == nil {
		return &CounterStateError{Reason: CounterStateNilContext}
	}
	return c.validateConfiguration()
}

func (c *Counter) validateConfiguration() error {
	if c.endpointErr != nil {
		return c.endpointErr
	}
	if c.endpoint == "" {
		return &CounterStateError{Reason: CounterStateMissingEndpoint}
	}
	if c.region == "" {
		return &CounterStateError{Reason: CounterStateMissingRegion}
	}
	if c.signer == nil {
		return &CounterStateError{Reason: CounterStateMissingAuthenticator}
	}
	if c.hc == nil {
		return &CounterStateError{Reason: CounterStateMissingHTTPDoer}
	}
	if c.timeout <= 0 {
		return &CounterStateError{Reason: CounterStateInvalidTimeout}
	}
	return nil
}

func (c *Counter) preflight(req inference.Request) ([]byte, error) {
	if err := c.checkBinding(req.Model); err != nil {
		return nil, err
	}
	if err := llm.ValidateModel(req.Model); err != nil {
		return nil, err
	}
	if !validBedrockModelID(req.Model.Name) {
		return nil, &model.ValidationError{
			Field:  "Name",
			Reason: "Bedrock model id must be 1-256 ASCII letters, digits, underscore, dot, hyphen, slash, or colon",
		}
	}
	switch req.Model.APIFormat {
	case model.APIFormatAnthropic:
		encoded, err := anthropicapi.EncodeRequest(req, false)
		if err != nil {
			return nil, err
		}
		invokeBody, err := toBedrockBody(encoded)
		if err != nil {
			return nil, err
		}
		return buildCountRequestEnvelope(invokeBody)
	case model.APIFormatBedrockConverse:
		converseBody, err := bedrockconverse.EncodeCountTokensInput(req)
		if err != nil {
			return nil, err
		}
		converseBody, err = c.options.applyConverseCountTokens(converseBody)
		if err != nil {
			return nil, err
		}
		return buildConverseCountRequestEnvelope(converseBody)
	default:
		return nil, &UnsupportedAPIFormatError{APIFormat: req.Model.APIFormat}
	}
}

func buildCountRequestEnvelope(invokeBody []byte) ([]byte, error) {
	if len(invokeBody) > maxInvokeModelTokensBodyBytes {
		return nil, &CounterRequestError{Reason: CounterRequestBodyTooLarge}
	}
	body, err := json.Marshal(countTokensRequest{
		Input: countTokensInput{
			InvokeModel: &invokeModelTokensRequest{Body: invokeBody},
		},
	})
	if err != nil {
		return nil, &CounterRequestError{Reason: CounterRequestEnvelopeEncoding, Err: err}
	}
	return body, nil
}

func buildConverseCountRequestEnvelope(converseBody []byte) ([]byte, error) {
	if len(converseBody) > maxInvokeModelTokensBodyBytes {
		return nil, &CounterRequestError{Reason: CounterRequestBodyTooLarge}
	}
	body, err := json.Marshal(countTokensRequest{
		Input: countTokensInput{Converse: json.RawMessage(converseBody)},
	})
	if err != nil {
		return nil, &CounterRequestError{Reason: CounterRequestEnvelopeEncoding, Err: err}
	}
	return body, nil
}

func (c *Counter) checkBinding(m model.Model) error {
	if llm.Provider(m.Provider) == llm.ProviderBedrock && m.BaseURL == "" {
		return nil
	}
	return &failure.ModelMismatchError{
		BoundProvider:   model.ProviderName(llm.ProviderBedrock),
		RequestProvider: m.Provider,
		BoundEndpoint:   c.endpoint,
		RequestEndpoint: m.BaseURL,
	}
}

func validBedrockModelID(modelID string) bool {
	if len(modelID) == 0 || len(modelID) > maxBedrockModelIDBytes {
		return false
	}
	for _, char := range modelID {
		if char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' {
			continue
		}
		switch char {
		case '_', '.', '-', '/', ':':
			continue
		default:
			return false
		}
	}
	return true
}

type countTokensRequest struct {
	Input countTokensInput `json:"input"`
}

type countTokensInput struct {
	InvokeModel *invokeModelTokensRequest `json:"invokeModel,omitempty"`
	Converse    json.RawMessage           `json:"converse,omitempty"`
}

type invokeModelTokensRequest struct {
	Body []byte `json:"body"`
}

func (c *Counter) do(req *http.Request) (content.TokenCount, error) {
	response, err := c.hc.Do(req)
	if err != nil {
		return 0, &failure.NetworkError{Err: err}
	}
	if response == nil || response.Body == nil {
		return 0, &CounterResponseError{Reason: CounterResponseMalformed}
	}
	defer response.Body.Close()
	body, tooLarge, err := readCountResponseBody(response.Body)
	if err != nil {
		return 0, &failure.NetworkError{Err: err}
	}
	if response.StatusCode/100 != 2 {
		return 0, &failure.APIError{
			Status:  response.StatusCode,
			Message: http.StatusText(response.StatusCode),
			Body:    body,
		}
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
		return body[:maxCountResponseBodyBytes], true, nil
	}
	return body, false, nil
}

type countTokensResponse struct {
	InputTokens countScalar `json:"inputTokens"`
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
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) < 2 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return 0, &CounterResponseError{Reason: CounterResponseMalformed}
	}
	if err := rejectDuplicateCount(body); err != nil {
		return 0, &CounterResponseError{Reason: CounterResponseDuplicateField, Err: err}
	}
	var response countTokensResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return 0, &CounterResponseError{Reason: CounterResponseMalformed, Err: err}
	}
	if !response.InputTokens.present {
		return 0, &CounterResponseError{Reason: CounterResponseMissingCount}
	}
	count, err := response.InputTokens.tokenCount()
	if err != nil {
		return 0, &CounterResponseError{Reason: CounterResponseInvalidCount, Err: err}
	}
	return count, nil
}

func rejectDuplicateCount(body []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(body))
	if _, err := decoder.Token(); err != nil {
		return nil
	}
	seen := false
	for decoder.More() {
		nameToken, err := decoder.Token()
		if err != nil {
			return nil
		}
		name, ok := nameToken.(string)
		if !ok {
			return nil
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil
		}
		if !strings.EqualFold(name, "inputTokens") {
			continue
		}
		if seen {
			return &CounterResponseError{Reason: CounterResponseDuplicateField}
		}
		seen = true
	}
	return nil
}

func (c countScalar) tokenCount() (content.TokenCount, error) {
	raw := bytes.TrimSpace(c.raw)
	if bytes.Equal(raw, []byte("null")) {
		return 0, countNormalizationError(usage.UsageNormalizationReasonNull)
	}
	if !countNumber(raw) {
		return 0, countNormalizationError(usage.UsageNormalizationReasonInvalidType)
	}
	if bytes.ContainsAny(raw, ".eE") {
		return 0, countNormalizationError(usage.UsageNormalizationReasonFractional)
	}
	value, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		if errors.Is(err, strconv.ErrRange) {
			return 0, countNormalizationError(usage.UsageNormalizationReasonOutOfRange)
		}
		return 0, countNormalizationError(usage.UsageNormalizationReasonInvalidType)
	}
	if value < 0 {
		return 0, &usage.UsageNormalizationError{
			Field:  usage.UsageNormalizationFieldInputTokens,
			Reason: usage.UsageNormalizationReasonNegative,
			Value:  value,
		}
	}
	return content.TokenCount(value), nil
}

func countNumber(raw []byte) bool {
	return len(raw) > 0 && (raw[0] == '-' || raw[0] >= '0' && raw[0] <= '9')
}

func countNormalizationError(reason usage.UsageNormalizationReason) error {
	return &usage.UsageNormalizationError{
		Field:  usage.UsageNormalizationFieldInputTokens,
		Reason: reason,
	}
}

// CounterCapability declares the CountTokens endpoint as the same region-routed
// Bedrock Runtime security boundary as InvokeModel.
func (c *Counter) CounterCapability() contextcount.CounterCapability {
	if c == nil || c.validateConfiguration() != nil {
		return contextcount.CounterCapability{}
	}
	return contextcount.CounterCapability{
		Provider:         contextcount.ProviderID(llm.ProviderBedrock),
		Transport:        contextcount.CounterTransportSameEndpoint,
		SecurityIdentity: counterSecurityIdentity(c.region, c.endpoint),
		Retention:        contextcount.RetentionLogged,
		TokenizerRev:     counterTokenizerRevision,
		Quality:          contextcount.CountQualityExactProvider,
	}
}

func counterSecurityIdentity(region, endpoint string) contextcount.SecurityIdentity {
	material := "provider=bedrock\nregion=" + region + "\nendpoint=" + endpoint + "\nauth=sigv4\ntransport=tls\npolicy=" + bedrockSecurityPolicyRevision
	return contextcount.SecurityIdentity(sha256.Sum256([]byte(material)))
}

func canonicalCounterEndpoint(endpoint string) (string, *CounterEndpointError) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", counterEndpointError(CounterEndpointMalformed)
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.User != nil {
		return "", counterEndpointError(CounterEndpointCredentials)
	}
	if parsed.Host == "" || parsed.Hostname() == "" {
		return "", counterEndpointError(CounterEndpointMissingHost)
	}
	if parsed.Path != "" && parsed.Path != "/" || parsed.RawPath != "" || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return "", counterEndpointError(CounterEndpointUnexpectedComponent)
	}
	host, endpointErr := canonicalEndpointHost(parsed.Hostname())
	if endpointErr != nil {
		return "", endpointErr
	}
	if transportErr := validateCounterTransport(parsed.Scheme, host); transportErr != nil {
		return "", transportErr
	}
	if !validEndpointPort(parsed.Port()) {
		return "", counterEndpointError(CounterEndpointMalformed)
	}
	parsed.Host = canonicalURLHost(host, parsed.Scheme, parsed.Port())
	parsed.Path = ""
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	return parsed.String(), nil
}

func validateCounterTransport(scheme, host string) *CounterEndpointError {
	switch scheme {
	case "https":
		return nil
	case "http":
		if isLoopbackHost(host) {
			return nil
		}
		return counterEndpointError(CounterEndpointInsecureTransport)
	default:
		return counterEndpointError(CounterEndpointUnsupportedScheme)
	}
}

func canonicalEndpointHost(host string) (string, *CounterEndpointError) {
	for _, char := range host {
		if char > 127 {
			return "", counterEndpointError(CounterEndpointNonASCIIHost)
		}
	}
	if strings.Contains(host, "%") {
		return "", counterEndpointError(CounterEndpointInvalidHost)
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.String(), nil
	}
	if strings.Contains(host, ":") {
		return "", counterEndpointError(CounterEndpointInvalidHost)
	}
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	if !validDNSHost(host) {
		return "", counterEndpointError(CounterEndpointInvalidHost)
	}
	return host, nil
}

func validDNSHost(host string) bool {
	if host == "" || len(host) > 253 || strings.Contains(host, "..") {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, char := range label {
			if (char < 'a' || char > 'z') && (char < '0' || char > '9') && char != '-' {
				return false
			}
		}
	}
	return true
}

func canonicalURLHost(host, scheme, port string) string {
	if scheme == "https" && port == "443" || scheme == "http" && port == "80" {
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
