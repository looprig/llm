package bedrock

import (
	"fmt"

	model "github.com/looprig/inference/model"
)

// UnsupportedAPIFormatError is a fail-closed rejection, before any I/O, of a
// request whose Model.APIFormat this client cannot honor. Bedrock currently
// supports the Anthropic-on-Bedrock InvokeModel dialect and native Converse;
// this error protects future/unknown formats from accidental fallback.
type UnsupportedAPIFormatError struct {
	APIFormat model.APIFormat
}

func (e *UnsupportedAPIFormatError) Error() string {
	return fmt.Sprintf("bedrock: API format %q is not implemented", e.APIFormat)
}

// RequestBuildError is a failure to CONSTRUCT the outbound HTTP request (a
// malformed endpoint/URL), kept distinct from *failure.NetworkError (reserved for
// transport failures out of hc.Do) so errors.As never misclassifies a config bug
// as a transport fault. Unwrap exposes the net/http cause.
type RequestBuildError struct {
	Err error
}

func (e *RequestBuildError) Error() string { return "bedrock: build request: " + e.Err.Error() }
func (e *RequestBuildError) Unwrap() error { return e.Err }

// ConfigError is a fail-closed rejection of an invalid bedrock.New configuration:
// an empty AWS region or empty SigV4 credentials. No Client is returned and no
// network object is created. Field names the offending input; Reason explains the
// constraint. Carries no secret (never the credential values themselves).
type ConfigError struct {
	Field  string
	Reason string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("bedrock: invalid config: %s: %s", e.Field, e.Reason)
}

// BodyTransformError wraps a failure to turn the Anthropic Messages body into the
// Bedrock InvokeModel body (unmarshal, field rewrite, or re-marshal). It is kept
// distinct from transport/API errors so a caller can tell a local encode fault
// from a wire fault. Err is the underlying cause.
type BodyTransformError struct {
	Err error
}

func (e *BodyTransformError) Error() string {
	return "bedrock: transform request body: " + e.Err.Error()
}

func (e *BodyTransformError) Unwrap() error { return e.Err }

// UnsupportedImageSourceError is a fail-closed rejection, before any I/O, of an
// Anthropic image block whose `source.type` Bedrock's InvokeModel contract cannot
// honor. Anthropic's first-party API accepts a remote {"type":"url"} source, so
// the shared anthropicapi encoder emits one, but Bedrock takes inline bytes only
// (its Converse ImageSource union is bytes | s3Location, and the Anthropic-on-
// Bedrock body accepts "base64"); the URL would otherwise reach Bedrock and draw
// an opaque HTTP 400. SourceType names the rejected discriminator and never the
// URL itself, so the error carries no caller content.
type UnsupportedImageSourceError struct {
	SourceType string
}

func (e *UnsupportedImageSourceError) Error() string {
	return fmt.Sprintf("bedrock: unsupported image source %q; Bedrock InvokeModel accepts inline base64 image bytes", e.SourceType)
}

// ThinkingBudgetError is a fail-closed rejection, before any I/O, of a Converse
// request whose reasoning budget is not smaller than its output cap. Anthropic
// states the rule as `max_tokens` must be greater than `thinking.budget_tokens`
// and enforces it with an HTTP 400; Anthropic-on-Bedrock inherits it.
//
// It is caught here rather than by the request schema because the two values
// live in different top-level objects and one of them is opaque:
// inferenceConfig.maxTokens is modelled, additionalModelRequestFields is a
// Smithy Document whose contents no schema constrains. Both numbers are carried
// on the error because a message that named only one of them would not say what
// to change.
type ThinkingBudgetError struct {
	MaxTokens    int
	BudgetTokens int
}

func (e *ThinkingBudgetError) Error() string {
	return fmt.Sprintf("bedrock: inferenceConfig.maxTokens (%d) must be greater than thinking.budget_tokens (%d)",
		e.MaxTokens, e.BudgetTokens)
}

// StreamingNotSupportedError is returned only for Anthropic-on-Bedrock
// InvokeModel requests. Native ConverseStream uses the event-stream codec.
type StreamingNotSupportedError struct{}

func (*StreamingNotSupportedError) Error() string {
	return "bedrock: Anthropic InvokeModel streaming is not supported; use native Bedrock ConverseStream"
}

// CounterStateReason classifies an unusable CountTokens counter boundary.
type CounterStateReason string

const (
	CounterStateNilReceiver          CounterStateReason = "nil counter"
	CounterStateNilContext           CounterStateReason = "nil context"
	CounterStateMissingEndpoint      CounterStateReason = "missing endpoint"
	CounterStateMissingRegion        CounterStateReason = "missing region"
	CounterStateMissingAuthenticator CounterStateReason = "missing authenticator"
	CounterStateMissingHTTPDoer      CounterStateReason = "missing HTTP doer"
	CounterStateInvalidTimeout       CounterStateReason = "invalid timeout"
)

// CounterStateError rejects invalid local state before encoding or I/O.
type CounterStateError struct {
	Reason CounterStateReason
}

func (e *CounterStateError) Error() string {
	return "bedrock: invalid CountTokens counter state: " + string(e.Reason)
}

// CounterEndpointReason classifies an endpoint that cannot safely receive a
// complete request. It deliberately carries no rejected endpoint text.
type CounterEndpointReason string

const (
	CounterEndpointMalformed           CounterEndpointReason = "malformed endpoint"
	CounterEndpointMissingHost         CounterEndpointReason = "missing endpoint host"
	CounterEndpointCredentials         CounterEndpointReason = "endpoint contains credentials"
	CounterEndpointUnsupportedScheme   CounterEndpointReason = "unsupported endpoint scheme"
	CounterEndpointInsecureTransport   CounterEndpointReason = "plaintext endpoint is not loopback"
	CounterEndpointNonASCIIHost        CounterEndpointReason = "endpoint host is not ASCII"
	CounterEndpointInvalidHost         CounterEndpointReason = "invalid endpoint host"
	CounterEndpointUnexpectedComponent CounterEndpointReason = "endpoint contains an unexpected path, query, or fragment"
)

// CounterEndpointError rejects unsafe routing without retaining credentials or
// provider input from the raw endpoint.
type CounterEndpointError struct {
	Reason CounterEndpointReason
}

func (e *CounterEndpointError) Error() string {
	return "bedrock: invalid CountTokens endpoint: " + string(e.Reason)
}

// CounterRequestReason classifies a local CountTokens request-envelope failure.
type CounterRequestReason string

const (
	CounterRequestBodyTooLarge     CounterRequestReason = "InvokeModel body exceeds 25000000 bytes"
	CounterRequestEnvelopeEncoding CounterRequestReason = "envelope encoding failed"
)

// CounterRequestError reports a local CountTokens envelope failure. It never
// retains request bytes, model ids, or the rejected body length.
type CounterRequestError struct {
	Reason CounterRequestReason
	Err    error
}

func (e *CounterRequestError) Error() string {
	if e == nil || e.Reason == "" {
		return "bedrock: encode CountTokens request envelope: unknown failure"
	}
	if e.Err == nil {
		return "bedrock: encode CountTokens request envelope: " + string(e.Reason)
	}
	return "bedrock: encode CountTokens request envelope: " + string(e.Reason) + ": " + e.Err.Error()
}

func (e *CounterRequestError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// CounterResponseReason classifies a successful response that cannot produce a
// trustworthy normalized input-token count.
type CounterResponseReason string

const (
	CounterResponseMalformed      CounterResponseReason = "malformed response"
	CounterResponseMissingCount   CounterResponseReason = "missing inputTokens"
	CounterResponseInvalidCount   CounterResponseReason = "invalid inputTokens"
	CounterResponseDuplicateField CounterResponseReason = "duplicate inputTokens"
	CounterResponseBodyTooLarge   CounterResponseReason = "response body too large"
)

// CounterResponseError omits provider-controlled bytes from Error while its
// typed cause remains available through errors.As.
type CounterResponseError struct {
	Reason CounterResponseReason
	Err    error
}

func (e *CounterResponseError) Error() string {
	if e.Err == nil {
		return "bedrock: CountTokens response: " + string(e.Reason)
	}
	return "bedrock: CountTokens response: " + string(e.Reason) + ": " + e.Err.Error()
}

func (e *CounterResponseError) Unwrap() error { return e.Err }
