package bedrock

import (
	"fmt"

	"github.com/looprig/inference"
)

// UnsupportedAPIFormatError is a fail-closed rejection, before any I/O, of a
// request whose Model.APIFormat this client cannot honor. Provider.supportsAPIFormat
// admits both APIFormatAnthropic and APIFormatBedrockConverse for Bedrock (so a
// Converse Model passes the model-validation preset), but this client implements
// only the Anthropic-native dialect for now; a Converse codec is a documented
// follow-up. Returning this — rather than silently Anthropic-encoding a Converse
// request — keeps the client from "silently doing less" than its declared contract.
// Carries the offending format so callers can branch via errors.As.
type UnsupportedAPIFormatError struct {
	APIFormat inference.APIFormat
}

func (e *UnsupportedAPIFormatError) Error() string {
	return fmt.Sprintf("bedrock: API format %q is not implemented; this client encodes only the Anthropic dialect (%q)", e.APIFormat, inference.APIFormatAnthropic)
}

// RequestBuildError is a failure to CONSTRUCT the outbound HTTP request (a
// malformed endpoint/URL), kept distinct from *inference.NetworkError (reserved for
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

// StreamingNotSupportedError is returned by Client.Stream: Bedrock streaming uses
// the AWS eventstream (application/vnd.amazon.eventstream) framing, which is a
// documented follow-up and is not yet implemented. Fail-closed: no stream is
// opened. A typed error so a caller can branch (errors.As) and fall back to Invoke.
type StreamingNotSupportedError struct{}

func (*StreamingNotSupportedError) Error() string {
	return "bedrock: streaming (AWS eventstream) is not yet implemented; use Invoke"
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

// CounterRequestError wraps an unexpected local failure to encode the fixed
// InvokeModel CountTokens envelope. It never retains request bytes.
type CounterRequestError struct {
	Err error
}

func (e *CounterRequestError) Error() string {
	return "bedrock: encode CountTokens request envelope: " + e.Err.Error()
}

func (e *CounterRequestError) Unwrap() error { return e.Err }

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
