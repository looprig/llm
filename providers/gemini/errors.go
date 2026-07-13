package gemini

import (
	"fmt"

	"github.com/looprig/inference"
)

// CounterResponseReason classifies a countTokens response that cannot produce a
// trustworthy normalized input-token count.
type CounterResponseReason string

const (
	CounterResponseMalformed    CounterResponseReason = "malformed response"
	CounterResponseMissingCount CounterResponseReason = "missing totalTokens"
	CounterResponseInvalidCount CounterResponseReason = "invalid totalTokens"
	CounterResponseBodyTooLarge CounterResponseReason = "response body too large"
)

// CounterResponseError reports an invalid successful countTokens response.
// Provider payload bytes are deliberately omitted so Error never leaks request
// or response data; Err carries only a typed/safe parsing cause.
type CounterResponseError struct {
	Reason CounterResponseReason
	Err    error
}

func (e *CounterResponseError) Error() string {
	if e.Err != nil {
		return "gemini: countTokens response: " + string(e.Reason) + ": " + e.Err.Error()
	}
	return "gemini: countTokens response: " + string(e.Reason)
}

func (e *CounterResponseError) Unwrap() error { return e.Err }

// UnsupportedAPIFormatError is a fail-closed rejection, before any I/O, of a
// request whose Model.APIFormat this client cannot honor. This client encodes only
// the Gemini generateContent dialect. Provider.supportsAPIFormat currently admits
// only APIFormatGemini for ProviderGoogle, so a ValidateModel-passing Google Model
// can never reach this guard — it is defense-in-depth (Open/Closed): should a second
// Google dialect ever be admitted upstream, this keeps the client from silently
// Gemini-encoding a request it does not understand. Carries the offending format so
// callers can branch via errors.As.
type UnsupportedAPIFormatError struct {
	APIFormat inference.APIFormat
}

func (e *UnsupportedAPIFormatError) Error() string {
	return fmt.Sprintf("gemini: API format %q is not implemented; this client encodes only the Gemini dialect (%q)", e.APIFormat, inference.APIFormatGemini)
}

// RequestBuildError is a failure to CONSTRUCT the outbound HTTP request (a
// malformed endpoint/URL), kept distinct from *inference.NetworkError (reserved for
// transport failures out of hc.Do) so errors.As never misclassifies a config bug
// as a transport fault. Unwrap exposes the net/http cause.
type RequestBuildError struct {
	Err error
}

func (e *RequestBuildError) Error() string { return "gemini: build request: " + e.Err.Error() }
func (e *RequestBuildError) Unwrap() error { return e.Err }
