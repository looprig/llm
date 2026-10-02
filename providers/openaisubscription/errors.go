package openaisubscription

import (
	"errors"
	"fmt"
)

// codeUsageLimitExceeded is the documented ChatGPT-plan usage limit code:
// https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery
const codeUsageLimitExceeded = "subscription_sharing_usage_limit_exceeded"

// ErrUsageLimit reports that the ChatGPT plan, or this app's limit on it, is
// exhausted. OpenAI's recovery is to pause requests that use the plan and
// point the user at ChatGPT Settings > Usage; it is not rate pressure and must
// not be retried.
var ErrUsageLimit = errors.New("openai-subscription: ChatGPT plan usage limit reached; see ChatGPT Settings > Usage")

// UsageLimitError is ErrUsageLimit with the bounded request ID when one was
// returned. It is deliberately not a *failure.APIError, so generic retry
// policies (which retry every HTTP 429) do not retry it.
type UsageLimitError struct{ RequestID string }

func (e *UsageLimitError) Error() string {
	if e == nil || e.RequestID == "" {
		return ErrUsageLimit.Error()
	}
	return ErrUsageLimit.Error() + " (request id " + e.RequestID + ")"
}
func (e *UsageLimitError) Unwrap() error { return ErrUsageLimit }

// ResponseError is a Responses failure (response.failed or a top-level error
// event) after the stream opened. Only bounded code tokens are kept: the
// provider's free-form message is never retained, so it cannot carry request
// text or credentials into logs. Param names the rejected input when the
// provider reports one (e.g. subscription_sharing_unsupported_capability).
type ResponseError struct{ Code, Param string }

func (e *ResponseError) Error() string {
	message := "openai-subscription: response failed"
	if e == nil {
		return message
	}
	if e.Code != "" {
		message += " (" + e.Code + ")"
	}
	if e.Param != "" {
		message += " param " + e.Param
	}
	return message
}
func (e *ResponseError) Format(s fmt.State, _ rune) { _, _ = s.Write([]byte(e.Error())) }
func (e *ResponseError) GoString() string           { return e.Error() }

// IncompleteError reports response.incomplete. OpenAI documents only
// response.completed as successful inference, so partial output is never
// returned as a result. Reason is the bounded incomplete_details.reason
// (e.g. max_output_tokens, content_filter) when present.
type IncompleteError struct{ Reason string }

func (e *IncompleteError) Error() string {
	if e == nil || e.Reason == "" {
		return "openai-subscription: response incomplete"
	}
	return "openai-subscription: response incomplete (" + e.Reason + ")"
}

func responseFailure(code, param string) error {
	code = safeToken(code)
	if code == codeUsageLimitExceeded {
		return &UsageLimitError{}
	}
	return &ResponseError{Code: code, Param: safeToken(param)}
}

// safeToken keeps a provider identifier only when it is a short machine token
// (letters, digits and _ . - [ ]); anything else is dropped, never echoed.
func safeToken(value string) string {
	if value == "" || len(value) > 128 {
		return ""
	}
	for _, c := range value {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '.' || c == '-' || c == '[' || c == ']') {
			return ""
		}
	}
	return value
}
