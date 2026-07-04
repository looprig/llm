package llm

import (
	"fmt"

	"github.com/looprig/inference"
)

// AttestationError is a TEE attestation failure.
// Fail-closed: a request must never be sent to the provider when this error is returned.
// Err may be nil when the failure has no underlying cause to chain.
//
// It lives in llm, not inference: attestation is provider-security policy (Phala/Chutes
// confidential inference), which the neutral model-call layer deliberately does not carry.
type AttestationError struct {
	Reason string
	Err    error
}

func (e *AttestationError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("llm: attestation error: %s: %v", e.Reason, e.Err)
	}
	return "llm: attestation error: " + e.Reason
}

func (e *AttestationError) Unwrap() error { return e.Err }

// AuthSigV4 classifies the AWS Signature Version 4 credential a Bedrock provider
// requires. It is an inference.AuthKind value defined here rather than in inference:
// inference owns only the generic credential kinds (AuthNone, AuthAPIKey), while
// provider-specific request-signing schemes are llm policy.
const AuthSigV4 inference.AuthKind = "sigv4"

// AuthRequiredError is returned by a provider factory when a provider that requires
// credentials is given none. Fail-closed. Carries no secret. Provider is the llm
// provider-policy label; Kind is the inference credential kind (including AuthSigV4).
type AuthRequiredError struct {
	Provider Provider
	Kind     inference.AuthKind
}

func (e *AuthRequiredError) Error() string {
	return fmt.Sprintf("provider %q requires %s credentials", e.Provider, e.Kind)
}
