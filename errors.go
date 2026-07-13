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

// CounterSupportReason classifies why an exact provider context counter is not
// available through the llm module. It never contains provider-controlled text.
type CounterSupportReason string

const (
	// CounterSupportExactUnavailable means the provider exposes no exact counter
	// that llm can construct. Consumers may explicitly inject a local estimator;
	// this error never substitutes one.
	CounterSupportExactUnavailable CounterSupportReason = "exact provider context counter unavailable"
	// CounterSupportAPIFormatUnavailable means the provider has an exact counter,
	// but it cannot encode the model's validated API dialect.
	CounterSupportAPIFormatUnavailable CounterSupportReason = "exact provider context counter unavailable for API format"
)

// CounterSupportError reports that a known provider has no exact context counter
// in llm for the selected dialect. All fields are secret-free and inspectable
// with errors.As.
type CounterSupportError struct {
	Provider  Provider
	Reason    CounterSupportReason
	APIFormat inference.APIFormat
}

func (e *CounterSupportError) Error() string {
	return fmt.Sprintf("provider %q context counter support for API format %q: %s", e.Provider, e.APIFormat, e.Reason)
}

// CounterDirectConstructionReason classifies why auto.NewCounter cannot build an
// exact counter from its (Model, APIKey) inputs.
type CounterDirectConstructionReason string

const (
	// CounterDirectConstructionNeedsSigV4 means construction requires AWS SigV4
	// credentials and a region, neither of which can be represented by APIKey.
	CounterDirectConstructionNeedsSigV4 CounterDirectConstructionReason = "requires AWS SigV4 credentials and region"
)

// CounterConstructor names a provider constructor without making it executable.
// It is a directive safe to expose in logs and user-facing configuration errors.
type CounterConstructor string

const (
	CounterConstructorBedrock CounterConstructor = "bedrock.NewCounter"
)

// CounterDirectConstructionError directs callers to an exact provider counter
// constructor whose required inputs cannot be supplied by auto.NewCounter.
type CounterDirectConstructionError struct {
	Provider Provider
	Reason   CounterDirectConstructionReason
	Use      CounterConstructor
}

func (e *CounterDirectConstructionError) Error() string {
	return fmt.Sprintf("provider %q context counter cannot be auto-constructed: %s; use %s", e.Provider, e.Reason, e.Use)
}
