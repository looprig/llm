package llm

import (
	"fmt"
	"strings"

	"github.com/looprig/credentials"
	model "github.com/looprig/inference/model"
)

// AuthBinding is the complete authority identity accepted by one provider
// transport. Scheme and usage are intentionally not enough to identify a
// credential: provider, transport, issuer, and audience are part of the
// binding as well.
type AuthBinding struct {
	Provider  string
	Transport string
	Scheme    credentials.Scheme
	Usage     credentials.UsageClass
	Issuer    string
	Audience  string
}

// Descriptor converts the policy binding into the credential descriptor used
// by a source and its leases. Label is presentation metadata and is therefore
// deliberately absent from an AuthBinding.
func (b AuthBinding) Descriptor() (credentials.Descriptor, error) {
	return credentials.NewDescriptor(b.Provider, b.Transport, b.Scheme, b.Usage, b.Issuer, b.Audience, "")
}

// Valid reports whether the complete binding can be represented by a safe
// credential descriptor.
func (b AuthBinding) Valid() bool {
	_, err := b.Descriptor()
	return err == nil
}

// AuthPolicy is the immutable set of complete credential identities accepted
// by one provider client. A policy with no accepted bindings is invalid.
type AuthPolicy struct {
	Accepted []AuthBinding
}

// Validate checks every accepted identity and rejects duplicate bindings. The
// input slice is never mutated; callers should treat a policy as immutable once
// it has been used to construct a client.
func (p AuthPolicy) Validate() error {
	if len(p.Accepted) == 0 {
		return &InvalidAuthPolicyError{Reason: "accepted binding set must not be empty"}
	}
	seen := make(map[string]struct{}, len(p.Accepted))
	for _, binding := range p.Accepted {
		descriptor, err := binding.Descriptor()
		if err != nil {
			return &InvalidAuthPolicyError{Reason: "accepted binding is invalid", Err: err}
		}
		key := descriptor.Canonical()
		if _, ok := seen[key]; ok {
			return &InvalidAuthPolicyError{Reason: "accepted bindings must be unique"}
		}
		seen[key] = struct{}{}
	}
	return nil
}

// Accepts reports whether descriptor has exactly one of the policy's accepted
// six-field identities. Descriptor.Label is presentation-only and is ignored.
func (p AuthPolicy) Accepts(descriptor credentials.Descriptor) bool {
	if !descriptor.Valid() || p.Validate() != nil {
		return false
	}
	for _, accepted := range p.Accepted {
		if descriptorMatchesBinding(descriptor, accepted) {
			return true
		}
	}
	return false
}

// Match validates descriptor against the policy and returns a safe typed
// mismatch error when it is not accepted.
func (p AuthPolicy) Match(descriptor credentials.Descriptor) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if !descriptor.Valid() {
		return &AuthPolicyMismatchError{Reason: "credential descriptor is invalid"}
	}
	if !p.Accepts(descriptor) {
		return &AuthPolicyMismatchError{Reason: "credential descriptor does not match the exact provider policy"}
	}
	return nil
}

func descriptorMatchesBinding(descriptor credentials.Descriptor, binding AuthBinding) bool {
	expected, err := binding.Descriptor()
	if err != nil {
		return false
	}
	return descriptor.Provider == expected.Provider &&
		descriptor.Transport == expected.Transport &&
		descriptor.Scheme == expected.Scheme &&
		descriptor.Usage == expected.Usage &&
		descriptor.Issuer == expected.Issuer &&
		descriptor.Audience == expected.Audience
}

// InvalidAuthPolicyError reports malformed provider policy metadata. It carries
// no credential or provider response data.
type InvalidAuthPolicyError struct {
	Reason string
	Err    error
}

func (e *InvalidAuthPolicyError) Error() string {
	if e == nil {
		return "llm: invalid auth policy"
	}
	if e.Err != nil {
		return "llm: invalid auth policy: " + e.Reason + ": " + e.Err.Error()
	}
	return "llm: invalid auth policy: " + e.Reason
}

func (e *InvalidAuthPolicyError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// AuthPolicyMismatchError reports a source or lease identity that is not
// authorized for the bound provider transport.
type AuthPolicyMismatchError struct{ Reason string }

func (e *AuthPolicyMismatchError) Error() string {
	if e == nil || e.Reason == "" {
		return "llm: credential binding does not match provider auth policy"
	}
	return "llm: credential binding does not match provider auth policy: " + e.Reason
}

// AuthPolicyForModel is the one-way compatibility bridge from the legacy
// Provider.RequiredAuth registry to an exact credentials policy. It is kept in
// llm because inference must remain unaware of provider identity and policy.
func AuthPolicyForModel(selected model.Model) (AuthPolicy, error) {
	if err := ValidateModel(selected); err != nil {
		return AuthPolicy{}, err
	}
	return Provider(selected.Provider).AuthPolicy(selected.APIFormat)
}

// authPolicyForProvider returns the exact policy bridge for a provider/API
// format pair. It deliberately derives the credential scheme from RequiredAuth
// so the legacy provider registry remains the single compatibility authority.
func authPolicyForProvider(p Provider, format model.APIFormat) (AuthPolicy, error) {
	if !p.supportsAPIFormat(format) {
		return AuthPolicy{}, &InvalidAuthPolicyError{Reason: fmt.Sprintf("provider %q does not support API format %q", p, format)}
	}
	legacyKind, err := p.RequiredAuth()
	if err != nil {
		return AuthPolicy{}, err
	}
	binding := AuthBinding{
		Provider:  string(p),
		Transport: transportIdentity(p, format),
		Issuer:    issuerIdentity(p),
		Audience:  audienceIdentity(p),
	}
	switch legacyKind {
	case authKindNone:
		binding.Scheme = credentials.SchemeNone
		binding.Usage = credentials.UsageLocal
		binding.Issuer = ""
		binding.Audience = ""
	case authKindAPIKey:
		binding.Scheme = credentials.SchemeAPIKey
		binding.Usage = credentials.UsageMeteredAPI
	case AuthOAuth:
		binding.Scheme = credentials.SchemeOAuth
		binding.Usage = credentials.UsageSubscription
	case AuthGCP:
		binding.Scheme = credentials.SchemeWorkloadIdentity
		binding.Usage = credentials.UsageMeteredAPI
	case AuthSigV4:
		binding.Scheme = credentials.SchemeSigV4
		binding.Usage = credentials.UsageMeteredAPI
	case AuthServiceKey, AuthToken:
		// These legacy kinds carry provider-specific bearer/service material. The
		// credentials descriptor still needs a concrete scheme; workload identity
		// is the conservative non-API-key classification until a provider-specific
		// source supplies a narrower policy.
		binding.Scheme = credentials.SchemeWorkloadIdentity
		binding.Usage = credentials.UsageMeteredAPI
	default:
		return AuthPolicy{}, &InvalidAuthPolicyError{Reason: fmt.Sprintf("unsupported legacy auth kind %q", legacyKind)}
	}
	policy := AuthPolicy{Accepted: []AuthBinding{binding}}
	if err := policy.Validate(); err != nil {
		return AuthPolicy{}, err
	}
	return policy, nil
}

// Keep the bridge readable without importing the legacy auth package into every
// policy comparison. These values are the stable names returned by RequiredAuth.
const (
	authKindNone   = "none"
	authKindAPIKey = "api_key"
)

func transportIdentity(provider Provider, format model.APIFormat) string {
	switch provider {
	case ProviderOpenAI:
		if format == model.APIFormatOpenAIResponses {
			return "responses"
		}
		return "chat"
	case ProviderAnthropic:
		return "messages"
	case ProviderGoogle:
		return "generate-content"
	case ProviderLMStudio, ProviderAtomicChat, ProviderLlamaCPP, ProviderOllama:
		return "local"
	default:
		identity := strings.ToLower(strings.TrimSpace(string(format)))
		if identity == "" {
			return "inference"
		}
		return identity
	}
}

func issuerIdentity(provider Provider) string {
	switch provider {
	case ProviderOpenAI:
		return "https://api.openai.com"
	case ProviderAnthropic:
		return "https://api.anthropic.com"
	case ProviderGoogle:
		return "https://generativelanguage.googleapis.com"
	default:
		return "https://" + string(provider) + ".invalid"
	}
}

func audienceIdentity(provider Provider) string {
	return "api://" + string(provider)
}
