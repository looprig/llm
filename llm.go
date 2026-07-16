// Package llm is the batteries-included provider SDK layered on the neutral
// github.com/looprig/inference model-call contract. It owns the provider POLICY
// that inference deliberately does not carry: the known-provider registry, the
// provider/API-format truth table, provider auth requirements, provider default
// endpoint behavior, and the fail-closed model-validation preset. It also hosts
// the self-contained provider-security machinery (aci, e2e, tee) and the
// provider-specific SigV4 authenticator (llm/auth). inference never imports llm.
package llm

// Provider names the concrete backend an llm/auto factory dispatches on. Unknown
// values are rejected by ValidateModel; a provider constructor additionally
// enforces each provider's auth requirement. It is the provider-policy analogue
// of the opaque model.ProviderName label: inference carries no provider
// constants, no auth requirements, and no default endpoints — those live here.
type Provider string

const (
	ProviderLMStudio   Provider = "lmstudio"
	ProviderPhala      Provider = "phala"
	ProviderChutes     Provider = "chutes"
	ProviderOpenRouter Provider = "openrouter"
	ProviderBedrock    Provider = "bedrock"
	// ProviderGoogle is Google's Gemini generateContent backend. The provider (the
	// backend "google") and the dialect (APIFormatGemini) are distinct axes: google
	// speaks only the Gemini wire format, authenticated with an x-goog-api-key header
	// (RequiredAuth → AuthAPIKey), and is served by the bespoke providers/gemini
	// client (the generic transport assumes a static /chat/completions path).
	ProviderGoogle Provider = "google"
)
