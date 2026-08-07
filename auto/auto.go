// Package auto is the composition root that selects and wires a concrete
// inference.Client for a validated Model. It imports every provider it can fully
// construct from (model, key) alone, so business logic depends only on the
// inference.Client interface — never on a concrete provider — and every
// credential/attestation decision is made here, once. Two providers it deliberately
// does NOT import take an input auto.New does not carry: Bedrock needs AWS SigV4
// credentials, and Phala needs an attestation acceptance Policy. For each, New
// dispatches to a typed construct-directly error (SigV4NotConstructibleError,
// PolicyNotConstructibleError) that directs the caller to the named constructor
// rather than building a fail-open client with defaulted credentials. It maps a
// Provider to its client and enforces the provider's fail-closed auth contract
// before any network object is constructed.
package auto

import (
	"fmt"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"

	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"

	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
	anthropicprovider "github.com/looprig/llm/providers/anthropic"
	atomicchat "github.com/looprig/llm/providers/atomic-chat"
	azureprovider "github.com/looprig/llm/providers/azure"
	azurecognitive "github.com/looprig/llm/providers/azure-cognitive-services"
	basetenprovider "github.com/looprig/llm/providers/baseten"
	cerebrasprovider "github.com/looprig/llm/providers/cerebras"
	"github.com/looprig/llm/providers/chutes"
	cloudflaregateway "github.com/looprig/llm/providers/cloudflare-ai-gateway"
	cloudflareworkers "github.com/looprig/llm/providers/cloudflare-workers-ai"
	cortecsprovider "github.com/looprig/llm/providers/cortecs"
	deepinfra "github.com/looprig/llm/providers/deepinfra"
	deepseekprovider "github.com/looprig/llm/providers/deepseek"
	digitaloceanprovider "github.com/looprig/llm/providers/digitalocean"
	fireworksprovider "github.com/looprig/llm/providers/fireworks"
	frogbotprovider "github.com/looprig/llm/providers/frogbot"
	geminiprovider "github.com/looprig/llm/providers/gemini"
	githubcopilot "github.com/looprig/llm/providers/github-copilot"
	gitlabprovider "github.com/looprig/llm/providers/gitlab"
	gmicloudprovider "github.com/looprig/llm/providers/gmicloud"
	vertexprovider "github.com/looprig/llm/providers/google-vertex"
	groqprovider "github.com/looprig/llm/providers/groq"
	heliconeprovider "github.com/looprig/llm/providers/helicone"
	huggingfaceprovider "github.com/looprig/llm/providers/huggingface"
	ionetprovider "github.com/looprig/llm/providers/ionet"
	llamaprovider "github.com/looprig/llm/providers/llama"
	llamacppprovider "github.com/looprig/llm/providers/llamacpp"
	llmgatewayprovider "github.com/looprig/llm/providers/llmgateway"
	minimaxprovider "github.com/looprig/llm/providers/minimax"
	moonshotprovider "github.com/looprig/llm/providers/moonshot"
	nebiusprovider "github.com/looprig/llm/providers/nebius"
	nvidiaprovider "github.com/looprig/llm/providers/nvidia"
	ollamaprovider "github.com/looprig/llm/providers/ollama"
	ollamacloudprovider "github.com/looprig/llm/providers/ollamacloud"
	openaiprovider "github.com/looprig/llm/providers/openai"
	opencodeprovider "github.com/looprig/llm/providers/opencode"
	opencodegoprovider "github.com/looprig/llm/providers/opencode-go"
	"github.com/looprig/llm/providers/openrouter"
	ovhcloudprovider "github.com/looprig/llm/providers/ovhcloud"
	p302ai "github.com/looprig/llm/providers/p302ai"
	sapprovider "github.com/looprig/llm/providers/sap-ai-core"
	scalewayprovider "github.com/looprig/llm/providers/scaleway"
	snowflakeprovider "github.com/looprig/llm/providers/snowflake-cortex"
	stackitprovider "github.com/looprig/llm/providers/stackit"
	syntheticprovider "github.com/looprig/llm/providers/synthetic"
	togetherprovider "github.com/looprig/llm/providers/together"
	veniceprovider "github.com/looprig/llm/providers/venice"
	vercelprovider "github.com/looprig/llm/providers/vercel"
	xaiprovider "github.com/looprig/llm/providers/xai"
	zaiprovider "github.com/looprig/llm/providers/zai"
	zenmuxprovider "github.com/looprig/llm/providers/zenmux"
)

// SigV4NotConstructibleError is returned by New for a provider whose required
// credential kind is AuthSigV4 (currently Bedrock). auto.New's only credential
// input is an auth.APIKey, which cannot carry AWS SigV4 credentials, so such a
// provider must be constructed directly via its own constructor (named by Use,
// e.g. "bedrock.New"). Fail-closed and directive — never a silent nil client.
// This is why auto does NOT import the bedrock package: it dispatches to an error,
// not to a constructor it cannot feed.
type SigV4NotConstructibleError struct {
	Provider llm.Provider
	Use      string
}

func (e *SigV4NotConstructibleError) Error() string {
	return fmt.Sprintf("provider %q requires AWS SigV4 credentials that auto.New cannot supply; construct it directly via %s", e.Provider, e.Use)
}

// PolicyNotConstructibleError is returned by New for a provider that needs an
// attestation acceptance Policy auto.New cannot supply (currently Phala). auto.New's
// inputs are (model, key) only — it carries no Policy — so the caller must construct the
// client directly via the named constructor with their own verified policy. Fail-closed
// and directive, never a silent client with a defaulted policy.
type PolicyNotConstructibleError struct {
	Provider llm.Provider
	Use      string
}

func (e *PolicyNotConstructibleError) Error() string {
	return fmt.Sprintf("provider %q requires an attestation policy that auto.New cannot supply; construct it directly via %s", e.Provider, e.Use)
}

// CredentialNotConstructibleError is returned when auto.New would have to read
// or exchange a provider-specific credential from process state. The auto API
// accepts only an explicit auth.APIKey value, so callers must obtain the
// provider's OAuth/GCP/service-key/account token and pass it explicitly.
type CredentialNotConstructibleError struct {
	Provider llm.Provider
	Kind     auth.AuthKind
	Use      string
}

func (e *CredentialNotConstructibleError) Error() string {
	return fmt.Sprintf("provider %q requires explicit %s credentials; auto.New will not discover them from the environment; construct it directly via %s", e.Provider, e.Kind, e.Use)
}

// Option customizes construction for a provider-specific branch.
type Option func(*options)

type options struct {
	openRouter []openrouter.Option
}

// WithOpenRouterOptions applies OpenRouter-specific headers and request-body
// options. It is used only when selected.Provider is OpenRouter; supplying it
// for another provider is rejected by New.
func WithOpenRouterOptions(opts ...openrouter.Option) Option {
	return func(config *options) {
		config.openRouter = append(config.openRouter, opts...)
	}
}

// New validates model, enforces the provider's fail-closed auth requirement, then
// constructs the concrete provider client. Ordered:
//  1. llm.ValidateModel — a self-contradictory or unknown-provider model yields a
//     *model.ValidationError before anything else.
//  2. Provider.RequiredAuth — an unknown provider fails closed with a
//     *model.ValidationError (never a permissive default).
//  3. A provider that requires an API key but is given none yields a
//     *llm.AuthRequiredError. A provider requiring a special credential kind
//     yields a *CredentialNotConstructibleError when the explicit credential is
//     absent; auto.New never discovers special credentials from the environment.
//  4. Dispatch on Provider to the concrete client.
//
// No live I/O happens here; the returned inference.Client performs its own
// per-request guards (binding, validation, auth) when Invoke/Stream is called.
func New(selected model.Model, key auth.APIKey, opts ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	p := llm.Provider(selected.Provider)
	kind, err := p.RequiredAuth()
	if err != nil {
		return nil, err
	}
	if key == "" {
		switch kind {
		case auth.AuthNone:
			// Local providers intentionally need no credential.
		case auth.AuthAPIKey:
			return nil, &llm.AuthRequiredError{Provider: p, Kind: kind}
		case llm.AuthSigV4:
			// Bedrock reaches its explicit dispatch below so callers receive
			// the provider-specific constructor directive.
		default:
			return nil, &CredentialNotConstructibleError{Provider: p, Kind: kind, Use: directCredentialConstructor(p)}
		}
	}
	var config options
	for _, opt := range opts {
		if opt != nil {
			opt(&config)
		}
	}
	if len(config.openRouter) > 0 && p != llm.ProviderOpenRouter {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: "OpenRouter options require provider \"openrouter\"",
		}
	}
	switch p {
	case llm.ProviderPhala:
		// Phala's client attests the TEE and enforces an attestation acceptance Policy.
		// auto.New's inputs are (model, key) only — it carries no Policy — so a Phala
		// client cannot be built here; direct the caller to phala.New (which takes their
		// own verified policy). Fail-closed with a directive typed error, never a silent
		// client with a defaulted policy that would fail open. This is why auto no longer
		// imports the phala package: it dispatches to an error, not to a constructor it
		// cannot feed.
		return nil, &PolicyNotConstructibleError{Provider: llm.ProviderPhala, Use: "phala.New"}
	case llm.ProviderChutes:
		return chutes.New(selected.BaseURL, string(key)), nil
	case llm.ProviderLMStudio:
		// LM Studio can speak either dialect (supportsAPIFormat admits both); genericHTTP
		// selects the codec by the model's declared APIFormat and fails closed on any
		// format with no codec, rather than silently mis-encoding. A local endpoint needs
		// no credentials.
		return genericHTTP(selected, auth.None())
	case llm.ProviderOpenRouter:
		// OpenRouter is an OpenAI-compatible aggregation gateway behind a Bearer key. The
		// fail-closed empty-key guard above (RequiredAuth → AuthAPIKey) already rejected a
		// missing key, so key is present here; wrap it as Bearer auth.
		if len(config.openRouter) > 0 {
			return openrouter.New(selected, key, config.openRouter...)
		}
		return genericHTTP(selected, auth.Key(key))
	case llm.ProviderOpenAI:
		return openaiprovider.New(selected, key)
	case llm.ProviderAzure:
		return azureprovider.New(selected, key)
	case llm.ProviderAnthropic:
		return anthropicprovider.New(selected, key)
	case llm.ProviderXAI:
		return xaiprovider.New(selected, key)
	case llm.ProviderAzureCognitiveServices:
		return azurecognitive.New(selected, key)
	case llm.Provider302AI:
		return p302ai.New(selected, key)
	case llm.ProviderAtomicChat:
		return atomicchat.New(selected, key)
	case llm.ProviderBaseten:
		return basetenprovider.New(selected, key)
	case llm.ProviderCerebras:
		return cerebrasprovider.New(selected, key)
	case llm.ProviderCloudflareAIGateway:
		return cloudflaregateway.New(selected, key)
	case llm.ProviderCloudflareWorkersAI:
		return cloudflareworkers.New(selected, key)
	case llm.ProviderCortecs:
		return cortecsprovider.New(selected, key)
	case llm.ProviderDeepSeek:
		return deepseekprovider.New(selected, key)
	case llm.ProviderDeepInfra:
		return deepinfra.New(selected, key)
	case llm.ProviderDigitalOcean:
		return digitaloceanprovider.New(selected, key)
	case llm.ProviderFrogBot:
		return frogbotprovider.New(selected, key)
	case llm.ProviderFireworks:
		return fireworksprovider.New(selected, key)
	case llm.ProviderGitLab:
		return gitlabprovider.New(selected, key)
	case llm.ProviderGitHubCopilot:
		return githubcopilot.New(selected, key)
	case llm.ProviderGMICloud:
		return gmicloudprovider.New(selected, key)
	case llm.ProviderGoogleVertex, llm.ProviderGoogleVertexAnthropic:
		return vertexprovider.New(selected, key)
	case llm.ProviderGroq:
		return groqprovider.New(selected, key)
	case llm.ProviderHuggingFace:
		return huggingfaceprovider.New(selected, key)
	case llm.ProviderHelicone:
		return heliconeprovider.New(selected, key)
	case llm.ProviderLlama:
		return llamaprovider.New(selected, key)
	case llm.ProviderLlamaCPP:
		return llamacppprovider.New(selected, key)
	case llm.ProviderIONet:
		return ionetprovider.New(selected, key)
	case llm.ProviderMoonshot:
		return moonshotprovider.New(selected, key)
	case llm.ProviderMiniMax:
		return minimaxprovider.New(selected, key)
	case llm.ProviderNVIDIA:
		return nvidiaprovider.New(selected, key)
	case llm.ProviderNebius:
		return nebiusprovider.New(selected, key)
	case llm.ProviderOllama:
		return ollamaprovider.New(selected, key)
	case llm.ProviderOllamaCloud:
		return ollamacloudprovider.New(selected, key)
	case llm.ProviderOpenCode:
		return opencodeprovider.New(selected, key)
	case llm.ProviderOpenCodeGo:
		return opencodegoprovider.New(selected, key)
	case llm.ProviderLLMGateway:
		return llmgatewayprovider.New(selected, key)
	case llm.ProviderSAP:
		serviceKey, parseErr := sapprovider.ParseServiceKey([]byte(key))
		if parseErr != nil {
			return nil, parseErr
		}
		return sapprovider.New(selected, serviceKey)
	case llm.ProviderSTACKIT:
		return stackitprovider.New(selected, key)
	case llm.ProviderOVHCloud:
		return ovhcloudprovider.New(selected, key)
	case llm.ProviderScaleway:
		return scalewayprovider.New(selected, key)
	case llm.ProviderSnowflakeCortex:
		return snowflakeprovider.New(selected, key)
	case llm.ProviderSynthetic:
		return syntheticprovider.New(selected, key)
	case llm.ProviderTogetherAI:
		return togetherprovider.New(selected, key)
	case llm.ProviderVenice:
		return veniceprovider.New(selected, key)
	case llm.ProviderVercel:
		return vercelprovider.New(selected, key)
	case llm.ProviderZAI:
		return zaiprovider.New(selected, key)
	case llm.ProviderZenMux:
		return zenmuxprovider.New(selected, key)
	case llm.ProviderGoogle:
		// Google's Gemini generateContent API is not plain codec-over-HTTP (per-model
		// ":generateContent" path + an x-goog-api-key header), so it uses the bespoke
		// providers/gemini client rather than genericHTTP. The empty-key guard above
		// (RequiredAuth → AuthAPIKey) already rejected a missing key; gemini.New re-checks
		// and fails closed on empty regardless.
		return geminiprovider.New(key)
	case llm.ProviderBedrock:
		// Bedrock's RequiredAuth is AuthSigV4, so the empty-APIKey guard above does not
		// fire and control reaches here. auto.New's only credential is an auth.APIKey,
		// which cannot carry AWS SigV4 credentials, so a Bedrock client cannot be built
		// here; direct the caller to bedrock.New (which takes auth.SigV4Credentials + a
		// region). Fail-closed with a directive typed error, not a silent nil.
		return nil, &SigV4NotConstructibleError{Provider: llm.ProviderBedrock, Use: "bedrock.New"}
	default:
		// Defensive: RequiredAuth above already rejects any provider not handled
		// here, so this is unreachable for a validated model — but a permissive
		// fall-through would fail open, so deny by default.
		return nil, &model.ValidationError{Field: "Provider", Reason: "unsupported provider"}
	}
}

func directCredentialConstructor(provider llm.Provider) string {
	switch provider {
	case llm.ProviderGitLab:
		return "gitlab.New"
	case llm.ProviderGitHubCopilot:
		return "githubcopilot.New"
	case llm.ProviderGoogleVertex, llm.ProviderGoogleVertexAnthropic:
		return "vertex.New"
	case llm.ProviderSAP:
		return "sapcore.New"
	case llm.ProviderSnowflakeCortex:
		return "snowflake.New"
	default:
		return "the provider's constructor"
	}
}

// genericHTTP builds the generic transport-backed client for a provider that speaks a
// plain codec-over-HTTP endpoint. It selects the wire codec by the model's declared
// APIFormat (failing closed if none is implemented) and injects the caller-supplied
// authenticator, so one construction serves both an unauthenticated local endpoint
// (LM Studio, auth.None) and a Bearer-key gateway (OpenRouter, auth.Key) — the auth
// decision stays at the composition root, not in the transport. The route is the
// static OpenAI/Anthropic-style chat path; the endpoint binds provider + format so the
// transport's per-request binding check catches a cross-wired Model.
func genericHTTP(model model.Model, a auth.Authenticator) (inference.Client, error) {
	codec, err := codecFor(model.APIFormat)
	if err != nil {
		return nil, err
	}
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = defaultGenericBaseURL(llm.Provider(model.Provider))
	}
	return transport.New(
		transport.Endpoint{
			BaseURL:   baseURL,
			Provider:  model.Provider,
			APIFormat: model.APIFormat,
		},
		route.StaticChat("/chat/completions"),
		codec,
		a,
	), nil
}

const (
	openRouterBaseURL = "https://openrouter.ai/api/v1"
	lmStudioBaseURL   = "http://localhost:1234/v1"
)

// defaultGenericBaseURL returns the canonical endpoint for a generic-transport
// provider, or "" if it has none (the caller then relies on an explicit base).
// INVARIANT: any generic-transport provider (one routed through genericHTTP) for
// which llm.Provider.allowsEmptyBaseURL() reports true MUST have a default here —
// otherwise an empty Model.BaseURL passes validation but yields a hostless endpoint
// that only fails at request time. The generic-transport providers are currently
// exactly {openrouter, lmstudio}, and both have a default below; the other
// empty-base providers (chutes, phala, google) self-default the base in their own
// dedicated clients, and bedrock is region-routed with no base.
func defaultGenericBaseURL(p llm.Provider) string {
	switch p {
	case llm.ProviderOpenRouter:
		return openRouterBaseURL
	case llm.ProviderLMStudio:
		return lmStudioBaseURL
	default:
		return ""
	}
}

// codecFor selects the wire codec for a generic (transport-backed) provider by its
// declared APIFormat. ValidateModel already admits every APIFormat the provider
// supports, and a provider may legitimately support a format auto cannot yet encode;
// codecFor is the fail-closed boundary that turns "no codec implemented" into a typed
// *model.ValidationError at construction rather than a silent wrong-dialect encode.
// Adding a new dialect is one new case here.
func codecFor(f model.APIFormat) (codec.Codec, error) {
	switch f {
	case model.APIFormatOpenAI:
		return openaiapi.Codec{}, nil
	case model.APIFormatAnthropic:
		return anthropicapi.Codec{}, nil
	case model.APIFormatGemini:
		return geminiapi.Codec{}, nil
	case model.APIFormatOpenAIResponses:
		return openairesponses.Codec{}, nil
	default:
		return nil, &model.ValidationError{Field: "APIFormat", Reason: "no codec implemented for this API format yet"}
	}
}
