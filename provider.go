package llm

import (
	auth "github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
)

// RequiresKey reports whether the provider needs an API key, and errors on an
// unknown provider so a newly added one must be classified here before it can be
// used. Hosted key providers (phala, chutes, openrouter, google) require a key; a
// local LM Studio endpoint does not. A bare default-false would fail open — the
// bug this method exists to prevent. This is the legacy boolean superseded by
// RequiredAuth (which is the real gate and can express non-key auth like SigV4).
func (p Provider) RequiresKey() (bool, error) {
	switch p {
	case ProviderLMStudio, ProviderAtomicChat, ProviderLlama, ProviderOllama:
		return false, nil
	case ProviderPhala, ProviderChutes, ProviderOpenRouter, ProviderOpenAI, ProviderAzure, ProviderAzureCognitiveServices, ProviderAnthropic, ProviderXAI, ProviderGoogle,
		Provider302AI, ProviderBaseten, ProviderCerebras, ProviderCloudflareAIGateway, ProviderCloudflareWorkersAI,
		ProviderCortecs, ProviderDeepSeek, ProviderDeepInfra, ProviderDigitalOcean, ProviderFrogBot, ProviderFireworks,
		ProviderGMICloud, ProviderGroq, ProviderHuggingFace, ProviderHelicone, ProviderIONet, ProviderMoonshot,
		ProviderMiniMax, ProviderNVIDIA, ProviderNebius, ProviderOllamaCloud, ProviderOpenCode, ProviderOpenCodeGo,
		ProviderLLMGateway, ProviderSTACKIT, ProviderOVHCloud, ProviderScaleway, ProviderTogetherAI, ProviderVenice,
		ProviderVercel, ProviderZAI, ProviderZenMux:
		return true, nil
	case ProviderGitLab, ProviderGitHubCopilot, ProviderGoogleVertex, ProviderGoogleVertexAnthropic, ProviderSAP, ProviderSnowflakeCortex:
		return false, nil
	case ProviderBedrock:
		// Bedrock authenticates with SigV4 credentials, not an API key; this legacy
		// boolean cannot express that. RequiredAuth() is the real gate (AuthSigV4).
		return false, nil
	default:
		return false, &model.ValidationError{Field: "Provider", Reason: "unknown provider; API-key policy undefined"}
	}
}

// supportsAPIFormat reports whether provider p is known to speak wire dialect f.
// It is fail-closed: an unknown provider supports no formats, so a newly added
// provider must be classified here before any Model naming it can validate.
func (p Provider) supportsAPIFormat(f model.APIFormat) bool {
	switch p {
	case ProviderPhala, ProviderChutes, ProviderOpenRouter, Provider302AI, ProviderAtomicChat, ProviderBaseten, ProviderCerebras,
		ProviderCloudflareWorkersAI, ProviderCortecs, ProviderDeepSeek, ProviderDigitalOcean, ProviderFrogBot, ProviderFireworks,
		ProviderGroq, ProviderHuggingFace, ProviderHelicone, ProviderLlama, ProviderIONet, ProviderMoonshot, ProviderNVIDIA,
		ProviderNebius, ProviderOllama, ProviderOllamaCloud, ProviderOpenCodeGo, ProviderSTACKIT, ProviderOVHCloud,
		ProviderScaleway, ProviderTogetherAI, ProviderVercel, ProviderZAI:
		return f == model.APIFormatOpenAI
	case ProviderOpenAI, ProviderAzure, ProviderXAI:
		return f == model.APIFormatOpenAIResponses
	case ProviderVenice:
		return f == model.APIFormatOpenAI || f == model.APIFormatOpenAIResponses
	case ProviderAnthropic, ProviderMiniMax, ProviderGMICloud:
		return f == model.APIFormatAnthropic
	case ProviderAzureCognitiveServices:
		return f == model.APIFormatOpenAI || f == model.APIFormatAnthropic
	case ProviderCloudflareAIGateway:
		return f == model.APIFormatOpenAI || f == model.APIFormatOpenAIResponses || f == model.APIFormatAnthropic
	case ProviderDeepInfra, ProviderGitLab, ProviderLLMGateway, ProviderZenMux:
		return f == model.APIFormatOpenAI || f == model.APIFormatAnthropic
	case ProviderGitHubCopilot:
		return f == model.APIFormatOpenAI || f == model.APIFormatOpenAIResponses
	case ProviderGoogleVertex:
		return f == model.APIFormatGemini || f == model.APIFormatAnthropic
	case ProviderGoogleVertexAnthropic:
		return f == model.APIFormatAnthropic
	case ProviderOpenCode:
		return f == model.APIFormatOpenAI || f == model.APIFormatOpenAIResponses || f == model.APIFormatAnthropic
	case ProviderSAP, ProviderSnowflakeCortex:
		return f == model.APIFormatOpenAI
	case ProviderLMStudio:
		return f == model.APIFormatOpenAI || f == model.APIFormatAnthropic
	case ProviderBedrock:
		// Bedrock supports both the existing Anthropic-on-Bedrock InvokeModel
		// dialect and the native Converse/ConverseStream dialect.
		return f == model.APIFormatAnthropic || f == APIFormatBedrockConverse
	case ProviderGoogle:
		// Google's generateContent backend speaks only the Gemini dialect.
		return f == model.APIFormatGemini
	default:
		return false
	}
}

// allowsEmptyBaseURL reports whether an empty Model.BaseURL is acceptable for the
// provider — true when the base is resolvable to a canonical endpoint: Bedrock is
// region-routed (no base), and every other current provider has a default endpoint
// the SDK supplies when BaseURL is empty. Fail-closed: an unknown/future provider
// with no default returns false, so ValidateModel keeps requiring an explicit base.
func (p Provider) allowsEmptyBaseURL() bool {
	switch p {
	case ProviderBedrock, ProviderChutes, ProviderPhala, ProviderOpenRouter, ProviderOpenAI, ProviderAzure, ProviderAzureCognitiveServices,
		ProviderAnthropic, ProviderXAI, ProviderLMStudio, ProviderGoogle, Provider302AI, ProviderAtomicChat, ProviderBaseten,
		ProviderCerebras, ProviderCloudflareAIGateway, ProviderCloudflareWorkersAI, ProviderCortecs, ProviderDeepSeek,
		ProviderDeepInfra, ProviderDigitalOcean, ProviderFrogBot, ProviderFireworks, ProviderGitLab, ProviderGitHubCopilot,
		ProviderGMICloud, ProviderGoogleVertex, ProviderGoogleVertexAnthropic, ProviderGroq, ProviderHuggingFace,
		ProviderHelicone, ProviderLlama, ProviderIONet, ProviderMoonshot, ProviderMiniMax, ProviderNVIDIA, ProviderNebius,
		ProviderOllama, ProviderOllamaCloud, ProviderOpenCode, ProviderOpenCodeGo, ProviderLLMGateway, ProviderSAP,
		ProviderSTACKIT, ProviderOVHCloud, ProviderScaleway, ProviderSnowflakeCortex, ProviderTogetherAI, ProviderVenice,
		ProviderVercel, ProviderZAI, ProviderZenMux:
		return true
	default:
		return false
	}
}

// RequiredAuth reports which credential kind the provider needs, erroring on an unknown provider
// so a newly added one must be classified here before use. Multi-auth-ready successor to
// RequiresKey; fail-closed by the same rationale (a permissive default would fail open).
func (p Provider) RequiredAuth() (auth.AuthKind, error) {
	switch p {
	case ProviderLMStudio, ProviderAtomicChat, ProviderLlama, ProviderOllama:
		return auth.AuthNone, nil
	case ProviderPhala, ProviderChutes, ProviderOpenRouter, ProviderOpenAI, ProviderAzure, ProviderAzureCognitiveServices, ProviderAnthropic, ProviderXAI, ProviderGoogle,
		Provider302AI, ProviderBaseten, ProviderCerebras, ProviderCloudflareAIGateway, ProviderCloudflareWorkersAI,
		ProviderCortecs, ProviderDeepSeek, ProviderDeepInfra, ProviderDigitalOcean, ProviderFrogBot, ProviderFireworks,
		ProviderGMICloud, ProviderGroq, ProviderHuggingFace, ProviderHelicone, ProviderIONet, ProviderMoonshot,
		ProviderMiniMax, ProviderNVIDIA, ProviderNebius, ProviderOllamaCloud, ProviderOpenCode, ProviderOpenCodeGo,
		ProviderLLMGateway, ProviderSTACKIT, ProviderOVHCloud, ProviderScaleway, ProviderTogetherAI, ProviderVenice,
		ProviderVercel, ProviderZAI, ProviderZenMux:
		return auth.AuthAPIKey, nil
	case ProviderGitLab, ProviderGitHubCopilot:
		return AuthOAuth, nil
	case ProviderGoogleVertex, ProviderGoogleVertexAnthropic:
		return AuthGCP, nil
	case ProviderSAP:
		return AuthServiceKey, nil
	case ProviderSnowflakeCortex:
		return AuthToken, nil
	case ProviderBedrock:
		// Bedrock authenticates with AWS SigV4, not a bearer API key; a generic auto
		// factory cannot supply SigV4 credentials, so a Bedrock client is built directly
		// via bedrock.New.
		return AuthSigV4, nil
	default:
		return "", &model.ValidationError{Field: "Provider", Reason: "unknown provider; auth policy undefined"}
	}
}
