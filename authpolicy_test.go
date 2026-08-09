package llm_test

import (
	"testing"

	"github.com/looprig/credentials"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

func TestAuthPolicyMatchesTheCompleteBindingTuple(t *testing.T) {
	t.Parallel()

	binding := llm.AuthBinding{
		Provider:  "openai",
		Transport: "responses",
		Scheme:    credentials.SchemeAPIKey,
		Usage:     credentials.UsageMeteredAPI,
		Issuer:    "https://api.openai.com",
		Audience:  "api://openai",
	}
	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{binding}}

	descriptor, err := credentials.NewDescriptor(
		binding.Provider,
		binding.Transport,
		binding.Scheme,
		binding.Usage,
		binding.Issuer,
		binding.Audience,
		"presentation-label",
	)
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Accepts(descriptor) {
		t.Fatalf("policy does not accept exact descriptor: %#v", descriptor)
	}

	for name, mutate := range map[string]func(*credentials.Descriptor){
		"provider":  func(d *credentials.Descriptor) { d.Provider = "anthropic" },
		"transport": func(d *credentials.Descriptor) { d.Transport = "chat" },
		"scheme":    func(d *credentials.Descriptor) { d.Scheme = credentials.SchemeOAuth },
		"usage":     func(d *credentials.Descriptor) { d.Usage = credentials.UsageSubscription },
		"issuer":    func(d *credentials.Descriptor) { d.Issuer = "https://other.example" },
		"audience":  func(d *credentials.Descriptor) { d.Audience = "api://other" },
	} {
		name, mutate := name, mutate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			mismatched := descriptor
			mutate(&mismatched)
			if policy.Accepts(mismatched) {
				t.Fatalf("policy accepted %s mismatch: %#v", name, mismatched)
			}
		})
	}
}

func TestAuthPolicyIgnoresDescriptorPresentationLabel(t *testing.T) {
	t.Parallel()

	binding := llm.AuthBinding{
		Provider:  "openai",
		Transport: "responses",
		Scheme:    credentials.SchemeAPIKey,
		Usage:     credentials.UsageMeteredAPI,
		Issuer:    "https://api.openai.com",
		Audience:  "api://openai",
	}
	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{binding}}
	descriptor, err := credentials.NewDescriptor(binding.Provider, binding.Transport, binding.Scheme, binding.Usage, binding.Issuer, binding.Audience, "label")
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Accepts(descriptor) {
		t.Fatal("policy rejected a descriptor that differs only by presentation label")
	}
}

func TestAuthPolicyRejectsInvalidAcceptedBinding(t *testing.T) {
	t.Parallel()

	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{{
		Provider:  "openai",
		Transport: "responses",
		Scheme:    credentials.SchemeAPIKey,
		Usage:     credentials.UsageMeteredAPI,
		Issuer:    "",
		Audience:  "api://openai",
	}}}
	if err := policy.Validate(); err == nil {
		t.Fatal("AuthPolicy.Validate() = nil for an invalid accepted binding")
	}
}

func TestAuthPolicyForModelBridgesLegacyRequiredAuth(t *testing.T) {
	t.Parallel()

	selected := modelForPolicyTest()
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	if len(policy.Accepted) != 1 {
		t.Fatalf("accepted bindings = %d, want one", len(policy.Accepted))
	}
	binding := policy.Accepted[0]
	if binding.Provider != string(llm.ProviderOpenAI) || binding.Scheme != credentials.SchemeAPIKey || binding.Usage != credentials.UsageMeteredAPI {
		t.Fatalf("legacy bridge binding = %#v", binding)
	}
	if binding.Transport == "" || binding.Issuer == "" || binding.Audience == "" {
		t.Fatalf("legacy bridge omitted exact identity fields: %#v", binding)
	}
	if binding.Audience != "https://api.openai.com" {
		t.Fatalf("default audience = %q, want resolved request origin", binding.Audience)
	}
}

func TestAuthPolicyBindsCustomRequestOriginExactly(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAI, "https://proxy.example.test/v1", "gpt-test")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	if got := policy.Accepted[0].Audience; got != "https://proxy.example.test" {
		t.Fatalf("custom audience = %q, want exact origin", got)
	}
	if got := policy.Accepted[0].Issuer; got != "https://api.openai.com" {
		t.Fatalf("custom issuer = %q, want reviewed OpenAI issuer", got)
	}
}

func TestAuthPolicyRejectsMalformedOrInsecureAuthenticatedOrigin(t *testing.T) {
	t.Parallel()

	tests := []string{"http://proxy.example.test/v1", "https://proxy.example.test/a?query=1", "https://proxy.example.test/a#fragment", "https://user:pass@proxy.example.test/v1", "https://"}
	for _, baseURL := range tests {
		baseURL := baseURL
		t.Run(baseURL, func(t *testing.T) {
			selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAI, baseURL, "gpt-test")
			if _, err := llm.AuthPolicyForModel(selected); err == nil {
				t.Fatalf("AuthPolicyForModel(%q) = nil error, want rejected origin", baseURL)
			}
		})
	}
}

func TestAuthPolicyAllowsExplicitLocalNoneOrigin(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(model.ProviderName(llm.ProviderLMStudio), model.APIFormatOpenAI, "http://127.0.0.1:1234/v1", "local-model")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	binding := policy.Accepted[0]
	if binding.Scheme != credentials.SchemeNone || binding.Issuer != "" || binding.Audience != "" {
		t.Fatalf("local policy = %#v, want explicit None with empty authority", binding)
	}
}

func TestProviderAuthPolicyRejectsUnsupportedFormat(t *testing.T) {
	t.Parallel()

	if policy, err := llm.ProviderOpenAI.AuthPolicy(model.APIFormatGemini); policy.Accepted != nil || err == nil {
		t.Fatalf("AuthPolicy(gemini) = (%#v, %v), want fail-closed unsupported-format error", policy, err)
	}
}

func TestAuthPolicyCanonicalizesDescriptorIdentifiers(t *testing.T) {
	t.Parallel()

	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{{
		Provider:  "OPENAI",
		Transport: "RESPONSES",
		Scheme:    credentials.SchemeAPIKey,
		Usage:     credentials.UsageMeteredAPI,
		Issuer:    "https://api.openai.com",
		Audience:  "https://api.openai.com",
	}}}
	descriptor, err := credentials.NewDescriptor("openai", "responses", credentials.SchemeAPIKey, credentials.UsageMeteredAPI, "https://api.openai.com", "https://api.openai.com", "")
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Accepts(descriptor) {
		t.Fatal("policy rejected canonical descriptor for case-insensitive binding identifiers")
	}
}

func modelForPolicyTest() model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "https://api.openai.com/v1", "gpt-test")
}
