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
		Audience:  "api://openai",
	}}}
	descriptor, err := credentials.NewDescriptor("openai", "responses", credentials.SchemeAPIKey, credentials.UsageMeteredAPI, "https://api.openai.com", "api://openai", "")
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
