package llm_test

import (
	"github.com/looprig/credentials"
	"github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"testing"
)

func TestSubscriptionProviderPolicy(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	b := policy.Accepted[0]
	if b.Provider != "openai-subscription" || b.Transport != "responses" || b.Scheme != credentials.SchemeOAuth || b.Usage != credentials.UsageSubscription || b.Issuer != "https://auth.openai.com" || b.Audience != "https://api.openai.com" {
		t.Fatalf("wrong binding: %+v", b)
	}
	for _, base := range []string{"https://chatgpt.com/backend-api/codex", "https://example.com/v1", "https://api.openai.com/v2"} {
		selected.BaseURL = base
		if llm.ValidateModel(selected) == nil {
			t.Fatalf("accepted endpoint %s", base)
		}
	}
	selected.BaseURL = ""
	selected.APIFormat = model.APIFormatOpenAI
	if llm.ValidateModel(selected) == nil {
		t.Fatal("accepted Chat Completions for subscription")
	}
}
