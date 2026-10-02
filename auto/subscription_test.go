package auto

import (
	"github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"testing"
)

func TestSubscriptionDispatch(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	if !dynamicPolicySupported(selected) {
		t.Fatal("subscription must support call-scoped credentials")
	}
	if _, err := New(selected, "api-key"); err == nil {
		t.Fatal("accepted API-key construction for subscription")
	}
	if _, err := constructInnerConfig(selected, "credential-source", options{}, true); err != nil {
		t.Fatal(err)
	}
	if _, err := NewCounter(selected, ""); err == nil {
		t.Fatal("subscription has no documented exact counter")
	} else if _, ok := err.(*llm.CounterSupportError); !ok {
		t.Fatalf("unclassified counter: %v", err)
	}
}
