package opencode_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/opencode"
)

func TestRejectsOpenCodeGoModelIdentity(t *testing.T) {
	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenCodeGo), model.APIFormatOpenAI, "https://example.test/v1", "model")
	if _, err := opencode.New(selected, auth.APIKey("key")); err == nil {
		t.Fatal("New() error = nil, want OpenCode constructor to reject opencode-go identity")
	}
}

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderOpenCode, auth.APIKey("key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return opencode.New(selected, key)
	})
}

func TestNewResponsesContract(t *testing.T) {
	contracttest.Responses(t, llm.ProviderOpenCode, auth.APIKey("key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return opencode.New(selected, key)
	})
}

func TestNewAnthropicContract(t *testing.T) {
	contracttest.Anthropic(t, llm.ProviderOpenCode, auth.APIKey("key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return opencode.New(selected, key)
	})
}
