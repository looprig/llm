package deepinfra_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/deepinfra"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderDeepInfra, auth.APIKey("deepinfra-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return deepinfra.New(selected, key)
	})
}

func TestNewAnthropicContract(t *testing.T) {
	contracttest.Anthropic(t, llm.ProviderDeepInfra, auth.APIKey("deepinfra-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return deepinfra.New(selected, key)
	})
}
