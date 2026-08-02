package azurecognitive_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/azure-cognitive-services"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAIWithHeader(t, llm.ProviderAzureCognitiveServices, auth.APIKey("azure-key"), "api-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return azurecognitive.New(selected, key, azurecognitive.WithResourceName("resource"))
	})
}

func TestNewAnthropicContract(t *testing.T) {
	contracttest.Anthropic(t, llm.ProviderAzureCognitiveServices, auth.APIKey("azure-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return azurecognitive.New(selected, key, azurecognitive.WithResourceName("resource"))
	})
}
