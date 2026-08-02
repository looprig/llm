package cerebras_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/cerebras"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderCerebras, auth.APIKey("key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return cerebras.New(selected, key)
	})
}
