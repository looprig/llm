package deepseek_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/deepseek"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderDeepSeek, auth.APIKey("key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return deepseek.New(selected, key)
	})
}
