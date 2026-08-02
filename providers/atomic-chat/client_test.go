package atomicchat_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/atomic-chat"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderAtomicChat, auth.APIKey(""), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return atomicchat.New(selected, key)
	})
}
