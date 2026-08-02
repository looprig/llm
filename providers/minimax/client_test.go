package minimax_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/minimax"
)

func TestNewAnthropicContract(t *testing.T) {
	contracttest.Anthropic(t, llm.ProviderMiniMax, auth.APIKey("minimax-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return minimax.New(selected, key)
	})
}
