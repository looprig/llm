package venice_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/venice"
)

func TestContracts(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderVenice, "venice-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return venice.New(selected, key)
	})
	contracttest.Responses(t, llm.ProviderVenice, "venice-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return venice.New(selected, key)
	})
}
