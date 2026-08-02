package vercel_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/vercel"
)

func TestContracts(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderVercel, "vercel-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return vercel.New(selected, key)
	})
	contracttest.Responses(t, llm.ProviderVercel, "vercel-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return vercel.New(selected, key)
	})
	contracttest.AnthropicBearer(t, llm.ProviderVercel, "vercel-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return vercel.New(selected, key)
	})
}
