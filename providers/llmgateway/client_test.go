package llmgateway_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/internal/simple"
	"github.com/looprig/llm/providers/llmgateway"
)

func TestContracts(t *testing.T) {
	contracttest.NoDefaultOpenCodeAttribution(t, llm.ProviderLLMGateway, "gateway-key", "HTTP-Referer", func(selected model.Model, key auth.APIKey, options ...simple.Option) (inference.Client, error) {
		return llmgateway.New(selected, key, options...)
	})
	contracttest.OpenAI(t, llm.ProviderLLMGateway, "gateway-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return llmgateway.New(selected, key)
	})
	contracttest.AnthropicBearer(t, llm.ProviderLLMGateway, "gateway-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return llmgateway.New(selected, key)
	})
}
