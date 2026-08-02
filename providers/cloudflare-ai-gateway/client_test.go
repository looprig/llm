package cloudflaregateway_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	cloudflaregateway "github.com/looprig/llm/providers/cloudflare-ai-gateway"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestContracts(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderCloudflareAIGateway, "cf-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return cloudflaregateway.New(selected, key, cloudflaregateway.WithAccountID("account"), cloudflaregateway.WithGatewayID("gateway"))
	})
	contracttest.Responses(t, llm.ProviderCloudflareAIGateway, "cf-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return cloudflaregateway.New(selected, key, cloudflaregateway.WithAccountID("account"), cloudflaregateway.WithGatewayID("gateway"))
	})
	contracttest.AnthropicBearer(t, llm.ProviderCloudflareAIGateway, "cf-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return cloudflaregateway.New(selected, key, cloudflaregateway.WithAccountID("account"), cloudflaregateway.WithGatewayID("gateway"))
	})
}
