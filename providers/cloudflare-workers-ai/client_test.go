package cloudflareworkers_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	cloudflareworkers "github.com/looprig/llm/providers/cloudflare-workers-ai"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderCloudflareWorkersAI, "cf-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return cloudflareworkers.New(selected, key, cloudflareworkers.WithAccountID("account"))
	})
}
