package githubcopilot_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/github-copilot"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestContracts(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderGitHubCopilot, "copilot-token", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return githubcopilot.New(selected, key)
	})
	contracttest.Responses(t, llm.ProviderGitHubCopilot, "copilot-token", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return githubcopilot.New(selected, key)
	})
	contracttest.AnthropicBearer(t, llm.ProviderGitHubCopilot, "copilot-token", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return githubcopilot.New(selected, key)
	})
}
