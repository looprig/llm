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

// TestSessionIDNotForwarded guards the other PatchHeaders user: Copilot's
// per-request header patch must not start carrying Request.SessionID.
func TestSessionIDNotForwarded(t *testing.T) {
	t.Parallel()
	contracttest.SessionIDNotForwarded(t, llm.ProviderGitHubCopilot, "copilot-token", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return githubcopilot.New(selected, key)
	})
}
