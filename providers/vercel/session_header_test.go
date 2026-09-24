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

// TestSessionIDNotForwarded guards the shared simple/compat path: a gateway
// that documents no session header must not receive Request.SessionID.
func TestSessionIDNotForwarded(t *testing.T) {
	t.Parallel()
	contracttest.SessionIDNotForwarded(t, llm.ProviderVercel, "vercel-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return vercel.New(selected, key)
	})
}
