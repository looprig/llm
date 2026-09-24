package opencode_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/internal/simple"
	"github.com/looprig/llm/providers/opencode"
)

// TestOpenCodeSessionHeaderContract pins the header OpenCode reads for
// per-conversation routing and cache affinity: Request.SessionID verbatim as
// x-opencode-session on every request format, absent without one, an explicit
// WithHeader winning, and an unsendable identity refused before any I/O.
func TestOpenCodeSessionHeaderContract(t *testing.T) {
	t.Parallel()
	contracttest.SessionHeader(t, llm.ProviderOpenCode, auth.APIKey("key"), "x-opencode-session", func(selected model.Model, key auth.APIKey, options ...simple.Option) (inference.Client, error) {
		return opencode.New(selected, key, options...)
	})
}
