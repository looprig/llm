package zenmux_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/internal/simple"
	"github.com/looprig/llm/providers/zenmux"
)

func TestContracts(t *testing.T) {
	contracttest.NoDefaultOpenCodeAttribution(t, llm.ProviderZenMux, "zen-key", "HTTP-Referer", func(selected model.Model, key auth.APIKey, options ...simple.Option) (inference.Client, error) {
		return zenmux.New(selected, key, options...)
	})
	contracttest.OpenAI(t, llm.ProviderZenMux, "zen-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return zenmux.New(selected, key)
	})
	contracttest.Anthropic(t, llm.ProviderZenMux, "zen-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return zenmux.New(selected, key)
	})
	contracttest.Responses(t, llm.ProviderZenMux, "zen-key", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return zenmux.New(selected, key)
	})
}
