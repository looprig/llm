package gmicloud_test

import (
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"testing"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/gmicloud"
	"github.com/looprig/llm/providers/internal/contracttest"
)

func TestNewAnthropicContract(t *testing.T) {
	contracttest.Anthropic(t, llm.ProviderGMICloud, auth.APIKey("gmi-key"), func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return gmicloud.New(selected, key)
	})
}
