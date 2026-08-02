package snowflake_test

import (
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/contracttest"
	"github.com/looprig/llm/providers/snowflake-cortex"
)

func TestOpenAIContract(t *testing.T) {
	contracttest.OpenAI(t, llm.ProviderSnowflakeCortex, "snowflake-token", func(selected model.Model, key auth.APIKey) (inference.Client, error) {
		return snowflake.New(selected, key, snowflake.WithAccount("org-account"))
	})
}
