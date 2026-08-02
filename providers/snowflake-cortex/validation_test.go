package snowflake_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	snowflake "github.com/looprig/llm/providers/snowflake-cortex"
)

func TestRejectsAccountThatContainsURLDelimiters(t *testing.T) {
	selected := model.CustomModel(model.ProviderName(llm.ProviderSnowflakeCortex), model.APIFormatOpenAI, "", "model")
	_, err := snowflake.New(selected, auth.APIKey("cortex-token"), snowflake.WithAccount("org/account"))
	var configErr *snowflake.ConfigurationError
	if !errors.As(err, &configErr) || configErr.Reason != snowflake.AccountInvalid {
		t.Fatalf("New() error = %T %v, want invalid account configuration", err, err)
	}
}
