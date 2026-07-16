package llm

import (
	"testing"

	model "github.com/looprig/inference/model"
)

// TestAPIFormatBedrockConverse pins the provider-named Converse dialect value and
// confirms it is distinct from the neutral dialect names that live in inference.
func TestAPIFormatBedrockConverse(t *testing.T) {
	t.Parallel()
	if APIFormatBedrockConverse != model.APIFormat("bedrock-converse") {
		t.Errorf("APIFormatBedrockConverse = %q, want %q", APIFormatBedrockConverse, "bedrock-converse")
	}
	for _, f := range []model.APIFormat{
		model.APIFormatOpenAI,
		model.APIFormatAnthropic,
		model.APIFormatGemini,
	} {
		if APIFormatBedrockConverse == f {
			t.Errorf("APIFormatBedrockConverse collides with neutral dialect %q", f)
		}
	}
}
