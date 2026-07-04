package llm

import (
	"testing"

	"github.com/looprig/inference"
)

// TestAPIFormatBedrockConverse pins the provider-named Converse dialect value and
// confirms it is distinct from the neutral dialect names that live in inference.
func TestAPIFormatBedrockConverse(t *testing.T) {
	t.Parallel()
	if APIFormatBedrockConverse != inference.APIFormat("bedrock-converse") {
		t.Errorf("APIFormatBedrockConverse = %q, want %q", APIFormatBedrockConverse, "bedrock-converse")
	}
	for _, f := range []inference.APIFormat{
		inference.APIFormatOpenAI,
		inference.APIFormatAnthropic,
		inference.APIFormatGemini,
	} {
		if APIFormatBedrockConverse == f {
			t.Errorf("APIFormatBedrockConverse collides with neutral dialect %q", f)
		}
	}
}
