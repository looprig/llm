package llm

import (
	"testing"

	model "github.com/looprig/inference/model"
)

func TestProviderAllowsEmptyBaseURL(t *testing.T) {
	tests := []struct {
		name string
		p    Provider
		want bool
	}{
		{"bedrock region-routed", ProviderBedrock, true},
		{"chutes self-defaults", ProviderChutes, true},
		{"phala self-defaults", ProviderPhala, true},
		{"openrouter self-defaults", ProviderOpenRouter, true},
		{"openai self-defaults", ProviderOpenAI, true},
		{"anthropic self-defaults", ProviderAnthropic, true},
		{"xai self-defaults", ProviderXAI, true},
		{"azure resolves resource endpoint", ProviderAzure, true},
		{"lmstudio self-defaults", ProviderLMStudio, true},
		{"google self-defaults", ProviderGoogle, true},
		{"unknown fails closed", Provider("nope"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.p.allowsEmptyBaseURL(); got != tt.want {
				t.Errorf("allowsEmptyBaseURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestProviderSupportsAPIFormat locks the provider/format truth table, including
// the provider-named APIFormatBedrockConverse constant that lives in llm.
func TestProviderSupportsAPIFormat(t *testing.T) {
	tests := []struct {
		name string
		p    Provider
		f    model.APIFormat
		want bool
	}{
		{"phala openai", ProviderPhala, model.APIFormatOpenAI, true},
		{"phala anthropic no", ProviderPhala, model.APIFormatAnthropic, false},
		{"chutes openai", ProviderChutes, model.APIFormatOpenAI, true},
		{"openrouter openai", ProviderOpenRouter, model.APIFormatOpenAI, true},
		{"openai responses", ProviderOpenAI, model.APIFormatOpenAIResponses, true},
		{"openai chat", ProviderOpenAI, model.APIFormatOpenAI, true},
		{"anthropic messages", ProviderAnthropic, model.APIFormatAnthropic, true},
		{"anthropic responses no", ProviderAnthropic, model.APIFormatOpenAIResponses, false},
		{"xai responses", ProviderXAI, model.APIFormatOpenAIResponses, true},
		{"xai chat", ProviderXAI, model.APIFormatOpenAI, true},
		{"azure responses", ProviderAzure, model.APIFormatOpenAIResponses, true},
		{"azure chat no", ProviderAzure, model.APIFormatOpenAI, false},
		{"lmstudio openai", ProviderLMStudio, model.APIFormatOpenAI, true},
		{"lmstudio anthropic", ProviderLMStudio, model.APIFormatAnthropic, true},
		{"lmstudio bedrock no", ProviderLMStudio, APIFormatBedrockConverse, false},
		{"bedrock anthropic", ProviderBedrock, model.APIFormatAnthropic, true},
		{"bedrock converse", ProviderBedrock, APIFormatBedrockConverse, true},
		{"bedrock openai no", ProviderBedrock, model.APIFormatOpenAI, false},
		{"google gemini", ProviderGoogle, model.APIFormatGemini, true},
		{"google openai no", ProviderGoogle, model.APIFormatOpenAI, false},
		{"unknown supports nothing", Provider("nope"), model.APIFormatOpenAI, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.p.supportsAPIFormat(tt.f); got != tt.want {
				t.Errorf("supportsAPIFormat(%q) = %v, want %v", tt.f, got, tt.want)
			}
		})
	}
}
