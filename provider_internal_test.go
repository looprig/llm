package llm

import (
	"testing"

	"github.com/looprig/inference"
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
		f    inference.APIFormat
		want bool
	}{
		{"phala openai", ProviderPhala, inference.APIFormatOpenAI, true},
		{"phala anthropic no", ProviderPhala, inference.APIFormatAnthropic, false},
		{"chutes openai", ProviderChutes, inference.APIFormatOpenAI, true},
		{"openrouter openai", ProviderOpenRouter, inference.APIFormatOpenAI, true},
		{"lmstudio openai", ProviderLMStudio, inference.APIFormatOpenAI, true},
		{"lmstudio anthropic", ProviderLMStudio, inference.APIFormatAnthropic, true},
		{"lmstudio bedrock no", ProviderLMStudio, APIFormatBedrockConverse, false},
		{"bedrock anthropic", ProviderBedrock, inference.APIFormatAnthropic, true},
		{"bedrock converse", ProviderBedrock, APIFormatBedrockConverse, true},
		{"bedrock openai no", ProviderBedrock, inference.APIFormatOpenAI, false},
		{"google gemini", ProviderGoogle, inference.APIFormatGemini, true},
		{"google openai no", ProviderGoogle, inference.APIFormatOpenAI, false},
		{"unknown supports nothing", Provider("nope"), inference.APIFormatOpenAI, false},
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
