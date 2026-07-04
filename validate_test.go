package llm_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/llm"
)

// pn converts an llm provider-policy label to the opaque inference.ProviderName a
// Model carries on the wire-metadata struct.
func pn(p llm.Provider) inference.ProviderName { return inference.ProviderName(p) }

// TestValidateModel exercises the provider-policy preset layered on inference's
// structural validation: known provider + supported APIFormat pair, non-empty
// Name, the empty-BaseURL provider-default rule, and the HTTPS-only BaseURL rule
// with its narrow loopback-only HTTP exception. Ported from the pre-split harness
// Model.Validate table.
func TestValidateModel(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		model   inference.Model
		wantErr bool
	}{
		{
			name:    "valid phala openai https",
			model:   inference.Model{Provider: pn(llm.ProviderPhala), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.phala.network/v1", Name: "zai-org/GLM-4.6"},
			wantErr: false,
		},
		{
			name:    "valid chutes openai https",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "moonshotai/Kimi-K2.6-TEE"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio openai http localhost",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://localhost:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio anthropic https",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatAnthropic, BaseURL: "https://lm.example.test", Name: "claude-local"},
			wantErr: false,
		},
		{
			name:    "boundary http 127.0.0.1 loopback allowed",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://127.0.0.1:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http uppercase LOCALHOST allowed",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://LOCALHOST:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http ipv6 loopback ::1 allowed",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://[::1]:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "custom origin validates identically (valid)",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "m", Origin: inference.OriginCustom},
			wantErr: false,
		},
		{
			name:    "error unknown provider",
			model:   inference.Model{Provider: pn(llm.Provider("bogus")), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error empty provider",
			model:   inference.Model{Provider: "", APIFormat: inference.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair phala anthropic",
			model:   inference.Model{Provider: pn(llm.ProviderPhala), APIFormat: inference.APIFormatAnthropic, BaseURL: "https://api.phala.network/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair chutes bedrock",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: llm.APIFormatBedrockConverse, BaseURL: "https://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair lmstudio bedrock",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: llm.APIFormatBedrockConverse, BaseURL: "http://localhost:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "valid bedrock anthropic empty baseurl (region-routed)",
			model:   inference.Model{Provider: pn(llm.ProviderBedrock), APIFormat: inference.APIFormatAnthropic, BaseURL: "", Name: "anthropic.claude-3-5-sonnet-20241022-v2:0"},
			wantErr: false,
		},
		{
			name:    "valid bedrock with explicit https baseurl still validates",
			model:   inference.Model{Provider: pn(llm.ProviderBedrock), APIFormat: inference.APIFormatAnthropic, BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Name: "anthropic.claude-3-5-sonnet-20241022-v2:0"},
			wantErr: false,
		},
		{
			name:    "error bedrock with non-loopback http baseurl (exception is empty-only)",
			model:   inference.Model{Provider: pn(llm.ProviderBedrock), APIFormat: inference.APIFormatAnthropic, BaseURL: "http://evil.example.com", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error empty name",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: ""},
			wantErr: true,
		},
		{
			name:    "valid chutes empty baseurl accepted",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio empty baseurl accepted",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "error unknown provider empty baseurl rejected",
			model:   inference.Model{Provider: pn(llm.Provider("bogus")), APIFormat: inference.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to remote host",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url with userinfo credentials",
			model:   inference.Model{Provider: pn(llm.ProviderPhala), APIFormat: inference.APIFormatOpenAI, BaseURL: "https://user:pass@evil.example.com/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to non-loopback ip",
			model:   inference.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: inference.APIFormatOpenAI, BaseURL: "http://127.0.0.2:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url not a url",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "://not-a-url", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url no scheme",
			model:   inference.Model{Provider: pn(llm.ProviderChutes), APIFormat: inference.APIFormatOpenAI, BaseURL: "api.chutes.ai", Name: "m"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := llm.ValidateModel(tt.model)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ValidateModel() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var ve *inference.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("ValidateModel() error is %T, want *inference.ValidationError", err)
				}
			}
		})
	}
}
