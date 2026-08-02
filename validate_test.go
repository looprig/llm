package llm_test

import (
	"errors"
	"testing"

	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

// pn converts an llm provider-policy label to the opaque model.ProviderName a
// Model carries on the wire-metadata struct.
func pn(p llm.Provider) model.ProviderName { return model.ProviderName(p) }

// TestValidateModel exercises the provider-policy preset layered on inference's
// structural validation: known provider + supported APIFormat pair, non-empty
// Name, the empty-BaseURL provider-default rule, and the HTTPS-only BaseURL rule
// with its narrow loopback-only HTTP exception. Ported from the pre-split harness
// Model.Validate table.
func TestValidateModel(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		model   model.Model
		wantErr bool
	}{
		{
			name:    "valid phala openai https",
			model:   model.Model{Provider: pn(llm.ProviderPhala), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.phala.network/v1", Name: "zai-org/GLM-4.6"},
			wantErr: false,
		},
		{
			name:    "valid chutes openai https",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "moonshotai/Kimi-K2.6-TEE"},
			wantErr: false,
		},
		{
			name:    "valid openai responses https",
			model:   model.Model{Provider: pn(llm.ProviderOpenAI), APIFormat: model.APIFormatOpenAIResponses, BaseURL: "https://api.openai.com/v1", Name: "gpt-5"},
			wantErr: false,
		},
		{
			name:    "valid anthropic messages https",
			model:   model.Model{Provider: pn(llm.ProviderAnthropic), APIFormat: model.APIFormatAnthropic, BaseURL: "https://api.anthropic.com/v1", Name: "claude-sonnet-4-6"},
			wantErr: false,
		},
		{
			name:    "valid xai responses https",
			model:   model.Model{Provider: pn(llm.ProviderXAI), APIFormat: model.APIFormatOpenAIResponses, BaseURL: "https://api.x.ai/v1", Name: "grok-4-5"},
			wantErr: false,
		},
		{
			name:    "valid azure responses https",
			model:   model.Model{Provider: pn(llm.ProviderAzure), APIFormat: model.APIFormatOpenAIResponses, BaseURL: "https://resource.openai.azure.com/openai/v1", Name: "gpt-4.1"},
			wantErr: false,
		},
		{
			name:    "valid azure responses empty baseurl for resource resolution",
			model:   model.Model{Provider: pn(llm.ProviderAzure), APIFormat: model.APIFormatOpenAIResponses, BaseURL: "", Name: "gpt-4.1"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio openai http localhost",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "http://localhost:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio anthropic https",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatAnthropic, BaseURL: "https://lm.example.test", Name: "claude-local"},
			wantErr: false,
		},
		{
			name:    "boundary http 127.0.0.1 loopback allowed",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "http://127.0.0.1:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http uppercase LOCALHOST allowed",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "http://LOCALHOST:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "boundary http ipv6 loopback ::1 allowed",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "http://[::1]:1234", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "custom origin validates identically (valid)",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: "m", Origin: model.OriginCustom},
			wantErr: false,
		},
		{
			name:    "error unknown provider",
			model:   model.Model{Provider: pn(llm.Provider("bogus")), APIFormat: model.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error empty provider",
			model:   model.Model{Provider: "", APIFormat: model.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair phala anthropic",
			model:   model.Model{Provider: pn(llm.ProviderPhala), APIFormat: model.APIFormatAnthropic, BaseURL: "https://api.phala.network/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair chutes bedrock",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: llm.APIFormatBedrockConverse, BaseURL: "https://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error unsupported pair lmstudio bedrock",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: llm.APIFormatBedrockConverse, BaseURL: "http://localhost:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "valid bedrock anthropic empty baseurl (region-routed)",
			model:   model.Model{Provider: pn(llm.ProviderBedrock), APIFormat: model.APIFormatAnthropic, BaseURL: "", Name: "anthropic.claude-3-5-sonnet-20241022-v2:0"},
			wantErr: false,
		},
		{
			name:    "valid bedrock with explicit https baseurl still validates",
			model:   model.Model{Provider: pn(llm.ProviderBedrock), APIFormat: model.APIFormatAnthropic, BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Name: "anthropic.claude-3-5-sonnet-20241022-v2:0"},
			wantErr: false,
		},
		{
			name:    "error bedrock with non-loopback http baseurl (exception is empty-only)",
			model:   model.Model{Provider: pn(llm.ProviderBedrock), APIFormat: model.APIFormatAnthropic, BaseURL: "http://evil.example.com", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error empty name",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "https://api.chutes.ai", Name: ""},
			wantErr: true,
		},
		{
			name:    "valid chutes empty baseurl accepted",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: false,
		},
		{
			name:    "valid lmstudio empty baseurl accepted",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "", Name: "qwen"},
			wantErr: false,
		},
		{
			name:    "error unknown provider empty baseurl rejected",
			model:   model.Model{Provider: pn(llm.Provider("bogus")), APIFormat: model.APIFormatOpenAI, BaseURL: "", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to remote host",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "http://api.chutes.ai", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url with userinfo credentials",
			model:   model.Model{Provider: pn(llm.ProviderPhala), APIFormat: model.APIFormatOpenAI, BaseURL: "https://user:pass@evil.example.com/v1", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error http to non-loopback ip",
			model:   model.Model{Provider: pn(llm.ProviderLMStudio), APIFormat: model.APIFormatOpenAI, BaseURL: "http://127.0.0.2:1234", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url not a url",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "://not-a-url", Name: "m"},
			wantErr: true,
		},
		{
			name:    "error base url no scheme",
			model:   model.Model{Provider: pn(llm.ProviderChutes), APIFormat: model.APIFormatOpenAI, BaseURL: "api.chutes.ai", Name: "m"},
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
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("ValidateModel() error is %T, want *model.ValidationError", err)
				}
			}
		})
	}
}
