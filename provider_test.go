package llm_test

import (
	"errors"
	"testing"

	auth "github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

func TestProviderRequiresKey(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		provider llm.Provider
		want     bool
		wantErr  bool
	}{
		{name: "lmstudio no key", provider: llm.ProviderLMStudio, want: false, wantErr: false},
		{name: "phala requires key", provider: llm.ProviderPhala, want: true, wantErr: false},
		{name: "chutes requires key", provider: llm.ProviderChutes, want: true, wantErr: false},
		{name: "openrouter requires key", provider: llm.ProviderOpenRouter, want: true, wantErr: false},
		{name: "openai requires key", provider: llm.ProviderOpenAI, want: true, wantErr: false},
		{name: "anthropic requires key", provider: llm.ProviderAnthropic, want: true, wantErr: false},
		{name: "xai requires key", provider: llm.ProviderXAI, want: true, wantErr: false},
		{name: "azure requires key", provider: llm.ProviderAzure, want: true, wantErr: false},
		{name: "google requires key", provider: llm.ProviderGoogle, want: true, wantErr: false},
		{name: "bedrock uses sigv4 not api key", provider: llm.ProviderBedrock, want: false, wantErr: false},
		{name: "unknown errors", provider: llm.Provider("bogus"), want: false, wantErr: true},
		{name: "empty errors", provider: llm.Provider(""), want: false, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := tt.provider.RequiresKey()
			if (err != nil) != tt.wantErr {
				t.Fatalf("RequiresKey() err = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("RequiresKey() = %v, want %v", got, tt.want)
			}
			if tt.wantErr {
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("error is %T, want *model.ValidationError", err)
				}
			}
		})
	}
}

func TestProviderRequiredAuth(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		provider llm.Provider
		want     auth.AuthKind
		wantErr  bool
	}{
		{name: "lmstudio needs none", provider: llm.ProviderLMStudio, want: auth.AuthNone},
		{name: "phala needs api key", provider: llm.ProviderPhala, want: auth.AuthAPIKey},
		{name: "chutes needs api key", provider: llm.ProviderChutes, want: auth.AuthAPIKey},
		{name: "openrouter needs api key", provider: llm.ProviderOpenRouter, want: auth.AuthAPIKey},
		{name: "openai needs api key", provider: llm.ProviderOpenAI, want: auth.AuthAPIKey},
		{name: "anthropic needs api key", provider: llm.ProviderAnthropic, want: auth.AuthAPIKey},
		{name: "xai needs api key", provider: llm.ProviderXAI, want: auth.AuthAPIKey},
		{name: "azure needs api key", provider: llm.ProviderAzure, want: auth.AuthAPIKey},
		{name: "bedrock needs sigv4", provider: llm.ProviderBedrock, want: llm.AuthSigV4},
		{name: "google needs api key", provider: llm.ProviderGoogle, want: auth.AuthAPIKey},
		{name: "empty is error", provider: "", wantErr: true},
		{name: "unknown is error", provider: "cohere", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := tt.provider.RequiredAuth()
			if (err != nil) != tt.wantErr {
				t.Fatalf("RequiredAuth() err = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("RequiredAuth() error = %v, want *model.ValidationError", err)
				}
			}
			if got != tt.want {
				t.Errorf("RequiredAuth() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestOpenAIResponsesAPIFormatIsDistinctFromChatCompletions(t *testing.T) {
	t.Parallel()

	if model.APIFormatOpenAIResponses == model.APIFormatOpenAI {
		t.Fatalf("Responses API format = %q, want a distinct format from Chat Completions", model.APIFormatOpenAIResponses)
	}
	if got, want := string(model.APIFormatOpenAIResponses), "openai-responses"; got != want {
		t.Fatalf("Responses API format = %q, want %q", got, want)
	}
}
