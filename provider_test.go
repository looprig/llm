package llm_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference"
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
				var ve *inference.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("error is %T, want *inference.ValidationError", err)
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
		want     inference.AuthKind
		wantErr  bool
	}{
		{name: "lmstudio needs none", provider: llm.ProviderLMStudio, want: inference.AuthNone},
		{name: "phala needs api key", provider: llm.ProviderPhala, want: inference.AuthAPIKey},
		{name: "chutes needs api key", provider: llm.ProviderChutes, want: inference.AuthAPIKey},
		{name: "openrouter needs api key", provider: llm.ProviderOpenRouter, want: inference.AuthAPIKey},
		{name: "bedrock needs sigv4", provider: llm.ProviderBedrock, want: llm.AuthSigV4},
		{name: "google needs api key", provider: llm.ProviderGoogle, want: inference.AuthAPIKey},
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
				var ve *inference.ValidationError
				if !errors.As(err, &ve) {
					t.Errorf("RequiredAuth() error = %v, want *inference.ValidationError", err)
				}
			}
			if got != tt.want {
				t.Errorf("RequiredAuth() = %q, want %q", got, tt.want)
			}
		})
	}
}
