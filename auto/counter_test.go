package auto

import (
	"errors"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	"github.com/looprig/llm"
	geminiprovider "github.com/looprig/llm/providers/gemini"
)

func TestNewCounterProviderMatrix(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		provider    llm.Provider
		model       inference.Model
		key         auth.APIKey
		wantGoogle  bool
		wantDirect  bool
		wantSupport bool
	}{
		{
			name:       "google exact counter",
			provider:   llm.ProviderGoogle,
			model:      counterModel(llm.ProviderGoogle, inference.APIFormatGemini, "https://generativelanguage.googleapis.com/v1beta"),
			key:        "google-test-key",
			wantGoogle: true,
		},
		{
			name:       "bedrock exact counter requires direct construction",
			provider:   llm.ProviderBedrock,
			model:      counterModel(llm.ProviderBedrock, inference.APIFormatAnthropic, ""),
			wantDirect: true,
		},
		{
			name:        "chutes has no exact counter",
			provider:    llm.ProviderChutes,
			model:       counterModel(llm.ProviderChutes, inference.APIFormatOpenAI, "https://api.chutes.ai"),
			wantSupport: true,
		},
		{
			name:        "phala has no exact counter",
			provider:    llm.ProviderPhala,
			model:       counterModel(llm.ProviderPhala, inference.APIFormatOpenAI, "https://api.phala.network/v1"),
			wantSupport: true,
		},
		{
			name:        "openrouter has no exact counter",
			provider:    llm.ProviderOpenRouter,
			model:       counterModel(llm.ProviderOpenRouter, inference.APIFormatOpenAI, "https://openrouter.ai/api/v1"),
			wantSupport: true,
		},
		{
			name:        "lmstudio has no provider exact counter",
			provider:    llm.ProviderLMStudio,
			model:       counterModel(llm.ProviderLMStudio, inference.APIFormatOpenAI, "http://localhost:1234/v1"),
			wantSupport: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := NewCounter(tt.model, tt.key)
			if tt.wantGoogle {
				if err != nil {
					t.Fatalf("NewCounter() error = %v", err)
				}
				if _, ok := got.(*geminiprovider.Counter); !ok {
					t.Fatalf("NewCounter() = %T, want *gemini.Counter", got)
				}
				if got.CounterCapability().Quality != inference.CountQualityExactProvider {
					t.Errorf("CounterCapability().Quality = %v, want exact provider", got.CounterCapability().Quality)
				}
				return
			}

			if got != nil {
				t.Fatalf("NewCounter() = %T alongside error, want nil", got)
			}
			if tt.wantDirect {
				var directErr *llm.CounterDirectConstructionError
				if !errors.As(err, &directErr) {
					t.Fatalf("NewCounter() error = %T, want *llm.CounterDirectConstructionError", err)
				}
				if directErr.Provider != tt.provider || directErr.Reason != llm.CounterDirectConstructionNeedsSigV4 || directErr.Use != llm.CounterConstructorBedrock {
					t.Errorf("CounterDirectConstructionError = %+v", directErr)
				}
				return
			}
			if tt.wantSupport {
				var supportErr *llm.CounterSupportError
				if !errors.As(err, &supportErr) {
					t.Fatalf("NewCounter() error = %T, want *llm.CounterSupportError", err)
				}
				if supportErr.Provider != tt.provider || supportErr.Reason != llm.CounterSupportExactUnavailable {
					t.Errorf("CounterSupportError = %+v", supportErr)
				}
			}
		})
	}
}

func TestNewCounterOrderedErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		model          inference.Model
		key            auth.APIKey
		wantValidation bool
		wantAuth       bool
		wantSupport    bool
		wantDirect     bool
	}{
		{
			name:           "unknown provider validates before support",
			model:          counterModel(llm.Provider("future"), inference.APIFormatOpenAI, "https://future.example.test"),
			key:            "key",
			wantValidation: true,
		},
		{
			name:           "self contradictory known provider validates before support",
			model:          counterModel(llm.ProviderGoogle, inference.APIFormatOpenAI, "https://generativelanguage.googleapis.com/v1beta"),
			key:            "key",
			wantValidation: true,
		},
		{
			name:           "empty model validates before support",
			model:          inference.Model{},
			key:            "key",
			wantValidation: true,
		},
		{
			name:     "google support reaches auth validation",
			model:    counterModel(llm.ProviderGoogle, inference.APIFormatGemini, ""),
			wantAuth: true,
		},
		{
			name:        "unsupported keyed provider reports support before missing key",
			model:       counterModel(llm.ProviderChutes, inference.APIFormatOpenAI, ""),
			wantSupport: true,
		},
		{
			name:       "bedrock reports direct construction independent of api key",
			model:      counterModel(llm.ProviderBedrock, inference.APIFormatAnthropic, ""),
			wantDirect: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := NewCounter(tt.model, tt.key)
			if got != nil {
				t.Fatalf("NewCounter() = %T alongside error, want nil", got)
			}
			if err == nil {
				t.Fatal("NewCounter() error = nil, want typed error")
			}
			var validationErr *inference.ValidationError
			var authErr *llm.AuthRequiredError
			var supportErr *llm.CounterSupportError
			var directErr *llm.CounterDirectConstructionError
			if errors.As(err, &validationErr) != tt.wantValidation || errors.As(err, &authErr) != tt.wantAuth || errors.As(err, &supportErr) != tt.wantSupport || errors.As(err, &directErr) != tt.wantDirect {
				t.Errorf("NewCounter() error = %T; validation=%v auth=%v support=%v direct=%v", err, errors.As(err, &validationErr), errors.As(err, &authErr), errors.As(err, &supportErr), errors.As(err, &directErr))
			}
		})
	}
}

func counterModel(provider llm.Provider, format inference.APIFormat, baseURL string) inference.Model {
	return inference.CustomModel(inference.ProviderName(provider), format, baseURL, "counter-test-model")
}
