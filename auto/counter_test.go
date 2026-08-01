package auto

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"

	contextcount "github.com/looprig/inference/contextcount"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
	anthropicprovider "github.com/looprig/llm/providers/anthropic"
	geminiprovider "github.com/looprig/llm/providers/gemini"
	openaiprovider "github.com/looprig/llm/providers/openai"
)

func TestNewCounterProviderMatrix(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		provider      llm.Provider
		model         model.Model
		key           auth.APIKey
		wantGoogle    bool
		wantDirect    bool
		wantSupport   bool
		supportReason llm.CounterSupportReason
	}{
		{
			name:          "phala openai has no exact counter",
			provider:      llm.ProviderPhala,
			model:         counterModel(llm.ProviderPhala, model.APIFormatOpenAI, "https://api.phala.network/v1"),
			wantSupport:   true,
			supportReason: llm.CounterSupportExactUnavailable,
		},
		{
			name:          "chutes openai has no exact counter",
			provider:      llm.ProviderChutes,
			model:         counterModel(llm.ProviderChutes, model.APIFormatOpenAI, "https://api.chutes.ai"),
			wantSupport:   true,
			supportReason: llm.CounterSupportExactUnavailable,
		},
		{
			name:          "openrouter openai has no exact counter",
			provider:      llm.ProviderOpenRouter,
			model:         counterModel(llm.ProviderOpenRouter, model.APIFormatOpenAI, "https://openrouter.ai/api/v1"),
			wantSupport:   true,
			supportReason: llm.CounterSupportExactUnavailable,
		},
		{
			name:       "google exact counter",
			provider:   llm.ProviderGoogle,
			model:      counterModel(llm.ProviderGoogle, model.APIFormatGemini, "https://generativelanguage.googleapis.com/v1beta"),
			key:        "google-test-key",
			wantGoogle: true,
		},
		{
			name:       "bedrock exact counter requires direct construction",
			provider:   llm.ProviderBedrock,
			model:      counterModel(llm.ProviderBedrock, model.APIFormatAnthropic, ""),
			wantDirect: true,
		},
		{
			name:          "bedrock converse has no exact counter for dialect",
			provider:      llm.ProviderBedrock,
			model:         counterModel(llm.ProviderBedrock, llm.APIFormatBedrockConverse, ""),
			wantSupport:   true,
			supportReason: llm.CounterSupportAPIFormatUnavailable,
		},
		{
			name:          "lmstudio openai has no provider exact counter",
			provider:      llm.ProviderLMStudio,
			model:         counterModel(llm.ProviderLMStudio, model.APIFormatOpenAI, "http://localhost:1234/v1"),
			wantSupport:   true,
			supportReason: llm.CounterSupportExactUnavailable,
		},
		{
			name:          "lmstudio anthropic has no provider exact counter",
			provider:      llm.ProviderLMStudio,
			model:         counterModel(llm.ProviderLMStudio, model.APIFormatAnthropic, "http://localhost:1234/v1"),
			wantSupport:   true,
			supportReason: llm.CounterSupportExactUnavailable,
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
				if got.CounterCapability().Quality != contextcount.CountQualityExactProvider {
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
				if supportErr.Provider != tt.provider || supportErr.Reason != tt.supportReason || supportErr.APIFormat != tt.model.APIFormat {
					t.Errorf("CounterSupportError = %+v, want provider %q reason %q API format %q", supportErr, tt.provider, tt.supportReason, tt.model.APIFormat)
				}
			}
		})
	}
}

func TestNewCounterPriorityProviders(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		model       model.Model
		key         auth.APIKey
		wantType    any
		wantSupport bool
	}{
		{
			name:     "openai responses exact counter",
			model:    counterModel(llm.ProviderOpenAI, model.APIFormatOpenAIResponses, ""),
			key:      "sk-openai-counter",
			wantType: (*openaiprovider.Counter)(nil),
		},
		{
			name:     "anthropic messages exact counter",
			model:    counterModel(llm.ProviderAnthropic, model.APIFormatAnthropic, ""),
			key:      "sk-ant-counter",
			wantType: (*anthropicprovider.Counter)(nil),
		},
		{
			name:        "xai responses has no exact counter",
			model:       counterModel(llm.ProviderXAI, model.APIFormatOpenAIResponses, ""),
			key:         "xai-counter",
			wantSupport: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := NewCounter(tt.model, tt.key)
			if tt.wantSupport {
				if got != nil {
					t.Fatalf("NewCounter() = %T alongside error, want nil", got)
				}
				var supportErr *llm.CounterSupportError
				if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderXAI || supportErr.Reason != llm.CounterSupportExactUnavailable {
					t.Fatalf("NewCounter() error = %T %v, want xAI unsupported counter", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewCounter() error = %v", err)
			}
			switch tt.wantType.(type) {
			case *openaiprovider.Counter:
				if _, ok := got.(*openaiprovider.Counter); !ok {
					t.Fatalf("NewCounter() = %T, want *openai.Counter", got)
				}
			case *anthropicprovider.Counter:
				if _, ok := got.(*anthropicprovider.Counter); !ok {
					t.Fatalf("NewCounter() = %T, want *anthropic.Counter", got)
				}
			}
			if err := got.CounterCapability().Validate(); err != nil {
				t.Fatalf("CounterCapability().Validate() error = %v", err)
			}
		})
	}
}

func TestNewCounterOrderedErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		model          model.Model
		key            auth.APIKey
		wantValidation bool
		wantAuth       bool
		wantSupport    bool
		wantDirect     bool
	}{
		{
			name:           "unknown provider validates before support",
			model:          counterModel(llm.Provider("future"), model.APIFormatOpenAI, "https://future.example.test"),
			key:            "key",
			wantValidation: true,
		},
		{
			name:           "self contradictory known provider validates before support",
			model:          counterModel(llm.ProviderGoogle, model.APIFormatOpenAI, "https://generativelanguage.googleapis.com/v1beta"),
			key:            "key",
			wantValidation: true,
		},
		{
			name:           "empty model validates before support",
			model:          model.Model{},
			key:            "key",
			wantValidation: true,
		},
		{
			name:     "google support reaches auth validation",
			model:    counterModel(llm.ProviderGoogle, model.APIFormatGemini, ""),
			wantAuth: true,
		},
		{
			name:        "unsupported keyed provider reports support before missing key",
			model:       counterModel(llm.ProviderChutes, model.APIFormatOpenAI, ""),
			wantSupport: true,
		},
		{
			name:       "bedrock reports direct construction independent of api key",
			model:      counterModel(llm.ProviderBedrock, model.APIFormatAnthropic, ""),
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
			var validationErr *model.ValidationError
			var authErr *llm.AuthRequiredError
			var supportErr *llm.CounterSupportError
			var directErr *llm.CounterDirectConstructionError
			if errors.As(err, &validationErr) != tt.wantValidation || errors.As(err, &authErr) != tt.wantAuth || errors.As(err, &supportErr) != tt.wantSupport || errors.As(err, &directErr) != tt.wantDirect {
				t.Errorf("NewCounter() error = %T; validation=%v auth=%v support=%v direct=%v", err, errors.As(err, &validationErr), errors.As(err, &authErr), errors.As(err, &supportErr), errors.As(err, &directErr))
			}
			if tt.wantAuth && (authErr.Provider != llm.ProviderGoogle || authErr.Kind != auth.AuthAPIKey) {
				t.Errorf("AuthRequiredError = %+v, want google API-key requirement", authErr)
			}
		})
	}
}

func TestResolveCounterDialects(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		provider       llm.Provider
		apiFormat      model.APIFormat
		key            auth.APIKey
		wantGoogle     bool
		wantDirect     bool
		wantSupport    bool
		wantValidation bool
		supportReason  llm.CounterSupportReason
	}{
		{
			name:       "google gemini constructs exact counter",
			provider:   llm.ProviderGoogle,
			apiFormat:  model.APIFormatGemini,
			key:        "google-test-key",
			wantGoogle: true,
		},
		{
			name:          "future google dialect fails closed",
			provider:      llm.ProviderGoogle,
			apiFormat:     model.APIFormat("future-google-dialect"),
			wantSupport:   true,
			supportReason: llm.CounterSupportAPIFormatUnavailable,
		},
		{
			name:       "bedrock anthropic directs construction",
			provider:   llm.ProviderBedrock,
			apiFormat:  model.APIFormatAnthropic,
			wantDirect: true,
		},
		{
			name:          "bedrock converse is unsupported",
			provider:      llm.ProviderBedrock,
			apiFormat:     llm.APIFormatBedrockConverse,
			wantSupport:   true,
			supportReason: llm.CounterSupportAPIFormatUnavailable,
		},
		{
			name:           "unknown provider fails closed",
			provider:       llm.Provider("future-provider"),
			apiFormat:      model.APIFormat("future-dialect"),
			wantValidation: true,
		},
		{
			name:          "future bedrock dialect fails closed",
			provider:      llm.ProviderBedrock,
			apiFormat:     model.APIFormat("future-bedrock-dialect"),
			wantSupport:   true,
			supportReason: llm.CounterSupportAPIFormatUnavailable,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := resolveCounter(tt.provider, tt.apiFormat, tt.key)
			if tt.wantGoogle {
				if err != nil {
					t.Fatalf("resolveCounter() error = %v", err)
				}
				if _, ok := got.(*geminiprovider.Counter); !ok {
					t.Fatalf("resolveCounter() = %T, want *gemini.Counter", got)
				}
				return
			}
			if got != nil {
				t.Fatalf("resolveCounter() = %T alongside error, want nil", got)
			}
			if tt.wantValidation {
				var validationErr *model.ValidationError
				if !errors.As(err, &validationErr) {
					t.Fatalf("resolveCounter() error = %T, want *model.ValidationError", err)
				}
				if validationErr.Field != "Provider" {
					t.Errorf("ValidationError.Field = %q, want Provider", validationErr.Field)
				}
				return
			}
			if tt.wantDirect {
				var directErr *llm.CounterDirectConstructionError
				if !errors.As(err, &directErr) {
					t.Fatalf("resolveCounter() error = %T, want *llm.CounterDirectConstructionError", err)
				}
				if directErr.Provider != tt.provider || directErr.Reason != llm.CounterDirectConstructionNeedsSigV4 || directErr.Use != llm.CounterConstructorBedrock {
					t.Errorf("CounterDirectConstructionError = %+v", directErr)
				}
				return
			}
			if tt.wantSupport {
				var supportErr *llm.CounterSupportError
				if !errors.As(err, &supportErr) {
					t.Fatalf("resolveCounter() error = %T, want *llm.CounterSupportError", err)
				}
				if supportErr.Provider != tt.provider || supportErr.Reason != tt.supportReason || supportErr.APIFormat != tt.apiFormat {
					t.Errorf("CounterSupportError = %+v, want provider %q reason %q API format %q", supportErr, tt.provider, tt.supportReason, tt.apiFormat)
				}
			}
		})
	}
}

func counterModel(provider llm.Provider, format model.APIFormat, baseURL string) model.Model {
	return model.CustomModel(model.ProviderName(provider), format, baseURL, "counter-test-model")
}
