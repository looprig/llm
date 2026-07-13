package auto

import (
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	"github.com/looprig/llm"
	geminiprovider "github.com/looprig/llm/providers/gemini"
)

// NewCounter validates model and resolves its exact provider context counter from
// the same (Model, APIKey) inputs accepted by New. It never performs provider I/O
// and never silently substitutes an estimator.
//
// Error ordering is intentional: model validation runs first so unknown or
// contradictory models remain validation failures. Exact-counter support is then
// classified before API-key presence, so a known unsupported provider cannot be
// mistaken for a supported counter with a missing key. Google is the only exact
// counter constructible from these inputs; its constructor performs the final
// fail-closed API-key validation.
func NewCounter(model inference.Model, key auth.APIKey) (inference.ContextCounter, error) {
	if err := llm.ValidateModel(model); err != nil {
		return nil, err
	}

	provider := llm.Provider(model.Provider)
	switch provider {
	case llm.ProviderGoogle:
		return geminiprovider.NewCounter(key)
	case llm.ProviderBedrock:
		switch model.APIFormat {
		case inference.APIFormatAnthropic:
			return nil, &llm.CounterDirectConstructionError{
				Provider: provider,
				Reason:   llm.CounterDirectConstructionNeedsSigV4,
				Use:      llm.CounterConstructorBedrock,
			}
		case llm.APIFormatBedrockConverse:
			return nil, &llm.CounterSupportError{
				Provider:  provider,
				Reason:    llm.CounterSupportAPIFormatUnavailable,
				APIFormat: model.APIFormat,
			}
		default:
			return nil, &inference.ValidationError{Field: "APIFormat", Reason: "context counter support is unclassified"}
		}
	case llm.ProviderChutes, llm.ProviderPhala, llm.ProviderOpenRouter, llm.ProviderLMStudio:
		return nil, &llm.CounterSupportError{
			Provider:  provider,
			Reason:    llm.CounterSupportExactUnavailable,
			APIFormat: model.APIFormat,
		}
	default:
		// ValidateModel rejects every provider not in llm's canonical registry.
		// Keep the default fail-closed so a future registry expansion cannot
		// accidentally acquire a counter or estimator through fallthrough.
		return nil, &inference.ValidationError{Field: "Provider", Reason: "context counter support is unclassified"}
	}
}
