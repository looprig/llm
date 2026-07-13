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
	return resolveCounter(llm.Provider(model.Provider), model.APIFormat, key)
}

// resolveCounter classifies an already validated provider/dialect pair. Keeping
// dialect dispatch explicit here makes registry expansion fail closed even before
// ValidateModel is updated to admit a new provider format.
func resolveCounter(provider llm.Provider, apiFormat inference.APIFormat, key auth.APIKey) (inference.ContextCounter, error) {
	switch provider {
	case llm.ProviderGoogle:
		if apiFormat != inference.APIFormatGemini {
			return nil, &llm.CounterSupportError{
				Provider:  provider,
				Reason:    llm.CounterSupportAPIFormatUnavailable,
				APIFormat: apiFormat,
			}
		}
		return geminiprovider.NewCounter(key)
	case llm.ProviderBedrock:
		if apiFormat == inference.APIFormatAnthropic {
			return nil, &llm.CounterDirectConstructionError{
				Provider: provider,
				Reason:   llm.CounterDirectConstructionNeedsSigV4,
				Use:      llm.CounterConstructorBedrock,
			}
		}
		return nil, &llm.CounterSupportError{
			Provider:  provider,
			Reason:    llm.CounterSupportAPIFormatUnavailable,
			APIFormat: apiFormat,
		}
	case llm.ProviderChutes, llm.ProviderPhala, llm.ProviderOpenRouter, llm.ProviderLMStudio:
		return nil, &llm.CounterSupportError{
			Provider:  provider,
			Reason:    llm.CounterSupportExactUnavailable,
			APIFormat: apiFormat,
		}
	default:
		// ValidateModel rejects every provider not in llm's canonical registry.
		// Keep the default fail-closed so a future registry expansion cannot
		// accidentally acquire a counter or estimator through fallthrough.
		return nil, &inference.ValidationError{Field: "Provider", Reason: "context counter support is unclassified"}
	}
}
