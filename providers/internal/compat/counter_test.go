package compat_test

import (
	"errors"
	"testing"

	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/compat"
)

func TestUnsupportedCounterIsTypedAndProviderBound(t *testing.T) {
	t.Parallel()

	counter, err := compat.UnsupportedCounter(llm.ProviderOpenRouter, model.APIFormatOpenAI)
	if counter != nil {
		t.Fatalf("UnsupportedCounter() = %T, want nil", counter)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) {
		t.Fatalf("UnsupportedCounter() error = %T, want *llm.CounterSupportError", err)
	}
	if supportErr.Provider != llm.ProviderOpenRouter || supportErr.APIFormat != model.APIFormatOpenAI || supportErr.Reason != llm.CounterSupportExactUnavailable {
		t.Errorf("CounterSupportError = %+v, want OpenRouter/OpenAI/exact-unavailable", supportErr)
	}
}
