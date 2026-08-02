package llamacpp_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/llamacpp"
)

func TestNewCounterIsExplicitlyUnsupported(t *testing.T) {
	counter, err := llamacpp.NewCounter(auth.APIKey("key"))
	if counter != nil || err == nil {
		t.Fatalf("NewCounter() = (%T, %v), want nil and typed error", counter, err)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderLlamaCPP {
		t.Fatalf("NewCounter() error = %T %v, want llama.cpp CounterSupportError", err, err)
	}
}
