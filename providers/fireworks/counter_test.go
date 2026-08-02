package fireworks_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/fireworks"
)

func TestNewCounterIsExplicitlyUnsupported(t *testing.T) {
	counter, err := fireworks.NewCounter(auth.APIKey("key"))
	if counter != nil || err == nil {
		t.Fatalf("NewCounter() = (%T, %v), want nil and typed error", counter, err)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderFireworks {
		t.Fatalf("NewCounter() error = %T %v, want provider-bound CounterSupportError", err, err)
	}
}
