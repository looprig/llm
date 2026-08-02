package digitalocean_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/digitalocean"
)

func TestNewCounterIsExplicitlyUnsupported(t *testing.T) {
	counter, err := digitalocean.NewCounter(auth.APIKey("key"))
	if counter != nil || err == nil {
		t.Fatalf("NewCounter() = (%T, %v), want nil and typed error", counter, err)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderDigitalOcean {
		t.Fatalf("NewCounter() error = %T %v, want provider-bound CounterSupportError", err, err)
	}
}
