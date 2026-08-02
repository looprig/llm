package gmicloud_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/gmicloud"
)

func TestNewCounterIsExplicitlyUnsupported(t *testing.T) {
	counter, err := gmicloud.NewCounter(auth.APIKey("key"))
	if counter != nil || err == nil {
		t.Fatalf("NewCounter() = (%T, %v), want nil and typed error", counter, err)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) || supportErr.Provider != llm.ProviderGMICloud {
		t.Fatalf("NewCounter() error = %T %v, want provider-bound CounterSupportError", err, err)
	}
	if supportErr.APIFormat != model.APIFormatOpenAI {
		t.Fatalf("NewCounter() APIFormat = %q, want %q", supportErr.APIFormat, model.APIFormatOpenAI)
	}
}
