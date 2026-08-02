package azure_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/azure"
)

func TestNewCounterReportsUnsupportedExactCounter(t *testing.T) {
	t.Parallel()

	counter, err := azure.NewCounter("azure-test-key")
	if counter != nil {
		t.Fatalf("NewCounter() = %T alongside error, want nil", counter)
	}
	var supportErr *llm.CounterSupportError
	if !errors.As(err, &supportErr) {
		t.Fatalf("NewCounter() error = %T %v, want *llm.CounterSupportError", err, err)
	}
	if supportErr.Provider != llm.ProviderAzure || supportErr.Reason != llm.CounterSupportExactUnavailable || supportErr.APIFormat != model.APIFormatOpenAIResponses {
		t.Fatalf("CounterSupportError = %+v, want Azure exact-counter-unavailable", supportErr)
	}
}
