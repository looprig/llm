package llm_test

import (
	"errors"
	"strings"
	"testing"

	auth "github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

func TestAuthRequiredError(t *testing.T) {
	t.Parallel()
	err := error(&llm.AuthRequiredError{Provider: llm.ProviderPhala, Kind: auth.AuthAPIKey})
	var are *llm.AuthRequiredError
	if !errors.As(err, &are) {
		t.Fatalf("errors.As failed for *AuthRequiredError")
	}
	if are.Provider != llm.ProviderPhala || are.Kind != auth.AuthAPIKey {
		t.Errorf("fields not preserved: %+v", are)
	}
	if msg := err.Error(); !strings.Contains(msg, "phala") || !strings.Contains(msg, string(auth.AuthAPIKey)) {
		t.Errorf("message missing provider/kind: %q", msg)
	}
}

func TestAttestationError(t *testing.T) {
	t.Parallel()
	cause := errors.New("boom")
	err := error(&llm.AttestationError{Reason: "quote_invalid", Err: cause})
	var ae *llm.AttestationError
	if !errors.As(err, &ae) {
		t.Fatalf("errors.As failed for *AttestationError")
	}
	if ae.Reason != "quote_invalid" {
		t.Errorf("Reason = %q, want quote_invalid", ae.Reason)
	}
	if !errors.Is(err, cause) {
		t.Errorf("Unwrap did not chain the cause")
	}
	if msg := err.Error(); !strings.Contains(msg, "quote_invalid") || !strings.Contains(msg, "boom") {
		t.Errorf("message missing reason/cause: %q", msg)
	}

	// Nil cause: message must still render without panicking and omit the cause.
	nilErr := &llm.AttestationError{Reason: "e2ee_failed"}
	if got := nilErr.Error(); !strings.Contains(got, "e2ee_failed") {
		t.Errorf("nil-cause message = %q, want it to contain the reason", got)
	}
	if nilErr.Unwrap() != nil {
		t.Errorf("Unwrap() on nil cause = %v, want nil", nilErr.Unwrap())
	}
}

// TestAuthSigV4Kind confirms the provider-specific SigV4 credential kind is an
// auth.AuthKind value defined in llm (inference owns only the generic kinds).
func TestAuthSigV4Kind(t *testing.T) {
	t.Parallel()
	if llm.AuthSigV4 == auth.AuthNone || llm.AuthSigV4 == auth.AuthAPIKey {
		t.Errorf("AuthSigV4 %q collides with a generic inference AuthKind", llm.AuthSigV4)
	}
	if string(llm.AuthSigV4) != "sigv4" {
		t.Errorf("AuthSigV4 = %q, want sigv4", llm.AuthSigV4)
	}
}

func TestCounterSupportError(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		err       *llm.CounterSupportError
		provider  llm.Provider
		reason    llm.CounterSupportReason
		apiFormat model.APIFormat
	}{
		{
			name:      "unsupported gateway dialect is inspectable",
			err:       &llm.CounterSupportError{Provider: llm.ProviderOpenRouter, Reason: llm.CounterSupportExactUnavailable, APIFormat: model.APIFormatOpenAI},
			provider:  llm.ProviderOpenRouter,
			reason:    llm.CounterSupportExactUnavailable,
			apiFormat: model.APIFormatOpenAI,
		},
		{
			name:      "zero value is safe",
			err:       &llm.CounterSupportError{},
			provider:  "",
			reason:    "",
			apiFormat: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := error(tt.err)
			var supportErr *llm.CounterSupportError
			if !errors.As(err, &supportErr) {
				t.Fatalf("errors.As(%T) failed for *CounterSupportError", err)
			}
			if supportErr.Provider != tt.provider || supportErr.Reason != tt.reason || supportErr.APIFormat != tt.apiFormat {
				t.Errorf("CounterSupportError = %+v, want provider %q reason %q API format %q", supportErr, tt.provider, tt.reason, tt.apiFormat)
			}
			if got := supportErr.Error(); got == "" {
				t.Error("CounterSupportError.Error() is empty")
			} else if tt.apiFormat != "" && !strings.Contains(got, string(tt.apiFormat)) {
				t.Errorf("CounterSupportError.Error() = %q, want API format %q", got, tt.apiFormat)
			}
		})
	}
}

func TestCounterDirectConstructionError(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		err      *llm.CounterDirectConstructionError
		provider llm.Provider
		reason   llm.CounterDirectConstructionReason
		use      llm.CounterConstructor
	}{
		{
			name: "bedrock directive is inspectable",
			err: &llm.CounterDirectConstructionError{
				Provider: llm.ProviderBedrock,
				Reason:   llm.CounterDirectConstructionNeedsSigV4,
				Use:      llm.CounterConstructorBedrock,
			},
			provider: llm.ProviderBedrock,
			reason:   llm.CounterDirectConstructionNeedsSigV4,
			use:      llm.CounterConstructorBedrock,
		},
		{
			name:     "zero value is safe",
			err:      &llm.CounterDirectConstructionError{},
			provider: "",
			reason:   "",
			use:      "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := error(tt.err)
			var directErr *llm.CounterDirectConstructionError
			if !errors.As(err, &directErr) {
				t.Fatalf("errors.As(%T) failed for *CounterDirectConstructionError", err)
			}
			if directErr.Provider != tt.provider || directErr.Reason != tt.reason || directErr.Use != tt.use {
				t.Errorf("CounterDirectConstructionError = %+v, want provider %q reason %q use %q", directErr, tt.provider, tt.reason, tt.use)
			}
			if got := directErr.Error(); got == "" {
				t.Error("CounterDirectConstructionError.Error() is empty")
			}
		})
	}
}
