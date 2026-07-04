package llm_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/llm"
)

func TestAuthRequiredError(t *testing.T) {
	t.Parallel()
	err := error(&llm.AuthRequiredError{Provider: llm.ProviderPhala, Kind: inference.AuthAPIKey})
	var are *llm.AuthRequiredError
	if !errors.As(err, &are) {
		t.Fatalf("errors.As failed for *AuthRequiredError")
	}
	if are.Provider != llm.ProviderPhala || are.Kind != inference.AuthAPIKey {
		t.Errorf("fields not preserved: %+v", are)
	}
	if msg := err.Error(); !strings.Contains(msg, "phala") || !strings.Contains(msg, string(inference.AuthAPIKey)) {
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
// inference.AuthKind value defined in llm (inference owns only the generic kinds).
func TestAuthSigV4Kind(t *testing.T) {
	t.Parallel()
	if llm.AuthSigV4 == inference.AuthNone || llm.AuthSigV4 == inference.AuthAPIKey {
		t.Errorf("AuthSigV4 %q collides with a generic inference AuthKind", llm.AuthSigV4)
	}
	if string(llm.AuthSigV4) != "sigv4" {
		t.Errorf("AuthSigV4 = %q, want sigv4", llm.AuthSigV4)
	}
}
