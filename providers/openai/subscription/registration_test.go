package subscription_test

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openai/subscription"
)

func TestOpenAIRegistrationGateIsBlockedAndBounded(t *testing.T) {
	t.Parallel()

	gate := subscription.OpenAIRegistration()
	if got, want := gate.Status(), subscription.StatusBlocked; got != want {
		t.Fatalf("Status() = %q, want %q", got, want)
	}
	if got, want := gate.Provider(), llm.ProviderOpenAI; got != want {
		t.Fatalf("Provider() = %q, want %q", got, want)
	}
	if got, want := gate.ReviewedAt(), time.Date(2026, time.August, 9, 0, 0, 0, 0, time.UTC); !got.Equal(want) {
		t.Fatalf("ReviewedAt() = %s, want %s", got, want)
	}

	urls := gate.EvidenceURLs()
	if len(urls) == 0 || len(urls) > 4 {
		t.Fatalf("EvidenceURLs() length = %d, want 1..4", len(urls))
	}
	wantURLs := map[string]bool{
		"https://learn.chatgpt.com/docs/auth":                                             true,
		"https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan": true,
		"https://learn.chatgpt.com/docs/enterprise/access-tokens":                         true,
		"https://help.openai.com/en/articles/20001410-sign-in-with-chatgpt":               true,
	}
	for _, raw := range urls {
		parsed, err := url.Parse(raw)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
			t.Errorf("EvidenceURLs() contains unsafe URL %q", raw)
		}
		if !wantURLs[raw] {
			t.Errorf("EvidenceURLs() contains unapproved URL %q", raw)
		}
	}

	urls[0] = "https://example.invalid/changed"
	if got := gate.EvidenceURLs()[0]; got == urls[0] {
		t.Fatal("EvidenceURLs() exposes mutable gate state")
	}
}

func TestRequireReturnsTypedRedactionSafeError(t *testing.T) {
	t.Parallel()

	err := subscription.OpenAIRegistration().Require()
	if err == nil {
		t.Fatal("Require() error = nil, want blocked registration error")
	}
	var unsupported *subscription.UnsupportedRegistrationError
	if !errors.As(err, &unsupported) {
		t.Fatalf("Require() error = %T, want *UnsupportedRegistrationError", err)
	}
	const requiredMessage = "no sanctioned third-party OAuth registration is available"
	if !strings.Contains(err.Error(), requiredMessage) {
		t.Fatalf("Require() error = %q, want phrase %q", err, requiredMessage)
	}

	for _, format := range []string{"%v", "%+v", "%#v", "%s", "%q", "%x", "%X", "%d", "%p"} {
		formatted := fmt.Sprintf(format, err)
		if strings.Contains(strings.ToLower(formatted), "identity") || strings.Contains(strings.ToLower(formatted), "secret") || strings.Contains(strings.ToLower(formatted), "token") {
			t.Errorf("fmt.Sprintf(%q, err) = %q, contains restricted material", format, formatted)
		}
		if strings.Contains(formatted, "%!") {
			t.Errorf("fmt.Sprintf(%q, err) = %q, contains formatting failure", format, formatted)
		}
	}

	var logs bytes.Buffer
	slog.New(slog.NewTextHandler(&logs, nil)).Error("registration", "error", err)
	if got := logs.String(); strings.Contains(strings.ToLower(got), "identity") || strings.Contains(strings.ToLower(got), "secret") || strings.Contains(strings.ToLower(got), "token") {
		t.Fatalf("slog output = %q, contains restricted material", got)
	}
}
