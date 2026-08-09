package subscription_test

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/anthropic/subscription"
)

func TestAnthropicRegistrationGateIsUnavailableAndBounded(t *testing.T) {
	t.Parallel()

	gate := subscription.AnthropicRegistration()
	if got, want := gate.Status(), subscription.StatusUnavailable; got != want {
		t.Fatalf("Status() = %q, want %q", got, want)
	}
	if got, want := gate.Provider(), llm.ProviderAnthropic; got != want {
		t.Fatalf("Provider() = %q, want %q", got, want)
	}
	if got, want := gate.ReviewedAt(), time.Date(2026, time.August, 9, 0, 0, 0, 0, time.UTC); !got.Equal(want) {
		t.Fatalf("ReviewedAt() = %s, want %s", got, want)
	}
	if got, want := gate.ReviewedDate(), subscription.ReviewedAtDate; got != want {
		t.Fatalf("ReviewedDate() = %q, want %q", got, want)
	}

	urls := gate.EvidenceURLs()
	if len(urls) == 0 || len(urls) > 5 {
		t.Fatalf("EvidenceURLs() length = %d, want 1..5", len(urls))
	}
	wantURLs := map[string]bool{
		subscription.EvidenceAuthenticationURL:  true,
		subscription.EvidenceLegalURL:           true,
		subscription.EvidenceAgentOverviewURL:   true,
		subscription.EvidenceAgentQuickstartURL: true,
		subscription.EvidenceThirdPartyUsageURL: true,
	}
	allowedHosts := map[string]bool{
		"code.claude.com":    true,
		"support.claude.com": true,
	}
	seen := make(map[string]bool, len(urls))
	for _, raw := range urls {
		parsed, err := url.Parse(raw)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
			t.Errorf("EvidenceURLs() contains unsafe URL %q", raw)
		}
		if !allowedHosts[parsed.Hostname()] {
			t.Errorf("EvidenceURLs() contains a non-provider source %q", raw)
		}
		if seen[raw] {
			t.Errorf("EvidenceURLs() contains duplicate source %q", raw)
		}
		seen[raw] = true
		if strings.Contains(parsed.Host, "api.") || strings.Contains(parsed.Path, "/oauth/") {
			t.Errorf("EvidenceURLs() exposes a provider transport or OAuth endpoint %q", raw)
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

func TestAnthropicRegistrationCannotConstructOrLogin(t *testing.T) {
	t.Parallel()

	var zero subscription.RegistrationGate
	for name, gate := range map[string]subscription.RegistrationGate{
		"published gate": subscription.AnthropicRegistration(),
		"zero gate":      zero,
	} {
		name, gate := name, gate
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			err := gate.Require()
			var unsupported *subscription.UnsupportedRegistrationError
			if !errors.As(err, &unsupported) {
				t.Fatalf("Require() error = %T, want *UnsupportedRegistrationError", err)
			}
		})
	}

	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	for _, name := range []string{"oauth.go", "client.go"} {
		path := filepath.Join(filepath.Dir(sourceFile), name)
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("subscription package exposes %s; unsupported registration must not have a login/client surface", name)
		}
	}
}

func TestAnthropicRegistrationDoesNotExposeIdentityOrEndpoints(t *testing.T) {
	t.Parallel()

	typeOfGate := reflect.TypeOf(subscription.RegistrationGate{})
	for index := 0; index < typeOfGate.NumField(); index++ {
		field := typeOfGate.Field(index)
		if field.PkgPath == "" {
			t.Fatalf("RegistrationGate field %q is exported; registration metadata must not expose client identity or endpoints", field.Name)
		}
	}
	for _, name := range []string{"ClientID", "ClientSecret", "Issuer", "Audience", "AuthorizationURL", "TokenURL", "RedirectURI", "EndpointOrigins", "GrantTypes", "Scopes"} {
		if _, ok := typeOfGate.MethodByName(name); ok {
			t.Fatalf("RegistrationGate method %q exposes unapproved registration metadata", name)
		}
	}

	formatted := fmt.Sprintf("%#v", subscription.AnthropicRegistration())
	for _, forbidden := range []string{"evidenceURLs", "client_id", "client secret", "issuer", "audience", "redirect_uri", "endpoint", "origin", "grant", "scope", "api.anthropic.com"} {
		if strings.Contains(strings.ToLower(formatted), forbidden) {
			t.Fatalf("formatted gate = %q, exposes forbidden registration detail %q", formatted, forbidden)
		}
	}
}

func TestRequireReturnsTypedRedactionSafeError(t *testing.T) {
	t.Parallel()

	err := subscription.AnthropicRegistration().Require()
	if err == nil {
		t.Fatal("Require() error = nil, want unavailable registration error")
	}
	var unsupported *subscription.UnsupportedRegistrationError
	if !errors.As(err, &unsupported) {
		t.Fatalf("Require() error = %T, want *UnsupportedRegistrationError", err)
	}
	const requiredMessage = "third-party subscription OAuth is not approved"
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
