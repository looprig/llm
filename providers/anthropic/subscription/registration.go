// Package subscription exposes the Anthropic subscription registration policy
// boundary. It deliberately has no credential, request, or discovery
// implementation: the current policy is to reject unsanctioned third-party
// registration attempts before they can become an inference flow.
package subscription

import (
	"time"

	"github.com/looprig/llm"
)

// Status is the immutable state of a subscription registration gate.
type Status string

const (
	// StatusUnavailable means the provider has not published a sanctioned
	// third-party subscription-registration contract.
	StatusUnavailable Status = "unavailable"
	// StatusBlocked means callers must not proceed through this gate.
	StatusBlocked Status = "blocked"

	// RegistrationStatusUnavailable and RegistrationStatusBlocked are explicit
	// aliases for callers that prefer the longer enum names.
	RegistrationStatusUnavailable = StatusUnavailable
	RegistrationStatusBlocked     = StatusBlocked
)

// ReviewedAtDate is the date on which the official evidence for this gate was
// reviewed. It is a date, not a timestamp, so it has no local-time ambiguity.
const ReviewedAtDate = "2026-08-09"

// The evidence set is deliberately fixed and bounded. These are documentation
// references only; no value from them is used to construct a request.
const (
	EvidenceAuthenticationURL  = "https://code.claude.com/docs/en/authentication"
	EvidenceLegalURL           = "https://code.claude.com/docs/en/legal-and-compliance"
	EvidenceAgentOverviewURL   = "https://code.claude.com/docs/en/agent-sdk/overview"
	EvidenceAgentQuickstartURL = "https://code.claude.com/docs/en/agent-sdk/quickstart"
	EvidencePlanURL            = "https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan"
)

const maxEvidenceURLs = 5

// RegistrationGate is an immutable, provider-specific registration policy.
// Its state is private so callers can only observe copies of the metadata.
// Require always fails closed for the current Anthropic subscription policy.
type RegistrationGate struct {
	status       Status
	provider     llm.Provider
	reviewedAt   time.Time
	evidenceURLs [maxEvidenceURLs]string
}

// AnthropicRegistration returns the current Anthropic subscription
// registration gate. Every call returns an independent value; its evidence
// accessor also returns a fresh copy.
func AnthropicRegistration() RegistrationGate {
	return RegistrationGate{
		status:     StatusUnavailable,
		provider:   llm.ProviderAnthropic,
		reviewedAt: time.Date(2026, time.August, 9, 0, 0, 0, 0, time.UTC),
		evidenceURLs: [maxEvidenceURLs]string{
			EvidenceAuthenticationURL,
			EvidenceLegalURL,
			EvidenceAgentOverviewURL,
			EvidenceAgentQuickstartURL,
			EvidencePlanURL,
		},
	}
}

// Status reports the gate's fail-closed status.
func (g RegistrationGate) Status() Status { return g.status }

// Provider reports the provider policy label covered by the gate.
func (g RegistrationGate) Provider() llm.Provider { return g.provider }

// ReviewedAt reports the UTC midnight at which the evidence was reviewed.
func (g RegistrationGate) ReviewedAt() time.Time { return g.reviewedAt }

// ReviewedDate reports the reviewed-at date in ISO-8601 calendar form.
func (g RegistrationGate) ReviewedDate() string {
	if g.reviewedAt.IsZero() {
		return ""
	}
	return g.reviewedAt.UTC().Format("2006-01-02")
}

// EvidenceURLs returns a bounded copy of the official evidence URL metadata.
func (g RegistrationGate) EvidenceURLs() []string {
	urls := make([]string, 0, maxEvidenceURLs)
	for _, raw := range g.evidenceURLs {
		if raw != "" {
			urls = append(urls, raw)
		}
	}
	return urls
}

// Require rejects use of this registration path. A zero-value gate also fails
// closed, so an uninitialized value cannot accidentally authorize a flow.
func (g RegistrationGate) Require() error {
	return &UnsupportedRegistrationError{}
}
