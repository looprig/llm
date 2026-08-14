//go:build live

package livetest

import (
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/looprig/core/content"
)

// tokenCount converts a catalogue integer to the neutral token count type,
// clamping a negative value to zero rather than wrapping the unsigned type.
func tokenCount(v int) content.TokenCount {
	if v <= 0 {
		return 0
	}
	return content.TokenCount(v)
}

var (
	secretsMu sync.RWMutex
	secrets   []string
)

// keyShaped matches the credential prefixes this workspace's catalogue is known
// to carry, as a second line of defence behind the exact-value scrub. The exact
// values are always registered; this pattern additionally catches a key that
// reached the output by some path the loader never saw (an echoed request body,
// a server error that quotes the token it rejected).
var keyShaped = regexp.MustCompile(`\b(?:sk-[A-Za-z0-9_\-]{8,}|cpk_[A-Za-z0-9_\-]{8,}|syn_[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})`)

// registerSecrets records every credential in the catalogue so scrub can remove
// it from any diagnostic these probes emit. Longest first, so a key that is a
// prefix of another cannot leave the longer key's tail behind.
func registerSecrets(cat *catalog) {
	secretsMu.Lock()
	defer secretsMu.Unlock()
	for _, row := range cat.Models {
		if len(row.APIKey) >= 8 {
			secrets = append(secrets, row.APIKey)
		}
	}
	sort.Slice(secrets, func(i, j int) bool { return len(secrets[i]) > len(secrets[j]) })
}

// scrub removes credentials from text destined for a test log or a report. It is
// the ONLY way any captured request body, response body, or error message
// reaches t.Log in this package: a live probe's whole value is the server's
// error text, and that text is untrusted with respect to what it echoes back.
func scrub(text string) string {
	secretsMu.RLock()
	known := secrets
	secretsMu.RUnlock()
	for _, secret := range known {
		if secret == "" {
			continue
		}
		text = strings.ReplaceAll(text, secret, redactionFor(secret))
	}
	return keyShaped.ReplaceAllString(text, "<redacted:key-shaped>")
}

// redactionFor names a credential by a short, non-reconstructable prefix so two
// different keys in one diagnostic remain distinguishable.
func redactionFor(secret string) string {
	prefix := secret
	if len(prefix) > 4 {
		prefix = prefix[:4]
	}
	return "<redacted:" + prefix + "...>"
}

// scrubBytes is scrub over a captured wire body, additionally bounding the
// result so one enormous error page cannot flood the test log.
func scrubBytes(body []byte) string {
	const maxLogged = 4096
	text := scrub(string(body))
	if len(text) > maxLogged {
		return text[:maxLogged] + "...<truncated>"
	}
	return text
}
