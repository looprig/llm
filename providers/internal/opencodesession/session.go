// Package opencodesession carries a conversation's identity to the OpenCode
// endpoints, which read it to keep one conversation on one route and cache
// prefix (OpenCode has said requests without it may eventually be refused).
// Both OpenCode provider packages — Zen (providers/opencode) and Go
// (providers/opencode-go) — register Patch, so the two gateways cannot drift
// into sending different spellings of the header.
package opencodesession

import (
	"net/http"

	"github.com/looprig/inference"
)

// Header is OpenCode's per-conversation request header.
const Header = "x-opencode-session"

// Patch copies the request's optional conversation identity,
// inference.Request.SessionID, verbatim into Header. It is used as a compat
// Definition's PatchHeaders, so it runs for every request format (Chat,
// Responses, Anthropic) and both Invoke and Stream, after WithHeader's static
// headers are in place.
//
// The value needs no checking here: every codec runs
// inference.ValidateRequestFeatures before the route is built, which refuses an
// identity that could not arrive byte-identical as a header value. Patch never
// logs it.
//
// An empty SessionID writes nothing. A caller with no conversation identity
// then sends exactly what it sent before the field existed, instead of
// asserting that every such request belongs to one shared conversation.
//
// A non-empty Header already present is left alone: WithHeader is how an
// operator pins the value deliberately, and a per-request default must not
// silently overwrite an explicit choice — the precedence github-copilot's
// header patch follows for the same reason. (An EMPTY WithHeader value does not
// suppress the identity; the way to send none is to leave SessionID empty.)
func Patch(req inference.Request, headers http.Header) {
	if req.SessionID == "" || headers.Get(Header) != "" {
		return
	}
	headers.Set(Header, req.SessionID)
}
