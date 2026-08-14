//go:build live

// Package livetest holds LIVE provider conformance probes: they open real
// network connections to real (paid) inference endpoints and are therefore
// gated behind the `live` build tag so no ordinary build, vet, or CI run can
// reach them. Run them deliberately:
//
//	go test -tags live -v -run TestLive ./providers/internal/livetest/...
//
// What these probes are for. Every other test in this repository holds an
// encoded request against a published JSON Schema, or decodes a fixture. Both
// answer "is this body legal per the spec?" — neither answers "does a server
// that actually implements the spec accept it?" A schema gate is only as strong
// as the schema (see providers/internal/conformance's unenforced-constraint
// report), and a fixture only proves we can read something someone already
// wrote down. These probes close that gap for the small number of endpoints the
// developer running them has credentials for.
//
// What they are NOT. Every endpoint reachable through the local model catalogue
// is a GATEWAY, not an origin API. An Anthropic-format endpoint fronting MiniMax
// proves that our encoder emits a body that gateway's Anthropic-compatible
// parser accepts; it does NOT prove api.anthropic.com would accept it, nor that
// the gateway enforces everything Anthropic does. Read every result as evidence
// about the specific (gateway, model) pair named in the subtest, and never
// generalize a pass to the origin vendor or a failure to the format.
//
// Credentials. Keys are read at run time from the developer's own carbon model
// catalogue (~/.looprig/carbon/models.json by default, overridable with
// LOOPRIG_LIVE_MODELS). Nothing here writes a key to a file, and everything the
// probes print goes through scrub, which replaces every loaded key value and
// every known key-shaped prefix with a redaction marker. A missing catalogue, a
// missing alias, or a missing key is a t.Skip, so the suite is a clean no-op for
// anyone who has not opted in.
//
// Traffic goes through a loopback recording proxy (see recorder.go) so a failure
// can report the exact request we emitted and the exact error body the server
// returned. failure.APIError is sanitized by construction — it deliberately
// retains no response body — and a status code alone cannot tell you WHICH field
// the server objected to, which is the only thing a conformance probe is for.
package livetest
