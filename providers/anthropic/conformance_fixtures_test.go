package anthropic_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/looprig/inference/codec/conformance"
)

// This file is the schema gate for the Anthropic Messages fixture corpus in
// testdata/. Every fixture is validated against the schema derived from the
// OpenAPI document Anthropic points its own SDK at, on every run, BEFORE any
// Looprig decoder is allowed to see it (decode_fixtures_test.go and
// encode_multimodal_test.go both re-validate at the point of use, so no fixture
// can reach a decoder without having passed the gate first).
//
// The corpus is swept as a directory rather than as a hand-maintained list, so
// adding a file to testdata/ without teaching the gate what it is fails the
// build rather than silently escaping validation.

const (
	kindMessage = "message"
	kindEvent   = "stream_event"
	kindRequest = "create_message_request"
)

// gateGapFixtures are wire-real payloads the official document cannot describe.
// Anthropic's MessageStreamEvent union declares exactly six members
// (message_start, message_delta, message_stop, content_block_start,
// content_block_delta, content_block_stop). "ping" and "error" are documented
// on the streaming endpoint and are handled by Anthropic's own SDK, but neither
// appears in the union, so the gate MUST reject them. They are kept as fixtures
// and their rejection is asserted, which pins the gap instead of hiding it: if a
// future spec refresh adds either member, TestGateGapEventsAreRejected fails and
// the fixtures move into the validated set.
var gateGapFixtures = map[string]bool{
	"event_ping.json":           true,
	"event_error.json":          true,
	"stream_ping_and_error.sse": true,
}

// gateRequest holds an ENCODED request body against Anthropic's own
// CreateMessageParams schema. This is the request half of the gate, and it is
// the stronger half: CreateMessageParams is additionalProperties:false and
// requires model/messages/max_tokens, and 83 of the document's 85 request
// object shapes are closed the same way. A response fixture tests our tolerance
// of what Anthropic sends; this tests what WE send, before it can draw a 400.
//
// Call it on the bytes the encoder produced, before any structural assertion,
// so a hand-written assertion can never certify a body the API would reject.
func gateRequest(t testing.TB, body []byte) {
	t.Helper()
	conformance.MustValidateRequest(t, "anthropic", kindRequest, body)
}

// fixture reads one checked-in fixture. The bytes on disk are the bytes
// validated: nothing is templated or generated at test time.
func fixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

// kindOf maps a fixture file name to the api-format kind it claims to be. The
// mapping is by prefix so the corpus is self-describing.
func kindOf(name string) (kind string, mustReject bool, ok bool) {
	switch {
	case strings.HasPrefix(name, "invalid_request_"):
		return kindRequest, true, true
	case strings.HasPrefix(name, "message_"):
		return kindMessage, false, true
	case strings.HasPrefix(name, "event_"):
		return kindEvent, false, true
	case strings.HasPrefix(name, "request_"):
		return kindRequest, false, true
	default:
		return "", false, false
	}
}

func corpus(t testing.TB) []string {
	t.Helper()
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("ReadDir(testdata) error = %v", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// TestEveryFixtureIsSpecLegal is the gate. A JSON fixture is held to its kind's
// schema; an .sse fixture is split into frames and every frame is held to the
// event union, with the SSE event name cross-checked against the payload's own
// discriminator.
func TestEveryFixtureIsSpecLegal(t *testing.T) {
	t.Parallel()

	seen := 0
	for _, name := range corpus(t) {
		if gateGapFixtures[name] {
			continue
		}
		if strings.HasSuffix(name, ".sse") {
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				if n := conformance.MustValidateStream(t, "anthropic", kindEvent, fixture(t, name)); n == 0 {
					t.Fatalf("%s validated no frames", name)
				}
			})
			seen++
			continue
		}
		kind, mustReject, ok := kindOf(name)
		if !ok {
			t.Errorf("fixture %s has no kind mapping; teach kindOf about it rather than letting it skip the gate", name)
			continue
		}
		seen++
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			payload := fixture(t, name)
			err := conformance.Validate("anthropic", kind, payload)
			if mustReject {
				if err == nil {
					t.Fatalf("gate accepted %s, which is not a legal %s payload", name, kind)
				}
				if !validationRejected(err) {
					t.Fatalf("gate errored on %s without reaching the schema, so this case proves nothing about %s:\n%v", name, kind, err)
				}
				t.Logf("gate correctly rejected %s:\n%v", name, err)
				return
			}
			if err != nil {
				t.Fatalf("%v", err)
			}
		})
	}
	if seen < 40 {
		t.Fatalf("swept only %d fixtures; the corpus is smaller than the suite expects", seen)
	}
}

// TestGateGapEventsAreRejected pins the one place the Anthropic corpus outruns
// the official document: "ping" and "error" are real stream events that
// MessageStreamEvent does not declare. Asserting the rejection means the gap is
// recorded as a fact about the spec, and that a spec refresh which closes it
// breaks this test rather than going unnoticed.
func TestGateGapEventsAreRejected(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"event_ping.json", "event_error.json"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate("anthropic", kindEvent, fixture(t, name))
			if err == nil {
				t.Fatalf("gate accepted %s; MessageStreamEvent now declares this member, so it should join the validated corpus", name)
			}
			if !validationRejected(err) {
				t.Fatalf("gate errored on %s without reaching the schema, so the recorded spec gap is unproven:\n%v", name, err)
			}
		})
	}

	// The mixed stream fails as a whole for the same reason: its ping frame is
	// individually illegal even though every other frame is fine.
	frames, err := conformance.ParseSSE(fixture(t, "stream_ping_and_error.sse"))
	if err != nil {
		t.Fatalf("ParseSSE(stream_ping_and_error.sse) error = %v", err)
	}
	rejected := 0
	for _, frame := range frames {
		if conformance.Validate("anthropic", kindEvent, frame.Data) != nil {
			rejected++
		}
	}
	if rejected != 2 {
		t.Fatalf("stream_ping_and_error.sse had %d gate-rejected frames, want exactly the ping and the error", rejected)
	}
}

// TestThinkingBlockRequiredFieldsArePinned states, as an executable fact, the
// constraint the encoder's dedicated thinking wire shapes exist to satisfy:
// RequestThinkingBlock requires signature AND thinking (so the empty-thinking
// "display: omitted" block still has to carry an explicit ""), and
// RequestRedactedThinkingBlock requires data. Both are additionalProperties:
// false, so a block that borrows another variant's field is also illegal.
func TestThinkingBlockRequiredFieldsArePinned(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		block   string
		accept  bool
		because string
	}{
		{
			name:    "thinking with empty text and a signature",
			block:   `{"type":"thinking","thinking":"","signature":"sig"}`,
			accept:  true,
			because: "the display:omitted shape current models return",
		},
		{
			name:    "thinking without the empty thinking key",
			block:   `{"type":"thinking","signature":"sig"}`,
			because: "omitempty dropping `thinking` is the HTTP 400 the encoder was repaired for",
		},
		{
			name:    "thinking without a signature",
			block:   `{"type":"thinking","thinking":"reasoned"}`,
			because: "signature is required",
		},
		{
			name:    "redacted_thinking without data",
			block:   `{"type":"redacted_thinking"}`,
			because: "data is required even when the opaque payload decoded empty",
		},
		{
			name:    "redacted_thinking with data",
			block:   `{"type":"redacted_thinking","data":"AAAA"}`,
			accept:  true,
			because: "the legal shape",
		},
		{
			name:    "thinking carrying a borrowed field",
			block:   `{"type":"thinking","thinking":"","signature":"sig","text":"leak"}`,
			because: "additionalProperties is false on the thinking variant",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"model":"claude-sonnet-4-5","max_tokens":16,"messages":[{"role":"assistant","content":[` + tc.block + `]}]}`)
			err := conformance.Validate("anthropic", kindRequest, body)
			if tc.accept && err != nil {
				t.Fatalf("gate rejected a legal block (%s):\n%v", tc.because, err)
			}
			if !tc.accept {
				if err == nil {
					t.Fatalf("gate accepted an illegal block (%s)", tc.because)
				}
				if !validationRejected(err) {
					t.Fatalf("gate errored without reaching the schema (%s), so this case proves nothing:\n%v", tc.because, err)
				}
			}
		})
	}
}

// TestCorpusFixturesAreCanonicalJSON keeps the corpus reviewable: a fixture that
// is not parseable JSON (or, for a stream, not parseable SSE) is a broken
// artifact regardless of what the schema says about it.
func TestCorpusFixturesAreCanonicalJSON(t *testing.T) {
	t.Parallel()

	for _, name := range corpus(t) {
		raw := fixture(t, name)
		if strings.HasSuffix(name, ".sse") {
			if _, err := conformance.ParseSSE(raw); err != nil {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		var probe any
		if err := json.Unmarshal(raw, &probe); err != nil {
			t.Errorf("%s: not valid JSON: %v", name, err)
		}
	}
}

// validationRejected reports whether err is a gate rejection rather than a
// lookup or parse failure, so a test that expects a rejection cannot be
// satisfied by a typo in the format or kind.
func validationRejected(err error) bool {
	var failure *conformance.Failure
	return errors.As(err, &failure)
}
