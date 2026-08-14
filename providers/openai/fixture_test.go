package openai_test

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/looprig/inference/codec/conformance"
)

// The rule this file exists to enforce: a fixture is evidence about Looprig's
// correctness only if the provider could really have sent it. Every fixture
// under testdata/ is therefore validated against OpenAI's own published schema
// on every run, BEFORE any Looprig decoder is allowed to see it. The decoder
// tests in chat_fixture_test.go and responses_fixture_test.go call
// chatFixture / chatStreamFixture / responsesFixture / responsesStreamFixture,
// which gate first and return the bytes second; there is no way to reach the
// bytes without passing the gate.

const (
	chatDir      = "testdata/chat"
	responsesDir = "testdata/responses"
)

// readFixture reads a checked-in fixture. Nothing is generated or templated:
// the bytes on disk are the bytes the gate validates and the decoder consumes.
func readFixture(t testing.TB, dir, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(%s/%s) error = %v", dir, name, err)
	}
	return raw
}

// chatFixture gate-validates a Chat Completions response fixture and returns it.
func chatFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw := readFixture(t, chatDir, name)
	conformance.MustValidate(t, "openai", "chat_completion", raw)
	return raw
}

// chatStreamFixture gate-validates every frame of a Chat Completions SSE
// fixture and returns the whole body.
func chatStreamFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw := readFixture(t, chatDir, name)
	conformance.MustValidateStream(t, "openai", "chat_completion_chunk", raw)
	return raw
}

// responsesFixture gate-validates a Responses response fixture and returns it.
func responsesFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw := readFixture(t, responsesDir, name)
	conformance.MustValidate(t, "openai-responses", "response", raw)
	return raw
}

// responsesStreamFixture gate-validates every frame of a Responses SSE fixture
// and returns the whole body. MustValidateStream additionally cross-checks each
// frame's SSE "event:" name against the payload's own `type` discriminator, so
// a frame that is individually legal but mislabelled still fails.
func responsesStreamFixture(t testing.TB, name string) []byte {
	t.Helper()
	raw := readFixture(t, responsesDir, name)
	conformance.MustValidateStream(t, "openai-responses", "stream_event", raw)
	return raw
}

// TestEveryFixtureIsALegalProviderPayload walks the corpus itself rather than a
// hand-maintained list, so a fixture added to testdata/ can never sit there
// ungated. It also pins the corpus size: silently losing fixtures is a
// regression in coverage that no individual decoder test would notice.
func TestEveryFixtureIsALegalProviderPayload(t *testing.T) {
	t.Parallel()

	cases := []struct {
		dir        string
		format     string
		objectKind string
		streamKind string
		want       int
	}{
		{chatDir, "openai", "chat_completion", "chat_completion_chunk", 30},
		{responsesDir, "openai-responses", "response", "stream_event", 30},
	}

	for _, tc := range cases {
		t.Run(tc.dir, func(t *testing.T) {
			t.Parallel()

			entries, err := os.ReadDir(tc.dir)
			if err != nil {
				t.Fatalf("ReadDir(%s) error = %v", tc.dir, err)
			}
			names := make([]string, 0, len(entries))
			for _, e := range entries {
				if !e.IsDir() {
					names = append(names, e.Name())
				}
			}
			sort.Strings(names)
			if len(names) != tc.want {
				t.Errorf("%s holds %d fixtures, want %d", tc.dir, len(names), tc.want)
			}

			for _, name := range names {
				raw := readFixture(t, tc.dir, name)
				switch {
				case strings.HasSuffix(name, ".sse"):
					if n := conformance.MustValidateStream(t, tc.format, tc.streamKind, raw); n == 0 {
						t.Errorf("%s/%s validated no frames", tc.dir, name)
					}
				case strings.HasSuffix(name, ".json"):
					conformance.MustValidate(t, tc.format, tc.objectKind, raw)
				default:
					t.Errorf("%s/%s is neither a .json nor a .sse fixture", tc.dir, name)
				}
			}
		})
	}
}
