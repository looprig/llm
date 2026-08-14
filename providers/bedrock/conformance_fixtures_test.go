package bedrock_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/looprig/inference/codec/conformance"
)

// This file is the schema gate for the Bedrock Converse fixture corpus in
// testdata/. Every fixture is validated against the schema derived from AWS's
// own Smithy 2.0 model of bedrock-runtime, on every run, BEFORE any Looprig
// decoder sees it. Fixtures reach the decoders only through the helpers in
// decode_fixtures_test.go, which validate again at the point of use.
//
// Bedrock does not use server-sent events: a ConverseStream frame is a separate
// event-stream message whose body is one member of ConverseStreamOutput. Stream
// fixtures are therefore either a single union-shaped object or a JSON array of
// them, never an .sse body.

const (
	kindResponse = "converse_response"
	kindStream   = "converse_stream_output"
	kindRequest  = "converse_request"
)

// gateConverseRequest holds an ENCODED Converse body against AWS's own
// ConverseRequest schema. It is the request half of the gate, and it catches
// our encoder rather than our tolerance of AWS's output.
//
// Read the verdict with the document's real strength in mind. AWS marks only
// modelId @required on ConverseRequest and modelId travels in the URI path, so
// the derived document requires NOTHING at the top level: a body with no
// messages passes. What it does enforce, and enforce hard, is @pattern,
// @length, enum membership and union arity — ToolUseId's anchored
// ^[a-zA-Z0-9_.:-]+$ with min 1 / max 64, ToolName's ^[a-zA-Z0-9_-]+$ with max
// 64, ImageSource being bytes|s3Location and nothing else, CachePointType's
// single "default", ToolResultStatus, ImageFormat, DocumentBlock.name's length
// cap. Those are where Bedrock's real 400s come from, so presence assertions
// stay the caller's job and are kept alongside every gate call below.
func gateConverseRequest(t testing.TB, body []byte) {
	t.Helper()
	conformance.MustValidateRequest(t, "bedrock-converse", kindRequest, body)
}

// gateInvokeModelBody holds an InvokeModel body against ANTHROPIC's
// CreateMessageParams, which is what that body actually is.
//
// Bedrock's InvokeModel route carries a first-party Anthropic Messages body
// with two transport edits: the model id moves to the URI path, and
// anthropic_version is added. Neither edit is describable by Anthropic's own
// document — CreateMessageParams requires model and is
// additionalProperties:false — so the two transport fields are reversed here
// and the remainder is held to the real schema. That is the whole point:
// everything the Anthropic encoder produced is validated, and the two fields
// the reversal touches are asserted directly by the transform tests.
func gateInvokeModelBody(t testing.TB, body []byte, modelID string) {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("InvokeModel body is not a JSON object: %v", err)
	}
	if _, present := fields["model"]; present {
		t.Fatalf("InvokeModel body carries a top-level model; Bedrock takes it in the URI: %s", body)
	}
	version, present := fields["anthropic_version"]
	if !present {
		t.Fatalf("InvokeModel body is missing anthropic_version: %s", body)
	}
	if string(version) != `"bedrock-2023-05-31"` {
		t.Fatalf("anthropic_version = %s, want \"bedrock-2023-05-31\"", version)
	}
	delete(fields, "anthropic_version")
	encoded, err := json.Marshal(modelID)
	if err != nil {
		t.Fatalf("marshal model id: %v", err)
	}
	fields["model"] = encoded
	restored, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("re-marshal InvokeModel body: %v", err)
	}
	conformance.MustValidateRequest(t, "anthropic", "create_message_request", restored)
}

func fixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

// kindOf maps a fixture file name to the api-format kind it claims to be.
func kindOf(name string) (kind string, mustReject bool, ok bool) {
	switch {
	case strings.HasPrefix(name, "invalid_request_"):
		return kindRequest, true, true
	case strings.HasPrefix(name, "converse_"):
		return kindResponse, false, true
	case strings.HasPrefix(name, "request_"):
		return kindRequest, false, true
	case strings.HasPrefix(name, "stream_"):
		return kindStream, false, true
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

// frames splits a fixture that holds a whole stream into its individual union
// frames; a fixture holding one frame yields a single-element slice. Keeping
// both forms behind one helper means the gate treats a lifecycle fixture exactly
// as it treats the frames it is built from.
func frames(t testing.TB, raw []byte) []json.RawMessage {
	t.Helper()
	if bytes.HasPrefix(bytes.TrimSpace(raw), []byte("[")) {
		var list []json.RawMessage
		if err := json.Unmarshal(raw, &list); err != nil {
			t.Fatalf("decode stream fixture: %v", err)
		}
		return list
	}
	return []json.RawMessage{raw}
}

// TestEveryFixtureIsSpecLegal is the gate.
func TestEveryFixtureIsSpecLegal(t *testing.T) {
	t.Parallel()

	seen := 0
	for _, name := range corpus(t) {
		kind, mustReject, ok := kindOf(name)
		if !ok {
			t.Errorf("fixture %s has no kind mapping; teach kindOf about it rather than letting it skip the gate", name)
			continue
		}
		seen++
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			raw := fixture(t, name)
			if kind == kindStream {
				for i, frame := range frames(t, raw) {
					if err := conformance.Validate("bedrock-converse", kind, frame); err != nil {
						t.Fatalf("frame %d:\n%v", i, err)
					}
				}
				return
			}
			err := conformance.Validate("bedrock-converse", kind, raw)
			if mustReject {
				if err == nil {
					t.Fatalf("gate accepted %s, which is not a legal %s payload", name, kind)
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

// TestToolUseIdConstraintsArePinned states the ToolUseId constraints from the
// Smithy model as executable facts: min 1, max 64, and the ANCHORED character
// class ^[a-zA-Z0-9_.:-]+$. The anchoring matters — an unanchored pattern would
// accept "bad id/here" because a legal substring exists inside it — so a value
// whose illegal characters sit in the middle is included deliberately.
func TestToolUseIdConstraintsArePinned(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		id     string
		accept bool
	}{
		{name: "typical Bedrock id", id: "tooluse_weather_1", accept: true},
		{name: "the full legal character class", id: "aZ0_.:-", accept: true},
		{name: "exactly 64 characters", id: strings.Repeat("a", 64), accept: true},
		{name: "65 characters exceeds maxLength", id: strings.Repeat("a", 65)},
		{name: "empty violates minLength", id: ""},
		{name: "a slash is outside the character class", id: "tooluse/weather"},
		{name: "an interior space proves the pattern is anchored", id: "tooluse weather"},
		{name: "a trailing newline proves the pattern is anchored", id: "tooluse_weather\n"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body, err := json.Marshal(map[string]any{"messages": []any{map[string]any{
				"role": "assistant",
				"content": []any{map[string]any{"toolUse": map[string]any{
					"toolUseId": tc.id, "name": "search", "input": map[string]any{},
				}}},
			}}})
			if err != nil {
				t.Fatalf("marshal probe: %v", err)
			}
			gateErr := conformance.Validate("bedrock-converse", kindRequest, body)
			if tc.accept && gateErr != nil {
				t.Fatalf("gate rejected a legal toolUseId %q:\n%v", tc.id, gateErr)
			}
			if !tc.accept && gateErr == nil {
				t.Fatalf("gate accepted an illegal toolUseId %q", tc.id)
			}
		})
	}
}

// TestUnionConstraintsArePinned records the Smithy enums and union memberships
// the corpus depends on. CachePointType in particular has exactly one legal
// value and it is lower-case "default", not "DEFAULT".
func TestUnionConstraintsArePinned(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		block  string
		accept bool
	}{
		{name: "cachePoint default", block: `{"cachePoint":{"type":"default"}}`, accept: true},
		{name: "cachePoint with a TTL", block: `{"cachePoint":{"type":"default","ttl":"1h"}}`, accept: true},
		{name: "cachePoint DEFAULT is not the enum value", block: `{"cachePoint":{"type":"DEFAULT"}}`},
		{name: "toolResult status success", block: `{"toolResult":{"toolUseId":"t1","status":"success","content":[{"text":"ok"}]}}`, accept: true},
		{name: "toolResult status error", block: `{"toolResult":{"toolUseId":"t1","status":"error","content":[{"text":"no"}]}}`, accept: true},
		{name: "toolResult status failure is not the enum value", block: `{"toolResult":{"toolUseId":"t1","status":"failure","content":[{"text":"no"}]}}`},
		{name: "image from inline bytes", block: `{"image":{"format":"png","source":{"bytes":"AAAA"}}}`, accept: true},
		{name: "image from s3", block: `{"image":{"format":"png","source":{"s3Location":{"uri":"s3://bucket-name/x.png"}}}}`, accept: true},
		// ImageSource is a union of bytes|s3Location ONLY. An Anthropic-shaped
		// URL image is legal first-party and illegal here; this is the exact
		// cross-dialect case providers/bedrock rejects locally.
		{name: "image from a URL has no ImageSource member", block: `{"image":{"format":"png","source":{"url":"https://example.com/x.png"}}}`},
		{name: "image format svg is not in ImageFormat", block: `{"image":{"format":"svg","source":{"bytes":"AAAA"}}}`},
		{name: "document from text", block: `{"document":{"format":"txt","name":"notes","source":{"text":"hi"}}}`, accept: true},
		{name: "document name at the 200 character cap", block: `{"document":{"format":"txt","name":"` + strings.Repeat("n", 200) + `","source":{"text":"hi"}}}`, accept: true},
		{name: "document name over the 200 character cap", block: `{"document":{"format":"txt","name":"` + strings.Repeat("n", 201) + `","source":{"text":"hi"}}}`},
		{name: "two union members set at once", block: `{"text":"a","image":{"format":"png","source":{"bytes":"AAAA"}}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"messages":[{"role":"user","content":[` + tc.block + `]}]}`)
			err := conformance.Validate("bedrock-converse", kindRequest, body)
			if tc.accept && err != nil {
				t.Fatalf("gate rejected a legal ContentBlock:\n%v", err)
			}
			if !tc.accept && err == nil {
				t.Fatalf("gate accepted an illegal ContentBlock: %s", tc.block)
			}
		})
	}
}

// TestGateCannotCheckDocumentNameContentRules records a real limit. The Smithy
// model documents that DocumentBlock.name must avoid consecutive whitespace and
// is restricted to alphanumerics plus a small punctuation set, but it expresses
// only length as a constraint — so the gate accepts a name the service will
// reject. The encoder is the only thing that can catch it, which is why
// bedrockconverse.validateDocumentName exists and is exercised separately.
func TestGateCannotCheckDocumentNameContentRules(t *testing.T) {
	t.Parallel()

	body := fixture(t, "request_document_name_consecutive_whitespace.json")
	if err := conformance.Validate("bedrock-converse", kindRequest, body); err != nil {
		t.Fatalf("expected the gate to accept a name with consecutive whitespace (length is its only name constraint):\n%v", err)
	}
}

// TestCorpusFixturesAreCanonicalJSON keeps the corpus reviewable.
func TestCorpusFixturesAreCanonicalJSON(t *testing.T) {
	t.Parallel()

	for _, name := range corpus(t) {
		var probe any
		if err := json.Unmarshal(fixture(t, name), &probe); err != nil {
			t.Errorf("%s: not valid JSON: %v", name, err)
		}
	}
}
