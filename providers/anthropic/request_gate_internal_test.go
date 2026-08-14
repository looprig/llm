package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/looprig/inference/codec/conformance"
)

// This file gives the package-internal tests the same request-direction gate
// the external suite gets from conformance_fixtures_test.go. Encode-side tests
// live on both sides of the package boundary — applyPromptCacheControl and the
// counter preflight are unexported — so the gate has to exist on both sides.

// gateRequestBody holds an encoded Messages body against Anthropic's own
// CreateMessageParams schema (additionalProperties:false, required
// model/messages/max_tokens).
func gateRequestBody(t testing.TB, body []byte) {
	t.Helper()
	conformance.MustValidateRequest(t, "anthropic", "create_message_request", body)
}

// gateRequestFields is gateRequestBody for the decomposed body the in-place
// cache-control rewrite works on.
func gateRequestFields(t testing.TB, fields map[string]json.RawMessage) {
	t.Helper()
	body, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("marshal request fields: %v", err)
	}
	gateRequestBody(t, body)
}

// gateCountTokensBody holds a /v1/messages/count_tokens body against
// CreateMessageParams.
//
// The gate has no count_tokens kind of its own: Anthropic's document models
// that endpoint with CountMessageTokensParams, which the derived schema tree
// does not carry. CountMessageTokensParams is CreateMessageParams minus
// max_tokens and the other generation-only fields, so the body is completed
// with the one property whose absence would otherwise fail the required list,
// and then held to the full request schema.
//
// Be precise about what that does and does not prove. It proves every
// structural, typed and patterned constraint on system, messages, tools and
// tool_choice — the same encoder produced them, so the same defects live there
// — and it rejects a property CreateMessageParams does not declare at all,
// because that object is additionalProperties:false. It does NOT catch a
// generation field leaking into the count body: max_tokens, temperature and
// stream are all legal CreateMessageParams properties, so only the caller's own
// field-name assertions can catch those. The one exception is max_tokens, which
// is checked here explicitly because injecting it would otherwise mask it.
func gateCountTokensBody(t testing.TB, body []byte) {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("count body is not a JSON object: %v", err)
	}
	if _, present := fields["max_tokens"]; present {
		t.Fatalf("count_tokens body carries max_tokens, which is a generation-only field: %s", body)
	}
	fields["max_tokens"] = json.RawMessage("1")
	completed, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("re-marshal count body: %v", err)
	}
	gateRequestBody(t, completed)
}
