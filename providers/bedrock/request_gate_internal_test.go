package bedrock

import (
	"encoding/json"
	"testing"

	"github.com/looprig/inference/codec/conformance"
)

// This file gives the package-internal tests the same request-direction gate
// the external suite gets from conformance_fixtures_test.go. The option
// appliers (applyConverse, applyConverseCountTokens) and the counter preflight
// are unexported, so the gate has to exist on both sides of the boundary.

// gateConverseBody holds an encoded Converse body against AWS's own
// ConverseRequest schema.
//
// Its strength is @pattern / @length / enum / union arity, not presence: AWS
// marks only modelId @required and that field travels in the URI path, so the
// document requires nothing at the top level. Presence stays the caller's
// assertion.
func gateConverseBody(t testing.TB, body []byte) {
	t.Helper()
	conformance.MustValidateRequest(t, "bedrock-converse", "converse_request", body)
}

// gateInvokeModelAnthropicBody holds a Bedrock InvokeModel body against
// Anthropic's CreateMessageParams, which is what that body is: a first-party
// Messages body with the model id lifted into the URI path and
// anthropic_version added. Both transport edits are reversed before the check,
// because Anthropic's own document can describe neither.
func gateInvokeModelAnthropicBody(t testing.TB, body []byte, modelID string) {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("InvokeModel body is not a JSON object: %v", err)
	}
	if _, present := fields["model"]; present {
		t.Fatalf("InvokeModel body carries a top-level model; Bedrock takes it in the URI: %s", body)
	}
	if got := string(fields["anthropic_version"]); got != `"bedrock-2023-05-31"` {
		t.Fatalf("anthropic_version = %s, want \"bedrock-2023-05-31\"", got)
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
