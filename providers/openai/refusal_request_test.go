package openai_test

import (
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	chat "github.com/looprig/inference/codec/openaiapi"
	responses "github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"

	"github.com/looprig/inference/codec/conformance"
)

// Refusal REQUEST coverage.
//
// A content.RefusalBlock in prior assistant history has to go back on the wire
// as the refusal each API declares, never as assistant text. These tests hold
// the encoded bodies against OpenAI's own request schemas, and — for the
// Responses dialect — MEASURE which refusal shapes that schema actually admits,
// because the encoder's fail-closed behaviour there rests on the answer.

func assistantRefusalRequest(name string, format model.APIFormat) inference.Request {
	return inference.Request{
		Model: model.CustomModel(model.ProviderName("openai"), format, "https://api.openai.com/v1", name),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "do the thing"},
			}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				&content.RefusalBlock{Text: "I'm sorry, I can't help with that."},
			}}},
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "try something else then"},
			}}},
		},
	}
}

// TestChatRefusalReplayIsALegalRequest holds the replayed refusal against
// OpenAI's own Chat Completions request schema before asserting anything about
// it: an assertion about a body the provider would reject proves nothing.
func TestChatRefusalReplayIsALegalRequest(t *testing.T) {
	t.Parallel()

	body, err := chat.EncodeRequest(assistantRefusalRequest("gpt-4.1", model.APIFormatOpenAI), false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateChatRequest(t, body)

	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if len(decoded.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(decoded.Messages))
	}
	got, ok := decoded.Messages[1]["refusal"]
	if !ok {
		t.Fatalf("assistant message %v carries no `refusal`", decoded.Messages[1])
	}
	if string(got) != `"I'm sorry, I can't help with that."` {
		t.Errorf("refusal = %s, want the refusal text verbatim", got)
	}
}

// TestChatRequestGateStrengthOnTheRefusalMember MEASURES what the Chat request
// gate actually enforces around `refusal` by feeding it deliberately wrong
// shapes, rather than assuming it is uniformly strict.
//
// Measured result: ChatCompletionRequestAssistantMessage declares `refusal` as
// string|null and the gate enforces that TYPE, but the object is OPEN — it
// carries no additionalProperties:false — so a misspelled member sails through.
// The type check is what catches an encoder that put the wrong Go value in the
// field; the encoder's own allowlist (a single *string populated only from a
// *content.RefusalBlock) is what carries the "no invented member" constraint,
// because the gate is blind to it.
func TestChatRequestGateStrengthOnTheRefusalMember(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantReject bool
	}{
		{
			name:       "refusal as a string is legal",
			body:       `{"model":"m","messages":[{"role":"assistant","content":"","refusal":"I cannot."}]}`,
			wantReject: false,
		},
		{
			name:       "refusal as an object is rejected on type",
			body:       `{"model":"m","messages":[{"role":"assistant","refusal":{"text":"I cannot."}}]}`,
			wantReject: true,
		},
		{
			name:       "refusal as a number is rejected on type",
			body:       `{"model":"m","messages":[{"role":"assistant","refusal":7}]}`,
			wantReject: true,
		},
		{
			name: "a misspelled member is NOT caught: the assistant message " +
				"object is open, so this constraint lives in the encoder",
			body:       `{"model":"m","messages":[{"role":"assistant","refusal_text":"I cannot."}]}`,
			wantReject: false,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate("openai", chatRequestKind, []byte(tt.body))
			if gotReject := err != nil; gotReject != tt.wantReject {
				t.Errorf("Validate() rejected = %v (err = %v), want rejected = %v", gotReject, err, tt.wantReject)
			}
		})
	}
}

// TestResponsesRefusalReplayHasNoLegalIDFreeWireForm is the measurement the
// Responses encoder's fail-closed refusal rests on.
//
// A `refusal` part is admissible only inside an OutputMessage, whose required
// set is ["id","type","role","content","status"]. Replayed assistant history
// carries no message id — no neutral block holds one — and EasyInputMessage,
// the only id-free assistant form, takes InputContent
// (input_text|input_image|input_file), which has no refusal member. Every
// id-free candidate is therefore a body the provider's own request schema
// rejects, which is why openairesponses fails closed instead of emitting one.
func TestResponsesRefusalReplayHasNoLegalIDFreeWireForm(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantReject bool
	}{
		{
			name:       "EasyInputMessage carrying a refusal part",
			body:       `{"model":"m","store":false,"input":[{"role":"assistant","content":[{"type":"refusal","refusal":"no"}]}]}`,
			wantReject: true,
		},
		{
			name:       "message item with a refusal part but no id",
			body:       `{"model":"m","store":false,"input":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"no"}]}]}`,
			wantReject: true,
		},
		{
			name:       "message item with an id but no status",
			body:       `{"model":"m","store":false,"input":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"refusal","refusal":"no"}]}]}`,
			wantReject: true,
		},
		{
			name:       "a full OutputMessage is the ONE legal form",
			body:       `{"model":"m","store":false,"input":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"no"}]}]}`,
			wantReject: false,
		},
		{
			name:       "an explanation-free refusal is still legal in that form",
			body:       `{"model":"m","store":false,"input":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":""}]}]}`,
			wantReject: false,
		},
		{
			name:       "omitting the required refusal member is rejected",
			body:       `{"model":"m","store":false,"input":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"refusal"}]}]}`,
			wantReject: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate("openai-responses", responsesRequestKind, []byte(tt.body))
			if gotReject := err != nil; gotReject != tt.wantReject {
				t.Errorf("Validate() rejected = %v (err = %v), want rejected = %v", gotReject, err, tt.wantReject)
			}
		})
	}
}

// TestResponsesRefusalReplayFailsClosed pins the encoder to that measurement:
// rather than emitting one of the rejected shapes, degrading the refusal to
// output_text, or dropping it, the request-replay direction refuses.
func TestResponsesRefusalReplayFailsClosed(t *testing.T) {
	t.Parallel()

	body, err := responses.EncodeRequest(assistantRefusalRequest("gpt-5", model.APIFormatOpenAIResponses), false)
	if err == nil {
		t.Fatalf("EncodeRequest() encoded a refusal replay as %s, want a typed refusal", body)
	}
	var unsupported *responses.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *openairesponses.UnsupportedBlockError", err, err)
	}
}

// TestResponsesRefusalResponseIsALegalOutputMessage covers the direction that
// CAN carry the refusal: when this process is the authority producing the
// response it synthesizes the OutputMessage id, so the refusal goes out as the
// native content part — and the encoded item is exactly the one shape the
// request schema admits, proving the part itself is well formed.
func TestResponsesRefusalResponseIsALegalOutputMessage(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	resp := &inference.Response{
		Message: &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
			&content.RefusalBlock{Text: "I'm sorry, I can't help with that."},
		}}},
		Model: "gpt-5",
	}
	if err := (responses.Codec{}).WriteResponse(rec, resp); err != nil {
		t.Fatalf("WriteResponse() error = %v", err)
	}

	var served struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &served); err != nil {
		t.Fatalf("unmarshal served response: %v", err)
	}
	if len(served.Output) != 1 {
		t.Fatalf("output = %s, want one item", rec.Body.String())
	}
	// Replay the served item straight back as a request input item: the
	// strictest available check that what we serve is what the API accepts.
	replay := []byte(`{"model":"gpt-5","store":false,"input":[` + string(served.Output[0]) + `]}`)
	gateResponsesRequest(t, replay)
}
