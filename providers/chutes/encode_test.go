package chutes

import (
	"bytes"
	"context"
	"crypto/mlkem"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"testing"

	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm/e2e"
)

// TestEncodeRequestPreservesStructuredOutputWithTools characterizes the Chutes
// extension boundary: the OpenAI request is embedded as-is and only the E2E
// response public key is added before the complete body is encrypted.
func TestEncodeRequestPreservesStructuredOutputWithTools(t *testing.T) {
	t.Parallel()

	output := inference.OutputSchema{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
		Strict: true,
	}
	wantToolSchema := json.RawMessage(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
	req := inference.Request{
		Model:  model.CustomModel("chutes", model.APIFormatOpenAI, "https://api.chutes.ai", "model", model.WithStructuredOutputWithTools()),
		Output: &output,
		Tools: []inference.Tool{{
			Name:   "lookup",
			Schema: wantToolSchema,
		}},
		ToolChoice: inference.ToolChoiceRequired,
	}

	body, err := encodeRequest(req, false, []byte("response-public-key"))
	if err != nil {
		t.Fatalf("encodeRequest() error = %v", err)
	}

	instanceKey, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatalf("generate instance key: %v", err)
	}
	var decrypted []byte
	c := &Client{
		apiBase: "https://api.chutes.test",
		apiKey:  "test-key",
		http: &http.Client{Transport: structuredRoundTripFunc(func(r *http.Request) (*http.Response, error) {
			sealed, readErr := io.ReadAll(r.Body)
			if readErr != nil {
				t.Fatalf("read sealed request: %v", readErr)
			}
			if len(sealed) < e2e.MLKEMCTSize {
				t.Fatalf("sealed request length = %d, want at least %d", len(sealed), e2e.MLKEMCTSize)
			}
			ciphertext := sealed[:e2e.MLKEMCTSize]
			shared, decapsulateErr := instanceKey.Decapsulate(ciphertext)
			if decapsulateErr != nil {
				t.Fatalf("decapsulate request key: %v", decapsulateErr)
			}
			opened, openErr := e2e.Open(shared, ciphertext, sealed[e2e.MLKEMCTSize:], []byte("e2e-req-v1"), true)
			if openErr != nil {
				t.Fatalf("decrypt request: %v", openErr)
			}
			decrypted = opened
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader(nil)),
				Request:    r,
			}, nil
		})},
	}
	session := &attestedSession{
		key:        instanceKey.EncapsulationKey().Bytes(),
		instanceID: "instance-1",
		nonces:     []string{"nonce-1"},
	}
	if _, status, _, err := c.invoke(context.Background(), "chute-1", session, body); err != nil || status != http.StatusOK {
		t.Fatalf("invoke() status = %d, error = %v, want 200 and nil", status, err)
	}

	var wire struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string          `json:"name"`
				Strict bool            `json:"strict"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
		Tools []struct {
			Function struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice    string `json:"tool_choice"`
		E2EResponsePK string `json:"e2e_response_pk"`
	}
	if err := json.Unmarshal(decrypted, &wire); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if wire.ResponseFormat.Type != "json_schema" || wire.ResponseFormat.JSONSchema.Name != "answer" || !wire.ResponseFormat.JSONSchema.Strict {
		t.Errorf("response_format = %+v, want strict json_schema named answer", wire.ResponseFormat)
	}
	assertJSONSemanticallyEqual(t, wire.ResponseFormat.JSONSchema.Schema, output.Schema)
	if len(wire.Tools) != 1 || wire.Tools[0].Function.Name != "lookup" {
		t.Errorf("tools = %+v, want one lookup function", wire.Tools)
	} else {
		assertJSONSemanticallyEqual(t, wire.Tools[0].Function.Parameters, wantToolSchema)
	}
	if wire.ToolChoice != "required" {
		t.Errorf("tool_choice = %q, want required", wire.ToolChoice)
	}
	if wire.E2EResponsePK == "" {
		t.Error("e2e_response_pk is empty")
	}
}

func assertJSONSemanticallyEqual(t *testing.T, got, want json.RawMessage) {
	t.Helper()
	var gotValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("unmarshal got JSON %q: %v", got, err)
	}
	var wantValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("unmarshal want JSON %q: %v", want, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("JSON = %s, want semantically %s", got, want)
	}
}

type structuredRoundTripFunc func(*http.Request) (*http.Response, error)

func (f structuredRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
