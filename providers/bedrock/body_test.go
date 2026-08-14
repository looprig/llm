package bedrock_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auth"
	"github.com/looprig/llm/providers/bedrock"
)

// TestBedrockBodyTransform verifies the Anthropic->Bedrock body rewrite via the
// exported Invoke path: the request that reaches the server must (a) drop the
// top-level "model" field, (b) carry "anthropic_version":"bedrock-2023-05-31", and
// (c) preserve the codec's other fields (messages, max_tokens). It is driven
// through a real httptest.Server so the transform is exercised end-to-end, not in
// isolation. Table covers the happy transform plus a body with an explicit
// max_tokens override that must survive the rewrite.
func TestBedrockBodyTransform(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		override     *model.Sampling
		wantMaxToken float64
	}{
		{name: "default max_tokens survives, model dropped, version added", wantMaxToken: 4096},
		{name: "explicit max_tokens override survives rewrite", override: &model.Sampling{MaxTokens: intptr(256)}, wantMaxToken: 256},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			srv, bodyCh := bodyCaptureServer(t)
			defer srv.Close()

			c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
			req := bedrockRequest("anthropic.claude-3-5-sonnet-20241022-v2:0")
			req.Override = tt.override

			if _, err := c.Invoke(context.Background(), req); err != nil {
				t.Fatalf("Invoke() err = %v, want nil", err)
			}

			raw := <-bodyCh
			gateInvokeModelBody(t, raw, "anthropic.claude-3-5-sonnet-20241022-v2:0")

			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatalf("unmarshal captured body: %v", err)
			}

			if _, ok := fields["model"]; ok {
				t.Error(`body still carries top-level "model"; Bedrock takes the model id in the URL`)
			}
			var version string
			if raw, ok := fields["anthropic_version"]; !ok {
				t.Error(`body missing "anthropic_version"`)
			} else if err := json.Unmarshal(raw, &version); err != nil || version != "bedrock-2023-05-31" {
				t.Errorf(`anthropic_version = %q (err %v), want "bedrock-2023-05-31"`, version, err)
			}
			if _, ok := fields["messages"]; !ok {
				t.Error(`body missing "messages" (codec field not preserved)`)
			}
			var maxTokens float64
			if raw, ok := fields["max_tokens"]; !ok {
				t.Error(`body missing "max_tokens"`)
			} else if err := json.Unmarshal(raw, &maxTokens); err != nil || maxTokens != tt.wantMaxToken {
				t.Errorf("max_tokens = %v (err %v), want %v", maxTokens, err, tt.wantMaxToken)
			}
		})
	}
}

// TestBedrockBodyTransformPreservesCodecOutput cross-checks that the transform is a
// pure add/remove: encoding the same request with the anthropicapi codec directly
// and diffing keys shows exactly {-model, +anthropic_version} and every other key
// byte-identical.
func TestBedrockBodyTransformPreservesCodecOutput(t *testing.T) {
	t.Parallel()

	// An adaptive-thinking model id, because these two tests need the body to
	// carry output_config: the codec emits output_config.effort only for the
	// adaptive dialect, and declaring that dialect on a Claude 3.5 id would be
	// a fixture claiming something the real model does not do.
	req := bedrockRequest("anthropic.claude-sonnet-5")
	req.Model.Caps.StructuredOutput = true
	req.Model.Caps.Thinking = true
	req.Model.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
	req.Model.Sampling.Effort = model.EffortHigh
	req.Output = &inference.OutputSchema{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
		Strict: true,
	}
	// The client encodes the invoke body via the free anthropicapi.EncodeRequest
	// (stream=false); use the same free function here so the cross-check compares
	// against the exact bytes the client rewrites.
	anthropicBody, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("codec EncodeRequest: %v", err)
	}
	// The pre-transform bytes are an ordinary first-party Messages body, so
	// they are gated directly: if the source of the rewrite is already illegal,
	// proving the rewrite preserved it proves nothing.
	conformance.MustValidateRequest(t, "anthropic", "create_message_request", anthropicBody)

	var anthropicFields map[string]json.RawMessage
	if err := json.Unmarshal(anthropicBody, &anthropicFields); err != nil {
		t.Fatalf("unmarshal anthropic body: %v", err)
	}

	srv, bodyCh := bodyCaptureServer(t)
	defer srv.Close()
	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
	if _, err := c.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke() err = %v", err)
	}
	rewritten := <-bodyCh
	gateInvokeModelBody(t, rewritten, "anthropic.claude-sonnet-5")

	var bedrockFields map[string]json.RawMessage
	if err := json.Unmarshal(rewritten, &bedrockFields); err != nil {
		t.Fatalf("unmarshal bedrock body: %v", err)
	}

	// Every anthropic field except "model" must survive byte-identical.
	for k, v := range anthropicFields {
		if k == "model" {
			continue
		}
		if !bytesEqualJSON(bedrockFields[k], v) {
			t.Errorf("field %q changed: bedrock=%s anthropic=%s", k, bedrockFields[k], v)
		}
	}
	// Bedrock adds exactly anthropic_version and drops model.
	if _, ok := bedrockFields["anthropic_version"]; !ok {
		t.Error("bedrock body missing anthropic_version")
	}
	if _, ok := bedrockFields["model"]; ok {
		t.Error("bedrock body still has model")
	}
	if len(bedrockFields) != len(anthropicFields) {
		// -model +anthropic_version nets to the same count.
		t.Errorf("field count = %d, want %d (=anthropic count; -model +anthropic_version)", len(bedrockFields), len(anthropicFields))
	}
}

// TestBedrockBodyTransformPreservesStructuredOutputAndEffort proves the Bedrock
// rewrite retains Anthropic's combined output_config object while changing only
// the transport-specific model/version fields.
func TestBedrockBodyTransformPreservesStructuredOutputAndEffort(t *testing.T) {
	t.Parallel()

	// An adaptive-thinking model id, because these two tests need the body to
	// carry output_config: the codec emits output_config.effort only for the
	// adaptive dialect, and declaring that dialect on a Claude 3.5 id would be
	// a fixture claiming something the real model does not do.
	req := bedrockRequest("anthropic.claude-sonnet-5")
	req.Model.Caps.StructuredOutput = true
	req.Model.Caps.Thinking = true
	req.Model.Caps.ThinkingDialect = model.ThinkingDialectAdaptive
	req.Model.Sampling.Effort = model.EffortHigh
	req.Output = &inference.OutputSchema{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
		Strict: true,
	}

	srv, bodyCh := bodyCaptureServer(t)
	defer srv.Close()
	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
	if _, err := c.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke() err = %v, want nil", err)
	}

	raw := <-bodyCh
	gateInvokeModelBody(t, raw, "anthropic.claude-sonnet-5")

	var wire struct {
		Model            json.RawMessage `json:"model"`
		AnthropicVersion string          `json:"anthropic_version"`
		OutputConfig     struct {
			Effort string `json:"effort"`
			Format struct {
				Type   string          `json:"type"`
				Schema json.RawMessage `json:"schema"`
			} `json:"format"`
		} `json:"output_config"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if wire.Model != nil {
		t.Errorf("model = %s, want omitted", wire.Model)
	}
	if wire.AnthropicVersion != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version = %q", wire.AnthropicVersion)
	}
	if wire.OutputConfig.Effort != "high" || wire.OutputConfig.Format.Type != "json_schema" || !json.Valid(wire.OutputConfig.Format.Schema) {
		t.Errorf("output_config = %+v, want effort high and valid json_schema format", wire.OutputConfig)
	}
}

// TestBedrockBodyTransformRejectsURLImageSource proves the InvokeModel path fails
// closed on an Anthropic image block whose source is a remote URL. Anthropic's
// first-party API accepts {"type":"url"}, so the shared anthropicapi encoder emits
// it, but Bedrock's ImageSource union is bytes | s3Location — the URL would reach
// Bedrock and draw an HTTP 400. The rejection must happen before any I/O, so the
// capture server must never see a request. The nested case covers a URL image
// carried inside a tool_result block's content.
func TestBedrockBodyTransformRejectsURLImageSource(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		messages content.AgenticMessages
	}{
		{
			name: "url image in a user message",
			messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{
					Role: content.RoleUser,
					Blocks: []content.Block{
						&content.TextBlock{Text: "describe this"},
						&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{URL: "https://example.com/x.png"}},
					},
				}},
			},
		},
		{
			name: "url image nested in a tool_result block",
			messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{
					Role:   content.RoleUser,
					Blocks: []content.Block{&content.TextBlock{Text: "screenshot it"}},
				}},
				&content.AIMessage{Message: content.Message{
					Role:   content.RoleAssistant,
					Blocks: []content.Block{&content.ToolUseBlock{ID: "toolu_1", Name: "screenshot"}},
				}},
				&content.ToolResultMessage{
					ToolUseID: "toolu_1",
					Message: content.Message{
						Role: content.RoleUser,
						Blocks: []content.Block{
							&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{URL: "https://example.com/x.png"}},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			srv, bodyCh := bodyCaptureServer(t)
			defer srv.Close()

			c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
			req := bedrockRequest("anthropic.claude-3-5-sonnet-20241022-v2:0")
			req.Messages = tt.messages

			_, err := c.Invoke(context.Background(), req)
			var unsupported *bedrock.UnsupportedImageSourceError
			if !errors.As(err, &unsupported) {
				t.Fatalf("Invoke() err = %v, want *bedrock.UnsupportedImageSourceError", err)
			}
			if unsupported.SourceType != "url" {
				t.Errorf("SourceType = %q, want %q", unsupported.SourceType, "url")
			}
			if !strings.Contains(unsupported.Error(), "base64") {
				t.Errorf("Error() = %q, want it to name the accepted inline base64 form", unsupported.Error())
			}
			select {
			case body := <-bodyCh:
				t.Errorf("request reached Bedrock before the local rejection: %s", body)
			default:
			}
		})
	}
}

// TestBedrockBodyTransformKeepsBase64ImageSource is the positive control for the
// URL rejection: an inline base64 image source is the form Bedrock accepts, so it
// must still encode and reach the wire unchanged.
func TestBedrockBodyTransformKeepsBase64ImageSource(t *testing.T) {
	t.Parallel()

	srv, bodyCh := bodyCaptureServer(t)
	defer srv.Close()

	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
	req := bedrockRequest("anthropic.claude-3-5-sonnet-20241022-v2:0")
	req.Messages = content.AgenticMessages{
		&content.UserMessage{Message: content.Message{
			Role: content.RoleUser,
			Blocks: []content.Block{
				&content.TextBlock{Text: "describe this"},
				&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{0x89, 'P', 'N', 'G'}}},
			},
		}},
	}

	if _, err := c.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke() err = %v, want nil", err)
	}

	raw := <-bodyCh
	// The positive control for the URL rejection is gated too: proving Bedrock
	// accepts the base64 form is only worth anything if the block we emit is
	// also a legal Anthropic Base64ImageSource.
	gateInvokeModelBody(t, raw, "anthropic.claude-3-5-sonnet-20241022-v2:0")

	var wire struct {
		Messages []struct {
			Content []struct {
				Type   string `json:"type"`
				Source struct {
					Type      string `json:"type"`
					MediaType string `json:"media_type"`
					Data      string `json:"data"`
				} `json:"source"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal captured body: %v", err)
	}
	if len(wire.Messages) != 1 || len(wire.Messages[0].Content) != 2 {
		t.Fatalf("wire messages = %+v, want 1 message with 2 blocks", wire.Messages)
	}
	image := wire.Messages[0].Content[1]
	if image.Type != "image" || image.Source.Type != "base64" {
		t.Fatalf("block = %+v, want a base64-source image block", image)
	}
	if image.Source.MediaType != string(content.MediaTypeImagePNG) {
		t.Errorf("media_type = %q, want %q", image.Source.MediaType, content.MediaTypeImagePNG)
	}
	if want := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G'}); image.Source.Data != want {
		t.Errorf("data = %q, want %q", image.Source.Data, want)
	}
}

func bytesEqualJSON(a, b json.RawMessage) bool {
	return string(a) == string(b)
}

// bedrockRequest builds a minimal valid ProviderBedrock request for name.
func bedrockRequest(name string) inference.Request {
	return inference.Request{
		Model: model.CustomModel(model.ProviderName(llm.ProviderBedrock), model.APIFormatAnthropic, "", name, model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}), model.WithTools(), model.WithImages()),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	}
}

func testCreds() auth.SigV4Credentials {
	return auth.SigV4Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}
}

func intptr(v int) *int { return &v }
