package anthropic_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
)

// This file is the request-direction half of the Anthropic conformance suite.
// Every case here encodes a real inference.Request and holds the resulting
// bytes against Anthropic's own CreateMessageParams schema BEFORE asserting
// anything about them, which is the stronger of the two directions: a response
// fixture tests our tolerance of what Anthropic sends, an encoded request tests
// whether Anthropic will accept what we send, and it says so before the bytes
// leave the process rather than after an HTTP 400.
//
// The request document is also far stricter than the response one.
// CreateMessageParams is additionalProperties:false and requires
// model/messages/max_tokens; 83 of the document's 85 request object shapes are
// closed the same way and 82 declare required properties. Every constraint
// exercised below is Anthropic's own, transcribed by the schema generator, not
// one this suite invented.

// gatedEncode encodes req, gates the bytes, and returns them.
func gatedEncode(t *testing.T, req inference.Request) []byte {
	t.Helper()
	body, err := anthropicapi.EncodeRequest(req, false)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateRequest(t, body)
	return body
}

func encodeModel(opts ...model.ModelOption) model.Model {
	base := []model.ModelOption{model.WithTools(), model.WithImages(), model.WithThinkingDialect(model.ThinkingDialectAdaptive)}
	base = append(base, opts...)
	return model.CustomModel("anthropic", model.APIFormatAnthropic, "https://example.test", "claude-sonnet-4-5", base...)
}

func userTurn(blocks ...content.Block) content.Conversation {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func assistantTurn(blocks ...content.Block) content.Conversation {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

var pngBytes = []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}

var pdfBytes = []byte{0x25, 0x50, 0x44, 0x46, 0x2d}

// TestEncodeRequestShapesAreSpecLegal walks the request shapes the Anthropic
// provider really emits and proves each one is a legal CreateMessageParams.
func TestEncodeRequestShapesAreSpecLegal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "inline image then text",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: pngBytes}},
					&content.TextBlock{Text: "Describe this image."},
				)},
			},
		},
		{
			name: "remote image url",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{URL: "https://example.com/diagram.png"}},
				)},
			},
		},
		{
			// The empty-thinking / explicit-signature shape is the one the
			// omitempty repair exists for: RequestThinkingBlock requires BOTH
			// signature and thinking, so a dropped "" is an HTTP 400.
			name: "thinking replay with an empty thinking body",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Continue."}),
					assistantTurn(
						// The signature carries the label of the dialect that
						// minted it; an unlabelled one is refused by the encoder,
						// because a signature is verified by its issuer and an
						// unsigned thinking block is a 400 in its own right.
						content.NewSignedThinkingBlock("", "sig", "anthropic", nil, ""),
						content.NewThinkingBlock("", "", json.RawMessage(`"cmVkYWN0ZWQ="`), "anthropic-redacted-thinking"),
						&content.TextBlock{Text: "Continuing."},
					),
				},
			},
		},
		{
			name: "tool use and tool result round trip",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Weather in Paris?"}),
					assistantTurn(content.NewToolUseBlock("toolu_01Weather", "get_weather", json.RawMessage(`{"city":"Paris"}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "toolu_01Weather",
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "18C and clear"}}},
					},
				},
				Tools: []inference.Tool{{Name: "get_weather", Description: "Look up weather", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "errored tool result",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Read /etc/shadow"}),
					assistantTurn(content.NewToolUseBlock("toolu_01Read", "read_file", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "toolu_01Read",
						IsError:   true,
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "permission denied"}}},
					},
				},
				Tools: []inference.Tool{{Name: "read_file", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "image nested inside a tool result",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Screenshot the page."}),
					assistantTurn(content.NewToolUseBlock("toolu_01Shot", "screenshot", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "toolu_01Shot",
						Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{
							&content.TextBlock{Text: "captured"},
							&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: pngBytes}},
						}},
					},
				},
				Tools: []inference.Tool{{Name: "screenshot", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			// RequestDocumentBlock is additionalProperties:false with required
			// [source, type], and Base64PDFSource is required
			// [data, media_type, type] with media_type const "application/pdf".
			// The gate checks all of that, including that `title` is a declared
			// property rather than an invented one.
			name: "pdf document with a title",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "quarterly report", Data: pdfBytes},
					&content.TextBlock{Text: "Summarize the filing."},
				)},
			},
		},
		{
			// PlainTextSource is the other reachable member, media_type const
			// "text/plain".
			name: "extracted-text document",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentText, Name: "notes", Text: "line one"},
					&content.TextBlock{Text: "Summarize the notes."},
				)},
			},
		},
		{
			// RequestToolResultBlock's content union lists RequestDocumentBlock
			// beside text and image, so a document survives a tool result too.
			name: "document nested inside a tool result",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Fetch the filing."}),
					assistantTurn(content.NewToolUseBlock("toolu_01Fetch", "fetch", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "toolu_01Fetch",
						Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{
							&content.TextBlock{Text: "fetched"},
							&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "filing", Data: pdfBytes},
						}},
					},
				},
				Tools: []inference.Tool{{Name: "fetch", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "cache breakpoints on system and the committed tail",
			req: inference.Request{
				Model:             encodeModel(model.WithPromptCaching()),
				System:            "You are precise.",
				TransientMessages: 1,
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "Long committed prefix."}),
					userTurn(&content.TextBlock{Text: "runtime"}),
				},
			},
		},
		{
			name: "in-thread system message folded into the system prefix",
			req: inference.Request{
				Model:  encodeModel(),
				System: "sys",
				Messages: content.AgenticMessages{
					&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{&content.TextBlock{Text: "mid"}}}},
					userTurn(&content.TextBlock{Text: "hi"}),
				},
			},
		},
		{
			name: "required tool choice and stop sequences",
			req: inference.Request{
				Model:      encodeModel(model.WithSampling(model.Sampling{Stop: []string{"END"}})),
				Messages:   content.AgenticMessages{userTurn(&content.TextBlock{Text: "hi"})},
				Tools:      []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
				ToolChoice: inference.ToolRequired(),
			},
		},
		{
			name: "named tool choice forcing one declared tool",
			req: inference.Request{
				Model:    encodeModel(),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "hi"})},
				Tools: []inference.Tool{
					{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)},
					{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)},
				},
				ToolChoice: inference.ToolNamed("search"),
			},
		},
		{
			name: "sampling at the interval boundaries",
			req: inference.Request{
				Model:    encodeModel(model.WithSampling(model.Sampling{Temperature: floatPtr(0), TopP: floatPtr(1)})),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "hi"})},
			},
		},
		{
			name: "adaptive thinking with a structured output format",
			req: inference.Request{
				Model:    encodeModel(model.WithStructuredOutputWithTools(), model.WithSampling(model.Sampling{Effort: model.EffortHigh})),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "hi"})},
				Output:   &inference.OutputSchema{Name: "answer", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)},
			},
		},
		{
			name: "tool-use id and tool name at the full legal character class",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "go"}),
					assistantTurn(content.NewToolUseBlock("aZ0_-", "aZ0_-", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "aZ0_-", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "tool name at the 128 character cap",
			req: inference.Request{
				Model:    encodeModel(),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: strings.Repeat("t", 128), Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gatedEncode(t, tc.req)
		})
	}
}

// TestEncodeRequestStreamingIsSpecLegal covers the other encode mode. The
// streaming body must carry "stream":true and still satisfy the closed
// CreateMessageParams object.
func TestEncodeRequestStreamingIsSpecLegal(t *testing.T) {
	t.Parallel()

	body, err := anthropicapi.EncodeRequest(inference.Request{
		Model:    encodeModel(),
		Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "hi"})},
	}, true)
	if err != nil {
		t.Fatalf("EncodeRequest(stream) error = %v", err)
	}
	gateRequest(t, body)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	if string(fields["stream"]) != "true" {
		t.Errorf("stream = %s, want true", fields["stream"])
	}
}

// TestEncodeRequestRefusesWireIllegalValues is the negative half, and it
// records five defects the request gate surfaced. In each case the encoder used
// to forward a caller-supplied value straight onto the wire, producing a body
// Anthropic's own document declares illegal; the assertion is now that the
// codec fails closed with a typed, nameable error before any I/O, exactly as it
// already did for empty text and unsupported block types.
//
// The values are not exotic. Every one of them is either a first-class value in
// the shared Looprig vocabulary (content.MediaTypeImageSVG), a shape another
// dialect legitimately produces (a Converse tool-use id containing "." or ":",
// an OpenAI temperature above 1), or a name a tool server really publishes (an
// MCP tool called "fs.read_file").
func TestEncodeRequestRefusesWireIllegalValues(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		req     inference.Request
		target  any
		because string
	}{
		{
			name: "svg image media type",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImageSVG, Source: content.ImageSource{Data: pngBytes}},
				)},
			},
			target:  new(*anthropicapi.UnsupportedImageMediaTypeError),
			because: "Base64ImageSource.media_type is an enum of jpeg/png/gif/webp; content.MediaTypeImageSVG has no member",
		},
		{
			name: "tool-use id carrying Converse's extra characters",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "go"}),
					assistantTurn(content.NewToolUseBlock("tooluse.weather:1", "get_weather", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "get_weather", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			target:  new(*anthropicapi.InvalidToolUseIDError),
			because: "tool_use.id is ^[a-zA-Z0-9_-]+$; Converse mints ids containing . and :",
		},
		{
			name: "tool-result tool_use_id carrying Converse's extra characters",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "go"}),
					assistantTurn(content.NewToolUseBlock("toolu_1", "get_weather", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "tooluse.weather:1",
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "ok"}}},
					},
				},
				Tools: []inference.Tool{{Name: "get_weather", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			target:  new(*anthropicapi.InvalidToolUseIDError),
			because: "tool_result.tool_use_id carries the same pattern as tool_use.id",
		},
		{
			// The empty id is the sharper case: `id` is omitempty on the wire
			// struct, so it did not travel as "" — the required property simply
			// disappeared, which is the same silent-drop failure the thinking
			// blocks were repaired for.
			name: "empty tool-use id",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{
					userTurn(&content.TextBlock{Text: "go"}),
					assistantTurn(content.NewToolUseBlock("", "t", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "t", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			target:  new(*anthropicapi.InvalidToolUseIDError),
			because: "omitempty dropped a required property rather than sending an empty one",
		},
		{
			name: "MCP-style dotted tool name",
			req: inference.Request{
				Model:    encodeModel(),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: "fs.read_file", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			target:  new(*anthropicapi.InvalidToolNameError),
			because: "Tool.name is ^[a-zA-Z0-9_-]{1,128}$",
		},
		{
			name: "tool name past the 128 character cap",
			req: inference.Request{
				Model:    encodeModel(),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: strings.Repeat("t", 129), Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			target:  new(*anthropicapi.InvalidToolNameError),
			because: "Tool.name has maxLength 128",
		},
		{
			name: "tool input schema that is not an object",
			req: inference.Request{
				Model:    encodeModel(),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: "t", Schema: json.RawMessage(`[]`)}},
			},
			target:  new(*anthropicapi.InvalidToolSchemaError),
			because: "InputSchema is typed object; bedrockconverse always checked this and anthropicapi did not",
		},
		{
			// The gate cannot see this one, and that is the point: there is no
			// audio shape in Anthropic's document for a schema to reject, so
			// the only place the limitation can be caught is the encoder. An
			// MCP tool returning audio produces this block, and the harness
			// persists it, so it recurs on every later turn of that session.
			name: "audio block, which the format does not model at all",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: []byte{0x49, 0x44, 0x33}},
				)},
			},
			target:  new(*anthropicapi.UnsupportedAudioError),
			because: "the Messages API document declares no audio content block in any shape",
		},
		{
			name: "non-pdf binary document",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentDOCX, Name: "spec", Data: []byte{0x50, 0x4b}},
				)},
			},
			target:  new(*anthropicapi.UnsupportedDocumentError),
			because: "Base64PDFSource.media_type is const application/pdf; no other binary document type has a source member",
		},
		{
			name: "markdown document body",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "readme", Text: "# Title"},
				)},
			},
			target:  new(*anthropicapi.UnsupportedDocumentError),
			because: "PlainTextSource.media_type is const text/plain; relabelling markdown would rewrite the caller's media type",
		},
		{
			name: "document title past the 500 character cap",
			req: inference.Request{
				Model: encodeModel(),
				Messages: content.AgenticMessages{userTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: strings.Repeat("n", 501), Data: pdfBytes},
				)},
			},
			target:  new(*anthropicapi.UnsupportedDocumentError),
			because: "RequestDocumentBlock.title has maxLength 500",
		},
		{
			name: "OpenAI-range temperature",
			req: inference.Request{
				Model:    encodeModel(model.WithSampling(model.Sampling{Temperature: floatPtr(1.7)})),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
			},
			target:  new(*anthropicapi.SamplingRangeError),
			because: "temperature is bounded to [0,1]; OpenAI's runs to 2",
		},
		{
			name: "top_p above one",
			req: inference.Request{
				Model:    encodeModel(model.WithSampling(model.Sampling{TopP: floatPtr(1.5)})),
				Messages: content.AgenticMessages{userTurn(&content.TextBlock{Text: "go"})},
			},
			target:  new(*anthropicapi.SamplingRangeError),
			because: "top_p is bounded to [0,1]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body, err := anthropicapi.EncodeRequest(tc.req, false)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted a body Anthropic rejects (%s): %s", tc.because, body)
			}
			if !errors.As(err, tc.target) {
				t.Fatalf("EncodeRequest() error = %T (%v), want %T (%s)", err, err, tc.target, tc.because)
			}
			if body != nil {
				t.Errorf("EncodeRequest() returned %d bytes alongside its error, want none", len(body))
			}
		})
	}
}

// TestToolChoiceGateStrength measures — rather than assumes — how much of the
// tool_choice contract the request gate actually enforces, by feeding it
// deliberately wrong shapes the encoder never produces.
//
// Anthropic's gate is the strongest of the four. ToolChoice is a real oneOf
// over four closed objects, so a missing `name`, a foreign dialect's spelling
// and an extra property are all rejected. The one thing it cannot see is
// whether the name refers to a declared tool: the schema has no cross-field
// constraint, so that is carried by inference.ValidateRequestFeatures instead.
func TestToolChoiceGateStrength(t *testing.T) {
	t.Parallel()

	const preamble = `{"model":"claude-sonnet-4-5","max_tokens":16,` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],` +
		`"tools":[{"name":"search","input_schema":{"type":"object"}}],`

	cases := []struct {
		name       string
		toolChoice string
		wantReject bool
		because    string
	}{
		{
			name:       "the shape the encoder emits",
			toolChoice: `{"type":"tool","name":"search"}`,
			because:    "ToolChoiceTool requires exactly type and name",
		},
		{
			name:       "tool choice with no name",
			toolChoice: `{"type":"tool"}`,
			wantReject: true,
			because:    "name is in ToolChoiceTool's required list",
		},
		{
			name:       "OpenAI Chat spelling",
			toolChoice: `{"type":"function","function":{"name":"search"}}`,
			wantReject: true,
			because:    "`function` is not a member of the ToolChoice union",
		},
		{
			name:       "Bedrock Converse spelling",
			toolChoice: `{"tool":{"name":"search"}}`,
			wantReject: true,
			because:    "every ToolChoice member requires `type`",
		},
		{
			name:       "name carried on the any variant",
			toolChoice: `{"type":"any","name":"search"}`,
			wantReject: true,
			because:    "ToolChoiceAny is additionalProperties:false",
		},
		{
			name:       "name that matches no declared tool",
			toolChoice: `{"type":"tool","name":"undeclared"}`,
			because:    "the schema carries no cross-field constraint; ValidateRequestFeatures does",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := []byte(preamble + `"tool_choice":` + tc.toolChoice + `}`)
			err := conformance.Validate("anthropic", "create_message_request", body)
			if tc.wantReject && err == nil {
				t.Fatalf("gate accepted %s, want rejection (%s)", tc.toolChoice, tc.because)
			}
			if !tc.wantReject && err != nil {
				t.Fatalf("gate rejected %s (%v), want acceptance (%s)", tc.toolChoice, err, tc.because)
			}
		})
	}
}

func floatPtr(v float64) *float64 { return &v }
