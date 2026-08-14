package bedrock_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/conformance"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

// This file is the request-direction half of the Bedrock conformance suite.
// Every case encodes a real inference.Request and holds the bytes against AWS's
// own ConverseRequest schema before asserting anything about them.
//
// Read the positive cases with the document's real strength in mind. AWS marks
// only modelId @required, and modelId travels in the URI path, so the derived
// request document requires NOTHING at the top level: a body with no messages
// passes the gate. What it enforces hard is @pattern, @length, enum membership
// and union arity, which is exactly where Bedrock's ValidationExceptions come
// from — ToolUseId's anchored ^[a-zA-Z0-9_.:-]+$ capped at 64, ToolName's
// ^[a-zA-Z0-9_-]+$ capped at 64, InferenceConfiguration's [0,1] temperature and
// topP and >= 1 maxTokens, SystemContentBlock's three-member union.

func gatedConverseEncode(t *testing.T, req inference.Request) []byte {
	t.Helper()
	body, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	gateConverseRequest(t, body)
	return body
}

func converseEncodeModel(opts ...model.ModelOption) model.Model {
	base := []model.ModelOption{
		model.WithTools(), model.WithImages(), model.WithThinking(),
		model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}),
	}
	base = append(base, opts...)
	return model.CustomModel(model.ProviderName(llm.ProviderBedrock), model.APIFormatBedrockConverse, "", converseModelID, base...)
}

func converseUserTurn(blocks ...content.Block) content.Conversation {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func converseAssistantTurn(blocks ...content.Block) content.Conversation {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

var conversePNG = []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}

var converseMP3 = []byte{0x49, 0x44, 0x33, 0x04}

// TestConverseEncodeShapesAreSpecLegal walks the request shapes the Converse
// path really emits.
func TestConverseEncodeShapesAreSpecLegal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "empty conversation still encodes messages as an array",
			req:  inference.Request{Model: converseEncodeModel()},
		},
		{
			name: "system prefix and a text turn",
			req: inference.Request{
				Model:    converseEncodeModel(),
				System:   "You are precise.",
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "hello"})},
			},
		},
		{
			name: "inline image bytes",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(
					&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: conversePNG}},
					&content.TextBlock{Text: "Describe this image."},
				)},
			},
		},
		{
			// ContentBlock declares an `audio` member, and AudioBlock requires
			// both format and source. The gate checks AudioFormat enum
			// membership, AudioSource's two-member union arity, and the
			// @length min 1 on the base64 bytes — every constraint the encoder
			// transcribed.
			name: "inline audio bytes",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(
					&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: converseMP3},
					&content.TextBlock{Text: "Transcribe this."},
				)},
			},
		},
		{
			// Every media type the shared vocabulary names, so a later AudioFormat
			// change shows up as a gate failure rather than a silent 400.
			name: "every mappable audio media type",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(
					&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioOGG, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioFLAC, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioAAC, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioMP4, Data: converseMP3},
					&content.AudioBlock{MediaType: content.MediaTypeAudioWebM, Data: converseMP3},
					&content.TextBlock{Text: "Transcribe all of them."},
				)},
			},
		},
		{
			// DocumentBlock requires name and source; DocumentSource is a
			// four-member union of which the encoder can select bytes and text.
			name: "documents in both reachable source members",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report-pdf", Data: []byte("%PDF-")},
					&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "notes-md", Text: "# Notes"},
					&content.TextBlock{Text: "Summarize the attachments."},
				)},
			},
		},
		{
			name: "reasoning replay with a signature",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "go"}),
					converseAssistantTurn(
						content.NewSignedThinkingBlock("reasoned", "sig", "bedrock-converse", nil, ""),
						&content.TextBlock{Text: "done"},
					),
				},
			},
		},
		{
			name: "tool use and tool result round trip",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "Weather in Paris?"}),
					converseAssistantTurn(content.NewToolUseBlock("tooluse_weather_1", "get_weather", json.RawMessage(`{"city":"Paris"}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "tooluse_weather_1",
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "18C"}}},
					},
				},
				Tools: []inference.Tool{{Name: "get_weather", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "errored tool result carries the error status",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "Read it"}),
					converseAssistantTurn(content.NewToolUseBlock("tooluse_read_1", "read_file", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "tooluse_read_1",
						IsError:   true,
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "denied"}}},
					},
				},
				Tools: []inference.Tool{{Name: "read_file", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			// The full legal ToolUseId class, including the "." and ":" that
			// Anthropic's narrower class excludes.
			name: "tool-use id at the full legal character class and length cap",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "go"}),
					converseAssistantTurn(content.NewToolUseBlock("aZ0_.:-", "lookup", json.RawMessage(`{}`), nil, "")),
					converseUserTurn(&content.TextBlock{Text: "next"}),
					converseAssistantTurn(content.NewToolUseBlock(strings.Repeat("a", 64), "lookup", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "tool name at the 64 character cap",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: strings.Repeat("t", 64), Schema: json.RawMessage(`{"type":"object"}`)}},
			},
		},
		{
			name: "sampling at the interval boundaries",
			req: inference.Request{
				Model: converseEncodeModel(model.WithSampling(model.Sampling{
					Temperature: converseFloatPtr(0), TopP: converseFloatPtr(1), MaxTokens: converseIntPtr(1),
				})),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
			},
		},
		{
			name: "structured output",
			req: inference.Request{
				Model:    converseEncodeModel(model.WithStructuredOutput()),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Output:   &inference.OutputSchema{Name: "answer", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)},
			},
		},
		{
			name: "required tool choice",
			req: inference.Request{
				Model:      converseEncodeModel(),
				Messages:   content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Tools:      []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
				ToolChoice: inference.ToolRequired(),
			},
		},
		{
			name: "named tool choice forcing one declared tool",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Tools: []inference.Tool{
					{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)},
					{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)},
				},
				ToolChoice: inference.ToolNamed("search"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gatedConverseEncode(t, tc.req)
		})
	}
}

// TestConverseEncodeCountTokensInputIsSpecLegal gates the count-tokens
// projection. It is the same conversation encoding with the generation-only
// fields removed, so it is held to the same schema.
func TestConverseEncodeCountTokensInputIsSpecLegal(t *testing.T) {
	t.Parallel()

	body, err := bedrockconverse.EncodeCountTokensInput(inference.Request{
		Model:    converseEncodeModel(model.WithSampling(model.Sampling{Temperature: converseFloatPtr(0.5)})),
		System:   "count this",
		Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "hello"})},
		Tools:    []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("EncodeCountTokensInput() error = %v", err)
	}
	gateConverseRequest(t, body)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	if _, present := fields["inferenceConfig"]; present {
		t.Error("count body carries inferenceConfig, which cannot affect an input count")
	}
}

// TestConverseEncodeRefusesWireIllegalValues is the negative half, and it
// records the defects the request gate surfaced in the Converse encoder. Each
// value used to be forwarded verbatim into a body AWS's own Smithy model
// declares illegal; each now fails closed before any I/O.
//
// The Converse encoder was already the more defensive of the two — it validated
// image formats, empty ids and non-object tool schemas long before this — so
// what the gate found here is the set of constraints it did NOT transcribe:
// every @pattern and @length on the identifiers, the sampling ranges, and the
// one omitempty that erases a union discriminator.
func TestConverseEncodeRefusesWireIllegalValues(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		req     inference.Request
		because string
	}{
		{
			name: "tool-use id past the 64 character cap",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "go"}),
					converseAssistantTurn(content.NewToolUseBlock(strings.Repeat("a", 65), "lookup", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolUseId has maxLength 64 and other dialects impose no such cap",
		},
		{
			name: "tool-use id outside the character class",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "go"}),
					converseAssistantTurn(content.NewToolUseBlock("call abc/def", "lookup", json.RawMessage(`{}`), nil, "")),
				},
				Tools: []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolUseId is the ANCHORED ^[a-zA-Z0-9_.:-]+$, so interior illegal characters reject",
		},
		{
			name: "tool-result tool_use_id outside the character class",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "go"}),
					converseAssistantTurn(content.NewToolUseBlock("ok_1", "lookup", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "bad id/here",
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "x"}}},
					},
				},
				Tools: []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolResultBlock.toolUseId carries the same constraint as ToolUseBlock's",
		},
		{
			name: "MCP-style dotted tool name",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: "fs.read_file", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolName is ^[a-zA-Z0-9_-]+$, which excludes the dot Converse's ToolUseId allows",
		},
		{
			name: "tool name past the 64 character cap",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
				Tools:    []inference.Tool{{Name: strings.Repeat("t", 65), Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolName has maxLength 64, half of Anthropic's 128",
		},
		{
			// The gate can see an illegal enum member but not an absent one:
			// AudioFormat is a closed enum, so a media type with no member has
			// no legal value to send at all. Only the encoder can catch it.
			name: "audio media type outside the AudioFormat enum",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.AudioBlock{MediaType: content.MediaType("audio/amr"), Data: converseMP3})},
			},
			because: "AudioFormat has no member for audio/amr, and format is @required",
		},
		{
			name: "empty audio payload",
			req: inference.Request{
				Model:    converseEncodeModel(),
				Messages: content.AgenticMessages{converseUserTurn(&content.AudioBlock{MediaType: content.MediaTypeAudioWAV})},
			},
			because: "AudioSource.bytes declares @length min 1",
		},
		{
			// ToolResultContentBlock is a different union from ContentBlock and
			// declares no audio member. This is the shape an MCP audio tool
			// result really produces.
			name: "audio inside a tool result",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					converseUserTurn(&content.TextBlock{Text: "listen"}),
					converseAssistantTurn(content.NewToolUseBlock("tooluse_listen", "listen", json.RawMessage(`{}`), nil, "")),
					&content.ToolResultMessage{
						ToolUseID: "tooluse_listen",
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.AudioBlock{MediaType: content.MediaTypeAudioMPEG, Data: converseMP3}}},
					},
				},
				Tools: []inference.Tool{{Name: "listen", Schema: json.RawMessage(`{"type":"object"}`)}},
			},
			because: "ToolResultContentBlock's union is json/text/image/document/video/searchResult",
		},
		{
			name: "OpenAI-range temperature",
			req: inference.Request{
				Model:    converseEncodeModel(model.WithSampling(model.Sampling{Temperature: converseFloatPtr(1.7)})),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
			},
			because: "InferenceConfiguration.temperature is bounded to [0,1]",
		},
		{
			name: "topP above one",
			req: inference.Request{
				Model:    converseEncodeModel(model.WithSampling(model.Sampling{TopP: converseFloatPtr(1.5)})),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
			},
			because: "InferenceConfiguration.topP is bounded to [0,1]",
		},
		{
			name: "zero maxTokens",
			req: inference.Request{
				Model:    converseEncodeModel(model.WithSampling(model.Sampling{MaxTokens: converseIntPtr(0)})),
				Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
			},
			because: "InferenceConfiguration.maxTokens has minimum 1",
		},
		{
			// SystemContentBlock.text is Converse's NonEmptyString AND the field
			// is omitempty on the wire shape, so an empty system text used to
			// encode to the bare object {} — a SystemContentBlock matching none
			// of its three union members.
			name: "empty in-thread system text",
			req: inference.Request{
				Model: converseEncodeModel(),
				Messages: content.AgenticMessages{
					&content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: []content.Block{&content.TextBlock{Text: ""}}}},
					converseUserTurn(&content.TextBlock{Text: "go"}),
				},
			},
			because: "omitempty erased the union discriminator, producing {}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body, err := bedrockconverse.EncodeRequest(tc.req)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted a body Bedrock rejects (%s): %s", tc.because, body)
			}
			if body != nil {
				t.Errorf("EncodeRequest() returned %d bytes alongside its error, want none", len(body))
			}
			if !strings.HasPrefix(err.Error(), "bedrockconverse: ") {
				t.Errorf("error = %q, want a typed bedrockconverse encode failure (%s)", err, tc.because)
			}
		})
	}
}

// TestConverseEncodeErrorsAreTyped keeps the new rejections usable by callers:
// each is an errors.As-able codec error, not a bare fmt.Errorf.
func TestConverseEncodeErrorsAreTyped(t *testing.T) {
	t.Parallel()

	_, err := bedrockconverse.EncodeRequest(inference.Request{
		Model:    converseEncodeModel(model.WithSampling(model.Sampling{Temperature: converseFloatPtr(2)})),
		Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
	})
	var encodeErr *bedrockconverse.EncodeError
	if !errors.As(err, &encodeErr) {
		t.Fatalf("error = %T (%v), want *bedrockconverse.EncodeError", err, err)
	}

	_, err = bedrockconverse.EncodeRequest(inference.Request{
		Model:    converseEncodeModel(),
		Messages: content.AgenticMessages{converseUserTurn(&content.TextBlock{Text: "go"})},
		Tools:    []inference.Tool{{Name: "fs.read", Schema: json.RawMessage(`{"type":"object"}`)}},
	})
	var schemaErr *bedrockconverse.ToolSchemaError
	if !errors.As(err, &schemaErr) {
		t.Fatalf("error = %T (%v), want *bedrockconverse.ToolSchemaError", err, err)
	}
}

func converseFloatPtr(v float64) *float64 { return &v }
func converseIntPtr(v int) *int           { return &v }

// TestToolChoiceGateStrength measures what the Converse request gate really
// enforces on toolChoice, by feeding it shapes the encoder never produces.
//
// This is the one place Converse's derived document is strict. ToolChoice is a
// Smithy union, and the generator turns that into a oneOf over required member
// keys, so a choice naming no member and a choice naming two are both
// rejected. SpecificToolChoice.name is @required and targets ToolName, whose
// ^[a-zA-Z0-9_-]+$ / 1..64 constraints survive into the schema — so an illegal
// name is caught here rather than by an HTTP 400. What the gate cannot see is
// whether the name matches a declared tool; inference.ValidateRequestFeatures
// carries that.
func TestToolChoiceGateStrength(t *testing.T) {
	t.Parallel()

	body := func(toolChoice string) []byte {
		return []byte(`{"messages":[{"role":"user","content":[{"text":"hi"}]}],` +
			`"toolConfig":{"tools":[{"toolSpec":{"name":"search","inputSchema":{"json":{"type":"object"}}}}],` +
			`"toolChoice":` + toolChoice + `}}`)
	}

	cases := []struct {
		name       string
		toolChoice string
		wantReject bool
		because    string
	}{
		{
			name:       "the shape the encoder emits",
			toolChoice: `{"tool":{"name":"search"}}`,
			because:    "SpecificToolChoice requires name and nothing else",
		},
		{
			name:       "tool member with no name",
			toolChoice: `{"tool":{}}`,
			wantReject: true,
			because:    "SpecificToolChoice.name is @required",
		},
		{
			name:       "no union member at all",
			toolChoice: `{}`,
			wantReject: true,
			because:    "ToolChoice's oneOf requires exactly one of any/auto/tool",
		},
		{
			name:       "two union members",
			toolChoice: `{"any":{},"tool":{"name":"search"}}`,
			wantReject: true,
			because:    "a Smithy union carries exactly one member; oneOf enforces the arity",
		},
		{
			name:       "Anthropic spelling",
			toolChoice: `{"type":"tool","name":"search"}`,
			wantReject: true,
			because:    "`type` is not a member of the union",
		},
		{
			name:       "name outside ToolName's pattern",
			toolChoice: `{"tool":{"name":"fs.read_file"}}`,
			wantReject: true,
			because:    "ToolName is ^[a-zA-Z0-9_-]+$; a dotted MCP tool name is illegal",
		},
		{
			name:       "name past ToolName's 64 character cap",
			toolChoice: `{"tool":{"name":"` + strings.Repeat("t", 65) + `"}}`,
			wantReject: true,
			because:    "ToolName has @length max 64",
		},
		{
			name:       "name that matches no declared tool",
			toolChoice: `{"tool":{"name":"undeclared"}}`,
			because:    "no cross-field constraint exists; ValidateRequestFeatures carries it",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := conformance.Validate("bedrock-converse", kindRequest, body(tc.toolChoice))
			if tc.wantReject && err == nil {
				t.Fatalf("gate accepted %s, want rejection (%s)", tc.toolChoice, tc.because)
			}
			if !tc.wantReject && err != nil {
				t.Fatalf("gate rejected %s (%v), want acceptance (%s)", tc.toolChoice, err, tc.because)
			}
		})
	}
}
