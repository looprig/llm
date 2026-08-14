package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	chat "github.com/looprig/inference/codec/openaiapi"
	responses "github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/openai"
)

// Multimodal REQUEST coverage.
//
// This file validates the bytes Looprig SENDS, not only the bytes it receives.
// The encoded body is held against OpenAI's own request schema before anything
// else is asserted, so an encoder that invents a member, omits a required one,
// or mis-spells a discriminator fails here rather than at the provider.
//
// It has a second job: the request fixtures under testdata/request are legal
// shapes taken from the spec that Looprig's neutral vocabulary CANNOT express
// (image detail, file_id images, file/document parts, audio parts). Gating
// them proves the shape is real and legal, which is what turns "we do not
// support this" from an assertion into a measured coverage gap.

const (
	chatRequestDir      = "testdata/request/chat"
	responsesRequestDir = "testdata/request/responses"

	chatRequestKind      = "chat_completion_request"
	responsesRequestKind = "create_response_request"
)

// pngDataURI is the one-pixel PNG the encode tests feed as inline image bytes.
var pngPixel = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
	0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89,
}

// captureRequest drives the real client against a server that records the
// request body, gate-validates that body against the named request schema, and
// returns it. The gate runs before the caller sees a single byte.
func captureRequest(t *testing.T, apiFormat model.APIFormat, format, kind string,
	responseFixtureDir, responseFixture string, req func(model.Model) inference.Request,
	modelOptions ...model.ModelOption) []byte {
	t.Helper()

	responseBody := readFixture(t, responseFixtureDir, responseFixture)
	bodies := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("read body: %v", err), http.StatusInternalServerError)
			return
		}
		bodies <- raw
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(responseBody)
	}))
	t.Cleanup(srv.Close)

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI), apiFormat, srv.URL+"/v1", "gpt-4.1",
		modelOptions...,
	)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), req(selected)); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	select {
	case body := <-bodies:
		conformance.MustValidateRequest(t, format, kind, body)
		return body
	default:
		t.Fatal("no request body captured")
		return nil
	}
}

// gateChatRequest holds an encoded Chat Completions body against OpenAI's own
// request schema. Call it on the captured bytes BEFORE asserting anything about
// them: an assertion about a body the provider would reject proves nothing.
func gateChatRequest(t testing.TB, body []byte) []byte {
	t.Helper()
	conformance.MustValidateRequest(t, "openai", chatRequestKind, body)
	return body
}

// gateResponsesRequest is gateChatRequest's Responses counterpart.
func gateResponsesRequest(t testing.TB, body []byte) []byte {
	t.Helper()
	conformance.MustValidateRequest(t, "openai-responses", responsesRequestKind, body)
	return body
}

func captureChatRequest(t *testing.T, blocks []content.Block) []byte {
	t.Helper()
	return captureRequest(t, model.APIFormatOpenAI, "openai", chatRequestKind,
		chatDir, "plain_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{
				Model: selected,
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{
						Role: content.RoleUser, Blocks: blocks,
					}},
				},
			}
		},
		model.WithImages(),
	)
}

func captureResponsesRequest(t *testing.T, blocks []content.Block) []byte {
	t.Helper()
	return captureRequest(t, model.APIFormatOpenAIResponses, "openai-responses", responsesRequestKind,
		responsesDir, "completed_text.json",
		func(selected model.Model) inference.Request {
			return inference.Request{
				Model: selected,
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{
						Role: content.RoleUser, Blocks: blocks,
					}},
				},
			}
		},
		model.WithImages(),
	)
}

// --- the fixture corpus of legal request shapes ---------------------------

// TestEveryMultimodalRequestFixtureIsLegal walks the request corpus and holds
// each file against the provider's own request schema, exactly as the response
// corpus is held against the response schema.
func TestEveryMultimodalRequestFixtureIsLegal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		dir    string
		format string
		kind   string
		want   int
	}{
		{chatRequestDir, "openai", chatRequestKind, 10},
		{responsesRequestDir, "openai-responses", responsesRequestKind, 7},
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
				conformance.MustValidateRequest(t, tc.format, tc.kind, readFixture(t, tc.dir, name))
			}
		})
	}
}

// --- Chat Completions: what Looprig actually encodes ----------------------

// chatUserParts pulls messages[0].content out of an encoded chat body as the
// content-part array. It fails the test if the encoder emitted the bare-string
// form instead.
func chatUserParts(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var decoded struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode request body: %v\nbody: %s", err, body)
	}
	if len(decoded.Messages) != 1 || decoded.Messages[0].Role != "user" {
		t.Fatalf("messages = %+v, want a single user message", decoded.Messages)
	}
	parts := make([]map[string]json.RawMessage, 0, len(decoded.Messages[0].Content))
	for _, raw := range decoded.Messages[0].Content {
		var part map[string]json.RawMessage
		if err := json.Unmarshal(raw, &part); err != nil {
			t.Fatalf("decode content part %s: %v", raw, err)
		}
		parts = append(parts, part)
	}
	return parts
}

func partString(t *testing.T, part map[string]json.RawMessage, key string) string {
	t.Helper()
	raw, ok := part[key]
	if !ok {
		t.Fatalf("content part %v has no %q", part, key)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("decode %q: %v", key, err)
	}
	return s
}

func chatImageURL(t *testing.T, part map[string]json.RawMessage) map[string]string {
	t.Helper()
	return chatPartObject(t, part, "image_url")
}

// chatPartObject pulls one nested object member out of a chat content part —
// image_url, file or input_audio, each of which nests its payload rather than
// keeping it flat the way Responses parts do.
func chatPartObject(t *testing.T, part map[string]json.RawMessage, key string) map[string]string {
	t.Helper()
	raw, ok := part[key]
	if !ok {
		t.Fatalf("content part %v has no %s", part, key)
	}
	var obj map[string]string
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("decode %s: %v", key, err)
	}
	return obj
}

func TestChatEncodesRemoteImageURL(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.TextBlock{Text: "What is in this picture?"},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/cat.png"}},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want text then image", len(parts))
	}
	if got := partString(t, parts[0], "type"); got != "text" {
		t.Errorf("part 0 type = %q, want text", got)
	}
	if got := partString(t, parts[1], "type"); got != "image_url" {
		t.Errorf("part 1 type = %q, want image_url", got)
	}
	if got := chatImageURL(t, parts[1])["url"]; got != "https://example.com/cat.png" {
		t.Errorf("image url = %q, want the remote URL verbatim", got)
	}
}

func TestChatEncodesInlineImageAsDataURI(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{Data: pngPixel}},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 1 {
		t.Fatalf("content parts = %d, want one image", len(parts))
	}
	url := chatImageURL(t, parts[0])["url"]
	const wantPrefix = "data:image/png;base64,"
	if len(url) <= len(wantPrefix) || url[:len(wantPrefix)] != wantPrefix {
		t.Errorf("image url = %q, want a %s data URI", url, wantPrefix)
	}
}

func TestChatEncodesMultipleImagesInOneMessage(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.TextBlock{Text: "Compare these."},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/a.png"}},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/b.png"}},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{Data: pngPixel}},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 4 {
		t.Fatalf("content parts = %d, want text plus three images", len(parts))
	}
	for i, want := range []string{"text", "image_url", "image_url", "image_url"} {
		if got := partString(t, parts[i], "type"); got != want {
			t.Errorf("part %d type = %q, want %q", i, got, want)
		}
	}
}

// TestChatPreservesInterleavedTextAndImageOrder pins that block order survives
// into the content-part array: an image described by the text that follows it
// is a different prompt from one described by the text before it.
func TestChatPreservesInterleavedTextAndImageOrder(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.TextBlock{Text: "Before."},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/mid.png"}},
		&content.TextBlock{Text: "After."},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 3 {
		t.Fatalf("content parts = %d, want text/image/text", len(parts))
	}
	if got := partString(t, parts[0], "text"); got != "Before." {
		t.Errorf("part 0 text = %q, want Before.", got)
	}
	if got := partString(t, parts[1], "type"); got != "image_url" {
		t.Errorf("part 1 type = %q, want image_url", got)
	}
	if got := partString(t, parts[2], "text"); got != "After." {
		t.Errorf("part 2 text = %q, want After.", got)
	}
}

// TestChatEmitsNoImageDetail records a coverage FINDING. The spec's
// image_url object carries an optional `detail` of auto|low|high, which is the
// only lever a caller has over vision token cost. content.ImageBlock has no
// field for it and openaiapi never emits it, so every image silently takes the
// provider's default. testdata/request/chat/image_detail_{low,high,auto}.json
// are gated proof that the shape is legal and simply unreachable from Looprig.
func TestChatEmitsNoImageDetail(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/cat.png"}},
	})

	if _, ok := chatImageURL(t, chatUserParts(t, body)[0])["detail"]; ok {
		t.Fatal("image_url.detail is now emitted — update this finding")
	}
}

// --- Responses: what Looprig actually encodes -----------------------------

func responsesInputParts(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var decoded struct {
		Input []struct {
			Type    string            `json:"type"`
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode request body: %v\nbody: %s", err, body)
	}
	if len(decoded.Input) != 1 || decoded.Input[0].Role != "user" {
		t.Fatalf("input = %+v, want a single user item", decoded.Input)
	}
	parts := make([]map[string]json.RawMessage, 0, len(decoded.Input[0].Content))
	for _, raw := range decoded.Input[0].Content {
		var part map[string]json.RawMessage
		if err := json.Unmarshal(raw, &part); err != nil {
			t.Fatalf("decode content part %s: %v", raw, err)
		}
		parts = append(parts, part)
	}
	return parts
}

func TestResponsesEncodesRemoteInputImage(t *testing.T) {
	t.Parallel()
	body := captureResponsesRequest(t, []content.Block{
		&content.TextBlock{Text: "What is in this picture?"},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/cat.png"}},
	})

	parts := responsesInputParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want input_text then input_image", len(parts))
	}
	if got := partString(t, parts[0], "type"); got != "input_text" {
		t.Errorf("part 0 type = %q, want input_text", got)
	}
	if got := partString(t, parts[1], "type"); got != "input_image" {
		t.Errorf("part 1 type = %q, want input_image", got)
	}
	if got := partString(t, parts[1], "image_url"); got != "https://example.com/cat.png" {
		t.Errorf("image_url = %q, want the remote URL verbatim", got)
	}
	// InputImageContent.required includes `detail`, so omitting it would be an
	// illegal payload. The encoder always sends "auto".
	if got := partString(t, parts[1], "detail"); got != "auto" {
		t.Errorf("detail = %q, want auto", got)
	}
}

func TestResponsesEncodesInlineInputImageAsDataURI(t *testing.T) {
	t.Parallel()
	body := captureResponsesRequest(t, []content.Block{
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{Data: pngPixel}},
	})

	parts := responsesInputParts(t, body)
	url := partString(t, parts[0], "image_url")
	const wantPrefix = "data:image/png;base64,"
	if len(url) <= len(wantPrefix) || url[:len(wantPrefix)] != wantPrefix {
		t.Errorf("image_url = %q, want a %s data URI", url, wantPrefix)
	}
}

func TestResponsesPreservesInterleavedTextAndImageOrder(t *testing.T) {
	t.Parallel()
	body := captureResponsesRequest(t, []content.Block{
		&content.TextBlock{Text: "Before."},
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/mid.png"}},
		&content.TextBlock{Text: "After."},
	})

	parts := responsesInputParts(t, body)
	if len(parts) != 3 {
		t.Fatalf("content parts = %d, want input_text/input_image/input_text", len(parts))
	}
	if got := partString(t, parts[0], "text"); got != "Before." {
		t.Errorf("part 0 text = %q, want Before.", got)
	}
	if got := partString(t, parts[1], "type"); got != "input_image" {
		t.Errorf("part 1 type = %q, want input_image", got)
	}
	if got := partString(t, parts[2], "text"); got != "After." {
		t.Errorf("part 2 text = %q, want After.", got)
	}
}

// TestResponsesEmitsNoImageFileID records a coverage FINDING: an input_image
// may reference an uploaded file by `file_id` instead of carrying a URL, and
// content.ImageSource has only URL and inline Data. Every Looprig image is
// therefore re-uploaded inline or fetched by URL, never referenced.
// testdata/request/responses/input_image_file_id.json is gated proof of the
// legal shape.
func TestResponsesEmitsNoImageFileID(t *testing.T) {
	t.Parallel()
	body := captureResponsesRequest(t, []content.Block{
		&content.ImageBlock{MediaType: content.MediaTypeImagePNG,
			Source: content.ImageSource{URL: "https://example.com/cat.png"}},
	})

	if _, ok := responsesInputParts(t, body)[0]["file_id"]; ok {
		t.Fatal("input_image.file_id is now emitted — update this finding")
	}
}

// --- documents and audio: what Looprig actually encodes -------------------

// pdfBytes is the inline document payload the encode tests feed.
var pdfBytes = []byte("%PDF-1.4\n")

// TestChatEncodesDocumentAsFilePart holds the file content part against the
// request schema. ChatCompletionRequestUserMessageContentPart's file member
// requires ["type","file"]; the inline form the neutral vocabulary can reach
// carries filename plus file_data.
func TestChatEncodesDocumentAsFilePart(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.TextBlock{Text: "Summarize the attached report."},
		&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: pdfBytes},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want text then file", len(parts))
	}
	if got := partString(t, parts[1], "type"); got != "file" {
		t.Fatalf("part 1 type = %q, want file", got)
	}
	file := chatPartObject(t, parts[1], "file")
	if file["filename"] != "report.pdf" {
		t.Errorf("file.filename = %q, want report.pdf", file["filename"])
	}
	want := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfBytes)
	if file["file_data"] != want {
		t.Errorf("file.file_data = %q, want the media type carried in a data URI", file["file_data"])
	}
}

// TestChatEncodesAudioAsInputAudioPart holds the audio content part against the
// request schema. input_audio.required is ["data","format"], and `format` is a
// closed two-member enum the gate does check.
func TestChatEncodesAudioAsInputAudioPart(t *testing.T) {
	t.Parallel()
	body := captureChatRequest(t, []content.Block{
		&content.TextBlock{Text: "Transcribe this."},
		&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: []byte("RIFF")},
	})

	parts := chatUserParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want text then input_audio", len(parts))
	}
	if got := partString(t, parts[1], "type"); got != "input_audio" {
		t.Fatalf("part 1 type = %q, want input_audio", got)
	}
	audio := chatPartObject(t, parts[1], "input_audio")
	if audio["format"] != "wav" {
		t.Errorf("input_audio.format = %q, want wav", audio["format"])
	}
	if want := base64.StdEncoding.EncodeToString([]byte("RIFF")); audio["data"] != want {
		t.Errorf("input_audio.data = %q, want %q", audio["data"], want)
	}
}

// TestChatRefusesAudioOutsideTheFormatEnum pins that the enum is enforced
// locally rather than relied on being caught by the gate: the encoder never
// produces a body for an mp4/flac/ogg AudioBlock at all.
func TestChatRefusesAudioOutsideTheFormatEnum(t *testing.T) {
	t.Parallel()

	err := invokeExpectingNoRequest(t, model.APIFormatOpenAI, &content.AudioBlock{
		MediaType: content.MediaTypeAudioFLAC, Data: []byte("fLaC"),
	})
	var chatErr *chat.UnsupportedBlockError
	if !errors.As(err, &chatErr) {
		t.Fatalf("error = %T %v, want *chat.UnsupportedBlockError", err, err)
	}
}

// TestResponsesEncodesDocumentAsInputFile holds the input_file content part
// against the request schema. InputFileContent.required is ["type"] alone.
func TestResponsesEncodesDocumentAsInputFile(t *testing.T) {
	t.Parallel()
	body := captureResponsesRequest(t, []content.Block{
		&content.TextBlock{Text: "Summarize the attached report."},
		&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: pdfBytes},
	})

	parts := responsesInputParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("content parts = %d, want input_text then input_file", len(parts))
	}
	if got := partString(t, parts[1], "type"); got != "input_file" {
		t.Fatalf("part 1 type = %q, want input_file", got)
	}
	if got := partString(t, parts[1], "filename"); got != "report.pdf" {
		t.Errorf("filename = %q, want report.pdf", got)
	}
	want := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfBytes)
	if got := partString(t, parts[1], "file_data"); got != want {
		t.Errorf("file_data = %q, want the media type carried in a data URI", got)
	}
}

// --- fail-closed behaviour where a modality has no wire home ---------------

// TestResponsesRefusesAudio records the one genuine wire limitation among the
// four dialect/modality pairs. The Responses input content union is
// input_text|input_image|input_file; the spec's InputAudio object is
// referenced only from EvalItemContentItem, the Evals API. So audio bound for a
// Responses target is refused with a typed error naming that, never degraded
// into a file or text part.
func TestResponsesRefusesAudio(t *testing.T) {
	t.Parallel()

	err := invokeExpectingNoRequest(t, model.APIFormatOpenAIResponses, &content.AudioBlock{
		MediaType: content.MediaTypeAudioWAV, Data: []byte("RIFF"),
	})
	var respErr *responses.UnsupportedBlockError
	if !errors.As(err, &respErr) {
		t.Fatalf("error = %T %v, want *responses.UnsupportedBlockError", err, err)
	}
	if respErr.Reason == "" {
		t.Error("UnsupportedBlockError.Reason is empty; it must name the limitation")
	}
}

// TestToolResultMediaBlocksFailClosed records the remaining gap, and it is a
// live one: mcp/pkg/harness builds AudioBlock and DocumentBlock values out of
// MCP tool results, and both dialects carry a tool result as plain text.
// ChatCompletionRequestToolMessageContentPart is a one-member union over the
// text part ("For tool messages, only type `text` is supported"), and this
// codec encodes a Responses function_call_output as the `output` string form.
// Both refuse rather than dropping the attachment, so the failure is visible.
//
// Responses' FunctionCallOutputItemParam.output does have a second form — an
// array of input_text|input_image|input_file — which would give a document
// tool result a home. Audio has none in either dialect.
func TestToolResultMediaBlocksFailClosed(t *testing.T) {
	t.Parallel()

	blocks := map[string]content.Block{
		"document": &content.DocumentBlock{
			MediaType: content.MediaTypeDocumentPDF,
			Name:      "report.pdf",
			Data:      pdfBytes,
		},
		"audio": &content.AudioBlock{
			MediaType: content.MediaTypeAudioWAV,
			Data:      []byte("RIFF"),
		},
	}
	formats := map[string]model.APIFormat{
		"chat":      model.APIFormatOpenAI,
		"responses": model.APIFormatOpenAIResponses,
	}

	for blockName, block := range blocks {
		for formatName, apiFormat := range formats {
			t.Run(formatName+"/"+blockName, func(t *testing.T) {
				t.Parallel()

				var reached bool
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					reached = true
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, "{}")
				}))
				t.Cleanup(srv.Close)

				selected := model.CustomModel(
					model.ProviderName(llm.ProviderOpenAI), apiFormat, srv.URL+"/v1", "gpt-4.1",
					model.WithImages(),
				)
				client, err := openai.New(selected, "sk-test")
				if err != nil {
					t.Fatalf("openai.New() error = %v", err)
				}
				_, err = client.Invoke(context.Background(), inference.Request{
					Model: selected,
					Messages: content.AgenticMessages{
						&content.ToolResultMessage{
							Message: content.Message{
								Role:   content.RoleTool,
								Blocks: []content.Block{block},
							},
							ToolUseID: "call_1",
						},
					},
				})
				if err == nil {
					t.Fatal("Invoke() error = nil, want a refusal to encode the block")
				}
				var chatErr *chat.UnsupportedBlockError
				var respErr *responses.UnsupportedBlockError
				if !errors.As(err, &chatErr) && !errors.As(err, &respErr) {
					t.Fatalf("error = %T %v, want an UnsupportedBlockError", err, err)
				}
				if reached {
					t.Error("the request was sent despite the unencodable block")
				}
			})
		}
	}
}

// invokeExpectingNoRequest drives a one-block user turn through the real
// client and fails the test if any request reaches the wire. It returns the
// error the caller then classifies.
func invokeExpectingNoRequest(t *testing.T, apiFormat model.APIFormat, block content.Block) error {
	t.Helper()

	var reached bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, "{}")
	}))
	t.Cleanup(srv.Close)

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI), apiFormat, srv.URL+"/v1", "gpt-4.1",
		model.WithImages(),
	)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "look"}, block},
			}},
		},
	})
	if err == nil {
		t.Fatal("Invoke() error = nil, want a refusal to encode the block")
	}
	if reached {
		t.Error("the request was sent despite the unencodable block")
	}
	return err
}

// TestImagesRequireTheModelCapability pins the guard that runs before either
// encoder: a model that does not advertise image input refuses the request
// rather than shipping an image the model cannot read.
func TestImagesRequireTheModelCapability(t *testing.T) {
	t.Parallel()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAI,
		"https://unused.invalid/v1", "gpt-4.1",
	)
	client, err := openai.New(selected, "sk-test")
	if err != nil {
		t.Fatalf("openai.New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role: content.RoleUser,
				Blocks: []content.Block{&content.ImageBlock{
					MediaType: content.MediaTypeImagePNG,
					Source:    content.ImageSource{URL: "https://example.com/cat.png"},
				}},
			}},
		},
	})
	var unsupported *inference.ImageInputUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %T %v, want *inference.ImageInputUnsupportedError", err, err)
	}
}
