package vertex_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/google-vertex"
)

// gateSentGeminiRequest holds the body Vertex actually put on the wire against
// Google's generateContent request schema. Vertex reuses the shared geminiapi
// codec, so the same encoder defects the gate catches on the Developer API
// route would otherwise reach Vertex unchecked. It reports non-fatally because
// it runs on the test server's goroutine.
//
// The gate's strength is the same here as everywhere else in the Gemini suite:
// types, enums, array-ness and $ref shapes are enforced; presence is not (the
// discovery document declares almost no required properties), and fields typed
// `any` — parametersJsonSchema, responseJsonSchema — are not inspected at all.
// Union arity is NOT enforced either: the derived request document contains no
// oneOf at all, so a Part carrying two members of Part.data validates cleanly
// and has to be pinned by an explicit assertion instead — see
// TestGeminiMediaRequestReachesVertexLegally.
func gateSentGeminiRequest(t *testing.T, r *http.Request) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read request body: %v", err)
		return
	}
	if err := conformance.Validate("gemini", "generate_content_request", body); err != nil {
		t.Errorf("the encoded Vertex Gemini request is not a legal payload: %v", err)
	}
}

func TestGeminiJSONAndSSE(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer vertex-token" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}
		if r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent" &&
			r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent" {
			t.Errorf("path = %q, want Vertex Gemini method path", r.URL.Path)
		}
		gateSentGeminiRequest(t, r)
		if r.URL.Path[len(r.URL.Path)-len("streamGenerateContent"):] == "streamGenerateContent" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"stream\"}]}}]}\n\n")
			// Terminal frame: an empty finishReason means the model has not
			// stopped generating, so the decoder treats a stream without one as
			// truncated rather than complete.
			_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"json"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":6,"totalTokenCount":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGoogleVertex), model.APIFormatGemini, srv.URL, "gemini-2.5-flash", model.WithTools())
	client, err := vertex.New(selected, auth.APIKey("vertex-token"), vertex.WithProject("project"), vertex.WithLocation("us-central1"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	// A thread and two tool declarations, so the gate above has something to
	// check: Vertex shares the geminiapi codec, including the choice between
	// FunctionDeclaration's projected `parameters` (the first tool, whose
	// schema fits Gemini's Schema dialect) and its verbatim
	// `parametersJsonSchema` (the second, which carries additionalProperties).
	request := inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
		Tools: []inference.Tool{
			{Name: "get_weather", Description: "forecast", Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
			{Name: "search", Description: "search", Schema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"additionalProperties":false}`)},
		},
	}
	response, err := client.Invoke(context.Background(), request)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 || response.Message.Blocks[0].(*content.TextBlock).Text != "json" {
		t.Fatalf("response = %#v, want json text", response)
	}
	reader, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("Stream.Next() error = %v", err)
	}
	if text, ok := chunk.(*content.TextChunk); !ok || text.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want stream text", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want EOF", err)
	}
}

// TestGeminiMediaRequestReachesVertexLegally puts the document and audio
// encodings through the Vertex route. Vertex shares the geminiapi codec, so a
// media part that is wrong is wrong on both routes; what differs is that only
// this test proves the bytes Vertex receives are a legal Gemini payload.
//
// The gate's strength on these parts, measured rather than assumed: it holds
// inlineData to the Blob $ref (an object, not a string) and both of its members
// to `string`. It does NOT check that a member is present, that `data` is really
// base64 (`format: byte` is an annotation this gate does not assert), that the
// mime is one Blob accepts, or that a part carries only one member of the
// Part.data union — Part has no oneOf in the discovery document. Those four are
// held by the encoder and by the codec's own tests instead.
//
// DocumentBlock.Name is deliberately absent from the encoding: Blob has no name
// member, and Part.partMetadata — the one field that could hold it — belongs to
// the v1beta Developer API message, not to the Vertex Part this same encoder
// feeds. It is dropped by decision, not by omission.
func TestGeminiMediaRequestReachesVertexLegally(t *testing.T) {
	pdf := []byte("%PDF-1.4\n")
	mp3 := []byte{0x49, 0x44, 0x33, 0x04}

	var sent []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		sent = body
		if err := conformance.Validate("gemini", "generate_content_request", body); err != nil {
			t.Errorf("the encoded Vertex Gemini media request is not a legal payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"read"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":6,"totalTokenCount":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGoogleVertex), model.APIFormatGemini, srv.URL, "gemini-2.5-flash", model.WithImages())
	client, err := vertex.New(selected, auth.APIKey("vertex-token"), vertex.WithProject("project"), vertex.WithLocation("us-central1"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	request := inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser,
			Blocks: []content.Block{
				&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "contract.pdf", Data: pdf},
				&content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: mp3},
				&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "notes.md", Text: "# notes"},
				&content.TextBlock{Text: "what do these say?"},
			},
		}}},
	}
	if _, err := client.Invoke(context.Background(), request); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	var envelope struct {
		Contents []struct {
			Parts []map[string]json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(sent, &envelope); err != nil {
		t.Fatalf("unmarshal sent body: %v", err)
	}
	if len(envelope.Contents) != 1 || len(envelope.Contents[0].Parts) != 4 {
		t.Fatalf("sent parts = %+v, want one turn of four parts", envelope.Contents)
	}
	dataMembers := []string{"text", "inlineData", "fileData", "functionCall", "functionResponse",
		"executableCode", "codeExecutionResult", "toolCall", "toolResponse"}
	for index, part := range envelope.Contents[0].Parts {
		var set []string
		for _, member := range dataMembers {
			if _, ok := part[member]; ok {
				set = append(set, member)
			}
		}
		if len(set) != 1 {
			t.Errorf("parts[%d] carries %v, want exactly one Part.data member", index, set)
		}
	}

	var blob struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"`
	}
	if err := json.Unmarshal(envelope.Contents[0].Parts[0]["inlineData"], &blob); err != nil {
		t.Fatalf("decode the document blob: %v", err)
	}
	if blob.MimeType != "application/pdf" {
		t.Errorf("parts[0].inlineData.mimeType = %q, want application/pdf", blob.MimeType)
	}
	if got, err := base64.StdEncoding.DecodeString(blob.Data); err != nil || !bytes.Equal(got, pdf) {
		t.Errorf("parts[0].inlineData.data = %q (err=%v), want the pdf bytes; the gate does not assert format: byte", blob.Data, err)
	}
	if err := json.Unmarshal(envelope.Contents[0].Parts[1]["inlineData"], &blob); err != nil {
		t.Fatalf("decode the audio blob: %v", err)
	}
	if blob.MimeType != "audio/wav" {
		t.Errorf("parts[1].inlineData.mimeType = %q, want audio/wav", blob.MimeType)
	}
}

func TestAnthropicJSONAndSSE(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer vertex-token" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}
		if r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-20250514:rawPredict" &&
			r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-20250514:streamRawPredict" {
			t.Errorf("path = %q, want Vertex Claude method path", r.URL.Path)
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("request JSON = %v", err)
		} else {
			var version string
			if err := json.Unmarshal(body["anthropic_version"], &version); err != nil || version != "vertex-2023-10-16" {
				t.Errorf("anthropic_version = %q, err=%v, want vertex-2023-10-16", version, err)
			}
		}
		if r.URL.Path[len(r.URL.Path)-len("streamRawPredict"):] == "streamRawPredict" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"model\":\"model\",\"usage\":{\"input_tokens\":4}}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream\"}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n")
			_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"json"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":6}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGoogleVertexAnthropic), model.APIFormatAnthropic, srv.URL, "claude-sonnet-4-20250514")
	client, err := vertex.New(selected, auth.APIKey("vertex-token"), vertex.WithProject("project"), vertex.WithLocation("us-central1"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 || response.Message.Blocks[0].(*content.TextBlock).Text != "json" {
		t.Fatalf("response = %#v, want json text", response)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = reader.Close() }()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("Stream.Next() error = %v", err)
	}
	if text, ok := chunk.(*content.TextChunk); !ok || text.Text != "stream" {
		t.Fatalf("stream chunk = %#v, want stream text", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want EOF", err)
	}
}
