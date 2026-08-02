package vertex_test

import (
	"context"
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

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/google-vertex"
)

func TestGeminiJSONAndSSE(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer vertex-token" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}
		if r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent" &&
			r.URL.Path != "/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent" {
			t.Errorf("path = %q, want Vertex Gemini method path", r.URL.Path)
		}
		if r.URL.Path[len(r.URL.Path)-len("streamGenerateContent"):] == "streamGenerateContent" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"stream\"}]}}]}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"json"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":6,"totalTokenCount":10}}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGoogleVertex), model.APIFormatGemini, srv.URL, "gemini-2.5-flash")
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
