package gemini_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	failure "github.com/looprig/inference/failure"
	stream "github.com/looprig/inference/stream"

	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
	gemini "github.com/looprig/llm/providers/gemini"
)

// Client must satisfy the inference.Client contract.
var _ inference.Client = (*gemini.Client)(nil)

const testKey = "AIza-test-key"

// geminiResponseJSON is a minimal valid non-streaming generateContent response the
// gemini codec decodes into a one-text-block AIMessage.
const geminiResponseJSON = `{
	"candidates": [
		{"content": {"role": "model", "parts": [{"text": "Hi there"}]}, "finishReason": "STOP", "index": 0}
	],
	"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
	"modelVersion": "gemini-2.5-flash"
}`

// geminiModel builds a Google/Gemini Model with the given model id. BaseURL is the
// real Gemini root so ValidateModel passes; the client routes to its bound
// (test-overridden) endpoint, not this BaseURL.
func geminiModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderGoogle), model.APIFormatGemini, "https://generativelanguage.googleapis.com/v1beta", name)
}

// geminiRequest builds a minimal one-user-message Request for model name.
func geminiRequest(name string) inference.Request {
	return inference.Request{
		Model: geminiModel(name),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	}
}

func readAll(t *testing.T, r *http.Request) []byte {
	t.Helper()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	return b
}

// capturedRequest records the wire-level facts a Gemini request must carry.
type capturedRequest struct {
	method      string
	escapedPath string
	rawQuery    string
	apiKey      string
	contentType string
	accept      string
	body        []byte
}

// TestGeminiInvoke covers the happy Invoke path end-to-end against an
// httptest.Server: the request is POSTed to /models/<id>:generateContent, carries
// the x-goog-api-key header and a JSON body with "contents", and a 200 Gemini
// response decodes to a text-block message.
func TestGeminiInvoke(t *testing.T) {
	t.Parallel()

	const modelID = "gemini-2.5-flash"

	capCh := make(chan capturedRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capCh <- capturedRequest{
			method:      r.Method,
			escapedPath: r.URL.EscapedPath(),
			rawQuery:    r.URL.RawQuery,
			apiKey:      r.Header.Get("x-goog-api-key"),
			contentType: r.Header.Get("Content-Type"),
			accept:      r.Header.Get("Accept"),
			body:        readAll(t, r),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, geminiResponseJSON)
	}))
	defer srv.Close()

	c := gemini.NewWithEndpoint(testKey, srv.URL)
	resp, err := c.Invoke(context.Background(), geminiRequest(modelID))
	if err != nil {
		t.Fatalf("Invoke() err = %v, want nil", err)
	}
	if resp == nil || resp.Message == nil || len(resp.Message.Blocks) != 1 {
		t.Fatalf("Invoke() resp = %+v, want a one-block message", resp)
	}
	if tb, ok := resp.Message.Blocks[0].(*content.TextBlock); !ok || tb.Text != "Hi there" {
		t.Errorf("decoded block = %+v, want TextBlock{Hi there}", resp.Message.Blocks[0])
	}
	if resp.Model != "gemini-2.5-flash" {
		t.Errorf("resp.Model = %q, want gemini-2.5-flash", resp.Model)
	}

	got := <-capCh
	if got.method != http.MethodPost {
		t.Errorf("method = %q, want POST", got.method)
	}
	// The model id is one PathEscaped segment; the ":generateContent" suffix stays
	// literal on the wire.
	wantPath := "/models/" + modelID + ":generateContent"
	if got.escapedPath != wantPath {
		t.Errorf("path = %q, want %q", got.escapedPath, wantPath)
	}
	if got.rawQuery != "" {
		t.Errorf("query = %q, want empty for generateContent", got.rawQuery)
	}
	if got.apiKey != testKey {
		t.Errorf("x-goog-api-key = %q, want %q", got.apiKey, testKey)
	}
	if got.contentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got.contentType)
	}
	if got.accept != "application/json" {
		t.Errorf("Accept = %q, want application/json", got.accept)
	}
	if !strings.Contains(string(got.body), `"contents"`) {
		t.Errorf("body = %s, want a JSON body containing \"contents\"", got.body)
	}
}

// TestGeminiInvokePreservesStructuredOutputWithTools captures the bespoke
// provider client's body and proves its preflight path retains every field the
// shared Gemini codec emits for the combined feature.
func TestGeminiInvokePreservesStructuredOutputWithTools(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCh <- readAll(t, r)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, geminiResponseJSON)
	}))
	defer srv.Close()

	req := geminiRequest("gemini-3-pro-preview")
	req.Model.Caps = model.Capabilities{
		Tools:                     true,
		StructuredOutput:          true,
		StructuredOutputWithTools: true,
	}
	req.Output = &inference.OutputSchema{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"},"details":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}},"required":["answer","details"],"additionalProperties":false}`),
		Strict: true,
	}
	wantProjectedSchema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"},"details":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}},"required":["answer","details"]}`)
	wantToolSchema := json.RawMessage(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
	req.Tools = []inference.Tool{{
		Name:   "lookup",
		Schema: wantToolSchema,
	}}
	req.ToolChoice = inference.ToolChoiceRequired

	c := gemini.NewWithEndpoint(testKey, srv.URL)
	resp, err := c.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke() error = %v, want nil", err)
	}
	if resp.FinishReason != stream.FinishReasonStop {
		t.Errorf("Invoke() FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonStop)
	}

	var wire struct {
		GenerationConfig struct {
			ResponseMIMEType   string          `json:"responseMimeType"`
			ResponseJSONSchema json.RawMessage `json:"responseJsonSchema"`
		} `json:"generationConfig"`
		Tools []struct {
			FunctionDeclarations []struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
		ToolConfig struct {
			FunctionCallingConfig struct {
				Mode string `json:"mode"`
			} `json:"functionCallingConfig"`
		} `json:"toolConfig"`
	}
	if err := json.Unmarshal(<-bodyCh, &wire); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	if wire.GenerationConfig.ResponseMIMEType != "application/json" {
		t.Errorf("generationConfig = %+v, want JSON MIME type and schema", wire.GenerationConfig)
	}
	assertJSONSemanticallyEqual(t, wire.GenerationConfig.ResponseJSONSchema, wantProjectedSchema)
	if len(wire.Tools) != 1 || len(wire.Tools[0].FunctionDeclarations) != 1 || wire.Tools[0].FunctionDeclarations[0].Name != "lookup" {
		t.Errorf("tools = %+v, want one lookup declaration", wire.Tools)
	} else {
		assertJSONSemanticallyEqual(t, wire.Tools[0].FunctionDeclarations[0].Parameters, wantToolSchema)
	}
	if wire.ToolConfig.FunctionCallingConfig.Mode != "ANY" {
		t.Errorf("toolConfig mode = %q, want ANY", wire.ToolConfig.FunctionCallingConfig.Mode)
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

// TestGeminiInvokeErrors covers the mapped failure modes: a non-2xx status maps to
// *failure.APIError (preserving status + body) and a transport failure (server
// closes the connection) maps to *failure.NetworkError.
func TestGeminiInvokeErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		serverStatus int
		serverBody   string
		hijack       bool
		wantAPIErr   bool
		wantAPICode  int
		wantNetErr   bool
	}{
		{
			name:         "non-2xx maps to APIError",
			serverStatus: http.StatusForbidden,
			serverBody:   `{"error":{"message":"api key not valid"}}`,
			wantAPIErr:   true,
			wantAPICode:  http.StatusForbidden,
		},
		{
			name:         "500 maps to APIError",
			serverStatus: http.StatusInternalServerError,
			serverBody:   `{"error":{"message":"boom"}}`,
			wantAPIErr:   true,
			wantAPICode:  http.StatusInternalServerError,
		},
		{
			name:       "transport failure maps to NetworkError",
			hijack:     true,
			wantNetErr: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.hijack {
					hj, ok := w.(http.Hijacker)
					if !ok {
						return
					}
					conn, _, _ := hj.Hijack()
					conn.Close()
					return
				}
				w.WriteHeader(tt.serverStatus)
				fmt.Fprint(w, tt.serverBody)
			}))
			defer srv.Close()

			c := gemini.NewWithEndpoint(testKey, srv.URL)
			_, err := c.Invoke(context.Background(), geminiRequest("gemini-2.5-flash"))

			switch {
			case tt.wantAPIErr:
				var apiErr *failure.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("err = %T, want *failure.APIError", err)
				}
				if apiErr.Status != tt.wantAPICode {
					t.Errorf("APIError.Status = %d, want %d", apiErr.Status, tt.wantAPICode)
				}
				if strings.Contains(err.Error(), "api key not valid") || strings.Contains(err.Error(), "boom") {
					t.Error("API error leaked provider payload")
				}
			case tt.wantNetErr:
				var netErr *failure.NetworkError
				if !errors.As(err, &netErr) {
					t.Fatalf("err = %T, want *failure.NetworkError", err)
				}
			}
		})
	}
}

// TestGeminiAPIErrorBodyBound locks the shared sanitized provider error policy
// used by Invoke and Stream. A hostile endpoint cannot force an unbounded
// allocation or make raw provider text observable through APIError.
func TestGeminiAPIErrorBodyBound(t *testing.T) {
	t.Parallel()

	const wantLimit = 1 << 20
	providerBody := strings.Repeat("x", wantLimit+128)
	tests := []struct {
		name string
		call func(*gemini.Client) error
	}{
		{name: "invoke", call: func(client *gemini.Client) error {
			_, err := client.Invoke(context.Background(), geminiRequest("gemini-2.5-flash"))
			return err
		}},
		{name: "stream", call: func(client *gemini.Client) error {
			_, err := client.Stream(context.Background(), geminiRequest("gemini-2.5-flash"))
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusBadGateway)
				_, _ = io.WriteString(w, providerBody)
			}))
			defer srv.Close()

			err := tt.call(gemini.NewWithEndpoint(testKey, srv.URL))
			var apiErr *failure.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %T, want *failure.APIError", err)
			}
			if strings.Contains(err.Error(), strings.Repeat("x", 64)) {
				t.Error("API error leaked provider body")
			}
		})
	}
}

// TestGeminiStream covers the happy Stream path against an SSE server: the request
// is POSTed to /models/<id>:streamGenerateContent?alt=sse with an SSE Accept header,
// and the streamed events decode, in order, into TextChunks.
func TestGeminiStream(t *testing.T) {
	t.Parallel()

	const modelID = "gemini-2.5-flash"

	capCh := make(chan capturedRequest, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capCh <- capturedRequest{
			method:      r.Method,
			escapedPath: r.URL.EscapedPath(),
			rawQuery:    r.URL.RawQuery,
			apiKey:      r.Header.Get("x-goog-api-key"),
			accept:      r.Header.Get("Accept"),
			body:        readAll(t, r),
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Two partial GenerateContentResponse events, SSE-framed.
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\", world\"}]}}]}\n\n")
	}))
	defer srv.Close()

	c := gemini.NewWithEndpoint(testKey, srv.URL)
	reader, err := c.Stream(context.Background(), geminiRequest(modelID))
	if err != nil {
		t.Fatalf("Stream() err = %v, want nil", err)
	}
	defer reader.Close()

	var texts []string
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("reader.Next() err = %v, want nil or EOF", err)
		}
		tc, ok := chunk.(*content.TextChunk)
		if !ok {
			t.Fatalf("chunk = %T, want *content.TextChunk", chunk)
		}
		texts = append(texts, tc.Text)
	}
	if strings.Join(texts, "") != "Hello, world" {
		t.Errorf("streamed text = %q, want %q", strings.Join(texts, ""), "Hello, world")
	}

	got := <-capCh
	if got.method != http.MethodPost {
		t.Errorf("method = %q, want POST", got.method)
	}
	wantPath := "/models/" + modelID + ":streamGenerateContent"
	if got.escapedPath != wantPath {
		t.Errorf("path = %q, want %q", got.escapedPath, wantPath)
	}
	if got.rawQuery != "alt=sse" {
		t.Errorf("query = %q, want alt=sse", got.rawQuery)
	}
	if got.apiKey != testKey {
		t.Errorf("x-goog-api-key = %q, want %q", got.apiKey, testKey)
	}
	if got.accept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", got.accept)
	}
}

// TestGeminiStreamErrorStatus locks the streaming non-2xx path: a non-2xx status is
// mapped to *failure.APIError (with the drained body) and no reader is returned.
func TestGeminiStreamErrorStatus(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer srv.Close()

	c := gemini.NewWithEndpoint(testKey, srv.URL)
	reader, err := c.Stream(context.Background(), geminiRequest("gemini-2.5-flash"))
	if reader != nil {
		t.Error("Stream() returned a non-nil reader alongside an error")
	}
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %T, want *failure.APIError", err)
	}
	if apiErr.Status != http.StatusTooManyRequests {
		t.Errorf("APIError.Status = %d, want %d", apiErr.Status, http.StatusTooManyRequests)
	}
	if strings.Contains(err.Error(), "rate limited") {
		t.Error("API error leaked provider payload")
	}
}

// TestGeminiPreIOGuards verifies the ordered fail-closed guards run before any
// network I/O and open no connection: a non-Google provider is a
// *failure.ModelMismatchError; an invalid Model (empty name) is a
// *model.ValidationError; and a non-Gemini API format on a Google model fails
// closed at ValidateModel (Google speaks only the Gemini dialect, so the format
// never reaches the client's UnsupportedAPIFormatError guard) — still a
// *model.ValidationError, still no network. Each case is exercised on both
// Invoke and Stream.
func TestGeminiPreIOGuards(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		mutate     func(*inference.Request)
		wantMM     bool
		wantValErr bool
	}{
		{
			name:   "wrong provider is a model mismatch",
			mutate: func(r *inference.Request) { r.Model.Provider = model.ProviderName(llm.ProviderChutes) },
			wantMM: true,
		},
		{
			name:       "empty name fails validation",
			mutate:     func(r *inference.Request) { r.Model.Name = "" },
			wantValErr: true,
		},
		{
			name:       "non-gemini format fails closed",
			mutate:     func(r *inference.Request) { r.Model.APIFormat = model.APIFormatOpenAI },
			wantValErr: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var called atomic.Bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called.Store(true)
				w.WriteHeader(http.StatusOK)
				fmt.Fprint(w, geminiResponseJSON)
			}))
			defer srv.Close()

			c := gemini.NewWithEndpoint(testKey, srv.URL)

			assert := func(t *testing.T, err error) {
				t.Helper()
				if tt.wantMM {
					var mm *failure.ModelMismatchError
					if !errors.As(err, &mm) {
						t.Fatalf("err = %T, want *failure.ModelMismatchError", err)
					}
				}
				if tt.wantValErr {
					var ve *model.ValidationError
					if !errors.As(err, &ve) {
						t.Fatalf("err = %T, want *model.ValidationError", err)
					}
				}
			}

			reqInvoke := geminiRequest("gemini-2.5-flash")
			tt.mutate(&reqInvoke)
			_, errInvoke := c.Invoke(context.Background(), reqInvoke)
			assert(t, errInvoke)

			reqStream := geminiRequest("gemini-2.5-flash")
			tt.mutate(&reqStream)
			readerStream, errStream := c.Stream(context.Background(), reqStream)
			if readerStream != nil {
				t.Error("Stream returned a non-nil reader despite a pre-I/O guard failure")
			}
			assert(t, errStream)

			if called.Load() {
				t.Error("network was called despite a pre-I/O guard failure")
			}
		})
	}
}

// TestGeminiNew covers the fail-closed constructor: an empty key yields
// *llm.AuthRequiredError and no client; a non-empty key yields a non-nil
// inference.Client.
func TestGeminiNew(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		key     auth.APIKey
		wantErr bool
	}{
		{name: "valid key", key: "AIza-key"},
		{name: "empty key fails closed", key: "", wantErr: true},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := gemini.New(tt.key)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New() err = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				if got != nil {
					t.Fatalf("New() returned non-nil client (%T) alongside an error", got)
				}
				var are *llm.AuthRequiredError
				if !errors.As(err, &are) {
					t.Fatalf("err = %T, want *llm.AuthRequiredError", err)
				}
				if are.Provider != llm.ProviderGoogle || are.Kind != auth.AuthAPIKey {
					t.Errorf("AuthRequiredError = {%q, %q}, want {google, api_key}", are.Provider, are.Kind)
				}
				return
			}
			if got == nil {
				t.Fatal("New() = nil, want non-nil client")
			}
		})
	}
}
