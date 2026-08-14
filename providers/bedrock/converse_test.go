package bedrock_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/bedrock"
)

const converseModelID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

const converseResponseJSON = `{
	"output":{"message":{"role":"assistant","content":[{"text":"Hello from Converse"}]}},
	"stopReason":"end_turn",
	"usage":{"inputTokens":5,"outputTokens":3}
}`

func converseRequest(name string) inference.Request {
	return inference.Request{
		Model: model.CustomModel(
			model.ProviderName(llm.ProviderBedrock),
			model.APIFormatBedrockConverse,
			"",
			name,
			model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}),
			model.WithImages(),
			model.WithTools(),
			model.WithThinking(),
		),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
	}
}

func eventStreamFrame(event, payload string) []byte {
	headers := bytes.NewBuffer(nil)
	writeHeader := func(name, value string) {
		headers.WriteByte(byte(len(name)))
		headers.WriteString(name)
		headers.WriteByte(7)
		var length [2]byte
		binary.BigEndian.PutUint16(length[:], uint16(len(value)))
		headers.Write(length[:])
		headers.WriteString(value)
	}
	writeHeader(":message-type", "event")
	writeHeader(":event-type", event)
	data := []byte(payload)
	total := 16 + headers.Len() + len(data)
	frame := make([]byte, total)
	binary.BigEndian.PutUint32(frame[:4], uint32(total))
	binary.BigEndian.PutUint32(frame[4:8], uint32(headers.Len()))
	binary.BigEndian.PutUint32(frame[8:12], crc32.ChecksumIEEE(frame[:8]))
	copy(frame[12:], headers.Bytes())
	copy(frame[12+headers.Len():], data)
	binary.BigEndian.PutUint32(frame[total-4:], crc32.ChecksumIEEE(frame[:total-4]))
	return frame
}

func eventStreamBody() []byte {
	frames := [][]byte{
		eventStreamFrame("messageStart", `{"role":"assistant"}`),
		eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"streamed"}}`),
		eventStreamFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		eventStreamFrame("messageStop", `{"stopReason":"end_turn"}`),
		eventStreamFrame("metadata", `{"usage":{"inputTokens":4,"outputTokens":2}}`),
	}
	var body []byte
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

func TestBedrockConverseInvoke(t *testing.T) {
	t.Parallel()

	requestCh := make(chan *http.Request, 1)
	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCh <- r.Clone(context.Background())
		bodyCh <- readAll(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseResponseJSON)
	}))
	defer srv.Close()

	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
	req := converseRequest(converseModelID)
	response, err := c.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Model != converseModelID {
		t.Errorf("response.Model = %q, want request model %q", response.Model, converseModelID)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 || response.Message.Blocks[0].(*content.TextBlock).Text != "Hello from Converse" {
		t.Fatalf("response.Message = %#v, want one Converse text block", response.Message)
	}

	got := <-requestCh
	if got.Method != http.MethodPost || got.URL.EscapedPath() != "/model/"+converseModelID+"/converse" {
		t.Errorf("request = %s %s, want POST /model/{id}/converse", got.Method, got.URL.EscapedPath())
	}
	if got.Header.Get("Content-Type") != "application/json" || got.Header.Get("Accept") != "application/json" {
		t.Errorf("headers = Content-Type %q Accept %q, want JSON", got.Header.Get("Content-Type"), got.Header.Get("Accept"))
	}
	if authz := got.Header.Get("Authorization"); !strings.HasPrefix(authz, "AWS4-HMAC-SHA256 ") || !strings.Contains(authz, "/us-east-1/bedrock/aws4_request") {
		t.Errorf("Authorization = %q, want Bedrock SigV4 scope", authz)
	}
	raw := <-bodyCh
	gateConverseRequest(t, raw)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("request body JSON = %v", err)
	}
	if _, ok := body["model"]; ok {
		t.Fatal("Converse body contains model; model ID belongs in URL")
	}
	if _, ok := body["messages"]; !ok {
		t.Fatal("Converse body missing messages")
	}
}

func TestBedrockConverseStream(t *testing.T) {
	t.Parallel()

	requestCh := make(chan *http.Request, 1)
	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCh <- r.Clone(context.Background())
		bodyCh <- readAll(t, r)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(eventStreamBody())
	}))
	defer srv.Close()

	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
	reader, err := c.Stream(context.Background(), converseRequest(converseModelID))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer reader.Close()
	chunk, err := reader.Next()
	if err != nil {
		t.Fatalf("first chunk error = %v", err)
	}
	if text, ok := chunk.(*content.TextChunk); !ok || text.Text != "streamed" {
		t.Fatalf("first chunk = %#v, want streamed text", chunk)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("terminal Next() error = %v, want io.EOF", err)
	}
	result, ok := reader.Result()
	if !ok || result.FinishReason != "stop" || result.Usage == nil || result.Usage.InputTokens != 4 || result.Model != converseModelID {
		t.Fatalf("stream result = %#v, ok=%v, want model/stop/usage", result, ok)
	}

	// ConverseStream is a distinct encode path (the streaming guardrail mode is
	// only legal there), so its body is gated separately from Converse's.
	gateConverseRequest(t, <-bodyCh)

	got := <-requestCh
	if got.URL.EscapedPath() != "/model/"+converseModelID+"/converse-stream" {
		t.Errorf("path = %q, want ConverseStream route", got.URL.EscapedPath())
	}
	if got.Header.Get("Accept") != "application/vnd.amazon.eventstream" {
		t.Errorf("Accept = %q, want application/vnd.amazon.eventstream", got.Header.Get("Accept"))
	}
	if authz := got.Header.Get("Authorization"); !strings.Contains(authz, "/us-east-1/bedrock/aws4_request") {
		t.Errorf("Authorization = %q, want Bedrock SigV4 scope", authz)
	}
}

func TestBedrockConverseOptions(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCh <- readAll(t, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseResponseJSON)
	}))
	defer srv.Close()

	budget := 1024
	request := converseRequest(converseModelID)
	request.System = "cache this system prefix"
	request.Model.Caps.StructuredOutput = true
	request.Output = &inference.OutputSchema{
		Name:        "answer",
		Description: "final answer",
		Schema:      json.RawMessage(`{"type":"object","additionalProperties":false}`),
	}
	c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL,
		bedrock.WithReasoning(bedrock.ReasoningOptions{Type: "enabled", BudgetTokens: &budget}),
		bedrock.WithAdditionalModelRequestFields(json.RawMessage(`{"temperature_top_k":50}`)),
		bedrock.WithAdditionalModelResponseFieldPaths("/stopReason", "/trace"),
		bedrock.WithGuardrail(bedrock.GuardrailOptions{Identifier: "guard", Version: "1", Trace: "enabled"}),
		bedrock.WithPerformanceLatency(bedrock.PerformanceLatencyOptimized),
		bedrock.WithServiceTier(bedrock.ServiceTierPriority),
		bedrock.WithRequestMetadata(map[string]string{"tenant": "test"}),
		bedrock.WithPromptCachePoint(bedrock.CachePointOptions{Type: "default"}),
	)
	if _, err := c.Invoke(context.Background(), request); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	raw := <-bodyCh
	// Every option this test switches on writes into the Converse body, so the
	// gate is the only thing that checks the shapes those options produce:
	// guardrailConfig's identifier/version patterns, serviceTier's enum,
	// performanceConfig's enum, requestMetadata's per-value pattern, and the
	// cachePoint block the prompt-cache option appends.
	gateConverseRequest(t, raw)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("body JSON = %v", err)
	}
	for _, field := range []string{"additionalModelRequestFields", "additionalModelResponseFieldPaths", "guardrailConfig", "performanceConfig", "serviceTier", "requestMetadata", "outputConfig", "system"} {
		if _, ok := body[field]; !ok {
			t.Errorf("body missing option field %q", field)
		}
	}
	var requestFields map[string]json.RawMessage
	if err := json.Unmarshal(body["additionalModelRequestFields"], &requestFields); err != nil {
		t.Fatalf("additionalModelRequestFields = %v", err)
	}
	var thinking map[string]json.RawMessage
	if err := json.Unmarshal(requestFields["thinking"], &thinking); err != nil {
		t.Fatalf("thinking option = %v", err)
	}
	if got := string(thinking["type"]); got != `"enabled"` || string(thinking["budget_tokens"]) != "1024" {
		t.Errorf("thinking = %s, want enabled/1024", thinking)
	}
	if string(requestFields["temperature_top_k"]) != "50" {
		t.Errorf("additional fields = %#v, custom field lost", requestFields)
	}
}

func TestBedrockConverseErrorsAndGuards(t *testing.T) {
	t.Parallel()

	t.Run("HTTP error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"message":"bad converse request"}`)
		}))
		defer srv.Close()
		c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
		_, err := c.Invoke(context.Background(), converseRequest(converseModelID))
		var apiErr *failure.APIError
		if err == nil || !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
			t.Fatalf("Invoke() error = %v, want HTTP API error", err)
		}
	})

	t.Run("malformed successful response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, `{`)
		}))
		defer srv.Close()
		c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
		_, err := c.Invoke(context.Background(), converseRequest(converseModelID))
		var decodeErr *bedrockconverse.DecodeError
		if !errors.As(err, &decodeErr) {
			t.Fatalf("Invoke() error = %T (%v), want DecodeError", err, err)
		}
	})

	t.Run("pre-IO wrong provider", func(t *testing.T) {
		var called atomic.Bool
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { called.Store(true) }))
		defer srv.Close()
		c := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
		req := converseRequest(converseModelID)
		req.Model.Provider = model.ProviderName(llm.ProviderGoogle)
		_, err := c.Stream(context.Background(), req)
		var mismatch interface{ Error() string }
		if !errors.As(err, &mismatch) || called.Load() {
			t.Fatalf("Stream() error = %v, called=%v; want pre-I/O binding error", err, called.Load())
		}
	})
}
