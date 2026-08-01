package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	contextcount "github.com/looprig/inference/contextcount"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

func TestCounterCountContext(t *testing.T) {
	selected := model.CustomModel(
		model.ProviderName(llm.ProviderAnthropic),
		model.APIFormatAnthropic,
		"",
		"claude-sonnet-4",
		model.WithTools(),
		model.WithThinking(),
		model.WithSampling(model.Sampling{Effort: model.EffortHigh}),
	)
	req := inference.Request{
		Model:  selected,
		System: "system",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
		Tools: []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)}},
	}

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodyCh <- body
		if got, want := r.URL.Path, "/v1/messages/count_tokens"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("x-api-key"), "sk-ant-counter"; got != want {
			t.Errorf("x-api-key = %q, want %q", got, want)
		}
		if got, want := r.Header.Get("anthropic-version"), anthropicVersion; got != want {
			t.Errorf("anthropic-version = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"input_tokens":123}`)
	}))
	defer srv.Close()

	selected.BaseURL = srv.URL + "/v1"
	counter := newCounter(auth.APIKey("sk-ant-counter"), srv.URL+"/v1")
	got, err := counter.CountContext(context.Background(), req)
	if err != nil {
		t.Fatalf("CountContext() error = %v", err)
	}
	want := contextcount.ContextCount{Model: selected.Key(), InputTokens: content.TokenCount(123), Quality: contextcount.CountQualityExactProvider}
	if got != want {
		t.Errorf("CountContext() = %+v, want %+v", got, want)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	for _, field := range []string{"model", "system", "messages", "tools"} {
		if _, ok := body[field]; !ok {
			t.Errorf("count request missing input field %q", field)
		}
	}
	for _, field := range []string{"max_tokens", "stream", "temperature", "top_p", "stop_sequences", "thinking", "output_config"} {
		if _, ok := body[field]; ok {
			t.Errorf("count request contains generation field %q", field)
		}
	}
}

func TestCounterResponseValidation(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		body   string
		reason CounterResponseReason
	}{
		{name: "malformed", body: `{"input_tokens":`, reason: CounterResponseMalformed},
		{name: "missing", body: `{}`, reason: CounterResponseMissingCount},
		{name: "fractional", body: `{"input_tokens":1.5}`, reason: CounterResponseInvalidCount},
		{name: "duplicate", body: `{"input_tokens":1,"input_tokens":2}`, reason: CounterResponseDuplicateField},
		{name: "trailing", body: `{"input_tokens":1}{}`, reason: CounterResponseMalformed},
		{name: "provider scalar is not echoed", body: `{"input_tokens":"private-input"}`, reason: CounterResponseInvalidCount},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decodeCountResponse([]byte(tt.body))
			if got != 0 {
				t.Errorf("decodeCountResponse() = %d on error, want zero", got)
			}
			var responseErr *CounterResponseError
			if !errors.As(err, &responseErr) || responseErr.Reason != tt.reason {
				t.Fatalf("error = %T %v, want CounterResponseError/%q", err, err, tt.reason)
			}
			if strings.Contains(err.Error(), "private-input") {
				t.Error("error leaked provider-controlled response value")
			}
		})
	}
}

func TestCounterAuthAndHTTPError(t *testing.T) {
	t.Parallel()
	valid := model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, "", "claude-sonnet-4")
	if counter, err := NewCounter(""); counter != nil || err == nil {
		t.Fatalf("NewCounter(empty key) = %T, %v, want auth error", counter, err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "provider failure", http.StatusBadRequest)
	}))
	defer srv.Close()
	valid.BaseURL = srv.URL + "/v1"
	_, err := newCounter("sk-ant-counter", srv.URL+"/v1").CountContext(context.Background(), inference.Request{Model: valid})
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
		t.Fatalf("HTTP error = %T %v, want *failure.APIError(400)", err, err)
	}
}
