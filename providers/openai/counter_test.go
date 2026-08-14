package openai

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
	"github.com/looprig/inference/codec/conformance"
	contextcount "github.com/looprig/inference/contextcount"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/llm"
)

func TestCounterCountContext(t *testing.T) {
	model := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAIResponses,
		"",
		"gpt-5",
		model.WithTools(),
		model.WithThinking(),
		model.WithSampling(model.Sampling{Effort: model.EffortHigh}),
	)
	req := inference.Request{
		Model:  model,
		System: "system",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role:   content.RoleUser,
				Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
			}},
		},
		Tools: []inference.Tool{{
			Name:   "lookup",
			Schema: json.RawMessage(`{"type":"object"}`),
		}},
	}

	t.Run("successful count and native request shape", func(t *testing.T) {
		bodyCh := make(chan []byte, 1)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			bodyCh <- body
			if got, want := r.URL.Path, "/v1/responses/input_tokens"; got != want {
				t.Errorf("path = %q, want %q", got, want)
			}
			if got, want := r.Header.Get("Authorization"), "Bearer sk-counter"; got != want {
				t.Errorf("Authorization = %q, want %q", got, want)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"object":"response.input_tokens","input_tokens":321}`)
		}))
		defer srv.Close()

		counter := newCounter(auth.APIKey("sk-counter"), srv.URL+"/v1")
		model.BaseURL = srv.URL + "/v1"
		got, err := counter.CountContext(context.Background(), req)
		if err != nil {
			t.Fatalf("CountContext() error = %v", err)
		}
		want := contextcount.ContextCount{Model: model.Key(), InputTokens: content.TokenCount(321), Quality: contextcount.CountQualityExactProvider}
		if got != want {
			t.Errorf("CountContext() = %+v, want %+v", got, want)
		}

		// The count preflight is a create_response_request with the generation
		// controls stripped, so it is held against that same request schema:
		// dropping fields must not produce a body OpenAI would reject.
		raw := <-bodyCh
		conformance.MustValidateRequest(t, "openai-responses", "create_response_request", raw)
		var body map[string]json.RawMessage
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request body JSON error = %v", err)
		}
		var modelName string
		decodeCounterField(t, body, "model", &modelName)
		if modelName != "gpt-5" {
			t.Errorf("model = %q, want gpt-5", modelName)
		}
		for _, field := range []string{"instructions", "input", "tools"} {
			if _, ok := body[field]; !ok {
				t.Errorf("count request missing input field %q", field)
			}
		}
		for _, field := range []string{"stream", "store", "max_output_tokens", "temperature", "top_p", "text", "reasoning", "service_tier", "metadata", "prompt_cache_key"} {
			if _, ok := body[field]; ok {
				t.Errorf("count request contains output-only field %q", field)
			}
		}
	})
}

func TestCounterResponseValidation(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		body   string
		reason CounterResponseReason
	}{
		{name: "malformed", body: `{"input_tokens":`, reason: CounterResponseMalformed},
		{name: "missing", body: `{"object":"response.input_tokens"}`, reason: CounterResponseMissingCount},
		{name: "null", body: `{"input_tokens":null}`, reason: CounterResponseInvalidCount},
		{name: "fractional", body: `{"input_tokens":1.5}`, reason: CounterResponseInvalidCount},
		{name: "negative", body: `{"input_tokens":-1}`, reason: CounterResponseInvalidCount},
		{name: "trailing", body: `{"input_tokens":1}{}`, reason: CounterResponseMalformed},
		{name: "duplicate", body: `{"input_tokens":1,"input_tokens":2}`, reason: CounterResponseDuplicateField},
		{name: "provider scalar is not echoed", body: `{"input_tokens":"private-input"}`, reason: CounterResponseInvalidCount},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decodeCountResponse([]byte(tt.body))
			if got != 0 {
				t.Errorf("decodeCountResponse() = %d on error, want zero", got)
			}
			var responseErr *CounterResponseError
			if !errors.As(err, &responseErr) {
				t.Fatalf("error = %T %v, want *CounterResponseError", err, err)
			}
			if responseErr.Reason != tt.reason {
				t.Errorf("reason = %q, want %q", responseErr.Reason, tt.reason)
			}
			if strings.Contains(err.Error(), "private-input") {
				t.Error("error leaked provider-controlled response value")
			}
		})
	}
}

func TestCounterHTTPErrorAndValidation(t *testing.T) {
	t.Parallel()
	valid := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "", "gpt-5")
	if counter, err := NewCounter(""); counter != nil || err == nil {
		t.Fatalf("NewCounter(empty key) = %T, %v, want typed auth error", counter, err)
	} else {
		var authErr *llm.AuthRequiredError
		if !errors.As(err, &authErr) || authErr.Provider != llm.ProviderOpenAI {
			t.Fatalf("NewCounter(empty key) error = %T %v, want OpenAI auth error", err, err)
		}
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "provider failure", http.StatusBadRequest)
	}))
	defer srv.Close()
	valid.BaseURL = srv.URL + "/v1"
	_, err := newCounter("sk-counter", srv.URL+"/v1").CountContext(context.Background(), inference.Request{Model: valid})
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadRequest {
		t.Fatalf("HTTP error = %T %v, want *failure.APIError(400)", err, err)
	}
}

func decodeCounterField(t *testing.T, body map[string]json.RawMessage, name string, out any) {
	t.Helper()
	raw, ok := body[name]
	if !ok {
		t.Fatalf("request body missing %q", name)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("decode %q: %v", name, err)
	}
}
