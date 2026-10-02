package openaisubscription

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/retry"
)

const completedEvent = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-test\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n"

func sseResponse(events string) *http.Response {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(events))}
}

func ptr[T any](v T) *T { return &v }

// The subscription route rejects max_output_tokens/temperature/top_p and flat
// top-level function tools:
// https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
func TestSubscriptionRequestShapeFollowsRouteRequirements(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test", model.WithTools())
	var body map[string]json.RawMessage
	c, err := New(selected, WithRoundTripper(roundTripFunc(func(r *http.Request) (*http.Response, error) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		// The vendored official schema models the additional_tools item
		// (AdditionalToolsItemParam: role developer, tools array).
		conformance.MustValidateRequest(t, "openai-responses", "create_response_request", raw)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		return sseResponse(completedEvent), nil
	})))
	if err != nil {
		t.Fatal(err)
	}
	request := inference.Request{
		Model:    selected,
		System:   "be brief",
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hi"}}}}},
		Tools:    []inference.Tool{{Name: "read_file", Description: "Read a file.", Schema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)}},
		Override: &model.Sampling{MaxTokens: ptr(128), Temperature: ptr(0.5), TopP: ptr(0.9)},
	}
	if _, err := c.InvokeWithAuth(context.Background(), request, httpauth.None()); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"max_output_tokens", "temperature", "top_p", "tools"} {
		if _, ok := body[field]; ok {
			t.Errorf("request carries unsupported top-level %q", field)
		}
	}
	if string(body["store"]) != "false" || string(body["stream"]) != "true" || string(body["instructions"]) != `"be brief"` {
		t.Fatalf("wrong store/stream/instructions: %s %s %s", body["store"], body["stream"], body["instructions"])
	}
	var input []map[string]json.RawMessage
	if err := json.Unmarshal(body["input"], &input); err != nil {
		t.Fatal(err)
	}
	if len(input) != 2 || string(input[0]["type"]) != `"additional_tools"` || string(input[0]["role"]) != `"developer"` || string(input[1]["role"]) != `"user"` {
		t.Fatalf("tools not supplied as a leading additional_tools item: %s", body["input"])
	}
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(input[0]["tools"], &tools); err != nil || len(tools) != 1 || string(tools[0]["type"]) != `"function"` || string(tools[0]["name"]) != `"read_file"` || len(tools[0]["parameters"]) == 0 {
		t.Fatalf("wrong additional tools: %s", input[0]["tools"])
	}
}

func TestSubscriptionRequestWithoutToolsHasNoToolItem(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	var body map[string]json.RawMessage
	c, _ := New(selected, WithRoundTripper(roundTripFunc(func(r *http.Request) (*http.Response, error) {
		raw, _ := io.ReadAll(r.Body)
		conformance.MustValidateRequest(t, "openai-responses", "create_response_request", raw)
		_ = json.Unmarshal(raw, &body)
		return sseResponse(completedEvent), nil
	})))
	if _, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None()); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body["input"]), "additional_tools") || body["tools"] != nil {
		t.Fatalf("tool-free request gained tools: %s", body["input"])
	}
}

// Only response.completed is a successful inference:
// https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
func TestSubscriptionRefusesIncompleteResponse(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	events := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n" +
		"data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-test\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[]}}\n\n"
	c, _ := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) { return sseResponse(events), nil })))
	_, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
	var incomplete *IncompleteError
	if !errors.As(err, &incomplete) || incomplete.Reason != "max_output_tokens" {
		t.Fatalf("Invoke() = %v, want *IncompleteError with reason", err)
	}
	reader, err := c.StreamWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	for {
		_, err = reader.Next()
		if err != nil {
			break
		}
	}
	if !errors.As(err, &incomplete) {
		t.Fatalf("Stream terminal = %v, want *IncompleteError", err)
	}
	if _, ok := reader.Result(); ok {
		t.Fatal("incomplete stream reported a result")
	}
}

func TestSubscriptionCompletedStreamStillSucceeds(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	events := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n" + completedEvent
	c, _ := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) { return sseResponse(events), nil })))
	response, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
	if err != nil {
		t.Fatal(err)
	}
	if response.Model != "gpt-test" || response.Usage == nil || response.Usage.InputTokens != 3 || response.FinishReason == "" {
		t.Fatalf("lost terminal metadata: %+v", response)
	}
}

func TestSubscriptionStreamErrorsKeepOnlySafeCodes(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	for name, events := range map[string]string{
		"response.failed": "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"Bearer secret-token echoed\"}}}\n\n",
		"error event":     "data: {\"type\":\"error\",\"code\":\"server_error\",\"param\":\"tools[0]\",\"message\":\"Bearer secret-token echoed\"}\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			c, _ := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) { return sseResponse(events), nil })))
			_, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
			var failed *ResponseError
			if !errors.As(err, &failed) || failed.Code != "server_error" {
				t.Fatalf("Invoke() = %v, want *ResponseError", err)
			}
			for _, format := range []string{"%v", "%+v", "%#v"} {
				if strings.Contains(fmt.Sprintf(format, err), "secret-token") {
					t.Fatalf("stream error retained provider message: %s", fmt.Sprintf(format, err))
				}
			}
		})
	}
}

// A plan usage limit is a pause, not rate pressure, whether it arrives before
// the stream opens (HTTP 429) or after it began (response.failed):
// https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery
func TestSubscriptionUsageLimitIsNotRetryable(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	httpLimit := func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 429, Header: http.Header{"X-Request-Id": []string{"req_123"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"type":"invalid_request_error","code":"subscription_sharing_usage_limit_exceeded","message":"limit"}}`))}, nil
	}
	streamLimit := func(*http.Request) (*http.Response, error) {
		return sseResponse("data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"subscription_sharing_usage_limit_exceeded\",\"message\":\"limit\"}}}\n\n"), nil
	}
	for name, rt := range map[string]roundTripFunc{"http 429": httpLimit, "response.failed": streamLimit} {
		t.Run(name, func(t *testing.T) {
			c, _ := New(selected, WithRoundTripper(rt))
			for _, call := range []func() error{
				func() error {
					_, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
					return err
				},
				func() error {
					_, err := c.InvokeWithAuthorizer(context.Background(), inference.Request{Model: selected}, httpauth.None())
					return err
				},
				func() error {
					reader, err := c.StreamWithAuthorizer(context.Background(), inference.Request{Model: selected}, httpauth.None())
					if err != nil {
						return err
					}
					defer reader.Close()
					for {
						if _, err := reader.Next(); err != nil {
							return err
						}
					}
				},
			} {
				err := call()
				var limit *UsageLimitError
				if !errors.As(err, &limit) || !errors.Is(err, ErrUsageLimit) {
					t.Fatalf("error = %v (%T), want *UsageLimitError", err, err)
				}
				var apiErr *failure.APIError
				if errors.As(err, &apiErr) || retry.Retryable(err) {
					t.Fatalf("usage limit is retryable: %v", err)
				}
			}
		})
	}
}

func TestSubscriptionTransientAdmissionStaysRetryable(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	c, _ := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 503, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":{"code":"subscription_sharing_usage_unavailable"}}`))}, nil
	})))
	_, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 503 || !retry.Retryable(err) {
		t.Fatalf("error = %v, want retryable 503 APIError", err)
	}
}
