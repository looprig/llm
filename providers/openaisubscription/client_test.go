package openaisubscription

import (
	"context"
	"encoding/json"
	"github.com/looprig/core/content"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
	"github.com/looprig/secrets"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestInvokeUsesSubscriptionStream(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test", model.WithTools())
	rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://api.openai.com/v1/responses" || r.Header.Get("Authorization") != "Bearer fixture-access" {
			t.Fatal("wrong subscription request")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["stream"] != true || body["store"] != false {
			t.Fatalf("wrong body: %v", body)
		}
		events := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n" +
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-test\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n"
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(events))}, nil
	})
	c, err := New(selected, WithRoundTripper(rt))
	if err != nil {
		t.Fatal(err)
	}
	secret, _ := secrets.New([]byte("fixture-access"))
	authorizer, _ := httpauth.Bearer(secret)
	response, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, authorizer)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Message.Blocks) != 1 || response.Usage == nil || response.Model != "gpt-test" {
		t.Fatalf("wrong response: %+v", response)
	}
	if response.Message.Role != content.RoleAssistant || response.Message.Usage == nil || response.Message.Blocks[0].(*content.TextBlock).Text != "hello" {
		t.Fatal("Invoke lost assistant message metadata")
	}
}

func TestSubscriptionRejectsTruncatedAndFailedStreams(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test")
	for _, events := range []string{"data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n", "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"failed\"}}}\n\n"} {
		c, err := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(events))}, nil
		})))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None()); err == nil {
			t.Fatal("accepted unsuccessful stream")
		}
	}
}
