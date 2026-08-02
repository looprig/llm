package cloudflaregateway_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	cloudflaregateway "github.com/looprig/llm/providers/cloudflare-ai-gateway"
)

func TestOptionsEncodeDocumentedGatewayHeadersAndReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer cf-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}
		checks := map[string]string{
			"cf-aig-gateway-id":  "gateway",
			"cf-aig-skip-cache":  "true",
			"cf-aig-cache-ttl":   "30",
			"cf-aig-cache-key":   "stable-key",
			"cf-aig-collect-log": "false",
			"cf-aig-metadata":    `{"trace":"test"}`,
		}
		for name, want := range checks {
			if got := r.Header.Get(name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("request JSON = %v", err)
		} else {
			var effort string
			if err := json.Unmarshal(body["reasoning_effort"], &effort); err != nil || effort != "high" {
				t.Errorf("reasoning_effort = %q, err=%v, want high", effort, err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderCloudflareAIGateway), model.APIFormatOpenAI, server.URL, "model")
	client, err := cloudflaregateway.New(
		selected,
		auth.APIKey("cf-key"),
		cloudflaregateway.WithGatewayID("gateway"),
		cloudflaregateway.WithMetadata(map[string]string{"trace": "test"}),
		cloudflaregateway.WithSkipCache(true),
		cloudflaregateway.WithCacheTTL(30),
		cloudflaregateway.WithCacheKey("stable-key"),
		cloudflaregateway.WithCollectLog(false),
		cloudflaregateway.WithReasoningEffort("high"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}
