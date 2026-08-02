package sap_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/sap-ai-core"
)

func TestServiceKeyJSONAndChatContract(t *testing.T) {
	var tokenCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/oauth/token":
			tokenCalls.Add(1)
			if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
				t.Errorf("token request = %s %s, want POST form", r.Method, r.Header.Get("Content-Type"))
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"access_token":"sap-access-token","expires_in":3600}`)
		case "/v2/chat":
			var body map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("request JSON: %v", err)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer sap-access-token" {
				t.Errorf("Authorization = %q, want bearer token", got)
			}
			if got := r.Header.Get("AI-Resource-Group"); got != "team" {
				t.Errorf("AI-Resource-Group = %q, want team", got)
			}
			if string(body["stream"]) == "true" {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"stream\"}}]}\n\n")
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"id","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"json"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	rawKey := []byte(`{"clientid":"client","clientsecret":"secret","url":"` + srv.URL + `","serviceurls":{"AI_API_URL":"` + srv.URL + `"}}`)
	serviceKey, err := sap.ParseServiceKey(rawKey)
	if err != nil {
		t.Fatalf("ParseServiceKey() error = %v", err)
	}
	selected := model.CustomModel(model.ProviderName(llm.ProviderSAP), model.APIFormatOpenAI, srv.URL, "gpt-4o")
	client, err := sap.New(selected, serviceKey, sap.WithResourceGroup("team"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 1 {
		t.Fatalf("response = %#v, want one block", response)
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
	if tokenCalls.Load() != 1 {
		t.Errorf("token requests = %d, want cached single token request", tokenCalls.Load())
	}
}

func TestDeploymentDiscovery(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/oauth/token":
			_, _ = fmt.Fprint(w, `{"access_token":"token","expires_in":3600}`)
		case "/v2/lm/deployments":
			if r.Header.Get("AI-Resource-Group") != "default" {
				t.Errorf("discovery resource group = %q, want default", r.Header.Get("AI-Resource-Group"))
			}
			_, _ = fmt.Fprint(w, `{"resources":[{"id":"deployment","deploymentUrl":"`+srv.URL+`/deployment","configurationName":"defaultOrchestrationConfig","status":"RUNNING"}]}`)
		case "/deployment/v2/chat":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"id","model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	serviceKey := sap.ServiceKey{ClientID: "client", ClientSecret: "secret", TokenURL: srv.URL}
	serviceKey.ServiceURLs.AIAPIURL = srv.URL
	selected := model.CustomModel(model.ProviderName(llm.ProviderSAP), model.APIFormatOpenAI, "", "gpt-4o")
	client, err := sap.New(selected, serviceKey)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() after discovery error = %v", err)
	}
}
