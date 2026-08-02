package gitlab_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/gitlab"
)

func TestOpenAIChatExchangesAndCachesDirectAccessToken(t *testing.T) {
	var exchangeCalls atomic.Int32
	var inferenceCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v4/ai/third_party_agents/direct_access":
			exchangeCalls.Add(1)
			if got := r.Header.Get("Authorization"); got != "Bearer gitlab-pat" {
				t.Errorf("exchange Authorization = %q, want PAT", got)
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("exchange body: %v", err)
			}
			flags, ok := body["feature_flags"].(map[string]any)
			if !ok || flags["duo_agent_platform"] != true {
				t.Errorf("feature_flags = %#v, want duo_agent_platform=true", body["feature_flags"])
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"token":"direct-token","headers":{"X-GitLab-Route":"agentic","x-api-key":"must-not-forward"}}`)
		case "/chat/completions":
			inferenceCalls.Add(1)
			if got := r.Header.Get("Authorization"); got != "Bearer direct-token" {
				t.Errorf("inference Authorization = %q, want exchanged token", got)
			}
			if got := r.Header.Get("X-GitLab-Route"); got != "agentic" {
				t.Errorf("X-GitLab-Route = %q, want agentic", got)
			}
			if got := r.Header.Get("X-Custom-Gateway"); got != "yes" {
				t.Errorf("X-Custom-Gateway = %q, want yes", got)
			}
			var body map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("inference body: %v", err)
			}
			if string(body["stream"]) == "true" {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"stream\"}}]}\n\n")
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"json"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, srv.URL, "duo-chat-gpt-5-1")
	client, err := gitlab.New(selected, auth.APIKey("gitlab-pat"), gitlab.WithInstanceURL(srv.URL), gitlab.WithFeatureFlag("duo_agent_platform", true), gitlab.WithAIGatewayHeader("X-Custom-Gateway", "yes"))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 6 {
		t.Fatalf("usage = %#v, want input 4/output 6", response.Usage)
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
	if exchangeCalls.Load() != 1 || inferenceCalls.Load() != 2 {
		t.Fatalf("exchange calls = %d, inference calls = %d, want 1 and 2", exchangeCalls.Load(), inferenceCalls.Load())
	}
}

func TestOpenAIChatSendsGitLabUpstreamModelID(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v4/ai/third_party_agents/direct_access" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"token":"direct-token","headers":{}}`)
			return
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("inference body: %v", err)
		}
		var modelID string
		if err := json.Unmarshal(body["model"], &modelID); err != nil {
			t.Fatalf("model field: %v", err)
		}
		if modelID != "gpt-5.1-2025-11-13" {
			t.Errorf("outbound model = %q, want mapped upstream ID", modelID)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"gpt-5.1-2025-11-13","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, server.URL, "duo-chat-gpt-5-1")
	client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(server.URL))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestOpenAIChatAllowsExplicitUpstreamModelIDOverride(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v4/ai/third_party_agents/direct_access" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"token":"direct-token","headers":{}}`)
			return
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("inference body: %v", err)
		}
		var modelID string
		_ = json.Unmarshal(body["model"], &modelID)
		if modelID != "provider-model" {
			t.Errorf("outbound model = %q, want provider-model", modelID)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"provider-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, server.URL, "custom-alias")
	client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(server.URL), gitlab.WithUpstreamModelID("provider-model", model.APIFormatOpenAI))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

func TestOpenAIChatRefreshesDirectAccessTokenOnceAfterInference401(t *testing.T) {
	var exchangeCalls atomic.Int32
	var inferenceCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v4/ai/third_party_agents/direct_access":
			call := exchangeCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"token":"direct-token-`+strings.TrimSpace(string(rune('0'+call)))+`","headers":{}}`)
		case "/chat/completions":
			call := inferenceCalls.Add(1)
			if call == 1 {
				http.Error(w, `{"error":"expired"}`, http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"id","model":"gpt-5.1-2025-11-13","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, server.URL, "duo-chat-gpt-5-1")
	client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(server.URL))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v, want one retry", err)
	}
	if got, want := exchangeCalls.Load(), int32(2); got != want {
		t.Fatalf("exchange calls = %d, want %d", got, want)
	}
	if got, want := inferenceCalls.Load(), int32(2); got != want {
		t.Fatalf("inference calls = %d, want %d", got, want)
	}
}

func TestOpenAIChatRefreshesDirectAccessTokenOnceAfterStream401(t *testing.T) {
	var exchangeCalls atomic.Int32
	var inferenceCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v4/ai/third_party_agents/direct_access":
			exchangeCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"token":"direct-token","headers":{}}`)
		case "/chat/completions":
			call := inferenceCalls.Add(1)
			if call == 1 {
				http.Error(w, `{"error":"expired"}`, http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, server.URL, "duo-chat-gpt-5-1")
	client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(server.URL))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	reader, err := client.Stream(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Stream() error = %v, want one retry", err)
	}
	defer func() { _ = reader.Close() }()
	if _, err := reader.Next(); err != nil {
		t.Fatalf("Stream.Next() error = %v", err)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want EOF", err)
	}
	if got, want := exchangeCalls.Load(), int32(2); got != want {
		t.Fatalf("exchange calls = %d, want %d", got, want)
	}
	if got, want := inferenceCalls.Load(), int32(2); got != want {
		t.Fatalf("inference calls = %d, want %d", got, want)
	}
}

func TestAnthropicAndResponsesRoutes(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		var exchangeCalls atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/v4/ai/third_party_agents/direct_access" {
				exchangeCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"token":"direct-token","headers":{"X-GitLab-Route":"anthropic"}}`)
				return
			}
			if r.URL.Path != "/messages" {
				t.Errorf("path = %q, want /messages", r.URL.Path)
			}
			if got := r.Header.Get("Authorization"); got != "Bearer direct-token" {
				t.Errorf("Authorization = %q, want exchanged token", got)
			}
			if got := r.Header.Get("anthropic-beta"); got != "context-1m-2025-08-07" {
				t.Errorf("anthropic-beta = %q, want context-1m-2025-08-07", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}`)
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatAnthropic, srv.URL, "duo-chat-sonnet-4-6")
		client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(srv.URL))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
		if err != nil || response == nil || response.Message == nil {
			t.Fatalf("Invoke() response=%#v err=%v", response, err)
		}
		if exchangeCalls.Load() != 1 {
			t.Fatalf("exchange calls = %d, want 1", exchangeCalls.Load())
		}
	})

	t.Run("responses", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/v4/ai/third_party_agents/direct_access" {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"token":"direct-token","headers":{}}`)
				return
			}
			if r.URL.Path != "/responses" {
				t.Errorf("path = %q, want /responses", r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"resp","object":"response","model":"model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`)
		}))
		defer srv.Close()
		selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAIResponses, srv.URL, "duo-chat-gpt-5-codex")
		client, err := gitlab.New(selected, "gitlab-pat", gitlab.WithInstanceURL(srv.URL))
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
	})
}

func TestDirectAccessExchangeErrorDoesNotExposeCredential(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v4/ai/third_party_agents/direct_access" {
			http.Error(w, "denied", http.StatusUnauthorized)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitLab), model.APIFormatOpenAI, srv.URL, "duo-chat-gpt-5-1")
	client, err := gitlab.New(selected, "secret-pat", gitlab.WithInstanceURL(srv.URL))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{Model: selected})
	var accessErr *gitlab.DirectAccessError
	if !errors.As(err, &accessErr) || accessErr.Status != http.StatusUnauthorized {
		t.Fatalf("Invoke() error = %T %v, want DirectAccessError(401)", err, err)
	}
	if strings.Contains(err.Error(), "secret-pat") {
		t.Fatalf("error exposed credential: %v", err)
	}
}
