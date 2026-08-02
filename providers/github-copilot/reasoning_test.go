package githubcopilot_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	githubcopilot "github.com/looprig/llm/providers/github-copilot"
)

func TestReasoningTextAliasIsNormalizedForJSONAndSSE(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-GitHub-Api-Version"); got != "2026-06-01" {
			t.Errorf("X-GitHub-Api-Version = %q, want 2026-06-01", got)
		}
		if got := r.Header.Get("x-initiator"); got != "agent" {
			t.Errorf("x-initiator = %q, want agent for an empty request", got)
		}
		var body []byte
		if r.URL.Path == "/chat/completions" {
			requestBody, _ := io.ReadAll(r.Body)
			if strings.Contains(string(requestBody), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprint(w, "data: {\"model\":\"model\",\"choices\":[{\"delta\":{\"reasoning_text\":\"think\"}}]}\n\n")
				_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
				return
			}
			body = []byte(`{"id":"id","model":"model","choices":[{"message":{"role":"assistant","reasoning_text":"think","content":"answer"},"finish_reason":"stop"}]}`)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitHubCopilot), model.APIFormatOpenAI, srv.URL, "model")
	client, err := githubcopilot.New(selected, "token")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Invoke(context.Background(), inference.Request{Model: selected})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(response.Message.Blocks) != 2 {
		t.Fatalf("blocks = %#v, want thinking and text", response.Message.Blocks)
	}
	if thinking, ok := response.Message.Blocks[0].(*content.ThinkingBlock); !ok || thinking.Thinking != "think" {
		t.Fatalf("first block = %#v, want thinking alias", response.Message.Blocks[0])
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
	if thinking, ok := chunk.(*content.ThinkingChunk); !ok || thinking.Thinking != "think" {
		t.Fatalf("stream chunk = %#v, want thinking alias", chunk)
	}
	second, err := reader.Next()
	if err != nil {
		t.Fatalf("second stream chunk error = %v", err)
	}
	if text, ok := second.(*content.TextChunk); !ok || text.Text != "answer" {
		t.Fatalf("second stream chunk = %#v, want answer text", second)
	}
	if _, err := reader.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("stream terminal error = %v, want EOF", err)
	}
}

func TestInitiatorTracksTheLatestUserPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-GitHub-Api-Version"); got != "2026-06-01" {
			t.Errorf("X-GitHub-Api-Version = %q, want 2026-06-01", got)
		}
		if got := r.Header.Get("x-initiator"); got != "user" {
			t.Errorf("x-initiator = %q, want user", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderGitHubCopilot), model.APIFormatOpenAI, srv.URL, "model")
	client, err := githubcopilot.New(selected, "token")
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role:   content.RoleUser,
			Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}
