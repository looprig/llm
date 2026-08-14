package simple_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/simple"
)

// Confirmed live against api.anthropic.com:
//
//	400 "`max_tokens` must be greater than `thinking.budget_tokens`"
//
// anthropicapi cannot produce it — it emits thinking:{"type":"adaptive"} and
// never writes budget_tokens — so the only way this adapter's callers reach a
// violating body is WithThinkingBudget, which injects a RAW
// thinking.budget_tokens body field for OpenAI-compatible and Anthropic-dialect
// passthrough. Six provider packages re-export it (llmgateway, minimax,
// fireworks, zenmux, cloudflare-ai-gateway, azure-cognitive-services, gitlab).
//
// The check has to live in the body patch rather than in the option itself: the
// budget is known when the option is built, the output cap only when a request
// is encoded. So the failure surfaces from Invoke/Stream, before the request is
// sent.

const (
	testProvider = llm.ProviderFireworks
	testModel    = "test-model"
)

func chatDefinition() simple.Definition {
	return simple.Definition{
		Provider:       testProvider,
		Authentication: auth.AuthNone,
		Authenticator:  func(auth.APIKey) (auth.Authenticator, error) { return auth.None(), nil },
	}
}

// echoServer records the request body it receives and answers with a minimal
// valid Chat Completions response.
func echoServer(t *testing.T, seen chan<- []byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		select {
		case seen <- body:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"id","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},`+
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func invokeWith(t *testing.T, baseURL string, sampling model.Sampling, caps model.Capabilities, options ...simple.Option) error {
	t.Helper()
	selected := model.CustomModel(model.ProviderName(testProvider), model.APIFormatOpenAI, baseURL, testModel)
	selected.Sampling = sampling
	selected.Caps = caps
	client, err := simple.New(selected, "", chatDefinition(), options...)
	if err != nil {
		t.Fatalf("simple.New() error = %v", err)
	}
	_, err = client.Invoke(context.Background(), inference.Request{
		Model: selected,
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hi"}},
		}}},
	})
	return err
}

func tokens(n int) *int { return &n }

func TestWithThinkingBudgetRejectsABudgetAtOrAboveTheOutputCap(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		caps       model.Capabilities
		maxTokens  int
		budget     int
		wantField  string
		wantMax    int
		wantBudget int
	}{
		{
			// A non-reasoning model keeps the legacy `max_tokens` spelling.
			name: "budget above max_tokens", maxTokens: 1024, budget: 2048,
			wantField: "max_tokens", wantMax: 1024, wantBudget: 2048,
		},
		{
			name: "budget exactly equal to max_tokens", maxTokens: 1024, budget: 1024,
			wantField: "max_tokens", wantMax: 1024, wantBudget: 1024,
		},
		{
			// A thinking-capable model gets `max_completion_tokens` instead —
			// and that is the spelling a budget-carrying request actually uses,
			// so a check that only knew `max_tokens` would miss every real case.
			name: "budget above max_completion_tokens", caps: model.Capabilities{Thinking: true},
			maxTokens: 512, budget: 4096,
			wantField: "max_completion_tokens", wantMax: 512, wantBudget: 4096,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			seen := make(chan []byte, 1)
			srv := echoServer(t, seen)
			err := invokeWith(t, srv.URL,
				model.Sampling{MaxTokens: tokens(tt.maxTokens)}, tt.caps,
				simple.WithThinkingBudget(tt.budget))
			if err == nil {
				t.Fatalf("Invoke() sent a body the provider answers with 400: %s", <-seen)
			}
			var budgetErr *simple.ThinkingBudgetError
			if !errors.As(err, &budgetErr) {
				t.Fatalf("Invoke() error = %v (%T), want *simple.ThinkingBudgetError", err, err)
			}
			if budgetErr.Field != tt.wantField || budgetErr.MaxTokens != tt.wantMax || budgetErr.BudgetTokens != tt.wantBudget {
				t.Errorf("ThinkingBudgetError = {%q, max %d, budget %d}, want {%q, max %d, budget %d}",
					budgetErr.Field, budgetErr.MaxTokens, budgetErr.BudgetTokens,
					tt.wantField, tt.wantMax, tt.wantBudget)
			}
			select {
			case body := <-seen:
				t.Errorf("the request reached the server despite the local rejection: %s", body)
			default:
			}
		})
	}
}

// TestWithThinkingBudgetAcceptsALegalBudget is the positive control: a rule that
// also rejects valid input is worse than the bug it closes. Each case must reach
// the server, and must still carry the thinking object the option exists to add.
func TestWithThinkingBudgetAcceptsALegalBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		sampling model.Sampling
		budget   int
	}{
		{name: "budget one token below the cap", sampling: model.Sampling{MaxTokens: tokens(1024)}, budget: 1023},
		{name: "budget far below the cap", sampling: model.Sampling{MaxTokens: tokens(8192)}, budget: 1024},
		// No cap in the request at all: nothing is violated, and the rule must
		// not invent a limit of its own.
		{name: "no max tokens at all", sampling: model.Sampling{}, budget: 8192},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			seen := make(chan []byte, 1)
			srv := echoServer(t, seen)
			if err := invokeWith(t, srv.URL, tt.sampling, model.Capabilities{},
				simple.WithThinkingBudget(tt.budget)); err != nil {
				t.Fatalf("Invoke() rejected a legal budget: %v", err)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(<-seen, &body); err != nil {
				t.Fatalf("decode sent body: %v", err)
			}
			var thinking struct {
				Type   string `json:"type"`
				Budget int    `json:"budget_tokens"`
			}
			if err := json.Unmarshal(body["thinking"], &thinking); err != nil {
				t.Fatalf("the option stopped writing thinking.budget_tokens: %v", err)
			}
			if thinking.Type != "enabled" || thinking.Budget != tt.budget {
				t.Errorf("thinking = %+v, want {enabled %d}", thinking, tt.budget)
			}
		})
	}
}

// TestWithThinkingBudgetSurvivesALaterBodyPatch pins the composition hazard the
// check exposed. compat.WithBodyPatch REPLACES the configured patch, and gitlab
// appends its own model-rewriting patch AFTER the caller's options — so a
// validating patch installed by WithThinkingBudget would have been silently
// dropped for exactly one of its seven consumers. simple.WithBodyPatch now
// CHAINS instead, in application order, so both patches run.
func TestWithThinkingBudgetSurvivesALaterBodyPatch(t *testing.T) {
	t.Parallel()

	seen := make(chan []byte, 1)
	srv := echoServer(t, seen)
	later := simple.WithBodyPatch(func(body map[string]json.RawMessage) error {
		body["model"] = json.RawMessage(`"rewritten"`)
		return nil
	})

	err := invokeWith(t, srv.URL, model.Sampling{MaxTokens: tokens(1024)}, model.Capabilities{},
		simple.WithThinkingBudget(2048), later)
	var budgetErr *simple.ThinkingBudgetError
	if !errors.As(err, &budgetErr) {
		t.Fatalf("Invoke() error = %v (%T), want *simple.ThinkingBudgetError; "+
			"a later WithBodyPatch clobbered the validating patch", err, err)
	}

	// The other direction: the later patch must still take effect on a legal
	// request, so chaining did not cost anyone their patch.
	if err := invokeWith(t, srv.URL, model.Sampling{MaxTokens: tokens(4096)}, model.Capabilities{},
		simple.WithThinkingBudget(2048), later); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-seen, &body); err != nil {
		t.Fatalf("decode sent body: %v", err)
	}
	if string(body["model"]) != `"rewritten"` {
		t.Errorf("model = %s, want \"rewritten\"; chaining dropped the later patch", body["model"])
	}
}

// TestWithBodyPatchChainsInApplicationOrder pins the ordering the chain
// guarantees, independently of the budget rule: an earlier patch runs first, so
// a later one can still overwrite what it wrote.
func TestWithBodyPatchChainsInApplicationOrder(t *testing.T) {
	t.Parallel()

	seen := make(chan []byte, 1)
	srv := echoServer(t, seen)
	first := simple.WithBodyPatch(func(body map[string]json.RawMessage) error {
		body["marker"] = json.RawMessage(`"first"`)
		body["only_first"] = json.RawMessage(`true`)
		return nil
	})
	second := simple.WithBodyPatch(func(body map[string]json.RawMessage) error {
		body["marker"] = json.RawMessage(`"second"`)
		return nil
	})
	if err := invokeWith(t, srv.URL, model.Sampling{}, model.Capabilities{}, first, second); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-seen, &body); err != nil {
		t.Fatalf("decode sent body: %v", err)
	}
	if string(body["marker"]) != `"second"` {
		t.Errorf("marker = %s, want \"second\"", body["marker"])
	}
	if string(body["only_first"]) != `true` {
		t.Errorf("only_first = %s, want true; the first patch was dropped", body["only_first"])
	}
}
