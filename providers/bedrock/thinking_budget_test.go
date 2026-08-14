package bedrock

import (
	"encoding/json"
	"errors"
	"testing"
)

// Confirmed live against api.anthropic.com:
//
//	400 "`max_tokens` must be greater than `thinking.budget_tokens`"
//
// Anthropic-on-Bedrock inherits the rule, and this is one of only two places in
// the tree that can produce a violating body: WithReasoning (and a hand-written
// WithAdditionalModelRequestFields) puts thinking.budget_tokens into
// additionalModelRequestFields, while maxTokens lives in a DIFFERENT top-level
// object, inferenceConfig. The shared bedrockconverse encoder writes
// inferenceConfig and never sees the option; the option writes
// additionalModelRequestFields and never saw inferenceConfig. applyConverse is
// the one place holding both, so the check has to live there.
//
// The conformance gate cannot help. Converse's ConverseRequest models
// additionalModelRequestFields as an opaque Document, so its contents are not
// even schema-shaped, let alone cross-checkable against inferenceConfig.maxTokens.
// TestTheConverseGateAcceptsABudgetAboveMaxTokens measures that rather than
// assuming it.

func budgetOf(v int) *int { return &v }

const converseBodyWithMaxTokens = `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"inferenceConfig":{"maxTokens":1024}}`

func TestApplyConverseRejectsAThinkingBudgetAtOrAboveMaxTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		configure  func(*config)
		wantMax    int
		wantBudget int
	}{
		{
			name:       "budget above maxTokens",
			configure:  func(c *config) { WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(2048)})(c) },
			wantMax:    1024,
			wantBudget: 2048,
		},
		{
			name:       "budget exactly equal to maxTokens",
			configure:  func(c *config) { WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(1024)})(c) },
			wantMax:    1024,
			wantBudget: 1024,
		},
		{
			name: "budget smuggled in through additionalModelRequestFields",
			configure: func(c *config) {
				WithAdditionalModelRequestFields(json.RawMessage(`{"thinking":{"type":"enabled","budget_tokens":4096}}`))(c)
			},
			wantMax:    1024,
			wantBudget: 4096,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var c config
			tt.configure(&c)
			body, err := c.applyConverse([]byte(converseBodyWithMaxTokens), false, nil)
			if err == nil {
				t.Fatalf("applyConverse() built a body Bedrock will reject with 400: %s", body)
			}
			var budgetErr *ThinkingBudgetError
			if !errors.As(err, &budgetErr) {
				t.Fatalf("applyConverse() error = %v (%T), want *bedrock.ThinkingBudgetError", err, err)
			}
			if budgetErr.MaxTokens != tt.wantMax || budgetErr.BudgetTokens != tt.wantBudget {
				t.Errorf("ThinkingBudgetError = {max %d, budget %d}, want {max %d, budget %d}",
					budgetErr.MaxTokens, budgetErr.BudgetTokens, tt.wantMax, tt.wantBudget)
			}
		})
	}
}

// TestApplyConverseAcceptsALegalThinkingBudget is the positive control: a rule
// that also rejects valid input is worse than the bug it closes. Every case
// here must still produce a body, and that body must still pass the gate.
func TestApplyConverseAcceptsALegalThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		body      string
		configure func(*config)
	}{
		{
			name:      "budget one token below maxTokens",
			body:      converseBodyWithMaxTokens,
			configure: func(c *config) { WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(1023)})(c) },
		},
		{
			name:      "budget far below maxTokens",
			body:      converseBodyWithMaxTokens,
			configure: func(c *config) { WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(256)})(c) },
		},
		{
			name:      "reasoning with no budget at all",
			body:      converseBodyWithMaxTokens,
			configure: func(c *config) { WithReasoning(ReasoningOptions{Type: "enabled"})(c) },
		},
		{
			name: "large budget with no inferenceConfig to contradict it",
			body: `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`,
			// Nothing in the body caps output, so nothing is violated. The check
			// must not invent a cap.
			configure: func(c *config) { WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(8192)})(c) },
		},
		{
			name: "large budget with an inferenceConfig that sets no maxTokens",
			body: `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"inferenceConfig":{"temperature":0.5}}`,
			configure: func(c *config) {
				WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(8192)})(c)
			},
		},
		{
			name:      "no reasoning option and no thinking field",
			body:      converseBodyWithMaxTokens,
			configure: func(c *config) { WithAdditionalModelRequestFields(json.RawMessage(`{"custom":true}`))(c) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var c config
			tt.configure(&c)
			body, err := c.applyConverse([]byte(tt.body), false, nil)
			if err != nil {
				t.Fatalf("applyConverse() rejected a legal body: %v", err)
			}
			gateConverseBody(t, body)
		})
	}
}

// TestApplyConverseCountTokensIgnoresTheBudgetRule pins that the CountTokens
// path is unaffected. Its body carries no inferenceConfig at all (an output cap
// cannot change an INPUT count, and counter_test.go asserts the field's
// absence), so there is nothing for the rule to compare against and it must
// stay out of the way rather than fail closed on a missing field.
func TestApplyConverseCountTokensIgnoresTheBudgetRule(t *testing.T) {
	t.Parallel()

	var c config
	WithReasoning(ReasoningOptions{BudgetTokens: budgetOf(8192)})(&c)
	if _, err := c.applyConverseCountTokens([]byte(`{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`), nil); err != nil {
		t.Fatalf("applyConverseCountTokens() error = %v", err)
	}
}

// TestTheConverseGateAcceptsABudgetAboveMaxTokens measures the gate rather than
// reasoning about it: additionalModelRequestFields is a Smithy Document, so its
// contents are unconstrained and no cross-object rule is expressible. The gate
// therefore validates a body the service rejects, which is precisely why the
// check above lives in the option path.
func TestTheConverseGateAcceptsABudgetAboveMaxTokens(t *testing.T) {
	t.Parallel()

	gateConverseBody(t, []byte(`{"messages":[{"role":"user","content":[{"text":"hi"}]}],`+
		`"inferenceConfig":{"maxTokens":1024},`+
		`"additionalModelRequestFields":{"thinking":{"type":"enabled","budget_tokens":8192}}}`))
	// Reaching here without a test failure IS the finding.
}
