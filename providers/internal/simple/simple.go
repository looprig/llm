// Package simple contains the common adapter used by provider packages whose
// documented endpoint is OpenAI Chat-compatible and uses either bearer or no
// authentication.
package simple

import (
	"encoding/json"
	"fmt"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	contextcount "github.com/looprig/inference/contextcount"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/compat"
)

type Definition = compat.Definition
type Option = compat.Option

func New(selected model.Model, key auth.APIKey, definition Definition, options ...Option) (inference.Client, error) {
	return compat.NewProvider(selected, key, definition, options...)
}

func NewCounter(provider llm.Provider) (contextcount.ContextCounter, error) {
	return compat.UnsupportedCounter(provider, model.APIFormatOpenAI)
}

func NewAnthropicCounter(provider llm.Provider) (contextcount.ContextCounter, error) {
	return compat.UnsupportedCounter(provider, model.APIFormatAnthropic)
}

func WithHeader(name, value string) Option { return compat.WithHeader(name, value) }

// WithBodyField is kept in the internal adapter so provider packages can expose
// only the stable fields their upstream documents, without importing compat.
func WithBodyField(name string, value any) Option { return compat.WithBodyField(name, value) }

// WithBodyPatch applies a provider-local patch after the common body fields are
// merged. Patches CHAIN in application order rather than replacing one another:
// compat.WithBodyPatch assigns, so the second of two patches silently discarded
// the first, and gitlab already relies on ordering it does not control (it
// appends its own model-rewriting patch AFTER the caller's options). Chaining
// here — the one place every provider package reaches WithBodyPatch through —
// makes an added patch additive, which is what a validating patch such as
// WithThinkingBudget's requires to be reliable.
func WithBodyPatch(patch func(map[string]json.RawMessage) error) Option {
	return func(options *compat.ProviderOptions) {
		options.PatchRequest = chainBodyPatch(options.PatchRequest, patch)
	}
}

// chainBodyPatch runs first then second, stopping at the first error. A nil
// half is skipped, so the zero state composes.
func chainBodyPatch(first, second func(map[string]json.RawMessage) error) func(map[string]json.RawMessage) error {
	switch {
	case first == nil:
		return second
	case second == nil:
		return first
	}
	return func(body map[string]json.RawMessage) error {
		if err := first(body); err != nil {
			return err
		}
		return second(body)
	}
}

func WithReasoningEffort(value string) Option {
	return compat.WithBodyField("reasoning_effort", value)
}

func WithReasoningEnabled(enabled bool) Option {
	return compat.WithBodyField("reasoning", map[string]bool{"enabled": enabled})
}

func WithThinking(enabled bool) Option {
	type thinking struct {
		Type string `json:"type"`
	}
	t := "disabled"
	if enabled {
		t = "enabled"
	}
	return compat.WithBodyField("thinking", thinking{Type: t})
}

// WithThinkingBudget injects a raw thinking.budget_tokens body field for the
// OpenAI-compatible and Anthropic-dialect targets that document it, and holds
// the result to Anthropic's rule that max_tokens must be GREATER than
// thinking.budget_tokens — violating it draws an HTTP 400 with that exact
// wording, confirmed live against api.anthropic.com.
//
// The rule is enforced from a body patch rather than from the option itself
// because only the patch sees both numbers: the budget is fixed when the option
// is built, while the output cap is written by the codec when a request is
// encoded. The failure therefore surfaces from Invoke/Stream, but still before
// any bytes leave the process.
func WithThinkingBudget(budget int) Option {
	field := compat.WithBodyField("thinking", map[string]any{
		"type":          "enabled",
		"budget_tokens": budget,
	})
	check := WithBodyPatch(checkThinkingBudget)
	return func(options *compat.ProviderOptions) {
		field(options)
		check(options)
	}
}

// ThinkingBudgetError is a fail-closed rejection, before any I/O, of a request
// whose reasoning budget is not smaller than its output cap. Field names the
// body member that carried the cap, because the three dialects this adapter
// speaks spell it differently and the caller needs to know which one it set.
// Both numbers are on the error: a message naming only one of them would not
// say what to change.
type ThinkingBudgetError struct {
	Field        string
	MaxTokens    int
	BudgetTokens int
}

func (e *ThinkingBudgetError) Error() string {
	return fmt.Sprintf("simple: %s (%d) must be greater than thinking.budget_tokens (%d)",
		e.Field, e.MaxTokens, e.BudgetTokens)
}

// maxTokenFields are every spelling of the output cap across the dialects
// compat can encode: Chat Completions writes max_tokens for a non-reasoning
// model and max_completion_tokens for a reasoning one, Responses writes
// max_output_tokens, and the Anthropic dialect writes max_tokens. Exactly one is
// ever present, but all four are checked so the rule cannot be defeated by a
// dialect switch — and max_completion_tokens in particular is the spelling a
// budget-carrying request actually uses.
var maxTokenFields = []string{"max_tokens", "max_completion_tokens", "max_output_tokens"}

// checkThinkingBudget refuses a body whose thinking.budget_tokens is not
// strictly below its output cap.
//
// It reads the budget back out of the BODY rather than closing over the option's
// value, so a budget written by a later WithBodyPatch is held to the same rule
// as one written by WithThinkingBudget.
//
// It is deliberately silent whenever it cannot see both numbers: no thinking
// object, no budget_tokens member, no cap field, or either value in a shape that
// is not a JSON number, all pass. A request with no declared output cap violates
// nothing, and inventing one — or rejecting an unrecognized shape — would turn a
// request the provider accepts into a local failure.
func checkThinkingBudget(body map[string]json.RawMessage) error {
	budget, ok := intMember(body["thinking"], "budget_tokens")
	if !ok {
		return nil
	}
	for _, field := range maxTokenFields {
		maxTokens, ok := decodeInt(body[field])
		if !ok || maxTokens > budget {
			continue
		}
		return &ThinkingBudgetError{Field: field, MaxTokens: maxTokens, BudgetTokens: budget}
	}
	return nil
}

// intMember reads one integer member out of a raw JSON object, reporting false
// for an absent object, a non-object, an absent member, or a non-integer member.
func intMember(object json.RawMessage, member string) (int, bool) {
	if len(object) == 0 {
		return 0, false
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(object, &decoded); err != nil {
		return 0, false
	}
	return decodeInt(decoded[member])
}

func decodeInt(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}
	return value, true
}

func WithThinkingEnabled(enabled bool) Option {
	return compat.WithBodyField("thinking", map[string]bool{"enabled": enabled})
}

func WithServiceTier(value string) Option {
	return compat.WithBodyField("service_tier", value)
}
