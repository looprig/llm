// Package simple contains the common adapter used by provider packages whose
// documented endpoint is OpenAI Chat-compatible and uses either bearer or no
// authentication.
package simple

import (
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

func WithHeader(name, value string) Option { return compat.WithHeader(name, value) }

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

func WithServiceTier(value string) Option {
	return compat.WithBodyField("service_tier", value)
}
