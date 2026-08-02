package fireworks

import "github.com/looprig/llm/providers/internal/simple"

func WithHeader(name, value string) Option    { return simple.WithHeader(name, value) }
func WithReasoningEffort(value string) Option { return simple.WithReasoningEffort(value) }
func WithThinking(enabled bool) Option        { return simple.WithThinking(enabled) }
