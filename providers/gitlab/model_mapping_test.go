package gitlab

import (
	"testing"

	model "github.com/looprig/inference/model"
)

func TestResolveModelMapsCurrentGitLabAliases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		format model.APIFormat
		wantID string
	}{
		{name: "duo-chat-opus-4-6", format: model.APIFormatAnthropic, wantID: "claude-opus-4-6"},
		{name: "duo-chat-sonnet-4-6", format: model.APIFormatAnthropic, wantID: "claude-sonnet-4-6"},
		{name: "duo-chat-opus-4-5", format: model.APIFormatAnthropic, wantID: "claude-opus-4-5-20251101"},
		{name: "duo-chat-sonnet-4-5", format: model.APIFormatAnthropic, wantID: "claude-sonnet-4-5-20250929"},
		{name: "duo-chat-haiku-4-5", format: model.APIFormatAnthropic, wantID: "claude-haiku-4-5-20251001"},
		{name: "duo-chat-gpt-5-1", format: model.APIFormatOpenAI, wantID: "gpt-5.1-2025-11-13"},
		{name: "duo-chat-gpt-5-2", format: model.APIFormatOpenAI, wantID: "gpt-5.2-2025-12-11"},
		{name: "duo-chat-gpt-5-mini", format: model.APIFormatOpenAI, wantID: "gpt-5-mini-2025-08-07"},
		{name: "duo-chat-gpt-5-codex", format: model.APIFormatOpenAIResponses, wantID: "gpt-5-codex"},
		{name: "duo-chat-gpt-5-2-codex", format: model.APIFormatOpenAIResponses, wantID: "gpt-5.2-codex"},
		{name: "duo-chat-gpt-5-3-codex", format: model.APIFormatOpenAIResponses, wantID: "gpt-5.3-codex"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := resolveModel(tt.name, tt.format, nil)
			if err != nil {
				t.Fatalf("resolveModel() error = %v", err)
			}
			if got != tt.wantID {
				t.Fatalf("resolveModel() = %q, want %q", got, tt.wantID)
			}
		})
	}
}

func TestResolveModelFailsClosedForUnknownAndMismatchedAliases(t *testing.T) {
	t.Parallel()

	if _, err := resolveModel("unknown-model", model.APIFormatOpenAI, nil); err == nil {
		t.Fatal("resolveModel(unknown) error = nil, want fail-closed mapping error")
	}
	if _, err := resolveModel("duo-chat-gpt-5-1", model.APIFormatOpenAIResponses, nil); err == nil {
		t.Fatal("resolveModel(family mismatch) error = nil, want mapping error")
	}
}

func TestResolveModelAllowsExplicitUpstreamOverride(t *testing.T) {
	t.Parallel()

	override := &modelOverride{ID: "provider-model", Format: model.APIFormatOpenAI}
	got, err := resolveModel("custom-alias", model.APIFormatOpenAI, override)
	if err != nil {
		t.Fatalf("resolveModel(override) error = %v", err)
	}
	if got != "provider-model" {
		t.Fatalf("resolveModel(override) = %q, want provider-model", got)
	}
	if _, err := resolveModel("custom-alias", model.APIFormatOpenAIResponses, override); err == nil {
		t.Fatal("resolveModel(override family mismatch) error = nil, want mapping error")
	}
}
