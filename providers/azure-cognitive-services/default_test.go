package azurecognitive

import "testing"

func TestDefaultOpenAIBaseURLUsesTheAzureV1Root(t *testing.T) {
	if got, want := defaultOpenAIBaseURL("resource"), "https://resource.cognitiveservices.azure.com/openai/v1"; got != want {
		t.Fatalf("defaultOpenAIBaseURL() = %q, want %q", got, want)
	}
}
