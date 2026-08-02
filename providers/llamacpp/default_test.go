package llamacpp_test

import (
	"testing"

	"github.com/looprig/llm/providers/llamacpp"
)

func TestDefaultBaseURLIsTheLocalLlamaServer(t *testing.T) {
	if got, want := llamacpp.DefaultBaseURL, "http://127.0.0.1:8080/v1"; got != want {
		t.Fatalf("DefaultBaseURL = %q, want %q", got, want)
	}
}
