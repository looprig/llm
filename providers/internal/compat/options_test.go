package compat_test

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looprig/llm/providers/internal/compat"
)

func TestConfigClonesHeadersAndPatchesBody(t *testing.T) {
	t.Parallel()

	headers := http.Header{"X-Test": []string{"one"}}
	cfg := compat.Config{
		Headers: headers,
		PatchRequest: func(body map[string]json.RawMessage) error {
			body["x"] = json.RawMessage(`true`)
			return nil
		},
	}
	clone := cfg.Clone()
	headers.Set("X-Test", "mutated")
	if got := clone.Headers.Get("X-Test"); got != "one" {
		t.Errorf("cloned header = %q, want one", got)
	}
	body := map[string]json.RawMessage{}
	if err := clone.PatchRequest(body); err != nil {
		t.Fatalf("PatchRequest() error = %v", err)
	}
	if string(body["x"]) != "true" {
		t.Errorf("patched body = %s, want true", body["x"])
	}
}
