package snowflake

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestNormalizeEmptyAssistantRoleJSON(t *testing.T) {
	got, err := normalizeEmptyAssistantRole([]byte(`{"choices":[{"delta":{"role":"","content":"ok"}}]}`))
	if err != nil {
		t.Fatalf("normalizeEmptyAssistantRole() error = %v", err)
	}
	if string(got) != `{"choices":[{"delta":{"content":"ok","role":"assistant"}}]}` {
		t.Fatalf("normalized JSON = %s, want assistant role", got)
	}
}

func TestNormalizeEmptyAssistantRoleStream(t *testing.T) {
	response := &http.Response{
		Header: http.Header{},
		Body:   io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"role\":\"\"}}]}\n\ndata: [DONE]\n\n")),
	}
	normalized, err := normalizeEmptyAssistantRoleStream(response)
	if err != nil {
		t.Fatalf("normalizeEmptyAssistantRoleStream() error = %v", err)
	}
	body, err := io.ReadAll(normalized.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if !strings.Contains(string(body), `"role":"assistant"`) {
		t.Fatalf("normalized stream = %s, want assistant role", body)
	}
}

func TestValidAccountAllowsUnderscoreWithinHostnameLabel(t *testing.T) {
	if !validAccount("org_account.region") {
		t.Fatal("validAccount(org_account.region) = false, want true")
	}
	if validAccount("org..account") {
		t.Fatal("validAccount(org..account) = true, want false")
	}
}
