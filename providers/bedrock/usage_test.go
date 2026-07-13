package bedrock_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/llm/providers/bedrock"
)

func TestBedrockInvokeUsageResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		usageJSON string
		wantUsage *content.Usage
		wantErr   bool
	}{
		{
			name:      "successful invoke preserves normalized response and message usage",
			usageJSON: `{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":2,"output_tokens":7}`,
			wantUsage: &content.Usage{InputTokens: 11, OutputTokens: 7, CacheReadTokens: 3, CacheCreationTokens: 2},
		},
		{
			name:      "malformed usage returns no response",
			usageJSON: `{"input_tokens":-1,"output_tokens":7}`,
			wantErr:   true,
		},
		{
			name:      "present-zero usage remains distinguishable",
			usageJSON: `{}`,
			wantUsage: &content.Usage{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"anthropic.claude-3-5-sonnet-20241022-v2:0","content":[{"type":"text","text":"hello"}],"usage":`+tt.usageJSON+`}`)
			}))
			defer srv.Close()

			client := bedrock.NewWithEndpoint(testCreds(), "us-east-1", srv.URL)
			resp, err := client.Invoke(context.Background(), bedrockRequest("anthropic.claude-3-5-sonnet-20241022-v2:0"))
			if tt.wantErr {
				if err == nil {
					t.Fatal("Invoke() error = nil, want malformed-usage error")
				}
				if resp != nil {
					t.Fatalf("Invoke() response = %+v, want nil after error", resp)
				}
				return
			}
			if err != nil {
				t.Fatalf("Invoke() error = %v", err)
			}
			if resp == nil || resp.Message == nil {
				t.Fatalf("Invoke() response = %+v, want message", resp)
			}
			if resp.Usage == nil || *resp.Usage != *tt.wantUsage {
				t.Errorf("Response.Usage = %+v, want %+v", resp.Usage, tt.wantUsage)
			}
			if resp.Message.Usage == nil || *resp.Message.Usage != *tt.wantUsage {
				t.Errorf("Response.Message.Usage = %+v, want %+v", resp.Message.Usage, tt.wantUsage)
			}
		})
	}
}
