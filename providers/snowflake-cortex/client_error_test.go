package snowflake

import (
	"testing"

	"github.com/looprig/inference/failure"
)

func TestIsConversationCompleteRequiresExplicitSafeCode(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  *failure.APIError
		want bool
	}{
		{
			name: "code",
			err:  &failure.APIError{Status: 400, Code: "conversation_complete"},
			want: true,
		},
		{
			name: "provider code alias",
			err:  &failure.APIError{Status: 400, ProviderCode: "conversation_complete"},
			want: true,
		},
		{
			name: "status only",
			err:  &failure.APIError{Status: 400},
			want: false,
		},
		{
			name: "wrong status",
			err:  &failure.APIError{Status: 401, Code: "conversation_complete"},
			want: false,
		},
		{
			name: "other code",
			err:  &failure.APIError{Status: 400, Code: "invalid_request_error"},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isConversationComplete(tt.err); got != tt.want {
				t.Fatalf("isConversationComplete(%#v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}
