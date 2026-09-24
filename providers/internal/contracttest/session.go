package contracttest

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/simple"
)

// contractSessionID is a representative Request.SessionID.
const contractSessionID = "6f1e2d3c-4b5a-4978-8675-6453422110ff"

// sessionHeaderFormats are the request formats a session-header contract is
// held across: the header is a transport concern, so it must not depend on
// which codec encoded the body.
var sessionHeaderFormats = []model.APIFormat{
	model.APIFormatOpenAI,
	model.APIFormatOpenAIResponses,
	model.APIFormatAnthropic,
}

// sessionHeaderServer answers every format with a minimal successful
// non-streaming response and records each request's headers.
type sessionHeaderServer struct {
	mu      sync.Mutex
	headers []http.Header
}

func (s *sessionHeaderServer) start(t *testing.T, apiFormat model.APIFormat) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.headers = append(s.headers, r.Header.Clone())
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch apiFormat {
		case model.APIFormatAnthropic:
			_, _ = io.WriteString(w, `{"id":"id","type":"message","role":"assistant","model":"model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`)
		case model.APIFormatOpenAIResponses:
			_, _ = io.WriteString(w, `{"id":"resp","object":"response","model":"model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
		default:
			_, _ = io.WriteString(w, `{"id":"id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (s *sessionHeaderServer) last(t *testing.T) http.Header {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.headers) == 0 {
		t.Fatal("no request reached the server")
	}
	return s.headers[len(s.headers)-1]
}

func (s *sessionHeaderServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.headers)
}

// SessionHeader verifies a provider that forwards inference.Request.SessionID
// in header, on every request format:
//   - absent when the request carries no SessionID, so a caller that predates
//     the field sends exactly what it sent before;
//   - the SessionID verbatim, as the header's only value, when it does;
//   - an explicit non-empty WithHeader value wins over the per-request one;
//   - an unsendable SessionID is refused locally with
//     *inference.InvalidSessionIDError and no request reaches the provider.
func SessionHeader(t *testing.T, provider llm.Provider, key auth.APIKey, header string, construct OptionConstructor) {
	t.Helper()
	for _, apiFormat := range sessionHeaderFormats {
		t.Run(string(apiFormat), func(t *testing.T) {
			t.Parallel()
			server := &sessionHeaderServer{}
			srv := server.start(t, apiFormat)
			selected := model.CustomModel(model.ProviderName(provider), apiFormat, srv.URL, "model")

			client, err := construct(selected, key)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
				t.Fatalf("Invoke() without SessionID error = %v", err)
			}
			if values := server.last(t).Values(header); len(values) != 0 {
				t.Errorf("%s = %q without a request SessionID, want absent", header, values)
			}

			if _, err := client.Invoke(context.Background(), inference.Request{Model: selected, SessionID: contractSessionID}); err != nil {
				t.Fatalf("Invoke() with SessionID error = %v", err)
			}
			if values := server.last(t).Values(header); len(values) != 1 || values[0] != contractSessionID {
				t.Errorf("%s = %q, want exactly [%q]", header, values, contractSessionID)
			}

			pinned, err := construct(selected, key, simple.WithHeader(header, "operator-pinned"))
			if err != nil {
				t.Fatalf("New() with explicit header error = %v", err)
			}
			if _, err := pinned.Invoke(context.Background(), inference.Request{Model: selected, SessionID: contractSessionID}); err != nil {
				t.Fatalf("Invoke() with pinned header error = %v", err)
			}
			if values := server.last(t).Values(header); len(values) != 1 || values[0] != "operator-pinned" {
				t.Errorf("%s = %q, want the explicit WithHeader value to win", header, values)
			}

			before := server.count()
			_, err = client.Invoke(context.Background(), inference.Request{Model: selected, SessionID: "id\r\nX-Injected: yes"})
			var invalid *inference.InvalidSessionIDError
			if !errors.As(err, &invalid) {
				t.Fatalf("Invoke() with unsendable SessionID error = %T %v, want *inference.InvalidSessionIDError", err, err)
			}
			if after := server.count(); after != before {
				t.Errorf("an unsendable SessionID reached the provider (%d request(s))", after-before)
			}
		})
	}
}

// SessionIDNotForwarded verifies a provider that documents no session header
// sends inference.Request.SessionID nowhere in its request headers: the
// identity is caller data, and a provider that has not said it reads it must
// not receive it.
func SessionIDNotForwarded(t *testing.T, provider llm.Provider, key auth.APIKey, construct Constructor) {
	t.Helper()
	for _, apiFormat := range sessionHeaderFormats {
		t.Run(string(apiFormat), func(t *testing.T) {
			t.Parallel()
			server := &sessionHeaderServer{}
			srv := server.start(t, apiFormat)
			selected := model.CustomModel(model.ProviderName(provider), apiFormat, srv.URL, "model")
			client, err := construct(selected, key)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if _, err := client.Invoke(context.Background(), inference.Request{Model: selected, SessionID: contractSessionID}); err != nil {
				t.Fatalf("Invoke() with SessionID error = %v", err)
			}
			for name, values := range server.last(t) {
				for _, value := range values {
					if value == contractSessionID {
						t.Errorf("header %s carries the request SessionID; this provider documents no session header", name)
					}
				}
			}
		})
	}
}
