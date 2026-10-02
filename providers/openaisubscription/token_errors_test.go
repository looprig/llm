package openaisubscription

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/looprig/credentials"
	"github.com/looprig/credentials/refresh"
	"github.com/looprig/secrets"
	"github.com/looprig/secrets/local"
)

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, errors.New("connection reset") }
func (failingBody) Close() error             { return nil }

func oauthError(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}
}

func subscriptionState(t *testing.T) refresh.State {
	t.Helper()
	host, _ := NewHostID()
	state, err := tokenState(tokenResponse{AccessToken: "old-access", RefreshToken: "old-refresh", ExpiresIn: 3600}, Registration{ClientID: "oaiapp_fixture", Subject: "subject", HostID: host, Scopes: scopes})
	if err != nil {
		t.Fatal(err)
	}
	return state
}

// Recovery per https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery
// (refresh errors) and the rotating-token contract in accounts-and-sessions.
func TestRefreshExchangeClassifiesTokenEndpointOutcomes(t *testing.T) {
	type class int
	const (
		ambiguous class = iota
		reauth
		temporary
		clientRejected
	)
	cases := map[string]struct {
		rt   roundTripFunc
		want class
	}{
		"lost response after write": {func(r *http.Request) (*http.Response, error) {
			httptrace.ContextClientTrace(r.Context()).WroteRequest(httptrace.WroteRequestInfo{})
			return nil, errors.New("connection reset")
		}, ambiguous},
		"custom transport failure of unknown progress": {func(*http.Request) (*http.Response, error) {
			return nil, errors.New("proxy hung up")
		}, ambiguous},
		"body lost after 200": {func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: http.Header{}, Body: failingBody{}}, nil
		}, ambiguous},
		"malformed 200": {func(*http.Request) (*http.Response, error) { return oauthError(200, `{`), nil }, ambiguous},
		"invalid 200 token": {func(*http.Request) (*http.Response, error) {
			return fixtureJSON(map[string]any{"access_token": "a", "token_type": "Bearer", "expires_in": 3600}), nil
		}, ambiguous},
		"dial failure": {func(*http.Request) (*http.Response, error) {
			return nil, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
		}, temporary},
		"dns failure": {func(*http.Request) (*http.Response, error) {
			return nil, &net.DNSError{Err: "no such host", Name: "auth.openai.com"}
		}, temporary},
		"invalid_grant": {func(*http.Request) (*http.Response, error) { return oauthError(400, `{"error":"invalid_grant"}`), nil }, reauth},
		"refresh_token_reused": {func(*http.Request) (*http.Response, error) {
			return oauthError(401, `{"error":{"code":"refresh_token_reused"}}`), nil
		}, reauth},
		"refresh_token_expired": {func(*http.Request) (*http.Response, error) {
			return oauthError(400, `{"error":"refresh_token_expired"}`), nil
		}, reauth},
		"refresh_token_invalidated": {func(*http.Request) (*http.Response, error) {
			return oauthError(400, `{"error":"refresh_token_invalidated"}`), nil
		}, reauth},
		"invalid_client":     {func(*http.Request) (*http.Response, error) { return oauthError(401, `{"error":"invalid_client"}`), nil }, clientRejected},
		"server unavailable": {func(*http.Request) (*http.Response, error) { return oauthError(503, `upstream`), nil }, temporary},
		"throttled":          {func(*http.Request) (*http.Response, error) { return oauthError(429, `{}`), nil }, temporary},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			exchange := refreshExchange(&http.Client{Transport: tc.rt})
			_, err := exchange(context.Background(), subscriptionState(t))
			if err == nil {
				t.Fatal("exchange succeeded")
			}
			got := map[class]bool{
				ambiguous:      errors.Is(err, refresh.ErrAmbiguousRotation),
				reauth:         errors.Is(err, refresh.ErrReauthenticationRequired),
				temporary:      errors.Is(err, refresh.ErrExchangeTemporary),
				clientRejected: errors.Is(err, ErrClient),
			}
			for c, ok := range got {
				if ok != (c == tc.want) {
					t.Fatalf("err = %v; class %d = %v, want only class %d", err, c, ok, tc.want)
				}
			}
			if strings.Contains(err.Error(), "old-refresh") || strings.Contains(err.Error(), "upstream") {
				t.Fatalf("error leaked token or body: %v", err)
			}
		})
	}
	t.Run("canceled before sending", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		sent := false
		_, err := refreshExchange(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			sent = true
			return nil, errors.New("unexpected")
		})})(ctx, subscriptionState(t))
		if sent || !errors.Is(err, context.Canceled) || errors.Is(err, refresh.ErrAmbiguousRotation) {
			t.Fatalf("err = %v sent = %v, want plain cancellation", err, sent)
		}
	})
	t.Run("canceled after sending", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		_, err := refreshExchange(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			httptrace.ContextClientTrace(r.Context()).WroteRequest(httptrace.WroteRequestInfo{})
			cancel()
			return nil, r.Context().Err()
		})})(ctx, subscriptionState(t))
		if !errors.Is(err, refresh.ErrAmbiguousRotation) || !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want ambiguous cancellation", err)
		}
	})
}

// A real http.Transport reports progress through httptrace, so a refused
// connection is known to precede sending.
func TestRefreshExchangeRealTransportRefusedConnectionIsTemporary(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	_ = listener.Close()
	transport := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}}
	defer transport.CloseIdleConnections()
	_, err = refreshExchange(&http.Client{Transport: transport})(context.Background(), subscriptionState(t))
	if !errors.Is(err, refresh.ErrExchangeTemporary) || errors.Is(err, refresh.ErrAmbiguousRotation) {
		t.Fatalf("err = %v, want temporary", err)
	}
}

func TestLostRefreshResponseIsNotResubmittedByAnotherSource(t *testing.T) {
	ctx := context.Background()
	store, err := local.New(filepath.Join(t.TempDir(), "state"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	initial := subscriptionState(t)
	initial.ExpiresAt = time.Now().Add(-time.Hour)
	value, _ := refresh.EncodeState(initial)
	stateRef, _ := secrets.NewReference("local", "credentials/subscription")
	if _, err := store.Put(ctx, stateRef, value, secrets.CreateOnlyPut()); err != nil {
		t.Fatal(err)
	}
	ref, _ := credentials.ParseReference("credential://openai-subscription/test")
	descriptor, _ := credentials.NewDescriptor("openai-subscription", "responses", credentials.SchemeOAuth, credentials.UsageSubscription, Issuer, "https://api.openai.com", "")
	exchanges := 0
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		exchanges++
		httptrace.ContextClientTrace(r.Context()).WroteRequest(httptrace.WroteRequestInfo{})
		return nil, errors.New("connection reset")
	})}
	newSource := func() credentials.Source {
		coordinator := refresh.NewProcessCoordinator()
		t.Cleanup(func() { _ = coordinator.Close() })
		source, err := NewSource(ctx, credentials.FactoryInput{Reference: ref, Descriptor: descriptor, State: stateRef, Resolver: store, Store: store, RefreshCoordinator: coordinator, StateSharing: credentials.SharingProcess, HTTPClient: client})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = source.Close() })
		return source
	}
	if _, err := newSource().Acquire(ctx); !errors.Is(err, refresh.ErrAmbiguousRotation) {
		t.Fatalf("Acquire() = %v, want ambiguous rotation", err)
	}
	if _, err := newSource().Acquire(ctx); !errors.Is(err, refresh.ErrAmbiguousRotation) {
		t.Fatalf("other source Acquire() = %v, want ambiguous rotation", err)
	}
	if exchanges != 1 {
		t.Fatalf("refresh token submitted %d times, want once", exchanges)
	}
}

func TestRefreshKeepsRotatedTokenWhenPlanScopeIsReduced(t *testing.T) {
	exchange := refreshExchange(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return fixtureJSON(map[string]any{"access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer", "expires_in": 3600, "scope": "openid resource.invoke"}), nil
	})})
	response, err := exchange(context.Background(), subscriptionState(t))
	if err != nil {
		t.Fatalf("exchange discarded a completed rotation: %v", err)
	}
	if string(response.RefreshToken.Bytes()) != "new-refresh" || !response.RefreshTokenSet {
		t.Fatal("rotated refresh token not returned for persistence")
	}
	var registration Registration
	if err := json.Unmarshal(response.ProviderData, &registration); err != nil || registration.Scopes != "openid resource.invoke" {
		t.Fatalf("reduced scopes not recorded: %q %v", registration.Scopes, err)
	}
	// The next use of the persisted state refuses plan usage, as a sign-in
	// that must re-grant chatgpt.tokens.use.direct.
	next := subscriptionState(t)
	next.ProviderData = response.ProviderData
	if _, err := RegistrationFromState(next); !errors.Is(err, ErrPermission) {
		t.Fatalf("RegistrationFromState() = %v, want ErrPermission", err)
	}
	_, err = exchange(context.Background(), next)
	if !errors.Is(err, ErrPermission) || !errors.Is(err, refresh.ErrReauthenticationRequired) {
		t.Fatalf("exchange on reduced scope = %v, want permission reauthentication", err)
	}
}

func TestLoginCodeExchangePreservesCancellationAndBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := exchangeTokens(ctx, &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("unexpected")
	})}, url.Values{}, true); !errors.Is(err, context.Canceled) || !errors.Is(err, credentials.ErrCanceled) {
		t.Fatalf("canceled code exchange = %v", err)
	}
	if _, err := exchangeTokens(context.Background(), &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return oauthError(503, ""), nil
	})}, url.Values{}, true); !errors.Is(err, ErrTokenUnavailable) {
		t.Fatalf("503 code exchange = %v, want ErrTokenUnavailable", err)
	}
}
