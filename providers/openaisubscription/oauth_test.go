package openaisubscription

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/looprig/credentials"
	"github.com/looprig/credentials/refresh"
	"github.com/looprig/secrets"
	"github.com/looprig/secrets/local"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixtureIDToken(t *testing.T, key *rsa.PrivateKey, clientID, nonce string, change func(map[string]any)) string {
	t.Helper()
	claims := map[string]any{"iss": Issuer, "aud": clientID, "sub": "fixture-subject", "nonce": nonce, "exp": time.Now().Add(time.Hour).Unix(), "iat": time.Now().Unix(), "email": "fixture@example.test"}
	if change != nil {
		change(claims)
	}
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","kid":"fixture-key"}`))
	raw, _ := json.Marshal(claims)
	input := header + "." + base64.RawURLEncoding.EncodeToString(raw)
	hash := sha256.Sum256([]byte(input))
	signature, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, hash[:])
	if err != nil {
		t.Fatal(err)
	}
	return input + "." + base64.RawURLEncoding.EncodeToString(signature)
}
func fixtureJSON(value any) *http.Response {
	raw, _ := json.Marshal(value)
	return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(string(raw)))}
}
func fixtureJWKS(key *rsa.PrivateKey) any {
	return map[string]any{"keys": []any{map[string]string{"kty": "RSA", "alg": "RS256", "use": "sig", "kid": "fixture-key", "n": base64.RawURLEncoding.EncodeToString(key.N.Bytes()), "e": base64.RawURLEncoding.EncodeToString(big.NewInt(int64(key.E)).Bytes())}}}
}

func TestLoginDynamicRegistration(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	host, err := NewHostID()
	if err != nil {
		t.Fatal(err)
	}
	var nonce, redirect, verifier string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.String() {
		case TokenURL:
			if err := r.ParseForm(); err != nil {
				t.Fatal(err)
			}
			if r.Form.Get("client_id") != "oaiapp_fixture" || r.Form.Get("resource") != BaseURL || r.Form.Get("redirect_uri") != redirect {
				t.Fatal("wrong exchange identity")
			}
			verifier = r.Form.Get("code_verifier")
			return fixtureJSON(map[string]any{"access_token": "fixture-access", "refresh_token": "fixture-refresh", "token_type": "Bearer", "expires_in": 3600, "scope": "openid offline_access resource.invoke chatgpt.tokens.use.direct", "id_token": fixtureIDToken(t, key, "oaiapp_fixture", nonce, nil)}), nil
		case JWKSURL:
			return fixtureJSON(fixtureJWKS(key)), nil
		default:
			t.Fatalf("unexpected request: %s", r.URL)
			return nil, fmt.Errorf("unexpected request")
		}
	})}
	state, registration, err := Login(context.Background(), LoginOptions{HostID: host, HTTPClient: client, OpenBrowser: func(raw string) error {
		u, _ := url.Parse(raw)
		q := u.Query()
		nonce = q.Get("nonce")
		redirect = q.Get("redirect_uri")
		if q.Get("client_id") != "dynamic_agent_client" || q.Get("ext_agent_host_id") != host || q.Get("agent_name_hint") != "Carbon" || q.Get("resource") != BaseURL || q.Get("code_challenge_method") != "S256" || nonce == "" {
			t.Fatal("wrong authorization request")
		}
		callback, _ := url.Parse(redirect)
		if callback.Hostname() != "127.0.0.1" || callback.Path != "/auth/callback" {
			t.Fatal("wrong callback")
		}
		callback.RawQuery = url.Values{"code": {"fixture-code"}, "state": {q.Get("state")}, "client_id": {"oaiapp_fixture"}}.Encode()
		response, err := http.Get(callback.String())
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode != 200 {
			t.Fatalf("callback status %d", response.StatusCode)
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if registration.ClientID != "oaiapp_fixture" || registration.Subject != "fixture-subject" || state.AccessToken.IsZero() || state.RefreshToken.IsZero() || !state.PersistAccessToken || verifier == "" {
		t.Fatal("registration not retained")
	}
}

func TestIDTokenValidation(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return fixtureJSON(fixtureJWKS(key)), nil })}
	for _, field := range []string{"iss", "aud", "nonce", "exp", "sub"} {
		token := fixtureIDToken(t, key, "oaiapp_fixture", "nonce", func(c map[string]any) {
			if field == "exp" {
				c[field] = 1
			} else {
				c[field] = ""
			}
		})
		if _, err := verifyIdentity(context.Background(), client, token, "oaiapp_fixture", "nonce"); err == nil {
			t.Fatalf("accepted invalid %s", field)
		}
	}
	token := fixtureIDToken(t, key, "oaiapp_fixture", "nonce", nil)
	if _, err := verifyIdentity(context.Background(), client, token+"x", "oaiapp_fixture", "nonce"); err == nil {
		t.Fatal("accepted bad signature")
	}
}

func TestSourceRefreshRotatesAndPersists(t *testing.T) {
	ctx := context.Background()
	store, err := local.New(filepath.Join(t.TempDir(), "state"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	host, _ := NewHostID()
	registration := Registration{ClientID: "oaiapp_fixture", Subject: "subject", HostID: host, Scopes: "resource.invoke chatgpt.tokens.use.direct"}
	initial, err := tokenState(tokenResponse{AccessToken: "old-access", RefreshToken: "old-refresh", ExpiresIn: 3600}, registration)
	if err != nil {
		t.Fatal(err)
	}
	initial.ExpiresAt = time.Now().Add(-time.Hour)
	value, err := refresh.EncodeState(initial)
	if err != nil {
		t.Fatal(err)
	}
	stateRef, _ := secrets.NewReference("local", "credentials/subscription")
	if _, err := store.Put(ctx, stateRef, value, secrets.CreateOnlyPut()); err != nil {
		t.Fatal(err)
	}
	ref, _ := credentials.ParseReference("credential://openai-subscription/test")
	descriptor, _ := credentials.NewDescriptor("openai-subscription", "responses", credentials.SchemeOAuth, credentials.UsageSubscription, Issuer, "https://api.openai.com", "")
	requests := 0
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requests++
		r.ParseForm()
		if r.URL.String() != TokenURL || r.Form.Get("client_id") != "oaiapp_fixture" || r.Form.Get("refresh_token") != "old-refresh" || r.Form.Has("scope") {
			t.Fatal("wrong refresh")
		}
		return fixtureJSON(map[string]any{"access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer", "expires_in": 3600}), nil
	})}
	coordinator := refresh.NewProcessCoordinator()
	defer coordinator.Close()
	source, err := NewSource(ctx, credentials.FactoryInput{Reference: ref, Descriptor: descriptor, State: stateRef, Resolver: store, Store: store, RefreshCoordinator: coordinator, StateSharing: credentials.SharingProcess, HTTPClient: client})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	lease, err := source.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequest("GET", BaseURL+"/models", nil)
	if err := lease.Authorizer().Authorize(ctx, request); err != nil || request.Header.Get("Authorization") != "Bearer new-access" {
		t.Fatal("new access token not published")
	}
	saved, err := store.Resolve(ctx, stateRef)
	if err != nil {
		t.Fatal(err)
	}
	next, err := refresh.DecodeState(saved.Value)
	if err != nil {
		t.Fatal(err)
	}
	if string(next.RefreshToken.Bytes()) != "new-refresh" || next.Generation == initial.Generation || requests != 1 {
		t.Fatal("rotation not saved")
	}
}

func TestModelsUseSelectedAccountAndFilterVisibility(t *testing.T) {
	// Reuse the real renewable Source and a local persisted token.
	ctx := context.Background()
	store, err := local.New(filepath.Join(t.TempDir(), "state"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	host, _ := NewHostID()
	registration := Registration{ClientID: "oaiapp_fixture", Subject: "subject", HostID: host, Scopes: "resource.invoke chatgpt.tokens.use.direct"}
	state, _ := tokenState(tokenResponse{AccessToken: "account-access", RefreshToken: "account-refresh", ExpiresIn: 3600}, registration)
	value, _ := refresh.EncodeState(state)
	stateRef, _ := secrets.NewReference("local", "credentials/account")
	store.Put(ctx, stateRef, value, secrets.CreateOnlyPut())
	ref, _ := credentials.ParseReference("credential://openai-subscription/test")
	descriptor, _ := credentials.NewDescriptor("openai-subscription", "responses", credentials.SchemeOAuth, credentials.UsageSubscription, Issuer, "https://api.openai.com", "")
	coordinator := refresh.NewProcessCoordinator()
	defer coordinator.Close()
	source, err := NewSource(ctx, credentials.FactoryInput{Reference: ref, Descriptor: descriptor, State: stateRef, Resolver: store, Store: store, RefreshCoordinator: coordinator, StateSharing: credentials.SharingProcess})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != BaseURL+"/models" || r.Header.Get("Authorization") != "Bearer account-access" {
			t.Fatal("wrong model account")
		}
		return fixtureJSON(map[string]any{"models": []any{map[string]string{"slug": "first", "display_name": "First", "visibility": "list"}, map[string]string{"slug": "hidden", "visibility": "hide"}, map[string]string{"slug": "second", "display_name": "Second", "visibility": "list"}}}), nil
	})}
	models, err := ListModels(ctx, source, client)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].Slug != "first" || models[1].Slug != "second" {
		t.Fatalf("wrong models: %v", models)
	}
}

func TestRevocationPinsIssuerAndUsesRefreshToken(t *testing.T) {
	host, _ := NewHostID()
	state, _ := tokenState(tokenResponse{AccessToken: "access", RefreshToken: "renewable", ExpiresIn: 3600}, Registration{ClientID: "oaiapp_fixture", Subject: "subject", HostID: host, Scopes: "resource.invoke chatgpt.tokens.use.direct"})
	for _, endpoint := range []string{Issuer + "/api/accounts/oauth/revoke", "https://evil.example/revoke"} {
		posts := 0
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.Method == "GET" {
				return fixtureJSON(map[string]string{"issuer": Issuer, "revocation_endpoint": endpoint}), nil
			}
			posts++
			r.ParseForm()
			if r.Form.Get("client_id") != "oaiapp_fixture" || r.Form.Get("token") != "renewable" || r.Form.Get("token_type_hint") != "refresh_token" {
				t.Fatal("wrong revocation authority")
			}
			return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(""))}, nil
		})}
		err := Revoke(context.Background(), client, state)
		if strings.Contains(endpoint, "evil") {
			if err == nil || posts != 0 {
				t.Fatal("sent token to wrong issuer")
			}
		} else if err != nil || posts != 1 {
			t.Fatalf("revocation failed: %v", err)
		}
	}
}

func TestSubscriptionSourceRejectsMissingDependencies(t *testing.T) {
	descriptor, _ := credentials.NewDescriptor("openai-subscription", "responses", credentials.SchemeOAuth, credentials.UsageSubscription, Issuer, "https://api.openai.com", "")
	if _, err := NewSource(context.Background(), credentials.FactoryInput{Descriptor: descriptor}); err == nil {
		t.Fatal("accepted missing factory dependencies")
	}
}

func TestLoginDenialDoesNotExchangeCode(t *testing.T) {
	host, _ := NewHostID()
	calls := 0
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { calls++; return fixtureJSON(nil), nil })}
	_, _, err := Login(context.Background(), LoginOptions{HostID: host, HTTPClient: client, OpenBrowser: func(raw string) error {
		u, _ := url.Parse(raw)
		q := u.Query()
		callback, _ := url.Parse(q.Get("redirect_uri"))
		callback.RawQuery = url.Values{"state": {q.Get("state")}, "error": {"access_denied"}}.Encode()
		response, err := http.Get(callback.String())
		if err != nil {
			return err
		}
		defer response.Body.Close()
		return nil
	}})
	if err == nil || calls != 0 {
		t.Fatal("exchanged denied authorization")
	}
}
