package openaisubscription

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"testing"
	"time"
)

func signES256(t *testing.T, key *ecdsa.PrivateKey, header string, claims map[string]any) string {
	t.Helper()
	raw, _ := json.Marshal(claims)
	input := base64.RawURLEncoding.EncodeToString([]byte(header)) + "." + base64.RawURLEncoding.EncodeToString(raw)
	digest := sha256.Sum256([]byte(input))
	r, s, err := ecdsa.Sign(rand.Reader, key, digest[:])
	if err != nil {
		t.Fatal(err)
	}
	signature := make([]byte, 64)
	r.FillBytes(signature[:32])
	s.FillBytes(signature[32:])
	return input + "." + base64.RawURLEncoding.EncodeToString(signature)
}

func TestIDTokenES256AndKeySelection(t *testing.T) {
	ecKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	rsaKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	coordinate := func(v []byte) string { return base64.RawURLEncoding.EncodeToString(v) }
	x, y := make([]byte, 32), make([]byte, 32)
	ecKey.X.FillBytes(x)
	ecKey.Y.FillBytes(y)
	ecJWK := map[string]string{"kty": "EC", "crv": "P-256", "alg": "ES256", "use": "sig", "kid": "ec-key", "x": coordinate(x), "y": coordinate(y)}
	rsaJWK := fixtureJWKS(rsaKey).(map[string]any)["keys"].([]any)[0]
	claims := map[string]any{"iss": Issuer, "aud": "oaiapp_fixture", "sub": "subject", "nonce": "nonce", "exp": time.Now().Add(time.Hour).Unix(), "iat": time.Now().Unix()}
	jwks := func(keys ...any) *http.Client {
		return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.String() != JWKSURL {
				t.Errorf("fetched keys from %s, want only the pinned JWKS", r.URL)
			}
			return fixtureJSON(map[string]any{"keys": keys}), nil
		})}
	}
	good := signES256(t, ecKey, `{"alg":"ES256","kid":"ec-key"}`, claims)
	if identity, err := verifyIdentity(context.Background(), jwks(ecJWK, rsaJWK), good, "oaiapp_fixture", "nonce"); err != nil || identity.Subject != "subject" {
		t.Fatalf("valid ES256 token: %v", err)
	}
	other, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	duplicate := map[string]string{}
	for k, v := range ecJWK {
		duplicate[k] = v
	}
	rejected := map[string]struct {
		token  string
		client *http.Client
	}{
		"ES256 wrong key":         {signES256(t, other, `{"alg":"ES256","kid":"ec-key"}`, claims), jwks(ecJWK)},
		"ES256 on RSA key":        {signES256(t, ecKey, `{"alg":"ES256","kid":"fixture-key"}`, claims), jwks(rsaJWK)},
		"RS256 on EC key":         {signFixture(t, rsaKey, `{"alg":"RS256","kid":"ec-key"}`, claims), jwks(ecJWK)},
		"unsupported ES384":       {signES256(t, ecKey, `{"alg":"ES384","kid":"ec-key"}`, claims), jwks(ecJWK)},
		"unsupported PS256":       {signFixture(t, rsaKey, `{"alg":"PS256","kid":"fixture-key"}`, claims), jwks(rsaJWK)},
		"key alg mismatch":        {good, jwks(map[string]string{"kty": "EC", "crv": "P-256", "alg": "RS256", "kid": "ec-key", "x": coordinate(x), "y": coordinate(y)})},
		"encryption key":          {good, jwks(map[string]string{"kty": "EC", "crv": "P-256", "use": "enc", "kid": "ec-key", "x": coordinate(x), "y": coordinate(y)})},
		"duplicate kid in JWKS":   {good, jwks(ecJWK, duplicate)},
		"jku header not followed": {signES256(t, other, `{"alg":"ES256","kid":"ec-key","jku":"https://evil.example/jwks"}`, claims), jwks(ecJWK)},
	}
	for name, tc := range rejected {
		if _, err := verifyIdentity(context.Background(), tc.client, tc.token, "oaiapp_fixture", "nonce"); !errors.Is(err, ErrIdentity) {
			t.Errorf("%s: err = %v, want ErrIdentity", name, err)
		}
	}
}

func TestLoginCallbackBindingAndReplay(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	host, _ := NewHostID()
	var nonce string
	exchanges := 0
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.String() {
		case TokenURL:
			exchanges++
			return fixtureJSON(map[string]any{"access_token": "a", "refresh_token": "r", "token_type": "Bearer", "expires_in": 3600, "scope": scopes, "id_token": fixtureIDToken(t, key, "oaiapp_fixture", nonce, nil)}), nil
		case JWKSURL:
			return fixtureJSON(fixtureJWKS(key)), nil
		}
		return nil, errors.New("unexpected request")
	})}
	statuses := map[string]int{}
	_, registration, err := Login(context.Background(), LoginOptions{HostID: host, HTTPClient: client, OpenBrowser: func(raw string) error {
		u, _ := url.Parse(raw)
		q := u.Query()
		nonce = q.Get("nonce")
		redirect, _ := url.Parse(q.Get("redirect_uri"))
		good := url.Values{"code": {"c"}, "state": {q.Get("state")}, "client_id": {"oaiapp_fixture"}}.Encode()
		send := func(name, method, path, host string) {
			target := *redirect
			target.Path = path
			target.RawQuery = good
			request, _ := http.NewRequest(method, target.String(), nil)
			if host != "" {
				request.Host = host
			}
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			_ = response.Body.Close()
			statuses[name] = response.StatusCode
		}
		send("wrong method", http.MethodPost, "/auth/callback", "")
		send("wrong path", http.MethodGet, "/callback", "")
		send("wrong host", http.MethodGet, "/auth/callback", "localhost:"+redirect.Port())
		send("dns rebinding host", http.MethodGet, "/auth/callback", "evil.example:"+redirect.Port())
		send("accepted", http.MethodGet, "/auth/callback", "")
		send("replay", http.MethodGet, "/auth/callback", "")
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"wrong method": 400, "wrong path": 400, "wrong host": 400, "dns rebinding host": 400, "accepted": 200, "replay": http.StatusConflict} {
		if statuses[name] != want {
			t.Errorf("%s: status %d, want %d", name, statuses[name], want)
		}
	}
	if exchanges != 1 || registration.ClientID != "oaiapp_fixture" {
		t.Fatalf("exchanges = %d, want exactly one code exchange", exchanges)
	}
}
