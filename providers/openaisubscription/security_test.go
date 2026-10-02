package openaisubscription

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
)

func TestInvokeDecodesToolCalls(t *testing.T) {
	events, err := os.ReadFile("../openai/testdata/responses/stream_function_call.sse")
	if err != nil {
		t.Fatal(err)
	}
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "gpt-test", model.WithTools())
	c, err := New(selected, WithRoundTripper(roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(string(events)))}, nil
	})))
	if err != nil {
		t.Fatal(err)
	}
	response, err := c.InvokeWithAuth(context.Background(), inference.Request{Model: selected}, httpauth.None())
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Message.Blocks) != 1 {
		t.Fatalf("blocks = %#v, want one tool use", response.Message.Blocks)
	}
	tool, ok := response.Message.Blocks[0].(*content.ToolUseBlock)
	if !ok || tool.ID != "call_s2" || tool.Name != "get_weather" || string(tool.Input) != `{"city":"Paris"}` {
		t.Fatalf("tool = %#v", response.Message.Blocks[0])
	}
}

func TestSubscriptionClientIgnoresConfiguredEndpoint(t *testing.T) {
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "https://example.com/v1", "gpt-test")
	if _, err := New(selected); err == nil {
		t.Fatal("constructed a subscription client for a non-canonical endpoint")
	}
	if _, err := New(model.CustomModel("openai", model.APIFormatOpenAIResponses, "", "gpt-test")); err == nil {
		t.Fatal("constructed a subscription client for the metered provider")
	}
}

// signFixture signs arbitrary header/claims with the fixture RSA key.
func signFixture(t *testing.T, key *rsa.PrivateKey, header string, claims map[string]any) string {
	t.Helper()
	raw, _ := json.Marshal(claims)
	input := base64.RawURLEncoding.EncodeToString([]byte(header)) + "." + base64.RawURLEncoding.EncodeToString(raw)
	hash := sha256.Sum256([]byte(input))
	signature, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, hash[:])
	if err != nil {
		t.Fatal(err)
	}
	return input + "." + base64.RawURLEncoding.EncodeToString(signature)
}

func TestIDTokenRejectsUnsafeHeadersAndAudiences(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return fixtureJSON(fixtureJWKS(key)), nil })}
	claims := func() map[string]any {
		return map[string]any{"iss": Issuer, "aud": "oaiapp_fixture", "sub": "subject", "nonce": "nonce", "exp": time.Now().Add(time.Hour).Unix(), "iat": time.Now().Unix()}
	}
	unsigned := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","kid":"fixture-key"}`)) + "." + base64.RawURLEncoding.EncodeToString(func() []byte { raw, _ := json.Marshal(claims()); return raw }()) + "."
	cases := map[string]string{
		"alg none":       unsigned,
		"alg HS256":      signFixture(t, key, `{"alg":"HS256","kid":"fixture-key"}`, claims()),
		"unknown kid":    signFixture(t, key, `{"alg":"RS256","kid":"other-key"}`, claims()),
		"missing kid":    signFixture(t, key, `{"alg":"RS256"}`, claims()),
		"crit header":    signFixture(t, key, `{"alg":"RS256","kid":"fixture-key","crit":["exp"]}`, claims()),
		"wrong key":      signFixture(t, other, `{"alg":"RS256","kid":"fixture-key"}`, claims()),
		"future iat":     signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["iat"] = time.Now().Add(time.Hour).Unix(); return c }()),
		"future nbf":     signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["nbf"] = time.Now().Add(time.Hour).Unix(); return c }()),
		"wrong azp":      signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["azp"] = "oaiapp_other"; return c }()),
		"multi aud":      signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["aud"] = []string{"oaiapp_fixture", "oaiapp_other"}; return c }()),
		"foreign aud":    signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["aud"] = []string{"oaiapp_other"}; return c }()),
		"wrong nonce":    signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); c["nonce"] = "replayed"; return c }()),
		"missing nonce":  signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); delete(c, "nonce"); return c }()),
		"missing exp":    signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any { c := claims(); delete(c, "exp"); return c }()),
		"non-jwt tokens": "a.b",
	}
	for name, token := range cases {
		if _, err := verifyIdentity(context.Background(), client, token, "oaiapp_fixture", "nonce"); !errors.Is(err, ErrIdentity) {
			t.Errorf("%s: err = %v, want ErrIdentity", name, err)
		}
	}
	multi := signFixture(t, key, `{"alg":"RS256","kid":"fixture-key"}`, func() map[string]any {
		c := claims()
		c["aud"] = []string{"oaiapp_fixture", "oaiapp_other"}
		c["azp"] = "oaiapp_fixture"
		return c
	}())
	if _, err := verifyIdentity(context.Background(), client, multi, "oaiapp_fixture", "nonce"); err != nil {
		t.Fatalf("multi-audience token with matching azp: %v", err)
	}
}

func TestRequestJSONRefusesRedirectsErrorsAndOversizedBodies(t *testing.T) {
	cases := map[string]*http.Response{
		"redirect":  {StatusCode: http.StatusFound, Header: http.Header{"Location": []string{"https://evil.example/token"}}, Body: io.NopCloser(strings.NewReader(""))},
		"error":     {StatusCode: http.StatusUnauthorized, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":"secret detail"}`))},
		"oversized": {StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"a":"` + strings.Repeat("x", maxResponseBytes) + `"}`))},
		"malformed": {StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{`))},
	}
	for name, response := range cases {
		requests := 0
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			requests++
			if r.URL.Host != "auth.openai.com" {
				t.Errorf("%s: followed redirect to %s", name, r.URL)
			}
			return response, nil
		})}
		var out map[string]any
		err := requestJSON(context.Background(), client, http.MethodPost, TokenURL, url.Values{"a": {"b"}}, nil, &out)
		if err == nil || requests != 1 {
			t.Errorf("%s: err = %v requests = %d", name, err, requests)
		}
		if err != nil && strings.Contains(err.Error(), "secret detail") {
			t.Errorf("%s: error leaked response body", name)
		}
	}
}

func TestTokenExchangeRequiresPlanPermissionAndBearer(t *testing.T) {
	for name, body := range map[string]map[string]any{
		"no plan scope": {"access_token": "a", "refresh_token": "r", "id_token": "i", "token_type": "Bearer", "expires_in": 3600, "scope": "openid resource.invoke"},
		"no invoke":     {"access_token": "a", "refresh_token": "r", "id_token": "i", "token_type": "Bearer", "expires_in": 3600, "scope": "openid chatgpt.tokens.use.direct"},
	} {
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return fixtureJSON(body), nil })}
		if _, err := exchangeTokens(context.Background(), client, url.Values{}, true); !errors.Is(err, ErrPermission) {
			t.Errorf("%s: err = %v, want ErrPermission", name, err)
		}
	}
	for name, body := range map[string]map[string]any{
		"not bearer":    {"access_token": "a", "refresh_token": "r", "id_token": "i", "token_type": "mac", "expires_in": 3600, "scope": scopes},
		"no refresh":    {"access_token": "a", "id_token": "i", "token_type": "Bearer", "expires_in": 3600, "scope": scopes},
		"no expiry":     {"access_token": "a", "refresh_token": "r", "id_token": "i", "token_type": "Bearer", "scope": scopes},
		"no id on code": {"access_token": "a", "refresh_token": "r", "token_type": "Bearer", "expires_in": 3600, "scope": scopes},
	} {
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return fixtureJSON(body), nil })}
		if _, err := exchangeTokens(context.Background(), client, url.Values{}, true); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// callbackLogin runs Login with a browser that delivers callback(q) and reports
// the status the loopback server returned, then cancels the flow.
func callbackLogin(t *testing.T, previous *Registration, callback func(url.Values) url.Values) (int, int, error) {
	t.Helper()
	host, _ := NewHostID()
	if previous != nil {
		previous.HostID = host
	}
	calls := 0
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { calls++; return fixtureJSON(nil), nil })}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	status := 0
	_, _, err := Login(ctx, LoginOptions{HostID: host, Previous: previous, HTTPClient: client, OpenBrowser: func(raw string) error {
		u, _ := url.Parse(raw)
		q := u.Query()
		redirect, _ := url.Parse(q.Get("redirect_uri"))
		redirect.RawQuery = callback(q).Encode()
		response, err := http.Get(redirect.String())
		if err != nil {
			return err
		}
		_ = response.Body.Close()
		status = response.StatusCode
		go func() { time.Sleep(50 * time.Millisecond); cancel() }()
		return nil
	}})
	return status, calls, err
}

func TestLoginCallbackRejectsForgedOrMismatchedResponses(t *testing.T) {
	cases := map[string]struct {
		previous *Registration
		query    func(url.Values) url.Values
	}{
		"wrong state": {nil, func(q url.Values) url.Values {
			return url.Values{"code": {"c"}, "state": {"forged"}, "client_id": {"oaiapp_fixture"}}
		}},
		"missing state": {nil, func(url.Values) url.Values { return url.Values{"code": {"c"}, "client_id": {"oaiapp_fixture"}} }},
		"duplicate state": {nil, func(q url.Values) url.Values {
			return url.Values{"code": {"c"}, "state": {q.Get("state"), q.Get("state")}, "client_id": {"oaiapp_fixture"}}
		}},
		"unissued client": {nil, func(q url.Values) url.Values {
			return url.Values{"code": {"c"}, "state": {q.Get("state")}, "client_id": {"dynamic_agent_client"}}
		}},
		"missing code": {nil, func(q url.Values) url.Values {
			return url.Values{"state": {q.Get("state")}, "client_id": {"oaiapp_fixture"}}
		}},
		"reauth client switch": {&Registration{ClientID: "oaiapp_fixture", Subject: "subject"}, func(q url.Values) url.Values {
			return url.Values{"code": {"c"}, "state": {q.Get("state")}, "client_id": {"oaiapp_other"}}
		}},
	}
	for name, tc := range cases {
		status, calls, err := callbackLogin(t, tc.previous, tc.query)
		if status != http.StatusBadRequest || calls != 0 || err == nil {
			t.Errorf("%s: status=%d exchanges=%d err=%v", name, status, calls, err)
		}
	}
}

func TestReauthenticationRefusesAccountSwitch(t *testing.T) {
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	host, _ := NewHostID()
	previous := &Registration{ClientID: "oaiapp_fixture", Subject: "original-subject", HostID: host, IDToken: "previous-id-token"}
	var nonce string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.String() {
		case TokenURL:
			return fixtureJSON(map[string]any{"access_token": "a", "refresh_token": "r", "token_type": "Bearer", "expires_in": 3600, "scope": scopes, "id_token": fixtureIDToken(t, key, "oaiapp_fixture", nonce, nil)}), nil
		case JWKSURL:
			return fixtureJSON(fixtureJWKS(key)), nil
		}
		return nil, fmt.Errorf("unexpected request %s", r.URL)
	})}
	_, _, err := Login(context.Background(), LoginOptions{HostID: host, Previous: previous, HTTPClient: client, OpenBrowser: func(raw string) error {
		u, _ := url.Parse(raw)
		q := u.Query()
		nonce = q.Get("nonce")
		if q.Get("client_id") != "oaiapp_fixture" || q.Has("agent_name_hint") || q.Get("id_token_hint") != "previous-id-token" {
			t.Errorf("wrong reauthorization request: %v", q)
		}
		redirect, _ := url.Parse(q.Get("redirect_uri"))
		redirect.RawQuery = url.Values{"code": {"c"}, "state": {q.Get("state")}}.Encode()
		response, err := http.Get(redirect.String())
		if err != nil {
			return err
		}
		return response.Body.Close()
	}})
	if !errors.Is(err, ErrIdentity) {
		t.Fatalf("err = %v, want ErrIdentity for a different subject", err)
	}
}

func TestRegistrationNeverFormatsSecrets(t *testing.T) {
	host, _ := NewHostID()
	r := Registration{ClientID: "oaiapp_fixture", Subject: "secret-subject", Email: "secret@example.test", HostID: host, IDToken: "secret-id-token"}
	for _, rendered := range []string{fmt.Sprint(r), fmt.Sprintf("%+v", r), fmt.Sprintf("%#v", r), r.LogValue().String(), fmt.Sprintf("%v", tokenResponse{AccessToken: "secret-access"})} {
		if strings.Contains(rendered, "secret") {
			t.Fatalf("rendered secret material: %s", rendered)
		}
	}
}
