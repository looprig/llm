package openaisubscription

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/looprig/credentials"
	"github.com/looprig/credentials/oauth"
	"github.com/looprig/credentials/refresh"
	"github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/secrets"
)

const (
	Issuer           = "https://auth.openai.com"
	AuthorizationURL = Issuer + "/api/accounts/authorize"
	TokenURL         = Issuer + "/api/accounts/oauth/token"
	JWKSURL          = Issuer + "/.well-known/jwks.json"
	DiscoveryURL     = Issuer + "/.well-known/openid-configuration"
	scopes           = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
	maxResponseBytes = 1 << 20
	maxErrorBytes    = 64 << 10
)

var (
	ErrLogin      = errors.New("openai-subscription: sign-in failed; retry login")
	ErrIdentity   = errors.New("openai-subscription: identity verification failed")
	ErrPermission = errors.New("openai-subscription: ChatGPT plan usage permission is required; sign in again and allow plan usage")
	ErrToken      = errors.New("openai-subscription: token exchange failed; sign in again")
	// ErrTokenUnavailable is a token-endpoint failure that did not consume the
	// grant (a connection that never sent the request, throttling or a server
	// error). Credentials are preserved; retry with bounded backoff.
	ErrTokenUnavailable = errors.New("openai-subscription: token endpoint temporarily unavailable; retry later")
	// ErrTokenUncertain is a refresh whose request may have reached OpenAI but
	// whose response was lost or unusable. The refresh token may already be
	// rotated, so it is never submitted again; sign in again.
	ErrTokenUncertain = errors.New("openai-subscription: token refresh outcome unknown; sign in again")
	// ErrClient is invalid_client: the saved client registration was
	// rejected. Retrying or refreshing cannot fix it.
	ErrClient = errors.New("openai-subscription: OAuth client registration rejected; check the saved client or sign in again")
	ErrModels = errors.New("openai-subscription: could not retrieve account models")
	ErrRevoke = errors.New("openai-subscription: remote revocation was not confirmed; disconnect Carbon in ChatGPT settings")
)

// Registration is private account continuity, never display metadata. Persist
// it only in protected state. Formatting and logging deliberately redact it.
type Registration struct {
	ClientID string `json:"client_id"`
	Subject  string `json:"subject"`
	Email    string `json:"email,omitempty"`
	HostID   string `json:"host_id"`
	IDToken  string `json:"id_token,omitempty"`
	Scopes   string `json:"scopes,omitempty"`
}

func (r Registration) String() string             { return "openai-subscription: account registration" }
func (r Registration) Format(s fmt.State, _ rune) { _, _ = io.WriteString(s, r.String()) }
func (r Registration) LogValue() slog.Value       { return slog.StringValue(r.String()) }

// AccountName is a stable non-identifying local label, distinct per client and account.
func (r Registration) AccountName() string {
	sum := sha256.Sum256([]byte(Issuer + "\x00" + r.ClientID + "\x00" + r.Subject))
	return "account-" + hex.EncodeToString(sum[:12])
}
func (r Registration) Validate() error {
	if !validClientID(r.ClientID) || r.Subject == "" || len(r.Subject) > 1024 || !validHostID(r.HostID) || len(r.IDToken) > 32<<10 || len(r.Email) > 1024 || len(r.Scopes) > 4096 {
		return ErrIdentity
	}
	return nil
}
func validClientID(value string) bool {
	if !strings.HasPrefix(value, "oaiapp_") || len(value) <= 7 || len(value) > 256 {
		return false
	}
	for _, c := range value {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}
func validHostID(value string) bool {
	if !strings.HasPrefix(value, "urn:uuid:") {
		return false
	}
	value = strings.TrimPrefix(value, "urn:uuid:")
	if len(value) != 36 || value[8] != '-' || value[13] != '-' || value[18] != '-' || value[23] != '-' || value[14] != '4' || !strings.ContainsRune("89ab", rune(value[19])) {
		return false
	}
	_, err := hex.DecodeString(strings.ReplaceAll(value, "-", ""))
	return err == nil
}
func NewHostID() (string, error) {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", ErrLogin
	}
	id[6] = (id[6] & 0x0f) | 0x40
	id[8] = (id[8] & 0x3f) | 0x80
	return fmt.Sprintf("urn:uuid:%x-%x-%x-%x-%x", id[0:4], id[4:6], id[6:8], id[8:10], id[10:16]), nil
}

type LoginOptions struct {
	HostID      string
	Previous    *Registration
	OpenBrowser func(string) error
	HTTPClient  *http.Client
}
type callbackResult struct{ code, clientID string }

// Login runs a bounded browser flow with a stable callback path. It leaves all
// local persistence to the caller and never mutates a selected account on failure.
func Login(ctx context.Context, options LoginOptions) (refresh.State, Registration, error) {
	if ctx == nil {
		return refresh.State{}, Registration{}, credentials.ErrNilContext
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	if !validHostID(options.HostID) || options.OpenBrowser == nil {
		return refresh.State{}, Registration{}, ErrLogin
	}
	if options.Previous != nil {
		if err := options.Previous.Validate(); err != nil {
			return refresh.State{}, Registration{}, err
		}
	}
	pkce, err := oauth.NewPKCE()
	if err != nil {
		return refresh.State{}, Registration{}, ErrLogin
	}
	nonce, err := oauth.NewState()
	if err != nil {
		return refresh.State{}, Registration{}, ErrLogin
	}
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return refresh.State{}, Registration{}, ErrLogin
	}
	redirect := "http://" + listener.Addr().String() + "/auth/callback"
	clientID := "dynamic_agent_client"
	if options.Previous != nil {
		clientID = options.Previous.ClientID
	}
	results := make(chan callbackResult, 1)
	var accepted bool
	var mu sync.Mutex
	server := &http.Server{ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, MaxHeaderBytes: 16 << 10}
	server.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "no-referrer")
		if r.Method != http.MethodGet || r.URL.Path != "/auth/callback" || r.Host != listener.Addr().String() || len(r.URL.RawQuery) > 8192 {
			http.Error(w, "Invalid callback", 400)
			return
		}
		q, err := url.ParseQuery(r.URL.RawQuery)
		if err != nil {
			http.Error(w, "Invalid callback", 400)
			return
		}
		for _, values := range q {
			if len(values) != 1 {
				http.Error(w, "Invalid callback", 400)
				return
			}
		}
		if oauth.ValidateState(pkce.State, q.Get("state")) != nil {
			http.Error(w, "Invalid callback", 400)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if accepted {
			http.Error(w, "Callback already used", http.StatusConflict)
			return
		}
		issued := q.Get("client_id")
		if options.Previous != nil && issued == "" {
			issued = clientID
		}
		if q.Get("error") != "" {
			accepted = true
			results <- callbackResult{}
			http.Error(w, "Sign-in declined", 400)
			return
		}
		if q.Get("code") == "" || len(q.Get("code")) > 4096 || !validClientID(issued) || options.Previous != nil && issued != clientID {
			http.Error(w, "Invalid callback", 400)
			return
		}
		accepted = true
		results <- callbackResult{q.Get("code"), issued}
		_, _ = io.WriteString(w, "Sign-in received. Return to Carbon to complete verification.")
	})
	defer server.Close()
	go func() { _ = server.Serve(listener) }()
	query := url.Values{"client_id": {clientID}, "ext_agent_host_id": {options.HostID}, "response_type": {"code"}, "redirect_uri": {redirect}, "scope": {scopes}, "resource": {BaseURL}, "state": {pkce.State}, "nonce": {nonce}, "code_challenge_method": {"S256"}, "code_challenge": {pkce.Challenge}}
	// A retained ID token is deliberately NOT sent as id_token_hint: this URL
	// becomes a browser-opener process argument, readable by process
	// inspection and command auditing. The hint is optional; without it a
	// returning sign-in shows the account selector and then redirects
	// (https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
	// The returned identity is still checked against Previous.Subject below.
	if options.Previous == nil {
		query.Set("agent_name_hint", "Carbon")
	}
	if err := options.OpenBrowser(AuthorizationURL + "?" + query.Encode()); err != nil {
		return refresh.State{}, Registration{}, ErrLogin
	}
	var callback callbackResult
	select {
	case callback = <-results:
	case <-ctx.Done():
		return refresh.State{}, Registration{}, credentials.NewCanceledError(ctx.Err())
	}
	if callback.code == "" {
		return refresh.State{}, Registration{}, ErrLogin
	}
	tokens, err := exchangeTokens(ctx, options.HTTPClient, url.Values{"grant_type": {"authorization_code"}, "client_id": {callback.clientID}, "code": {callback.code}, "code_verifier": {pkce.Verifier}, "redirect_uri": {redirect}, "resource": {BaseURL}}, true)
	if err != nil {
		return refresh.State{}, Registration{}, err
	}
	identity, err := verifyIdentity(ctx, options.HTTPClient, tokens.IDToken, callback.clientID, nonce)
	if err != nil {
		return refresh.State{}, Registration{}, err
	}
	if options.Previous != nil && identity.Subject != options.Previous.Subject {
		return refresh.State{}, Registration{}, ErrIdentity
	}
	registration := Registration{ClientID: callback.clientID, Subject: identity.Subject, Email: identity.Email, HostID: options.HostID, IDToken: tokens.IDToken, Scopes: tokens.Scope}
	if err := registration.Validate(); err != nil {
		return refresh.State{}, Registration{}, err
	}
	state, err := tokenState(tokens, registration)
	return state, registration, err
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int64  `json:"expires_in"`
	Scope        string `json:"scope"`
}

func (t tokenResponse) String() string             { return "openai-subscription: token response" }
func (t tokenResponse) Format(s fmt.State, _ rune) { _, _ = io.WriteString(s, t.String()) }
func hasPlanPermission(scope string) bool {
	var invoke, plan bool
	for _, s := range strings.Fields(scope) {
		invoke = invoke || s == "resource.invoke"
		plan = plan || s == "chatgpt.tokens.use.direct"
	}
	return invoke && plan
}
func exchangeTokens(ctx context.Context, client *http.Client, form url.Values, initial bool) (tokenResponse, error) {
	var tokens tokenResponse
	if err := requestJSON(ctx, client, http.MethodPost, TokenURL, form, nil, &tokens); err != nil {
		return tokenResponse{}, tokenFailure(err, !initial)
	}
	if !strings.EqualFold(tokens.TokenType, "Bearer") || tokens.AccessToken == "" || tokens.RefreshToken == "" || tokens.ExpiresIn <= 0 || tokens.ExpiresIn > 86400 || len(tokens.AccessToken) > 32<<10 || len(tokens.RefreshToken) > 32<<10 {
		// A 200 means OpenAI processed the grant; a refresh token may have
		// rotated even though this response cannot be used.
		return tokenResponse{}, tokenFailure(&requestFailure{sent: true, invalid: true}, !initial)
	}
	if initial && tokens.IDToken == "" {
		return tokenResponse{}, ErrIdentity
	}
	if initial && !hasPlanPermission(tokens.Scope) {
		return tokenResponse{}, ErrPermission
	}
	return tokens, nil
}
func tokenState(tokens tokenResponse, registration Registration) (refresh.State, error) {
	access, err := secrets.New([]byte(tokens.AccessToken))
	if err != nil {
		return refresh.State{}, ErrToken
	}
	renewable, err := secrets.New([]byte(tokens.RefreshToken))
	if err != nil {
		return refresh.State{}, ErrToken
	}
	generationID, err := oauth.NewState()
	if err != nil {
		return refresh.State{}, ErrToken
	}
	generation, err := credentials.NewGeneration(generationID)
	if err != nil {
		return refresh.State{}, ErrToken
	}
	data, err := json.Marshal(registration)
	if err != nil {
		return refresh.State{}, ErrToken
	}
	state := refresh.State{Schema: refresh.StateSchemaV1, Generation: generation, AccessToken: access, RefreshToken: renewable, ExpiresAt: time.Now().UTC().Add(time.Duration(tokens.ExpiresIn) * time.Second), ProviderData: data, PersistAccessToken: true}
	if err := state.Validate(); err != nil {
		return refresh.State{}, ErrToken
	}
	return state, nil
}
func RegistrationFromState(state refresh.State) (Registration, error) {
	var r Registration
	if err := json.Unmarshal(state.ProviderData, &r); err != nil {
		return r, ErrIdentity
	}
	if err := r.Validate(); err != nil {
		return Registration{}, err
	}
	if !hasPlanPermission(r.Scopes) {
		return Registration{}, ErrPermission
	}
	return r, nil
}

// NewSource adapts provider-owned continuity to the existing coordinated,
// versioned refresh machinery. The coordinator must cover the state store.
func NewSource(ctx context.Context, input credentials.FactoryInput) (credentials.Source, error) {
	if ctx == nil {
		return nil, credentials.ErrNilContext
	}
	if input.Resolver == nil || input.Store == nil || input.RefreshCoordinator == nil {
		return nil, credentials.ErrFactoryConstruction
	}
	selected := model.CustomModel("openai-subscription", model.APIFormatOpenAIResponses, "", "policy")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil || len(policy.Accepted) != 1 {
		return nil, credentials.ErrFactoryConstruction
	}
	expected, err := policy.Accepted[0].Descriptor()
	if err != nil || input.Descriptor.BindingCanonical() != expected.BindingCanonical() {
		return nil, credentials.ErrFactoryUnsupported
	}
	record, err := input.Resolver.Resolve(ctx, input.State)
	if err != nil {
		return nil, err
	}
	state, err := refresh.DecodeState(record.Value)
	if err != nil {
		return nil, err
	}
	if _, err := RegistrationFromState(state); err != nil {
		return nil, err
	}
	return refresh.New(refresh.Options{Context: ctx, Reference: input.Reference, Descriptor: input.Descriptor, State: input.State, Store: input.Store, Resolver: input.Resolver, Preconditions: input.Preconditions, Coordinator: input.RefreshCoordinator, StateSharing: input.StateSharing, Clock: input.Clock, ExpirySkew: time.Minute, PersistAccessToken: true, Exchange: refreshExchange(input.HTTPClient)})
}

// refreshExchange is the provider half of refresh.Source. Every failure is
// classified for the source (see tokenFailure), which records an ambiguous
// or rejected grant durably so no process submits it again.
func refreshExchange(client *http.Client) refresh.ExchangeFunc {
	return func(ctx context.Context, state refresh.State) (refresh.TokenResponse, error) {
		r, err := RegistrationFromState(state)
		if err != nil {
			// Missing plan permission or unusable continuity: only a new
			// sign-in can repair it.
			return refresh.TokenResponse{}, classify(err, refresh.ErrReauthenticationRequired)
		}
		raw := state.RefreshToken.Bytes()
		defer clear(raw)
		tokens, err := exchangeTokens(ctx, client, url.Values{"grant_type": {"refresh_token"}, "client_id": {r.ClientID}, "refresh_token": {string(raw)}, "resource": {BaseURL}}, false)
		if err != nil {
			return refresh.TokenResponse{}, err
		}
		// OAuth permits scope omission on refresh; it retains the original
		// grant. A reduced grant is recorded rather than refused: the
		// rotation already happened, so discarding it would lose the only
		// usable refresh token. RegistrationFromState refuses the stored
		// state with ErrPermission at its next use, which asks for a sign-in
		// that re-grants plan usage (OpenAI: retain the sign-in, mark plan
		// usage disabled).
		if tokens.Scope != "" {
			r.Scopes = tokens.Scope
		}
		// Retain the last verified ID token; a refresh token response is not a
		// new interactive authentication or account switch.
		next, err := tokenState(tokens, r)
		if err != nil {
			return refresh.TokenResponse{}, classify(ErrTokenUncertain, refresh.ErrAmbiguousRotation)
		}
		return refresh.TokenResponse{AccessToken: next.AccessToken, RefreshToken: next.RefreshToken, RefreshTokenSet: true, Generation: next.Generation, ExpiresAt: next.ExpiresAt, ProviderData: next.ProviderData, PersistAccessToken: true}, nil
	}
}

// unusableGrantCodes are OpenAI's documented "clear unusable tokens and
// repeat OAuth" refresh errors:
// https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery
var unusableGrantCodes = map[string]bool{
	"invalid_grant": true, "invalid_refresh_token": true, "token_expired": true,
	"refresh_token_expired": true, "refresh_token_invalidated": true, "refresh_token_reused": true,
}

// tokenFailure maps a token-endpoint failure to a public error plus the
// credentials/refresh class that drives recovery. refreshing is false for the
// interactive authorization-code exchange, where nothing is reused.
func tokenFailure(err error, refreshing bool) error {
	var failure *requestFailure
	if !errors.As(err, &failure) {
		return err
	}
	switch {
	case failure.status != 0:
		switch {
		case unusableGrantCodes[failure.code]:
			return classify(ErrToken, refresh.ErrReauthenticationRequired)
		case failure.code == "invalid_client":
			return ErrClient
		case failure.status == http.StatusTooManyRequests || failure.status >= 500:
			return classify(ErrTokenUnavailable, refresh.ErrExchangeTemporary)
		default:
			return ErrToken
		}
	case failure.sent && refreshing:
		// The grant may have been consumed; this takes precedence over a
		// cancellation that arrived after the request was sent.
		return classify(ErrTokenUncertain, refresh.ErrAmbiguousRotation, failure.canceled)
	case failure.canceled != nil:
		return credentials.NewCanceledError(failure.canceled)
	case failure.sent:
		// An authorization code is single-use; a lost response means login
		// must start over.
		return ErrToken
	default:
		return classify(ErrTokenUnavailable, refresh.ErrExchangeTemporary)
	}
}

// classifiedError keeps a provider-facing message while matching the
// recovery classes it carries with errors.Is.
type classifiedError struct {
	public  error
	classes []error
}

func classify(public error, classes ...error) error {
	e := &classifiedError{public: public}
	for _, class := range classes {
		if class != nil {
			e.classes = append(e.classes, class)
		}
	}
	return e
}
func (e *classifiedError) Error() string { return e.public.Error() }
func (e *classifiedError) Unwrap() []error {
	return append([]error{e.public}, e.classes...)
}

// requestFailure is requestJSON's bounded failure description. It never
// retains a response body, header, or the request.
type requestFailure struct {
	// sent reports that the request may have reached the server.
	sent bool
	// status and code describe a definitive non-200 response; code is set
	// only when it is a short machine token.
	status int
	code   string
	// invalid is a 200 whose body could not be read or decoded.
	invalid bool
	// canceled is the caller's context error when it ended the request.
	canceled error
}

func (f *requestFailure) Error() string {
	if f.status != 0 {
		return fmt.Sprintf("openai-subscription: request failed with status %d", f.status)
	}
	return "openai-subscription: request failed"
}

// requestJSON never follows provider redirects or retains response bodies in
// errors, and bounds both time and response size. Failures are a
// *requestFailure that records whether the request may have been sent.
func requestJSON(ctx context.Context, client *http.Client, method, endpoint string, form url.Values, auth func(*http.Request) error, out any) error {
	if ctx == nil {
		return credentials.ErrNilContext
	}
	if err := ctx.Err(); err != nil {
		return &requestFailure{canceled: err}
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	var wrote atomic.Bool
	traceCtx := httptrace.WithClientTrace(timeoutCtx, &httptrace.ClientTrace{
		WroteRequest: func(httptrace.WroteRequestInfo) { wrote.Store(true) },
	})
	var body io.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	}
	req, err := http.NewRequestWithContext(traceCtx, method, endpoint, body)
	if err != nil {
		return &requestFailure{}
	}
	if form != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	if auth != nil {
		if err := auth(req); err != nil {
			return &requestFailure{}
		}
	}
	hc := http.Client{Timeout: 30 * time.Second}
	if client != nil {
		hc = *client
	}
	hc.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := hc.Do(req)
	if err != nil {
		return &requestFailure{sent: wrote.Load() || !reportsWriteProgress(hc.Transport) && !failedBeforeSending(err), canceled: ctx.Err()}
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBytes))
		defer clear(raw)
		return &requestFailure{sent: true, status: response.StatusCode, code: oauthErrorCode(raw)}
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil || len(raw) > maxResponseBytes {
		return &requestFailure{sent: true, invalid: true, canceled: ctx.Err()}
	}
	defer clear(raw)
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return &requestFailure{sent: true, invalid: true}
	}
	return nil
}

// reportsWriteProgress reports whether the transport calls httptrace's
// WroteRequest, so its absence proves the request was not sent. A
// caller-owned RoundTripper gives no such guarantee.
func reportsWriteProgress(rt http.RoundTripper) bool {
	if rt == nil {
		return true
	}
	_, ok := rt.(*http.Transport)
	return ok
}

// failedBeforeSending recognizes failures that precede any request byte: name
// resolution and connection establishment.
func failedBeforeSending(err error) bool {
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	var opErr *net.OpError
	return errors.As(err, &opErr) && opErr.Op == "dial"
}

// oauthErrorCode extracts a bounded OAuth error code from either the RFC 6749
// shape {"error":"invalid_grant"} or a nested {"error":{"code":"..."}}.
func oauthErrorCode(raw []byte) string {
	var body struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(raw, &body) != nil || len(body.Error) == 0 {
		return ""
	}
	var code string
	if json.Unmarshal(body.Error, &code) != nil {
		var nested struct {
			Code string `json:"code"`
		}
		if json.Unmarshal(body.Error, &nested) != nil {
			return ""
		}
		code = nested.Code
	}
	return safeToken(code)
}
