// Package subscription contains provider-neutral, fixture-only contracts for
// credential-backed subscription transports. It deliberately knows no real
// provider origin, OAuth registration, account identity, or model catalogue.
package subscription

import (
	"bytes"
	"context"
	"crypto/x509"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

const (
	defaultFixtureModel = "fixture-model"
	maxFixtureRequest   = 1 << 20
)

// fixtureFiles deliberately contains only bounded, checked-in wire examples.
// No provider response is fetched at test time.
//
//go:embed testdata/*
var fixtureFiles embed.FS

// Constructor builds one provider-bound client for a model and fixture source.
// The fixture server is passed explicitly so a provider test can provide its
// TLS roots and any transport-only test options without the contract runner
// knowing provider constructor details.
type Constructor func(model.Model, credentials.Source, *Server) (inference.Client, error)

// SourceFactory creates the explicit credential source used for one contract
// case. Tests normally return a deterministic fixture source; production
// credential stores and provider account selection are intentionally outside
// this package.
type SourceFactory func(credentials.Descriptor) (credentials.Source, error)

// Contract describes one provider-neutral subscription transport matrix. The
// values are caller-supplied identity metadata; no provider defaults are
// embedded here.
type Contract struct {
	Provider    string
	Transport   string
	Issuer      string
	Scheme      credentials.Scheme
	Usage       credentials.UsageClass
	Formats     []model.APIFormat
	ModelName   string
	NewSource   SourceFactory
	Constructor Constructor
}

// Witness is the fail-closed certification handle for an enabled subscription
// constructor. Provider package tests should construct one with their explicit
// contract and call Run; a provider is not contract-certified merely because
// its constructor compiles.
type Witness struct {
	contract Contract
}

// NewWitness validates the explicit contract that a provider package test
// intends to certify. It never supplies provider identity, endpoints, or
// model defaults.
func NewWitness(contract Contract) (Witness, error) {
	if err := ValidateContract(contract); err != nil {
		return Witness{}, err
	}
	return Witness{contract: contract}, nil
}

// Validate rechecks the witness contract before a provider test runs it.
func (w Witness) Validate() error { return ValidateContract(w.contract) }

// Run executes the certified provider contract against the supplied fixture.
func (w Witness) Run(t *testing.T, server *Server) {
	t.Helper()
	if err := w.Validate(); err != nil {
		t.Fatal(err)
	}
	Run(t, w.contract, server)
}

// ValidateContract checks the bounded, explicit inputs needed by Run. It does
// not perform network I/O or infer an endpoint, credential, or model.
func ValidateContract(contract Contract) error {
	if strings.TrimSpace(contract.Provider) == "" {
		return errors.New("subscription contract: provider is required")
	}
	if strings.TrimSpace(contract.Transport) == "" {
		return errors.New("subscription contract: transport is required")
	}
	if contract.Scheme == "" || !contract.Scheme.Valid() {
		return errors.New("subscription contract: valid credential scheme is required")
	}
	if contract.Usage == "" || !contract.Usage.Valid() {
		return errors.New("subscription contract: valid credential usage is required")
	}
	if _, err := credentials.NewDescriptor(contract.Provider, contract.Transport, contract.Scheme, contract.Usage, contract.Issuer, "https://fixture.invalid", ""); err != nil {
		return fmt.Errorf("subscription contract: invalid binding: %w", err)
	}
	if len(contract.Formats) == 0 {
		return errors.New("subscription contract: at least one API format is required")
	}
	seenFormats := make(map[model.APIFormat]struct{}, len(contract.Formats))
	for _, format := range contract.Formats {
		if _, ok := seenFormats[format]; ok {
			return fmt.Errorf("subscription contract: duplicate API format %q", format)
		}
		seenFormats[format] = struct{}{}
	}
	if contract.NewSource == nil {
		return errors.New("subscription contract: source factory is required")
	}
	if contract.Constructor == nil {
		return errors.New("subscription contract: constructor is required")
	}
	return nil
}

// Server is a loopback-only TLS provider fixture. The handler and request
// capture are owned by the contract runner; callers only use URL and RootCAs
// when adapting a provider constructor.
type Server struct {
	tls                 *httptest.Server
	mu                  sync.Mutex
	requests            []requestRecord
	mode                fixtureMode
	redirect            string
	remaining           int
	streamStarted       chan struct{}
	streamClosed        chan struct{}
	streamStart         *sync.Once
	streamClose         *sync.Once
	concurrentTarget    int
	concurrentCount     int
	concurrentRelease   chan struct{}
	concurrentAbort     chan struct{}
	concurrentAbortOnce *sync.Once
	errorBodyClosed     chan struct{}
	errorBodyCloseOnce  *sync.Once
}

type fixtureMode uint8

const (
	fixtureModeSuccess fixtureMode = iota
	fixtureModeAuthOnce
	fixtureModeBoundedError
	fixtureModeTrackedError
	fixtureModeRedirect
	fixtureModeStreamHold
	fixtureModeConcurrentAuth
)

type requestRecord struct {
	method string
	path   string
	header http.Header
	body   []byte
}

// NewServer starts a loopback TLS fixture. httptest's generated certificate is
// the only root exposed by RootCAs; callers must opt into it explicitly.
func NewServer(t testing.TB) *Server {
	t.Helper()
	s := &Server{}
	s.tls = httptest.NewTLSServer(http.HandlerFunc(s.handle))
	return s
}

// URL returns the exact loopback TLS origin of the fixture.
func (s *Server) URL() string {
	if s == nil {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tls == nil {
		return ""
	}
	return s.tls.URL
}

// RootCAs returns a fresh pool containing only the fixture's certificate.
func (s *Server) RootCAs() *x509.CertPool {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tls == nil {
		return nil
	}
	roots := x509.NewCertPool()
	roots.AddCert(s.tls.Certificate())
	return roots
}

// Close stops the fixture server. It is safe to call more than once.
func (s *Server) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	tls := s.tls
	s.tls = nil
	s.mu.Unlock()
	if tls == nil {
		return nil
	}
	tls.Close()
	return nil
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(io.LimitReader(r.Body, maxFixtureRequest+1))
	if len(body) > maxFixtureRequest {
		body = body[:maxFixtureRequest]
	}
	record := requestRecord{method: r.Method, path: r.URL.Path, header: r.Header.Clone(), body: append([]byte(nil), body...)}

	s.mu.Lock()
	s.requests = append(s.requests, record)
	mode, target := s.mode, s.redirect
	if mode == fixtureModeAuthOnce && s.remaining > 0 {
		s.remaining--
	} else if mode == fixtureModeAuthOnce {
		mode = fixtureModeSuccess
	}
	var release chan struct{}
	if mode == fixtureModeConcurrentAuth {
		if s.concurrentCount < s.concurrentTarget {
			s.concurrentCount++
			release = s.concurrentRelease
			if s.concurrentCount == s.concurrentTarget {
				close(s.concurrentRelease)
			}
		} else {
			mode = fixtureModeSuccess
		}
	}
	streamStarted, streamClosed := s.streamStarted, s.streamClosed
	streamStart, streamClose := s.streamStart, s.streamClose
	concurrentAbort := s.concurrentAbort
	s.mu.Unlock()
	if release != nil {
		select {
		case <-release:
			mode = fixtureModeAuthOnce
		case <-concurrentAbort:
			return
		case <-r.Context().Done():
			s.abortConcurrent()
			return
		}
	}

	switch mode {
	case fixtureModeAuthOnce:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"code":"authentication_error","message":"fixture-provider-secret"}}`)
	case fixtureModeBoundedError:
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Request-ID", "fixture-request-id")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":"internal_server_error","message":"fixture-provider-secret"}}`+strings.Repeat(" ", 128<<10))
	case fixtureModeTrackedError:
		s.writeTrackedError(w, r)
	case fixtureModeRedirect:
		w.Header().Set("Location", target)
		w.WriteHeader(http.StatusTemporaryRedirect)
	case fixtureModeStreamHold:
		s.writeHeldStream(w, r, record.path, streamStarted, streamClosed, streamStart, streamClose)
	default:
		s.writeSuccess(w, r.URL.Path, body)
	}
}

func (s *Server) writeTrackedError(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", "fixture-request-id")
	w.WriteHeader(http.StatusInternalServerError)
	data := []byte(`{"error":{"code":"internal_server_error","message":"fixture-provider-secret"}}` + strings.Repeat(" ", 64<<10+1024))
	for len(data) > 0 {
		chunk := data
		if len(chunk) > 4096 {
			chunk = chunk[:4096]
		}
		if _, err := w.Write(chunk); err != nil {
			s.closeErrorBody()
			return
		}
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		data = data[len(chunk):]
	}
	select {
	case <-r.Context().Done():
		s.closeErrorBody()
	case <-time.After(2 * time.Second):
	}
}

func (s *Server) writeHeldStream(w http.ResponseWriter, r *http.Request, path string, started, closed chan struct{}, startOnce, closeOnce *sync.Once) {
	name, ok := fixtureName(path, true)
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	data, err := fixtureFiles.ReadFile("testdata/" + name)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	if boundary := bytes.Index(data, []byte("\n\n")); boundary >= 0 {
		data = data[:boundary+2]
	}
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = w.Write(data)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	startOnce.Do(func() { close(started) })
	select {
	case <-r.Context().Done():
	case <-time.After(10 * time.Second):
	}
	closeOnce.Do(func() { close(closed) })
}

func (s *Server) writeSuccess(w http.ResponseWriter, path string, body []byte) {
	var request struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	name, ok := fixtureName(path, request.Stream)
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	data, err := fixtureFiles.ReadFile("testdata/" + name)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	if request.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	_, _ = w.Write(data)
}

func fixtureName(path string, streaming bool) (string, bool) {
	suffix := "_invoke.json"
	if streaming {
		suffix = "_stream.sse"
	}
	switch {
	case path == "/messages" || strings.HasSuffix(path, "/messages"):
		return "anthropic" + suffix, true
	case path == "/chat/completions" || strings.HasSuffix(path, "/chat/completions"):
		return "openai_chat" + suffix, true
	case path == "/responses" || strings.HasSuffix(path, "/responses"):
		return "openai_responses" + suffix, true
	default:
		return "", false
	}
}

func (s *Server) reset() {
	s.mu.Lock()
	s.requests = nil
	s.mode = fixtureModeSuccess
	s.redirect = ""
	s.remaining = 0
	s.streamStarted = nil
	s.streamClosed = nil
	s.streamStart = nil
	s.streamClose = nil
	s.concurrentTarget = 0
	s.concurrentCount = 0
	s.concurrentRelease = nil
	s.concurrentAbort = nil
	s.concurrentAbortOnce = nil
	s.errorBodyClosed = nil
	s.errorBodyCloseOnce = nil
	s.mu.Unlock()
}

func (s *Server) setMode(mode fixtureMode, target string) {
	s.mu.Lock()
	s.mode = mode
	s.redirect = target
	if mode == fixtureModeAuthOnce {
		s.remaining = 1
	} else {
		s.remaining = 0
	}
	s.mu.Unlock()
}

func (s *Server) setStreamHold() {
	s.mu.Lock()
	s.mode = fixtureModeStreamHold
	s.streamStarted = make(chan struct{})
	s.streamClosed = make(chan struct{})
	s.streamStart = &sync.Once{}
	s.streamClose = &sync.Once{}
	s.mu.Unlock()
}

func (s *Server) setTrackedError() {
	s.mu.Lock()
	s.mode = fixtureModeTrackedError
	s.errorBodyClosed = make(chan struct{})
	s.errorBodyCloseOnce = &sync.Once{}
	s.mu.Unlock()
}

func (s *Server) closeErrorBody() {
	s.mu.Lock()
	closed, once := s.errorBodyClosed, s.errorBodyCloseOnce
	s.mu.Unlock()
	if closed != nil && once != nil {
		once.Do(func() { close(closed) })
	}
}

func (s *Server) waitErrorBodyClosed(timeout time.Duration) bool {
	s.mu.Lock()
	closed := s.errorBodyClosed
	s.mu.Unlock()
	if closed == nil {
		return false
	}
	select {
	case <-closed:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (s *Server) setConcurrentAuth(target int) {
	s.mu.Lock()
	s.mode = fixtureModeConcurrentAuth
	s.concurrentTarget = target
	s.concurrentCount = 0
	s.concurrentRelease = make(chan struct{})
	s.concurrentAbort = make(chan struct{})
	s.concurrentAbortOnce = &sync.Once{}
	s.mu.Unlock()
}

func (s *Server) abortConcurrent() {
	s.mu.Lock()
	abort, once := s.concurrentAbort, s.concurrentAbortOnce
	s.mu.Unlock()
	if abort != nil && once != nil {
		once.Do(func() { close(abort) })
	}
}

func (s *Server) waitStreamStarted(timeout time.Duration) bool {
	s.mu.Lock()
	started := s.streamStarted
	s.mu.Unlock()
	if started == nil {
		return false
	}
	select {
	case <-started:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (s *Server) waitStreamClosed(timeout time.Duration) bool {
	s.mu.Lock()
	closed := s.streamClosed
	s.mu.Unlock()
	if closed == nil {
		return false
	}
	select {
	case <-closed:
		return true
	case <-time.After(timeout):
		return false
	}
}

func (s *Server) records() []requestRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]requestRecord, len(s.requests))
	for i, record := range s.requests {
		out[i] = requestRecord{method: record.method, path: record.path, header: record.header.Clone(), body: append([]byte(nil), record.body...)}
	}
	return out
}

// Run executes the complete contract matrix against server.
func Run(t *testing.T, contract Contract, server *Server) {
	t.Helper()
	if err := Execute(contract, server); err != nil {
		t.Fatal(err)
	}
}

// RunContract is a descriptive alias for Run.
func RunContract(t *testing.T, contract Contract, server *Server) {
	Run(t, contract, server)
}

// Execute validates a contract and runs its complete fixture-only matrix.
func Execute(contract Contract, server *Server) error {
	if err := ValidateContract(contract); err != nil {
		return err
	}
	if server == nil || server.URL() == "" {
		return errors.New("subscription contract: fixture server is required")
	}
	if err := validateFixtureOrigin(server.URL()); err != nil {
		return err
	}
	for _, format := range contract.Formats {
		if err := executeFormat(contract, server, format); err != nil {
			return fmt.Errorf("subscription contract %q: %w", format, err)
		}
	}
	return nil
}

func validateFixtureOrigin(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("subscription contract: fixture must be an exact HTTPS origin")
	}
	switch strings.ToLower(u.Hostname()) {
	case "127.0.0.1", "::1", "localhost":
		return nil
	default:
		return errors.New("subscription contract: fixture must be loopback-only")
	}
}

func executeFormat(contract Contract, server *Server, format model.APIFormat) error {
	if format != model.APIFormatAnthropic && format != model.APIFormatOpenAI && format != model.APIFormatOpenAIResponses {
		return fmt.Errorf("unsupported ingress API format %q", format)
	}
	name := contract.ModelName
	if name == "" {
		name = defaultFixtureModel
	}
	maxTokens := 256
	selected := model.CustomModel(model.ProviderName(contract.Provider), format, server.URL(), name,
		model.WithTools(), model.WithThinkingDialect(model.ThinkingDialectAdaptive), model.WithImages(), model.WithPromptCaching(),
		model.WithStructuredOutputWithTools(), model.WithSampling(model.Sampling{MaxTokens: &maxTokens, Effort: model.EffortHigh}),
	)
	descriptor, err := credentials.NewDescriptor(contract.Provider, contract.Transport, contract.Scheme, contract.Usage, contract.Issuer, server.URL(), "")
	if err != nil {
		return fmt.Errorf("descriptor: %w", err)
	}
	request := featureRequestForContract(selected)

	// The normal client covers Invoke, Stream, and exact endpoint pinning.
	server.reset()
	normal, tracked, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return err
	}
	defer tracked.Close()
	server.setMode(fixtureModeSuccess, "")
	response, err := normal.Invoke(context.Background(), request)
	if err != nil {
		return fmt.Errorf("invoke: %w", err)
	}
	if err := validateResponse(format, response); err != nil {
		return fmt.Errorf("invoke response: %w", err)
	}
	records := server.records()
	if len(records) != 1 {
		return fmt.Errorf("invoke wire requests = %d, want 1", len(records))
	}
	if err := validateWireRequest(format, records[0].body, false); err != nil {
		return fmt.Errorf("invoke request: %w", err)
	}

	server.reset()
	server.setMode(fixtureModeSuccess, "")
	reader, err := normal.Stream(context.Background(), request)
	if err != nil {
		return fmt.Errorf("stream: %w", err)
	}
	chunks, result, ok, err := consumeFixtureStream(reader)
	if err != nil {
		return fmt.Errorf("stream read: %w", err)
	}
	if !ok {
		return errors.New("stream result missing")
	}
	if err := validateChunksAndResult(format, chunks, result); err != nil {
		return fmt.Errorf("stream result: %w", err)
	}
	records = server.records()
	if len(records) != 1 {
		return fmt.Errorf("stream wire requests = %d, want 1", len(records))
	}
	if err := validateWireRequest(format, records[0].body, true); err != nil {
		return fmt.Errorf("stream request: %w", err)
	}

	// Streaming non-2xx responses are bounded and sanitized before a reader is
	// returned, just like Invoke failures.
	server.reset()
	server.setMode(fixtureModeBoundedError, "")
	if _, err := normal.Stream(context.Background(), request); err == nil {
		return errors.New("stream bounded error: Stream returned nil")
	} else if err := validateFixtureAPIError(err, http.StatusInternalServerError, "internal_server_error", "fixture-request-id"); err != nil {
		return fmt.Errorf("stream bounded error: %w", err)
	}

	// A request Model that changes the bound endpoint must fail before any wire
	// request. This checks exact endpoint pinning independently of redirects.
	server.reset()
	wrongModel := request.Model.Clone()
	wrongModel.BaseURL = server.URL() + "/other-origin"
	wrongRequest := request
	wrongRequest.Model = wrongModel
	if _, err := normal.Invoke(context.Background(), wrongRequest); err == nil {
		return errors.New("endpoint pinning: mismatched model endpoint was accepted")
	} else {
		var mismatch *failure.ModelMismatchError
		if !errors.As(err, &mismatch) {
			return fmt.Errorf("endpoint pinning error = %T, want model mismatch", err)
		}
	}
	if got := len(server.records()); got != 0 {
		return fmt.Errorf("endpoint pinning sent %d wire requests", got)
	}

	// Bounded/sanitized API errors use a separate constructor/source so this
	// case cannot accidentally consume the normal call's source generation.
	server.reset()
	errorClient, errorSource, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("bounded-error constructor: %w", err)
	}
	defer errorSource.Close()
	server.setMode(fixtureModeBoundedError, "")
	if _, err := errorClient.Invoke(context.Background(), request); err == nil {
		return errors.New("bounded error: Invoke returned nil")
	} else {
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) {
			return fmt.Errorf("bounded error = %T, want sanitized APIError", err)
		}
		if apiErr.Status != http.StatusInternalServerError || apiErr.Code != "internal_server_error" || apiErr.RequestID != "fixture-request-id" || strings.Contains(err.Error(), "fixture-provider-secret") {
			return fmt.Errorf("bounded error leaked or had wrong status: %v", err)
		}
	}
	server.reset()
	server.setTrackedError()
	if _, err := errorClient.Invoke(context.Background(), request); err == nil {
		return errors.New("tracked Invoke error: Invoke returned nil")
	} else if err := validateFixtureAPIError(err, http.StatusInternalServerError, "internal_server_error", "fixture-request-id"); err != nil {
		return fmt.Errorf("tracked Invoke error: %w", err)
	}
	if !server.waitErrorBodyClosed(2 * time.Second) {
		return errors.New("tracked Invoke error body was not closed")
	}
	server.reset()
	server.setTrackedError()
	if _, err := errorClient.Stream(context.Background(), request); err == nil {
		return errors.New("tracked Stream error: Stream returned nil")
	} else if err := validateFixtureAPIError(err, http.StatusInternalServerError, "internal_server_error", "fixture-request-id"); err != nil {
		return fmt.Errorf("tracked Stream error: %w", err)
	}
	if !server.waitErrorBodyClosed(2 * time.Second) {
		return errors.New("tracked Stream error body was not closed")
	}

	// Redirects are surfaced as a bounded APIError; no credential-bearing
	// request may follow the Location to another origin.
	redirectTarget, targetCalls := newRedirectTarget()
	defer redirectTarget.Close()
	server.reset()
	redirectClient, redirectSource, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("redirect constructor: %w", err)
	}
	defer redirectSource.Close()
	server.setMode(fixtureModeRedirect, redirectTarget.URL)
	if _, err := redirectClient.Invoke(context.Background(), request); err == nil {
		return errors.New("redirect rejection: Invoke returned nil")
	} else {
		var apiErr *failure.APIError
		if !errors.As(err, &apiErr) || apiErr.Status != http.StatusTemporaryRedirect {
			return fmt.Errorf("redirect error = %T %v, want 307 APIError", err, err)
		}
	}
	if got := targetCalls(); got != 0 {
		return fmt.Errorf("redirect target received %d requests", got)
	}
	server.setMode(fixtureModeRedirect, redirectTarget.URL)
	if _, err := redirectClient.Stream(context.Background(), request); err == nil {
		return errors.New("stream redirect rejection: Stream returned nil")
	} else if err := validateFixtureAPIError(err, http.StatusTemporaryRedirect, "", ""); err != nil {
		return fmt.Errorf("stream redirect rejection: %w", err)
	}
	if got := targetCalls(); got != 0 {
		return fmt.Errorf("stream redirect target received %d requests", got)
	}

	if err := exerciseStreamLifecycle(contract, selected, descriptor, server, request); err != nil {
		return err
	}

	// One logical call may perform one recovery wire attempt, but must account
	// for exactly one invalidation/acquire pair. Invoke and Stream each get their
	// own logical call in this matrix.
	server.reset()
	recoveryClient, recoverySource, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("recovery constructor: %w", err)
	}
	defer recoverySource.Close()
	server.setMode(fixtureModeAuthOnce, "")
	if _, err := recoveryClient.Invoke(context.Background(), request); err != nil {
		return fmt.Errorf("invoke recovery: %w", err)
	}
	gets, invalidates := recoverySource.counts()
	if gets != 2 || invalidates != 1 {
		return fmt.Errorf("invoke recovery accounting = acquires %d/invalidate %d, want 2/1", gets, invalidates)
	}
	server.reset()
	server.setMode(fixtureModeAuthOnce, "")
	reader, err = recoveryClient.Stream(context.Background(), request)
	if err != nil {
		return fmt.Errorf("stream recovery: %w", err)
	}
	_, _, _, err = consumeFixtureStream(reader)
	if err != nil {
		return fmt.Errorf("stream recovery read: %w", err)
	}
	gets, invalidates = recoverySource.counts()
	if gets != 4 || invalidates != 2 {
		return fmt.Errorf("stream recovery accounting = acquires %d/invalidate %d, want 4/2", gets, invalidates)
	}

	// Concurrent initial auth failures share one generation invalidation. Every
	// logical call gets at most one recovery wire attempt, with no unbounded
	// retry fan-out.
	const concurrentCalls = 8
	server.reset()
	concurrentClient, concurrentSource, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("concurrent recovery constructor: %w", err)
	}
	defer concurrentSource.Close()
	server.setConcurrentAuth(concurrentCalls)
	concurrentCtx, concurrentCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer concurrentCancel()
	start := make(chan struct{})
	results := make(chan error, concurrentCalls)
	for i := 0; i < concurrentCalls; i++ {
		go func() {
			<-start
			_, callErr := concurrentClient.Invoke(concurrentCtx, request)
			results <- callErr
		}()
	}
	close(start)
	for i := 0; i < concurrentCalls; i++ {
		select {
		case callErr := <-results:
			if callErr != nil {
				concurrentCancel()
				return fmt.Errorf("concurrent recovery call: %w", callErr)
			}
		case <-concurrentCtx.Done():
			concurrentCancel()
			return fmt.Errorf("concurrent recovery deadline: %w", concurrentCtx.Err())
		}
	}
	gets, invalidates = concurrentSource.counts()
	if gets != concurrentCalls*2 || invalidates != 1 {
		return fmt.Errorf("concurrent recovery accounting = acquires %d/invalidate %d, want %d/1", gets, invalidates, concurrentCalls*2)
	}
	if got, want := len(server.records()), concurrentCalls*2; got != want {
		return fmt.Errorf("concurrent recovery wire requests = %d, want %d", got, want)
	}
	return nil
}

func validateFixtureAPIError(err error, status int, code, requestID string) error {
	var apiErr *failure.APIError
	if !errors.As(err, &apiErr) {
		return fmt.Errorf("error = %T, want sanitized APIError", err)
	}
	if apiErr.Status != status || code != "" && apiErr.Code != code || requestID != "" && apiErr.RequestID != requestID || strings.Contains(err.Error(), "fixture-provider-secret") {
		return fmt.Errorf("error leaked or had wrong fields: %v", err)
	}
	return nil
}

func exerciseStreamLifecycle(contract Contract, selected model.Model, descriptor credentials.Descriptor, server *Server, request inference.Request) error {
	client, source, err := constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("stream cancellation constructor: %w", err)
	}
	defer source.Close()
	server.reset()
	server.setStreamHold()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reader, err := client.Stream(ctx, request)
	if err != nil {
		return fmt.Errorf("stream cancellation open: %w", err)
	}
	if !server.waitStreamStarted(2 * time.Second) {
		_ = reader.Close()
		return errors.New("stream cancellation fixture did not start")
	}
	cancel()
	_ = reader.Close()
	if !server.waitStreamClosed(2 * time.Second) {
		return errors.New("stream cancellation did not close response body")
	}

	client, source, err = constructTracked(contract, selected, descriptor, server)
	if err != nil {
		return fmt.Errorf("early-close constructor: %w", err)
	}
	defer source.Close()
	server.reset()
	server.setStreamHold()
	reader, err = client.Stream(context.Background(), request)
	if err != nil {
		return fmt.Errorf("early-close open: %w", err)
	}
	if !server.waitStreamStarted(2 * time.Second) {
		_ = reader.Close()
		return errors.New("early-close fixture did not start")
	}
	nextDone := make(chan error, 1)
	go func() {
		_, nextErr := reader.Next()
		nextDone <- nextErr
	}()
	if err := reader.Close(); err != nil {
		return fmt.Errorf("early-close Close: %w", err)
	}
	if !server.waitStreamClosed(2 * time.Second) {
		return errors.New("early-close did not close response body")
	}
	select {
	case <-nextDone:
	case <-time.After(2 * time.Second):
		return errors.New("early-close left blocked StreamReader.Next")
	}
	return nil
}

// contractSignatureFormat labels this fixture's reasoning signature with the
// dialect under test. A signature is verified by the endpoint that minted it,
// so a matrix fixture reused across dialects cannot share one label: the
// Anthropic and Bedrock encoders refuse a signature that is not theirs, which
// is exactly the protection the label exists to give. Dialects with no
// signature wire field get none and never read the field.
func contractSignatureFormat(format model.APIFormat) string {
	switch format {
	case model.APIFormatAnthropic:
		return "anthropic"
	case model.APIFormatBedrockConverse:
		return "bedrock-converse"
	default:
		return ""
	}
}

func featureRequestForContract(selected model.Model) inference.Request {
	return inference.Request{
		Model:  selected,
		System: "Use the fixture tool and return a structured answer.",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
				&content.TextBlock{Text: "inspect this image"},
				&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte("fixture-image")}},
			}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{
				content.NewSignedThinkingBlock("prior reasoning", "fixture-signature",
					contractSignatureFormat(selected.APIFormat), nil, ""),
				&content.TextBlock{Text: "calling lookup"},
				&content.ToolUseBlock{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"value":"input"}`)},
			}}},
			&content.ToolResultMessage{Message: content.Message{Role: content.RoleTool, Blocks: []content.Block{&content.TextBlock{Text: "tool result"}}}, ToolUseID: "call_1"},
		},
		Tools:      []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`)}},
		Output:     &inference.OutputSchema{Name: "answer", Strict: true, Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)},
		ToolChoice: inference.ToolRequired(),
	}
}

func constructTracked(contract Contract, selected model.Model, descriptor credentials.Descriptor, server *Server) (inference.Client, *trackedSource, error) {
	source, err := contract.NewSource(descriptor)
	if err != nil {
		return nil, nil, fmt.Errorf("source: %w", err)
	}
	if source == nil {
		return nil, nil, errors.New("source factory returned nil")
	}
	tracked := &trackedSource{inner: source}
	client, err := contract.Constructor(selected, tracked, server)
	if err != nil {
		_ = tracked.Close()
		return nil, nil, err
	}
	if client == nil {
		_ = tracked.Close()
		return nil, nil, errors.New("constructor returned nil client")
	}
	return client, tracked, nil
}

func consumeFixtureStream(reader *stream.StreamReader[content.Chunk]) ([]content.Chunk, stream.StreamResult, bool, error) {
	if reader == nil {
		return nil, stream.StreamResult{}, false, errors.New("nil stream reader")
	}
	defer func() { _ = reader.Close() }()
	var chunks []content.Chunk
	for {
		chunk, err := reader.Next()
		if errors.Is(err, io.EOF) {
			result, ok := reader.Result()
			return chunks, result, ok, nil
		}
		if err != nil {
			return nil, stream.StreamResult{}, false, err
		}
		chunks = append(chunks, chunk)
	}
}

func validateResponse(format model.APIFormat, response *inference.Response) error {
	if response == nil || response.Message == nil || len(response.Message.Blocks) < 3 {
		return errors.New("response lacks text/thinking/tool blocks")
	}
	if response.Model != defaultFixtureModel || response.FinishReason != stream.FinishReasonToolUse {
		return fmt.Errorf("model/finish = %q/%q", response.Model, response.FinishReason)
	}
	if response.Usage == nil {
		return errors.New("response usage is nil")
	}
	want := expectedUsage(format)
	if *response.Usage != want {
		return fmt.Errorf("usage = %+v, want %+v", *response.Usage, want)
	}
	return nil
}

func validateChunksAndResult(format model.APIFormat, chunks []content.Chunk, result stream.StreamResult) error {
	var textSeen, thinkingSeen, toolSeen bool
	for _, chunk := range chunks {
		switch typed := chunk.(type) {
		case *content.TextChunk:
			textSeen = textSeen || typed.Text != ""
		case *content.ThinkingChunk:
			thinkingSeen = thinkingSeen || typed.Thinking != ""
		case *content.ToolUseChunk:
			toolSeen = toolSeen || typed.ID != "" || typed.Name != "" || typed.InputJSON != ""
		}
	}
	if !textSeen || !thinkingSeen || !toolSeen {
		return fmt.Errorf("chunks text/thinking/tool = %v/%v/%v", textSeen, thinkingSeen, toolSeen)
	}
	if result.Model != defaultFixtureModel || result.FinishReason != stream.FinishReasonToolUse || result.Usage == nil {
		return errors.New("stream result lacks model/finish/usage")
	}
	want := expectedUsage(format)
	if *result.Usage != want {
		return fmt.Errorf("stream usage = %+v, want %+v", *result.Usage, want)
	}
	return nil
}

func expectedUsage(format model.APIFormat) content.Usage {
	switch format {
	case model.APIFormatAnthropic:
		return content.Usage{InputTokens: 8, OutputTokens: 6, CacheReadTokens: 2, CacheCreationTokens: 1}
	case model.APIFormatOpenAI:
		return content.Usage{InputTokens: 8, OutputTokens: 6, CacheReadTokens: 2, CacheCreationTokens: 1, ReasoningTokens: 2}
	default:
		return content.Usage{InputTokens: 8, OutputTokens: 6, CacheReadTokens: 2, ReasoningTokens: 2}
	}
}

func validateWireRequest(format model.APIFormat, body []byte, streaming bool) error {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return err
	}
	if streaming {
		var got bool
		if err := json.Unmarshal(top["stream"], &got); err != nil || !got {
			return errors.New("stream flag missing")
		}
	}
	raw := string(body)
	for _, marker := range []string{"lookup", "call_1"} {
		if !strings.Contains(raw, marker) {
			return fmt.Errorf("request missing %q", marker)
		}
	}
	switch format {
	case model.APIFormatAnthropic:
		for _, marker := range []string{"\"tools\"", "\"thinking\"", "\"output_config\"", "\"cache_control\"", "\"type\":\"image\""} {
			if !strings.Contains(raw, marker) {
				return fmt.Errorf("anthropic request missing %s", marker)
			}
		}
	case model.APIFormatOpenAI:
		for _, marker := range []string{"\"tools\"", "\"response_format\"", "\"reasoning_effort\"", "\"image_url\""} {
			if !strings.Contains(raw, marker) {
				return fmt.Errorf("openai chat request missing %s", marker)
			}
		}
	case model.APIFormatOpenAIResponses:
		for _, marker := range []string{"\"tools\"", "\"text\"", "\"reasoning\"", "\"input_image\"", "\"store\":false"} {
			if !strings.Contains(raw, marker) {
				return fmt.Errorf("openai responses request missing %s", marker)
			}
		}
	}
	return nil
}

func newRedirectTarget() (*httptest.Server, func() int) {
	var calls sync.Mutex
	count := 0
	target := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Lock()
		count++
		calls.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	return target, func() int {
		calls.Lock()
		defer calls.Unlock()
		return count
	}
}

// trackedSource preserves the dynamic-source recovery interfaces while
// counting the contract's acquire/invalidate calls. It is used by the full
// runner below and intentionally keeps no credential material.
type trackedSource struct {
	inner       credentials.Source
	mu          sync.Mutex
	gets        int
	invalidates int
}

func (s *trackedSource) Reference() credentials.Reference   { return s.inner.Reference() }
func (s *trackedSource) Descriptor() credentials.Descriptor { return s.inner.Descriptor() }
func (s *trackedSource) Acquire(ctx context.Context) (credentials.Lease, error) {
	s.mu.Lock()
	s.gets++
	s.mu.Unlock()
	return s.inner.Acquire(ctx)
}
func (s *trackedSource) Invalidate(ctx context.Context, generation credentials.Generation, failure credentials.Failure) error {
	s.mu.Lock()
	s.invalidates++
	s.mu.Unlock()
	return s.inner.Invalidate(ctx, generation, failure)
}
func (s *trackedSource) Close() error { return s.inner.Close() }
func (s *trackedSource) CanRecover(failure credentials.Failure) bool {
	if recoverable, ok := s.inner.(interface {
		CanRecover(credentials.Failure) bool
	}); ok {
		return recoverable.CanRecover(failure)
	}
	return false
}
func (s *trackedSource) counts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gets, s.invalidates
}

var _ credentials.Source = (*trackedSource)(nil)
