package gemini

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	geminicodec "github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/llm"
)

const counterTestKey auth.APIKey = "AIza-counter-test-key"

var _ inference.ContextCounter = (*Counter)(nil)

type counterCapturedRequest struct {
	method      string
	escapedPath string
	rawQuery    string
	apiKey      string
	contentType string
	accept      string
	body        []byte
}

func TestCounterCountContext(t *testing.T) {
	t.Parallel()

	temperature := 0.25
	maxTokens := 128
	req := inference.Request{
		Model: inference.CustomModel(
			inference.ProviderName(llm.ProviderGoogle),
			inference.APIFormatGemini,
			defaultBaseURL,
			"publishers/google:prod/models/gemini 2.5",
			inference.WithTools(),
			inference.WithSampling(inference.Sampling{Temperature: &temperature, MaxTokens: &maxTokens}),
		),
		System: "You are concise.",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "hi"}}}, Usage: &content.Usage{InputTokens: 99, OutputTokens: 1}},
		},
		Tools: []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: []byte(`{"type":"object","properties":{"q":{"type":"string"}}}`)}},
	}
	wantBody, err := geminicodec.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	tests := []struct {
		name       string
		response   string
		wantTokens content.TokenCount
	}{
		{name: "complete request and nonzero count", response: `{"totalTokens":321}`, wantTokens: 321},
		{name: "present zero count", response: `{"totalTokens":0}`, wantTokens: 0},
		{name: "maximum count", response: `{"totalTokens":9223372036854775807}`, wantTokens: 9223372036854775807},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			captured := make(chan counterCapturedRequest, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				captured <- counterCapturedRequest{
					method:      r.Method,
					escapedPath: r.URL.EscapedPath(),
					rawQuery:    r.URL.RawQuery,
					apiKey:      r.Header.Get(apiKeyHeader),
					contentType: r.Header.Get("Content-Type"),
					accept:      r.Header.Get("Accept"),
					body:        body,
				}
				w.Header().Set("Content-Type", contentTypeJSON)
				_, _ = io.WriteString(w, tt.response)
			}))
			defer srv.Close()

			counter := newCounter(counterTestKey, srv.URL)
			got, err := counter.CountContext(context.Background(), req)
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			want := inference.ContextCount{Model: req.Model.Key(), InputTokens: tt.wantTokens, Quality: inference.CountQualityExactProvider}
			if got != want {
				t.Errorf("CountContext() = %+v, want %+v", got, want)
			}

			wire := <-captured
			if wire.method != http.MethodPost {
				t.Errorf("method = %q, want POST", wire.method)
			}
			wantPath := "/models/" + "publishers%2Fgoogle:prod%2Fmodels%2Fgemini%202.5" + ":countTokens"
			if wire.escapedPath != wantPath {
				t.Errorf("escaped path = %q, want %q", wire.escapedPath, wantPath)
			}
			if wire.rawQuery != "" {
				t.Errorf("query = %q, want empty", wire.rawQuery)
			}
			if wire.apiKey != string(counterTestKey) {
				t.Errorf("x-goog-api-key = %q, want configured key", wire.apiKey)
			}
			if wire.contentType != contentTypeJSON || wire.accept != contentTypeJSON {
				t.Errorf("content headers = (%q, %q), want application/json", wire.contentType, wire.accept)
			}
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(wire.body, &envelope); err != nil {
				t.Fatalf("countTokens request body unmarshal error = %v", err)
			}
			if len(envelope) != 1 {
				t.Fatalf("countTokens request fields = %d, want only generateContentRequest: %s", len(envelope), wire.body)
			}
			nested, ok := envelope["generateContentRequest"]
			if !ok {
				t.Fatalf("countTokens request = %s, want generateContentRequest wrapper", wire.body)
			}
			if !bytes.Equal(nested, wantBody) {
				t.Errorf("nested generateContentRequest = %s, want byte-identical complete inference body %s", nested, wantBody)
			}
		})
	}
}

func TestCounterPreflight(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		mutate     func(*inference.Request)
		wantMM     bool
		wantVal    bool
		wantEncode bool
	}{
		{name: "wrong provider", mutate: func(req *inference.Request) { req.Model.Provider = inference.ProviderName(llm.ProviderChutes) }, wantMM: true},
		{name: "empty model name", mutate: func(req *inference.Request) { req.Model.Name = "" }, wantVal: true},
		{name: "wrong API format", mutate: func(req *inference.Request) { req.Model.APIFormat = inference.APIFormatOpenAI }, wantVal: true},
		{name: "invalid model URL", mutate: func(req *inference.Request) { req.Model.BaseURL = "http://example.com" }, wantVal: true},
		{name: "nil conversation cannot be encoded", mutate: func(req *inference.Request) { req.Messages = content.AgenticMessages{nil} }, wantEncode: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var called atomic.Bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				called.Store(true)
				_, _ = io.WriteString(w, `{"totalTokens":1}`)
			}))
			defer srv.Close()

			req := counterRequest("gemini-2.5-flash")
			tt.mutate(&req)
			_, err := newCounter(counterTestKey, srv.URL).CountContext(context.Background(), req)
			if err == nil {
				t.Fatal("CountContext() error = nil, want pre-I/O failure")
			}
			if tt.wantMM {
				var target *inference.ModelMismatchError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *inference.ModelMismatchError", err)
				}
			}
			if tt.wantVal {
				var target *inference.ValidationError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *inference.ValidationError", err)
				}
			}
			if tt.wantEncode {
				var target *geminicodec.EncodeError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *geminiapi.EncodeError", err)
				}
			}
			if called.Load() {
				t.Error("provider called despite pre-I/O failure")
			}
		})
	}
}

func TestCounterResponseValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		response   string
		wantReason CounterResponseReason
		wantNorm   inference.UsageNormalizationReason
	}{
		{name: "malformed JSON", response: `{"totalTokens":`, wantReason: CounterResponseMalformed},
		{name: "missing total tokens", response: `{}`, wantReason: CounterResponseMissingCount},
		{name: "explicit null", response: `{"totalTokens":null}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonNull},
		{name: "fractional", response: `{"totalTokens":1.5}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonFractional},
		{name: "exponent", response: `{"totalTokens":1e3}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonFractional},
		{name: "negative", response: `{"totalTokens":-1}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonNegative},
		{name: "positive out of range", response: `{"totalTokens":9223372036854775808}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonOutOfRange},
		{name: "negative out of range", response: `{"totalTokens":-9223372036854775809}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonOutOfRange},
		{name: "string", response: `{"totalTokens":"private-input"}`, wantReason: CounterResponseInvalidCount, wantNorm: inference.UsageNormalizationReasonInvalidType},
		{name: "trailing JSON", response: `{"totalTokens":1}{"totalTokens":2}`, wantReason: CounterResponseMalformed},
		{name: "oversized success body", response: strings.Repeat(" ", maxCountResponseBodyBytes+1), wantReason: CounterResponseBodyTooLarge},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, tt.response)
			}))
			defer srv.Close()

			got, err := newCounter(counterTestKey, srv.URL).CountContext(context.Background(), counterRequest("gemini-2.5-flash"))
			if got != (inference.ContextCount{}) {
				t.Errorf("CountContext() = %+v on error, want zero", got)
			}
			var responseErr *CounterResponseError
			if !errors.As(err, &responseErr) {
				t.Fatalf("error = %T %v, want *CounterResponseError", err, err)
			}
			if responseErr.Reason != tt.wantReason {
				t.Errorf("CounterResponseError.Reason = %q, want %q", responseErr.Reason, tt.wantReason)
			}
			if strings.Contains(err.Error(), "private-input") {
				t.Errorf("error exposes provider scalar: %q", err)
			}
			if tt.wantNorm != "" {
				var normErr *inference.UsageNormalizationError
				if !errors.As(err, &normErr) {
					t.Fatalf("error = %T, want wrapped *inference.UsageNormalizationError", err)
				}
				if normErr.Field != inference.UsageNormalizationFieldInputTokens || normErr.Reason != tt.wantNorm {
					t.Errorf("normalization error = %+v, want InputTokens/%q", normErr, tt.wantNorm)
				}
			}
		})
	}
}

func TestCounterTransportErrors(t *testing.T) {
	t.Parallel()

	readFailure := errors.New("read failed")
	tests := []struct {
		name       string
		client     *http.Client
		ctx        func() (context.Context, context.CancelFunc)
		wantAPI    bool
		wantStatus int
		wantNet    bool
		wantCause  error
		wantBody   int
	}{
		{
			name: "bounded non-2xx API body",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusTooManyRequests, Body: io.NopCloser(strings.NewReader(strings.Repeat("x", maxAPIErrorBodyBytes+128))), Header: make(http.Header)}, nil
			})},
			ctx:     func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			wantAPI: true, wantStatus: http.StatusTooManyRequests, wantBody: maxAPIErrorBodyBytes,
		},
		{
			name: "network failure",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, readFailure
			})},
			ctx:     func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			wantNet: true, wantCause: readFailure,
		},
		{
			name: "response read failure",
			client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: &failingReadCloser{err: readFailure}, Header: make(http.Header)}, nil
			})},
			ctx:     func() (context.Context, context.CancelFunc) { return context.WithCancel(context.Background()) },
			wantNet: true, wantCause: readFailure,
		},
		{
			name: "already canceled context",
			client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				<-req.Context().Done()
				return nil, req.Context().Err()
			})},
			ctx: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, func() {}
			},
			wantNet: true, wantCause: context.Canceled,
		},
		{
			name: "deadline",
			client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				<-req.Context().Done()
				return nil, req.Context().Err()
			})},
			ctx: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Nanosecond)
			},
			wantNet: true, wantCause: context.DeadlineExceeded,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			ctx, cancel := tt.ctx()
			defer cancel()
			counter := newCounter(counterTestKey, "http://127.0.0.1")
			counter.hc = tt.client
			_, err := counter.CountContext(ctx, counterRequest("gemini-2.5-flash"))
			if tt.wantAPI {
				var apiErr *inference.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %T, want *inference.APIError", err)
				}
				if apiErr.Status != tt.wantStatus || len(apiErr.Body) != tt.wantBody || apiErr.Message != string(apiErr.Body) {
					t.Errorf("APIError = status:%d body:%d message:%d, want status:%d body/message:%d", apiErr.Status, len(apiErr.Body), len(apiErr.Message), tt.wantStatus, tt.wantBody)
				}
			}
			if tt.wantNet {
				var netErr *inference.NetworkError
				if !errors.As(err, &netErr) {
					t.Fatalf("error = %T, want *inference.NetworkError", err)
				}
				if !errors.Is(err, tt.wantCause) {
					t.Errorf("error = %v, want wrapped cause %v", err, tt.wantCause)
				}
			}
		})
	}
}

func TestCounterCapability(t *testing.T) {
	t.Parallel()

	const endpoint = "https://generativelanguage.googleapis.com/v1beta/"
	first := newCounter("key-one", endpoint).CounterCapability()
	second := newCounter("key-two", strings.TrimSuffix(endpoint, "/")).CounterCapability()
	differentEndpoint := newCounter("key-one", "https://example.googleapis.com/v1beta").CounterCapability()
	defaultPortEndpoint := newCounter("key-one", "HTTPS://GENERATIVELANGUAGE.GOOGLEAPIS.COM:443/v1beta").CounterCapability()

	tests := []struct {
		name      string
		got       inference.CounterCapability
		wantEqual *inference.CounterCapability
		wantDiff  *inference.CounterCapability
	}{
		{name: "structurally valid", got: first},
		{name: "same endpoint ignores credential", got: second, wantEqual: &first},
		{name: "scheme host and default port normalize", got: defaultPortEndpoint, wantEqual: &first},
		{name: "different endpoint has different identity", got: differentEndpoint, wantDiff: &first},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if err := tt.got.Validate(); err != nil {
				t.Fatalf("CounterCapability().Validate() error = %v", err)
			}
			if tt.got.Provider != inference.ProviderID(llm.ProviderGoogle) || tt.got.Transport != inference.CounterTransportSameEndpoint || tt.got.Retention != inference.RetentionLogged || tt.got.TokenizerRev != counterTokenizerRevision || tt.got.Quality != inference.CountQualityExactProvider {
				t.Errorf("CounterCapability() = %+v, want google/same-endpoint/logged/pinned-revision/exact-provider", tt.got)
			}
			if tt.got.SecurityIdentity == (inference.SecurityIdentity{}) {
				t.Error("SecurityIdentity is zero")
			}
			if tt.wantEqual != nil && tt.got != *tt.wantEqual {
				t.Errorf("capability = %+v, want deterministic %+v", tt.got, *tt.wantEqual)
			}
			if tt.wantDiff != nil && tt.got.SecurityIdentity == tt.wantDiff.SecurityIdentity {
				t.Error("different endpoint retained the same SecurityIdentity")
			}
			keyDigest := sha256.Sum256([]byte("key-one"))
			if tt.got.SecurityIdentity == inference.SecurityIdentity(keyDigest) {
				t.Error("SecurityIdentity is a credential digest")
			}
		})
	}
}

func TestCounterCanonicalEndpointRoute(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		suffix string
	}{
		{name: "canonical endpoint", suffix: ""},
		{name: "trailing slash removed", suffix: "/"},
		{name: "empty query marker removed", suffix: "/?"},
		{name: "query and fragment removed", suffix: "/?credential=private#fragment-secret"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			pathCh := make(chan string, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				pathCh <- req.URL.EscapedPath() + "?" + req.URL.RawQuery
				_, _ = io.WriteString(w, `{"totalTokens":1}`)
			}))
			defer srv.Close()

			base := newCounter(counterTestKey, srv.URL)
			counter := newCounter(counterTestKey, srv.URL+tt.suffix)
			if counter.CounterCapability().SecurityIdentity != base.CounterCapability().SecurityIdentity {
				t.Error("equivalent endpoint produced a different SecurityIdentity")
			}
			_, err := counter.CountContext(context.Background(), counterRequest("gemini-2.5-flash"))
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if got, want := <-pathCh, "/models/gemini-2.5-flash:countTokens?"; got != want {
				t.Errorf("wire route = %q, want canonical %q", got, want)
			}
		})
	}
}

func TestCounterRejectsUnsafeEndpoint(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		endpoint string
		secret   string
		reason   CounterEndpointReason
	}{
		{name: "userinfo credentials", endpoint: "https://agent:private-password@example.com/v1beta", secret: "private-password", reason: CounterEndpointCredentials},
		{name: "non-loopback plaintext", endpoint: "http://example.com/v1beta", reason: CounterEndpointInsecureTransport},
		{name: "missing host", endpoint: "https:///v1beta", reason: CounterEndpointMissingHost},
		{name: "unsupported scheme", endpoint: "ftp://example.com/v1beta", reason: CounterEndpointUnsupportedScheme},
		{name: "malformed URL", endpoint: "://bad-endpoint", reason: CounterEndpointMalformed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var called atomic.Bool
			counter := newCounter(counterTestKey, tt.endpoint)
			counter.hc = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				called.Store(true)
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"totalTokens":1}`)), Header: make(http.Header)}, nil
			})}

			_, err := counter.CountContext(context.Background(), counterRequest("gemini-2.5-flash"))
			if err == nil {
				t.Fatal("CountContext() error = nil, want endpoint rejection")
			}
			var endpointErr *CounterEndpointError
			if !errors.As(err, &endpointErr) {
				t.Fatalf("error = %T, want *CounterEndpointError", err)
			}
			if endpointErr.Reason != tt.reason {
				t.Errorf("CounterEndpointError.Reason = %q, want %q", endpointErr.Reason, tt.reason)
			}
			if called.Load() {
				t.Error("transport called for unsafe endpoint")
			}
			if tt.secret != "" && strings.Contains(err.Error(), tt.secret) {
				t.Errorf("endpoint error exposes credential: %q", err)
			}
			if tt.secret != "" && strings.Contains(counter.endpoint, tt.secret) {
				t.Errorf("counter transport config retains credential: %q", counter.endpoint)
			}
			if got := counter.CounterCapability(); got != (inference.CounterCapability{}) {
				t.Errorf("CounterCapability() = %+v for unsafe endpoint, want zero metadata", got)
			}
		})
	}
}

func TestCounterConstructionAndClientSeparation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		key         auth.APIKey
		wantErr     bool
		wantCounter bool
		checkClient bool
	}{
		{name: "counter construction", key: counterTestKey, wantCounter: true},
		{name: "counter requires credential", key: "", wantErr: true},
		{name: "ordinary client exposes no counter capability", key: counterTestKey, checkClient: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if tt.checkClient {
				client, err := New(tt.key)
				if err != nil {
					t.Fatalf("New() error = %v", err)
				}
				if _, ok := client.(inference.ContextCounter); ok {
					t.Fatal("ordinary Gemini inference client unexpectedly implements ContextCounter")
				}
				return
			}

			counter, err := NewCounter(tt.key)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewCounter() error = %v, wantErr %v", err, tt.wantErr)
			}
			if (counter != nil) != tt.wantCounter {
				t.Errorf("NewCounter() counter present = %v, want %v", counter != nil, tt.wantCounter)
			}
			if tt.wantErr {
				var authErr *llm.AuthRequiredError
				if !errors.As(err, &authErr) {
					t.Fatalf("error = %T, want *llm.AuthRequiredError", err)
				}
			}
		})
	}
}

func FuzzCounterResponse(f *testing.F) {
	seeds := []string{
		`{}`, `{"totalTokens":0}`, `{"totalTokens":null}`, `{"totalTokens":-1}`,
		`{"totalTokens":1.5}`, `{"totalTokens":9223372036854775808}`, `{"totalTokens":"value"}`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, input string) {
		_, err := decodeCountResponse([]byte(input))
		if err == nil {
			return
		}
		var responseErr *CounterResponseError
		if !errors.As(err, &responseErr) {
			t.Fatalf("decodeCountResponse() error = %T %v, want *CounterResponseError", err, err)
		}
	})
}

func FuzzCounterEndpoint(f *testing.F) {
	seeds := []string{
		defaultBaseURL,
		"HTTPS://GENERATIVELANGUAGE.GOOGLEAPIS.COM:443/v1beta/",
		"http://127.0.0.1:8080/?credential=private#fragment",
		"https://agent:private@example.com/v1beta",
		"http://example.com/v1beta",
		"://bad-endpoint",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, input string) {
		canonical, endpointErr := canonicalCounterEndpoint(input)
		if endpointErr != nil {
			if canonical != "" {
				t.Fatalf("canonical endpoint = %q alongside %v", canonical, endpointErr)
			}
			return
		}
		parsed, err := url.Parse(canonical)
		if err != nil {
			t.Fatalf("canonical endpoint %q does not parse: %v", canonical, err)
		}
		if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
			t.Fatalf("canonical endpoint retains unsafe URL metadata: %q", canonical)
		}
		if parsed.Path != "" && strings.HasSuffix(parsed.Path, "/") {
			t.Fatalf("canonical endpoint retains trailing slash: %q", canonical)
		}
		again, againErr := canonicalCounterEndpoint(canonical)
		if againErr != nil || again != canonical {
			t.Fatalf("canonicalization is not idempotent: first=%q second=%q error=%v", canonical, again, againErr)
		}
	})
}

func counterRequest(modelName string) inference.Request {
	return inference.Request{
		Model: inference.CustomModel(inference.ProviderName(llm.ProviderGoogle), inference.APIFormatGemini, defaultBaseURL, modelName),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}}},
		},
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

type failingReadCloser struct{ err error }

func (r *failingReadCloser) Read([]byte) (int, error) { return 0, r.err }
func (r *failingReadCloser) Close() error             { return nil }
