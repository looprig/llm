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
	failure "github.com/looprig/inference/failure"

	"github.com/looprig/inference/codec/conformance"
	geminicodec "github.com/looprig/inference/codec/geminiapi"
	contextcount "github.com/looprig/inference/contextcount"
	model "github.com/looprig/inference/model"
	usage "github.com/looprig/inference/usage"
	"github.com/looprig/llm"
)

// gateCountTokensBody validates the generateContentRequest a countTokens
// envelope wraps against Google's own request schema. countTokens has no
// document of its own in the gate, but its payload IS a GenerateContentRequest
// (plus the model resource name), so the encoder is held to exactly the same
// standard on the counting route as on the inference one — which matters,
// because the counter reuses geminiapi.EncodeRequest verbatim.
func gateCountTokensBody(t *testing.T, body []byte) {
	t.Helper()
	var envelope struct {
		GenerateContentRequest json.RawMessage `json:"generateContentRequest"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Errorf("countTokens body unmarshal error = %v", err)
		return
	}
	if len(envelope.GenerateContentRequest) == 0 {
		t.Errorf("countTokens body = %s, want a generateContentRequest wrapper", body)
		return
	}
	if err := conformance.Validate("gemini", "generate_content_request", envelope.GenerateContentRequest); err != nil {
		t.Errorf("the encoded countTokens payload is not a legal Gemini request: %v", err)
	}
}

// gateSentCountTokens gates the body an httptest handler received. Like
// gateSentRequest in the gemini_test suite it reports non-fatally, because it
// runs on the server's goroutine.
func gateSentCountTokens(t *testing.T, r *http.Request) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read countTokens request body: %v", err)
		return
	}
	gateCountTokensBody(t, body)
}

const counterTestKey auth.APIKey = "AIza-counter-test-key"

var _ contextcount.ContextCounter = (*Counter)(nil)

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

	req := completeCounterRequest("publishers/google:prod/models/gemini 2.5")
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
			want := contextcount.ContextCount{Model: req.Model.Key(), InputTokens: tt.wantTokens, Quality: contextcount.CountQualityExactProvider}
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
			gateCountTokensBody(t, wire.body)
			assertCountGenerateRequest(t, nested, wantBody, "models/"+req.Model.Name)
		})
	}
}

func TestCounterCountRequestModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		modelName string
	}{
		{name: "ordinary model", modelName: "gemini-2.5-flash"},
		{name: "resource-like model", modelName: "publishers/google:prod/models/gemini 2.5"},
		{name: "JSON escaped model", modelName: "gemini-\"quoted\"\\branch<&"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			req := completeCounterRequest(tt.modelName)
			codecBody, err := geminicodec.EncodeRequest(req)
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			bodyCh := make(chan []byte, 1)
			pathCh := make(chan string, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				body, _ := io.ReadAll(request.Body)
				bodyCh <- body
				pathCh <- request.URL.EscapedPath()
				_, _ = io.WriteString(w, `{"totalTokens":1}`)
			}))
			defer srv.Close()

			_, err = newCounter(counterTestKey, srv.URL).CountContext(context.Background(), req)
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if got, want := <-pathCh, "/models/"+url.PathEscape(tt.modelName)+":countTokens"; got != want {
				t.Errorf("wire route = %q, want %q", got, want)
			}

			body := <-bodyCh
			gateCountTokensBody(t, body)
			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(body, &envelope); err != nil {
				t.Fatalf("countTokens body unmarshal error = %v", err)
			}
			if len(envelope) != 1 {
				t.Fatalf("countTokens wrapper fields = %d, want 1", len(envelope))
			}
			nested, ok := envelope["generateContentRequest"]
			if !ok {
				t.Fatal("countTokens body lacks generateContentRequest")
			}
			assertCountGenerateRequest(t, nested, codecBody, "models/"+tt.modelName)
		})
	}
}

func TestCounterRequestEnvelopeErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body []byte
	}{
		{name: "empty body", body: nil},
		{name: "null body", body: []byte(`null`)},
		{name: "array body", body: []byte(`[]`)},
		{name: "scalar body", body: []byte(`"value"`)},
		{name: "malformed object", body: []byte(`{"contents":`)},
		{name: "multiple values", body: []byte(`{} {}`)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := wrapGenerateContentRequest("model", tt.body)
			if got != nil {
				t.Errorf("wrapGenerateContentRequest() = %s on error, want nil", got)
			}
			var requestErr *CounterRequestError
			if !errors.As(err, &requestErr) {
				t.Fatalf("error = %T, want *CounterRequestError", err)
			}
			if requestErr.Reason != CounterRequestGenerateBodyInvalid {
				t.Errorf("CounterRequestError.Reason = %q, want %q", requestErr.Reason, CounterRequestGenerateBodyInvalid)
			}
		})
	}
}

func TestCounterRequestModelCollision(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body []byte
	}{
		{name: "exact model", body: []byte(`{"model":"existing","contents":[]}`)},
		{name: "uppercase model", body: []byte(`{"MODEL":"existing","contents":[]}`)},
		{name: "title case model", body: []byte(`{"contents":[],"Model":"existing"}`)},
		{name: "duplicate exact model", body: []byte(`{"model":"first","model":"second"}`)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := wrapGenerateContentRequest("requested-model", tt.body)
			if got != nil {
				t.Errorf("wrapGenerateContentRequest() = %s on collision, want nil", got)
			}
			var requestErr *CounterRequestError
			if !errors.As(err, &requestErr) {
				t.Fatalf("error = %T, want *CounterRequestError", err)
			}
			if requestErr.Reason != CounterRequestModelCollision {
				t.Errorf("CounterRequestError.Reason = %q, want %q", requestErr.Reason, CounterRequestModelCollision)
			}
			if strings.Contains(err.Error(), "existing") || strings.Contains(err.Error(), "requested-model") {
				t.Errorf("collision error exposes model data: %q", err)
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
		{name: "wrong provider", mutate: func(req *inference.Request) { req.Model.Provider = model.ProviderName(llm.ProviderChutes) }, wantMM: true},
		{name: "empty model name", mutate: func(req *inference.Request) { req.Model.Name = "" }, wantVal: true},
		{name: "wrong API format", mutate: func(req *inference.Request) { req.Model.APIFormat = model.APIFormatOpenAI }, wantVal: true},
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
				var target *failure.ModelMismatchError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *failure.ModelMismatchError", err)
				}
			}
			if tt.wantVal {
				var target *model.ValidationError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *model.ValidationError", err)
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
		wantNorm   usage.UsageNormalizationReason
	}{
		{name: "malformed JSON", response: `{"totalTokens":`, wantReason: CounterResponseMalformed},
		{name: "missing total tokens", response: `{}`, wantReason: CounterResponseMissingCount},
		{name: "explicit null", response: `{"totalTokens":null}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonNull},
		{name: "fractional", response: `{"totalTokens":1.5}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonFractional},
		{name: "exponent", response: `{"totalTokens":1e3}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonFractional},
		{name: "negative", response: `{"totalTokens":-1}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonNegative},
		{name: "positive out of range", response: `{"totalTokens":9223372036854775808}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonOutOfRange},
		{name: "negative out of range", response: `{"totalTokens":-9223372036854775809}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonOutOfRange},
		{name: "string", response: `{"totalTokens":"private-input"}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonInvalidType},
		{name: "trailing JSON", response: `{"totalTokens":1}{"totalTokens":2}`, wantReason: CounterResponseMalformed},
		{name: "oversized success body", response: strings.Repeat(" ", maxCountResponseBodyBytes+1), wantReason: CounterResponseBodyTooLarge},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gateSentCountTokens(t, r)
				_, _ = io.WriteString(w, tt.response)
			}))
			defer srv.Close()

			got, err := newCounter(counterTestKey, srv.URL).CountContext(context.Background(), counterRequest("gemini-2.5-flash"))
			if got != (contextcount.ContextCount{}) {
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
				var normErr *usage.UsageNormalizationError
				if !errors.As(err, &normErr) {
					t.Fatalf("error = %T, want wrapped *usage.UsageNormalizationError", err)
				}
				if normErr.Field != usage.UsageNormalizationFieldInputTokens || normErr.Reason != tt.wantNorm {
					t.Errorf("normalization error = %+v, want InputTokens/%q", normErr, tt.wantNorm)
				}
			}
		})
	}
}

func TestCounterRejectsDuplicateCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		response string
	}{
		{name: "same value", response: `{"totalTokens":1,"totalTokens":1}`},
		{name: "different values", response: `{"totalTokens":1,"totalTokens":2}`},
		{name: "duplicate around another field", response: `{"totalTokens":1,"ignored":true,"totalTokens":2}`},
		{name: "reverse field order", response: `{"ignored":true,"totalTokens":1,"totalTokens":1}`},
		{name: "mixed case after canonical", response: `{"totalTokens":1,"TOTALTOKENS":2}`},
		{name: "mixed case before canonical", response: `{"TotalTokens":1,"totalTokens":2}`},
		{name: "multiple casing variants", response: `{"totaltokens":1,"tOtAlToKeNs":2,"TOTALTOKENS":3}`},
		{name: "escaped canonical spelling", response: `{"totalTokens":1,"total\u0054okens":2}`},
		{name: "escaped mixed case spelling", response: `{"TOTAL\u0054OKENS":1,"totalTokens":2}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := decodeCountResponse([]byte(tt.response))
			if got != 0 {
				t.Errorf("decodeCountResponse() = %d on duplicate, want zero", got)
			}
			var responseErr *CounterResponseError
			if !errors.As(err, &responseErr) {
				t.Fatalf("error = %T, want *CounterResponseError", err)
			}
			if responseErr.Reason != CounterResponseDuplicateField {
				t.Errorf("CounterResponseError.Reason = %q, want %q", responseErr.Reason, CounterResponseDuplicateField)
			}
			var fieldErr *CounterResponseFieldError
			if !errors.As(err, &fieldErr) {
				t.Fatalf("error chain lacks *CounterResponseFieldError: %v", err)
			}
			if fieldErr.Field != CounterResponseFieldTotalTokens || fieldErr.Reason != CounterResponseFieldDuplicate {
				t.Errorf("CounterResponseFieldError = %+v, want totalTokens/duplicate", fieldErr)
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
				var apiErr *failure.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %T, want *failure.APIError", err)
				}
				if apiErr.Status != tt.wantStatus {
					t.Errorf("APIError.Status = %d, want %d", apiErr.Status, tt.wantStatus)
				}
				if strings.Contains(err.Error(), strings.Repeat("x", 64)) {
					t.Error("API error leaked provider body")
				}
			}
			if tt.wantNet {
				var netErr *failure.NetworkError
				if !errors.As(err, &netErr) {
					t.Fatalf("error = %T, want *failure.NetworkError", err)
				}
				if !errors.Is(err, tt.wantCause) {
					t.Errorf("error = %v, want wrapped cause %v", err, tt.wantCause)
				}
			}
		})
	}
}

func TestCounterStateValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		ctx         context.Context
		build       func(*atomic.Bool) *Counter
		wantReason  CounterStateReason
		wantZeroCap bool
	}{
		{name: "nil receiver", ctx: context.Background(), build: func(*atomic.Bool) *Counter { return nil }, wantReason: CounterStateNilReceiver, wantZeroCap: true},
		{name: "nil context", ctx: nil, build: validCounterWithSpy, wantReason: CounterStateNilContext},
		{name: "missing endpoint", ctx: context.Background(), build: func(called *atomic.Bool) *Counter {
			counter := validCounterWithSpy(called)
			counter.endpoint = ""
			return counter
		}, wantReason: CounterStateMissingEndpoint, wantZeroCap: true},
		{name: "missing authenticator", ctx: context.Background(), build: func(called *atomic.Bool) *Counter {
			counter := validCounterWithSpy(called)
			counter.auth = nil
			return counter
		}, wantReason: CounterStateMissingAuthenticator, wantZeroCap: true},
		{name: "missing HTTP doer", ctx: context.Background(), build: func(called *atomic.Bool) *Counter {
			counter := validCounterWithSpy(called)
			counter.hc = nil
			return counter
		}, wantReason: CounterStateMissingHTTPDoer, wantZeroCap: true},
		{name: "zero timeout", ctx: context.Background(), build: func(called *atomic.Bool) *Counter {
			counter := validCounterWithSpy(called)
			counter.timeout = 0
			return counter
		}, wantReason: CounterStateInvalidTimeout, wantZeroCap: true},
		{name: "negative timeout", ctx: context.Background(), build: func(called *atomic.Bool) *Counter {
			counter := validCounterWithSpy(called)
			counter.timeout = -time.Second
			return counter
		}, wantReason: CounterStateInvalidTimeout, wantZeroCap: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var called atomic.Bool
			counter := tt.build(&called)
			got, err := counter.CountContext(tt.ctx, counterRequest("gemini-2.5-flash"))
			if got != (contextcount.ContextCount{}) {
				t.Errorf("CountContext() = %+v on invalid state, want zero", got)
			}
			var stateErr *CounterStateError
			if !errors.As(err, &stateErr) {
				t.Fatalf("error = %T %v, want *CounterStateError", err, err)
			}
			if stateErr.Reason != tt.wantReason {
				t.Errorf("CounterStateError.Reason = %q, want %q", stateErr.Reason, tt.wantReason)
			}
			if called.Load() {
				t.Error("transport called for invalid counter state")
			}
			if tt.wantZeroCap && counter.CounterCapability() != (contextcount.CounterCapability{}) {
				t.Errorf("CounterCapability() = %+v for invalid counter state, want zero", counter.CounterCapability())
			}
		})
	}
}

func TestCounterTimeout(t *testing.T) {
	t.Parallel()

	const (
		shortTimeout   = 20 * time.Millisecond
		callerTimeout  = 10 * time.Millisecond
		longTimeout    = 250 * time.Millisecond
		safetyRelease  = 250 * time.Millisecond
		maximumElapsed = 150 * time.Millisecond
	)
	tests := []struct {
		name           string
		phase          counterDelayPhase
		counterTimeout time.Duration
		callerTimeout  time.Duration
	}{
		{name: "background context times out waiting for headers", phase: delayResponseHeaders, counterTimeout: shortTimeout},
		{name: "background context times out reading body", phase: delayResponseBody, counterTimeout: shortTimeout},
		{name: "earlier caller deadline wins", phase: delayResponseHeaders, counterTimeout: longTimeout, callerTimeout: callerTimeout},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			release := make(chan struct{})
			timer := time.AfterFunc(safetyRelease, func() { close(release) })
			defer timer.Stop()
			srv := httptest.NewServer(delayedCounterHandler(tt.phase, release))
			defer srv.Close()

			counter := newCounter(counterTestKey, srv.URL)
			counter.timeout = tt.counterTimeout
			ctx := context.Background()
			cancel := func() {}
			if tt.callerTimeout > 0 {
				ctx, cancel = context.WithTimeout(ctx, tt.callerTimeout)
			}
			defer cancel()

			started := time.Now()
			got, err := counter.CountContext(ctx, counterRequest("gemini-2.5-flash"))
			elapsed := time.Since(started)
			if got != (contextcount.ContextCount{}) {
				t.Errorf("CountContext() = %+v on timeout, want zero", got)
			}
			var networkErr *failure.NetworkError
			if !errors.As(err, &networkErr) {
				t.Fatalf("error = %T %v, want *failure.NetworkError", err, err)
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("error = %v, want wrapped context deadline", err)
			}
			if elapsed > maximumElapsed {
				t.Errorf("CountContext() elapsed = %s, want internal/earlier caller deadline under %s", elapsed, maximumElapsed)
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
		got       contextcount.CounterCapability
		wantEqual *contextcount.CounterCapability
		wantDiff  *contextcount.CounterCapability
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
			if tt.got.Provider != contextcount.ProviderID(llm.ProviderGoogle) || tt.got.Transport != contextcount.CounterTransportSameEndpoint || tt.got.Retention != contextcount.RetentionLogged || tt.got.TokenizerRev != counterTokenizerRevision || tt.got.Quality != contextcount.CountQualityExactProvider {
				t.Errorf("CounterCapability() = %+v, want google/same-endpoint/logged/pinned-revision/exact-provider", tt.got)
			}
			if tt.got.SecurityIdentity == (contextcount.SecurityIdentity{}) {
				t.Error("SecurityIdentity is zero")
			}
			if tt.wantEqual != nil && tt.got != *tt.wantEqual {
				t.Errorf("capability = %+v, want deterministic %+v", tt.got, *tt.wantEqual)
			}
			if tt.wantDiff != nil && tt.got.SecurityIdentity == tt.wantDiff.SecurityIdentity {
				t.Error("different endpoint retained the same SecurityIdentity")
			}
			keyDigest := sha256.Sum256([]byte("key-one"))
			if tt.got.SecurityIdentity == contextcount.SecurityIdentity(keyDigest) {
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
				gateSentCountTokens(t, req)
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

func TestCounterEndpointEquivalence(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		left         string
		right        string
		wantEndpoint string
	}{
		{name: "DNS case trailing dot default port and dot path", left: "HTTPS://EXAMPLE.COM.:443/v1beta/./", right: "https://example.com/v1beta", wantEndpoint: "https://example.com/v1beta"},
		{name: "IPv6 compression and default port", left: "https://[0:0:0:0:0:0:0:1]:443/v1beta", right: "https://[::1]/v1beta", wantEndpoint: "https://[::1]/v1beta"},
		{name: "path duplicate and parent segments", left: "https://example.com/v1beta//child/../", right: "https://example.com/v1beta", wantEndpoint: "https://example.com/v1beta"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			left := newCounter(counterTestKey, tt.left)
			right := newCounter(counterTestKey, tt.right)
			if left.endpoint != tt.wantEndpoint || right.endpoint != tt.wantEndpoint {
				t.Errorf("canonical endpoints = (%q, %q), want %q", left.endpoint, right.endpoint, tt.wantEndpoint)
			}
			if left.CounterCapability() != right.CounterCapability() {
				t.Errorf("equivalent endpoints produced different capabilities: left=%+v right=%+v", left.CounterCapability(), right.CounterCapability())
			}
			requestURL := make(chan string, 1)
			left.hc = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				requestURL <- req.URL.String()
				gateSentCountTokens(t, req)
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"totalTokens":1}`)), Header: make(http.Header)}, nil
			})}
			_, err := left.CountContext(context.Background(), counterRequest("gemini-2.5-flash"))
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if got, want := <-requestURL, tt.wantEndpoint+"/models/gemini-2.5-flash:countTokens"; got != want {
				t.Errorf("request URL = %q, want %q", got, want)
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
		{name: "non-ASCII host", endpoint: "https://éxample.com/v1beta", reason: CounterEndpointNonASCIIHost},
		{name: "IPv6 zone", endpoint: "https://[fe80::1%25en0]/v1beta", reason: CounterEndpointInvalidHost},
		{name: "double DNS trailing dot", endpoint: "https://example.com../v1beta", reason: CounterEndpointInvalidHost},
		{name: "ambiguous escaped path", endpoint: "https://example.com/v1%62eta", reason: CounterEndpointAmbiguousPath},
		{name: "invalid named port", endpoint: "https://example.com:bad/v1beta", reason: CounterEndpointMalformed},
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
			if got := counter.CounterCapability(); got != (contextcount.CounterCapability{}) {
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
				if _, ok := client.(contextcount.ContextCounter); ok {
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
		`{"totalTokens":1,"totalTokens":1}`, `{"totalTokens":1,"totalTokens":2}`,
		`{"ignored":true,"totalTokens":1,"totalTokens":1}`,
		`{"totalTokens":1,"TOTALTOKENS":2}`, `{"TotalTokens":1,"totalTokens":2}`,
		`{"totaltokens":1,"tOtAlToKeNs":2,"TOTALTOKENS":3}`,
		`{"totalTokens":1,"total\u0054okens":2}`, `{"TOTAL\u0054OKENS":1,"totalTokens":2}`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, input string) {
		duplicate := hasDuplicateTotalTokens([]byte(input))
		_, err := decodeCountResponse([]byte(input))
		if duplicate {
			var fieldErr *CounterResponseFieldError
			if !errors.As(err, &fieldErr) || fieldErr.Field != CounterResponseFieldTotalTokens || fieldErr.Reason != CounterResponseFieldDuplicate {
				t.Fatalf("duplicate totalTokens error = %T %v, want typed duplicate field chain", err, err)
			}
			return
		}
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
		"HTTPS://EXAMPLE.COM.:443/v1beta/./",
		"https://[0:0:0:0:0:0:0:1]:443/v1beta",
		"http://127.0.0.1:8080/?credential=private#fragment",
		"https://agent:private@example.com/v1beta",
		"http://example.com/v1beta",
		"https://éxample.com/v1beta",
		"https://[fe80::1%25en0]/v1beta",
		"https://example.com/v1%62eta",
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
		if parsed.RawPath != "" || parsed.Path != canonicalEndpointPath(parsed.Path) {
			t.Fatalf("canonical endpoint retains ambiguous path: %q", canonical)
		}
		for _, char := range parsed.Hostname() {
			if char > 127 {
				t.Fatalf("canonical endpoint retains non-ASCII host: %q", canonical)
			}
		}
		again, againErr := canonicalCounterEndpoint(canonical)
		if againErr != nil || again != canonical {
			t.Fatalf("canonicalization is not idempotent: first=%q second=%q error=%v", canonical, again, againErr)
		}
	})
}

func counterRequest(modelName string) inference.Request {
	return inference.Request{
		Model: model.CustomModel(model.ProviderName(llm.ProviderGoogle), model.APIFormatGemini, defaultBaseURL, modelName),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}}},
		},
	}
}

func completeCounterRequest(modelName string) inference.Request {
	temperature := 0.25
	maxTokens := 128
	return inference.Request{
		Model: model.CustomModel(
			model.ProviderName(llm.ProviderGoogle),
			model.APIFormatGemini,
			defaultBaseURL,
			modelName,
			model.WithTools(),
			model.WithSampling(model.Sampling{Temperature: &temperature, MaxTokens: &maxTokens}),
		),
		System: "You are concise.",
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}}}},
			&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "hi"}}}, Usage: &content.Usage{InputTokens: 99, OutputTokens: 1}},
		},
		Tools: []inference.Tool{{Name: "lookup", Description: "look up a value", Schema: []byte(`{"type":"object","properties":{"q":{"type":"string"}}}`)}},
	}
}

func assertCountGenerateRequest(t *testing.T, nested, codecBody []byte, wantModel string) {
	t.Helper()
	var nestedFields map[string]json.RawMessage
	if err := json.Unmarshal(nested, &nestedFields); err != nil {
		t.Fatalf("generateContentRequest unmarshal error = %v", err)
	}
	modelJSON, ok := nestedFields["model"]
	if !ok {
		t.Fatal("generateContentRequest lacks model")
	}
	var gotModel string
	if err := json.Unmarshal(modelJSON, &gotModel); err != nil {
		t.Fatalf("generateContentRequest.model unmarshal error = %v", err)
	}
	if gotModel != wantModel {
		t.Errorf("generateContentRequest.model = %q, want %q", gotModel, wantModel)
	}

	var codecFields map[string]json.RawMessage
	if err := json.Unmarshal(codecBody, &codecFields); err != nil {
		t.Fatalf("codec body unmarshal error = %v", err)
	}
	if len(nestedFields) != len(codecFields)+1 {
		t.Fatalf("generateContentRequest fields = %d, want codec fields %d plus model", len(nestedFields), len(codecFields))
	}
	for field, want := range codecFields {
		got, ok := nestedFields[field]
		if !ok {
			t.Errorf("generateContentRequest lost codec field %q", field)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("generateContentRequest field %q = %s, want byte-equivalent %s", field, got, want)
		}
	}

	escapedModel, err := json.Marshal(wantModel)
	if err != nil {
		t.Fatalf("json.Marshal(model) error = %v", err)
	}
	expected := make([]byte, 0, len(codecBody)+len(escapedModel)+10)
	expected = append(expected, `{"model":`...)
	expected = append(expected, escapedModel...)
	expected = append(expected, ',')
	expected = append(expected, codecBody[1:]...)
	if !bytes.Equal(nested, expected) {
		t.Errorf("generateContentRequest bytes = %s, want lossless deterministic merge %s", nested, expected)
	}
}

func validCounterWithSpy(called *atomic.Bool) *Counter {
	counter := newCounter(counterTestKey, defaultBaseURL)
	counter.hc = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		called.Store(true)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"totalTokens":1}`)), Header: make(http.Header)}, nil
	})}
	return counter
}

func hasDuplicateTotalTokens(body []byte) bool {
	decoder := json.NewDecoder(bytes.NewReader(body))
	start, err := decoder.Token()
	if err != nil {
		return false
	}
	delim, ok := start.(json.Delim)
	if !ok || delim != '{' {
		return false
	}
	count := 0
	for decoder.More() {
		nameToken, tokenErr := decoder.Token()
		if tokenErr != nil {
			return false
		}
		name, ok := nameToken.(string)
		if !ok {
			return false
		}
		var value json.RawMessage
		if decodeErr := decoder.Decode(&value); decodeErr != nil {
			return false
		}
		if strings.EqualFold(name, string(CounterResponseFieldTotalTokens)) {
			count++
		}
	}
	return count > 1
}

type counterDelayPhase uint8

const (
	delayResponseHeaders counterDelayPhase = iota + 1
	delayResponseBody
)

func delayedCounterHandler(phase counterDelayPhase, release <-chan struct{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if phase == delayResponseBody {
			w.Header().Set("Content-Type", contentTypeJSON)
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"totalTokens":`)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
		select {
		case <-req.Context().Done():
			return
		case <-release:
		}
		if phase == delayResponseBody {
			_, _ = io.WriteString(w, `1}`)
			return
		}
		_, _ = io.WriteString(w, `{"totalTokens":1}`)
	})
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

type failingReadCloser struct{ err error }

func (r *failingReadCloser) Read([]byte) (int, error) { return 0, r.err }
func (r *failingReadCloser) Close() error             { return nil }
