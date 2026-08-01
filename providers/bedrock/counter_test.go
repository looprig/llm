package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	contextcount "github.com/looprig/inference/contextcount"
	failure "github.com/looprig/inference/failure"
	model "github.com/looprig/inference/model"
	usage "github.com/looprig/inference/usage"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auth"
)

var _ contextcount.ContextCounter = (*Counter)(nil)

const counterModelID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

func TestCounterEnvelopeMatchesInvokeBody(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		req  inference.Request
	}{
		{name: "complete request", req: richCounterRequest(counterModelID)},
		{name: "minimal request", req: counterRequest(counterModelID)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var captured []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				captured, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", contentTypeJSON)
				_, _ = io.WriteString(w, `{"inputTokens":37}`)
			}))
			defer srv.Close()

			counter := newCounter(counterTestCreds(), "us-east-1", srv.URL)
			got, err := counter.CountContext(context.Background(), tt.req)
			if err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			if got != (contextcount.ContextCount{Model: tt.req.Model.Key(), InputTokens: 37, Quality: contextcount.CountQualityExactProvider}) {
				t.Errorf("CountContext() = %+v, want exact count 37 for request model", got)
			}

			anthropicBody, err := anthropicapi.EncodeRequest(tt.req, false)
			if err != nil {
				t.Fatalf("anthropicapi.EncodeRequest() error = %v", err)
			}
			wantInvokeBody, err := toBedrockBody(anthropicBody)
			if err != nil {
				t.Fatalf("toBedrockBody() error = %v", err)
			}
			var envelope struct {
				Input struct {
					InvokeModel struct {
						Body string `json:"body"`
					} `json:"invokeModel"`
				} `json:"input"`
			}
			if err := json.Unmarshal(captured, &envelope); err != nil {
				t.Fatalf("count envelope is invalid JSON: %v; body=%s", err, captured)
			}
			decoded, err := base64.StdEncoding.DecodeString(envelope.Input.InvokeModel.Body)
			if err != nil {
				t.Fatalf("invokeModel.body is not base64: %v", err)
			}
			if !bytes.Equal(decoded, wantInvokeBody) {
				t.Errorf("decoded count body = %s, want exact InvokeModel body %s", decoded, wantInvokeBody)
			}
		})
	}
}

func TestCounterConverseEnvelopeUsesNativeUnion(t *testing.T) {
	t.Parallel()

	m := model.CustomModel(
		model.ProviderName(llm.ProviderBedrock),
		model.APIFormatBedrockConverse,
		"",
		counterModelID,
		model.WithTools(),
		model.WithStructuredOutputWithTools(),
	)
	req := inference.Request{
		Model:  m,
		System: "count this",
		Messages: content.AgenticMessages{&content.UserMessage{Message: content.Message{
			Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
		}}},
		Tools:  []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)}},
		Output: &inference.OutputSchema{Name: "answer", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)},
	}
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		_, _ = io.WriteString(w, `{"inputTokens":29}`)
	}))
	defer srv.Close()

	counter := newCounter(counterTestCreds(), "us-east-1", srv.URL)
	got, err := counter.CountContext(context.Background(), req)
	if err != nil {
		t.Fatalf("CountContext() error = %v", err)
	}
	if got.InputTokens != 29 || got.Quality != contextcount.CountQualityExactProvider {
		t.Errorf("CountContext() = %+v, want exact 29", got)
	}
	var envelope struct {
		Input struct {
			Converse    json.RawMessage `json:"converse"`
			InvokeModel json.RawMessage `json:"invokeModel"`
		} `json:"input"`
	}
	if err := json.Unmarshal(captured, &envelope); err != nil {
		t.Fatalf("count envelope JSON = %v", err)
	}
	if len(envelope.Input.Converse) == 0 || string(envelope.Input.InvokeModel) != "" {
		t.Fatalf("input union = converse:%s invokeModel:%s, want Converse only", envelope.Input.Converse, envelope.Input.InvokeModel)
	}
	var converse map[string]json.RawMessage
	if err := json.Unmarshal(envelope.Input.Converse, &converse); err != nil {
		t.Fatalf("input.converse JSON = %v", err)
	}
	for _, field := range []string{"messages", "system", "toolConfig"} {
		if _, ok := converse[field]; !ok {
			t.Errorf("input.converse missing %q", field)
		}
	}
	for _, field := range []string{"inferenceConfig", "outputConfig"} {
		if _, ok := converse[field]; ok {
			t.Errorf("input.converse unexpectedly includes %q", field)
		}
	}
}

func TestCounterRouteAndSigV4(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		modelID   string
		wantPath  string
		session   string
		wantToken bool
	}{
		{name: "colon model id", modelID: counterModelID, wantPath: "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/count-tokens"},
		{name: "slash model id", modelID: "provider/model:v1", wantPath: "/model/provider%2Fmodel:v1/count-tokens"},
		{name: "inference profile ARN", modelID: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/team/model", wantPath: "/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2Fteam%2Fmodel/count-tokens", session: "session-token", wantToken: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			capture := make(chan *http.Request, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capture <- r.Clone(context.Background())
				_, _ = io.WriteString(w, `{"inputTokens":1}`)
			}))
			defer srv.Close()
			creds := counterTestCreds()
			creds.SessionToken = tt.session
			counter := newCounter(creds, "us-east-1", srv.URL)
			if _, err := counter.CountContext(context.Background(), counterRequest(tt.modelID)); err != nil {
				t.Fatalf("CountContext() error = %v", err)
			}
			got := <-capture
			if got.Method != http.MethodPost {
				t.Errorf("method = %q, want POST", got.Method)
			}
			if got.URL.EscapedPath() != tt.wantPath {
				t.Errorf("escaped path = %q, want %q", got.URL.EscapedPath(), tt.wantPath)
			}
			if got.Header.Get("Content-Type") != contentTypeJSON || got.Header.Get("Accept") != contentTypeJSON {
				t.Errorf("JSON headers = Content-Type %q Accept %q", got.Header.Get("Content-Type"), got.Header.Get("Accept"))
			}
			authorization := got.Header.Get("Authorization")
			if !strings.HasPrefix(authorization, "AWS4-HMAC-SHA256 ") || !strings.Contains(authorization, "/us-east-1/bedrock/aws4_request") {
				t.Errorf("Authorization = %q, want Bedrock SigV4 scope", authorization)
			}
			if got.Header.Get("X-Amz-Date") == "" {
				t.Error("X-Amz-Date is missing")
			}
			if present := got.Header.Get("X-Amz-Security-Token") != ""; present != tt.wantToken {
				t.Errorf("security token present = %v, want %v", present, tt.wantToken)
			}
			if strings.Contains(authorization, creds.SecretAccessKey) || creds.SessionToken != "" && strings.Contains(authorization, creds.SessionToken) {
				t.Error("Authorization leaked secret credentials")
			}
		})
	}
}

func TestCounterPreflightRejectsBeforeIO(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		mutate     func(*inference.Request)
		wantModel  bool
		wantValid  bool
		wantFormat bool
	}{
		{name: "provider mismatch", mutate: func(r *inference.Request) { r.Model.Provider = model.ProviderName(llm.ProviderGoogle) }, wantModel: true},
		{name: "missing model name", mutate: func(r *inference.Request) { r.Model.Name = "" }, wantValid: true},
		{name: "model name contains query delimiter", mutate: func(r *inference.Request) { r.Model.Name = "model?credential" }, wantValid: true},
		{name: "model name exceeds AWS limit", mutate: func(r *inference.Request) { r.Model.Name = strings.Repeat("a", 257) }, wantValid: true},
		{name: "unknown API format", mutate: func(r *inference.Request) { r.Model.APIFormat = model.APIFormat("future-bedrock-dialect") }, wantValid: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var calls atomic.Int64
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				_, _ = io.WriteString(w, `{"inputTokens":1}`)
			}))
			defer srv.Close()
			counter := newCounter(counterTestCreds(), "us-east-1", srv.URL)
			req := counterRequest(counterModelID)
			tt.mutate(&req)
			_, err := counter.CountContext(context.Background(), req)
			if tt.wantModel {
				var target *failure.ModelMismatchError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *failure.ModelMismatchError", err)
				}
			}
			if tt.wantValid {
				var target *model.ValidationError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *model.ValidationError", err)
				}
			}
			if tt.wantFormat {
				var target *UnsupportedAPIFormatError
				if !errors.As(err, &target) {
					t.Fatalf("error = %T, want *UnsupportedAPIFormatError", err)
				}
			}
			if calls.Load() != 0 {
				t.Errorf("network calls = %d, want zero", calls.Load())
			}
		})
	}
}

func TestCounterEndpointBinding(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		requestEndpoint func(string) string
	}{
		{name: "equal-looking bound endpoint is still forbidden", requestEndpoint: func(bound string) string { return bound }},
		{name: "different endpoint is forbidden", requestEndpoint: func(string) string { return "https://bedrock-runtime.us-west-2.amazonaws.com" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			counter := newCounter(counterTestCreds(), "us-east-1", defaultEndpoint("us-east-1"))
			var signCalls atomic.Int64
			var doCalls atomic.Int64
			counter.signer = authenticatorFunc(func(context.Context, *http.Request) error {
				signCalls.Add(1)
				return nil
			})
			counter.hc = doerFunc(func(*http.Request) (*http.Response, error) {
				doCalls.Add(1)
				return nil, errors.New("unexpected CountTokens I/O")
			})

			req := counterRequest(counterModelID)
			req.Model.BaseURL = tt.requestEndpoint(counter.endpoint)
			// Audio is intentionally unsupported by the Anthropic encoder. Receiving
			// ModelMismatchError proves endpoint binding runs before body encoding.
			req.Messages[0].(*content.UserMessage).Blocks = []content.Block{
				&content.AudioBlock{MediaType: content.MediaType("audio/wav"), Data: []byte("not encoded")},
			}
			_, err := counter.CountContext(context.Background(), req)
			var mismatch *failure.ModelMismatchError
			if !errors.As(err, &mismatch) {
				t.Fatalf("CountContext() error = %T, want *failure.ModelMismatchError before encoding", err)
			}
			if mismatch.BoundEndpoint != counter.endpoint {
				t.Errorf("BoundEndpoint = %q, want %q", mismatch.BoundEndpoint, counter.endpoint)
			}
			if mismatch.RequestEndpoint != req.Model.BaseURL {
				t.Errorf("RequestEndpoint = %q, want %q", mismatch.RequestEndpoint, req.Model.BaseURL)
			}
			if signCalls.Load() != 0 {
				t.Errorf("Authorize calls = %d, want zero", signCalls.Load())
			}
			if doCalls.Load() != 0 {
				t.Errorf("HTTP Do calls = %d, want zero", doCalls.Load())
			}
		})
	}
}

func TestCounterRequestEnvelopeBodyBounds(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		size       int
		wantReason CounterRequestReason
	}{
		{name: "empty body is accepted", size: 0},
		{name: "maximum body is accepted", size: maxInvokeModelTokensBodyBytes},
		{name: "one byte over maximum is rejected", size: maxInvokeModelTokensBodyBytes + 1, wantReason: CounterRequestBodyTooLarge},
	}
	const emptyEnvelope = `{"input":{"invokeModel":{"body":""}}}`
	const bodySecret = "request-body-secret"
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			invokeBody := make([]byte, tt.size)
			if tt.wantReason != "" {
				copy(invokeBody, bodySecret)
			}
			body, err := buildCountRequestEnvelope(invokeBody)
			if tt.wantReason != "" {
				var requestErr *CounterRequestError
				if !errors.As(err, &requestErr) {
					t.Fatalf("buildCountRequestEnvelope() error = %T, want *CounterRequestError", err)
				}
				if requestErr.Reason != tt.wantReason {
					t.Errorf("CounterRequestError.Reason = %q, want %q", requestErr.Reason, tt.wantReason)
				}
				if body != nil {
					t.Errorf("buildCountRequestEnvelope() body length = %d on rejection, want nil", len(body))
				}
				if strings.Contains(err.Error(), bodySecret) {
					t.Errorf("CounterRequestError leaked request body bytes: %q", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildCountRequestEnvelope() error = %v", err)
			}
			wantLength := len(emptyEnvelope) + base64.StdEncoding.EncodedLen(tt.size)
			if len(body) != wantLength {
				t.Errorf("envelope length = %d, want %d for %d-byte binary body", len(body), wantLength, tt.size)
			}
			if tt.size == 0 && string(body) != emptyEnvelope {
				t.Errorf("empty body envelope = %s, want %s", body, emptyEnvelope)
			}
		})
	}
}

func TestCounterRequestErrorIsSafe(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  *CounterRequestError
	}{
		{name: "zero value", err: &CounterRequestError{}},
		{name: "reason without cause", err: &CounterRequestError{Reason: CounterRequestBodyTooLarge}},
		{name: "encoding cause", err: &CounterRequestError{Reason: CounterRequestEnvelopeEncoding, Err: errors.New("safe encoding cause")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.err.Error()
			if got == "" {
				t.Error("CounterRequestError.Error() is empty")
			}
			if strings.Contains(got, "request-body-secret") {
				t.Errorf("CounterRequestError.Error() leaked body bytes: %q", got)
			}
		})
	}
}

func TestCounterResponseCounts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		body string
		want content.TokenCount
	}{
		{name: "zero", body: `{"inputTokens":0}`, want: 0},
		{name: "ordinary", body: `{"inputTokens":42}`, want: 42},
		{name: "maximum signed integer", body: `{"inputTokens":9223372036854775807}`, want: content.TokenCount(math.MaxInt64)},
		{name: "unknown fields ignored", body: `{"future":true,"inputTokens":9}`, want: 9},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := decodeCountResponse([]byte(tt.body))
			if err != nil {
				t.Fatalf("decodeCountResponse() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("decodeCountResponse() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCounterResponseRejectsAmbiguity(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		body       string
		wantReason CounterResponseReason
		wantNorm   usage.UsageNormalizationReason
		secret     string
	}{
		{name: "missing", body: `{}`, wantReason: CounterResponseMissingCount},
		{name: "null", body: `{"inputTokens":null}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonNull},
		{name: "fractional", body: `{"inputTokens":1.5}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonFractional},
		{name: "exponent", body: `{"inputTokens":1e3}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonFractional},
		{name: "negative", body: `{"inputTokens":-1}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonNegative},
		{name: "positive out of range", body: `{"inputTokens":9223372036854775808}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonOutOfRange},
		{name: "negative out of range", body: `{"inputTokens":-9223372036854775809}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonOutOfRange},
		{name: "string", body: `{"inputTokens":"provider-secret-value"}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonInvalidType, secret: "provider-secret-value"},
		{name: "boolean", body: `{"inputTokens":true}`, wantReason: CounterResponseInvalidCount, wantNorm: usage.UsageNormalizationReasonInvalidType},
		{name: "duplicate exact", body: `{"inputTokens":1,"inputTokens":2}`, wantReason: CounterResponseDuplicateField},
		{name: "duplicate case variant", body: `{"inputTokens":1,"InputTokens":2}`, wantReason: CounterResponseDuplicateField},
		{name: "malformed", body: `{"inputTokens":`, wantReason: CounterResponseMalformed},
		{name: "top-level array", body: `[]`, wantReason: CounterResponseMalformed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := decodeCountResponse([]byte(tt.body))
			var responseErr *CounterResponseError
			if !errors.As(err, &responseErr) {
				t.Fatalf("error = %T, want *CounterResponseError", err)
			}
			if responseErr.Reason != tt.wantReason {
				t.Errorf("reason = %q, want %q", responseErr.Reason, tt.wantReason)
			}
			if tt.wantNorm != "" {
				var norm *usage.UsageNormalizationError
				if !errors.As(err, &norm) {
					t.Fatalf("error chain lacks *usage.UsageNormalizationError: %v", err)
				}
				if norm.Field != usage.UsageNormalizationFieldInputTokens || norm.Reason != tt.wantNorm {
					t.Errorf("normalization = %+v, want InputTokens/%s", norm, tt.wantNorm)
				}
			}
			if tt.secret != "" && strings.Contains(err.Error(), tt.secret) {
				t.Errorf("error leaked provider value: %v", err)
			}
		})
	}
}

func TestCounterHTTPFailures(t *testing.T) {
	t.Parallel()

	providerSecret := strings.Repeat("provider-secret-payload", 5000)
	tests := []struct {
		name       string
		status     int
		body       string
		wantAPI    bool
		wantTooBig bool
	}{
		{name: "unsupported model remains API error", status: http.StatusBadRequest, body: `{"message":"unsupported model"}`, wantAPI: true},
		{name: "forbidden remains API error", status: http.StatusForbidden, body: `{"message":"forbidden"}`, wantAPI: true},
		{name: "large API response is bounded", status: http.StatusBadRequest, body: providerSecret, wantAPI: true},
		{name: "large successful response is rejected", status: http.StatusOK, body: strings.Repeat("x", maxCountResponseBodyBytes+1), wantTooBig: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			defer srv.Close()
			counter := newCounter(counterTestCreds(), "us-east-1", srv.URL)
			_, err := counter.CountContext(context.Background(), counterRequest(counterModelID))
			if tt.wantAPI {
				var apiErr *failure.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %T, want *failure.APIError", err)
				}
				if apiErr.Status != tt.status {
					t.Errorf("status = %d, want %d", apiErr.Status, tt.status)
				}
				if len(apiErr.Body) > maxCountResponseBodyBytes {
					t.Errorf("API body length = %d, exceeds bound", len(apiErr.Body))
				}
				if strings.Contains(err.Error(), "provider-secret-payload") {
					t.Errorf("API error string leaked provider payload")
				}
			}
			if tt.wantTooBig {
				var responseErr *CounterResponseError
				if !errors.As(err, &responseErr) || responseErr.Reason != CounterResponseBodyTooLarge {
					t.Fatalf("error = %v, want body-too-large CounterResponseError", err)
				}
			}
		})
	}
}

func TestCounterContextAndTransportFailures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		configure   func(*Counter) (context.Context, context.CancelFunc)
		slowBody    bool
		wantCause   error
		wantState   CounterStateReason
		wantNetwork bool
	}{
		{name: "pre-canceled caller context", configure: func(_ *Counter) (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx, func() {}
		}, wantCause: context.Canceled, wantNetwork: true},
		{name: "caller deadline wins", configure: func(c *Counter) (context.Context, context.CancelFunc) {
			c.timeout = time.Second
			return context.WithTimeout(context.Background(), 10*time.Millisecond)
		}, wantCause: context.DeadlineExceeded, wantNetwork: true},
		{name: "internal ceiling covers slow header", configure: func(c *Counter) (context.Context, context.CancelFunc) {
			c.timeout = 10 * time.Millisecond
			return context.Background(), func() {}
		}, wantCause: context.DeadlineExceeded, wantNetwork: true},
		{name: "internal ceiling covers slow body", configure: func(c *Counter) (context.Context, context.CancelFunc) {
			c.timeout = 10 * time.Millisecond
			return context.Background(), func() {}
		}, slowBody: true, wantCause: context.DeadlineExceeded, wantNetwork: true},
		{name: "nil context", configure: func(_ *Counter) (context.Context, context.CancelFunc) { return nil, func() {} }, wantState: CounterStateNilContext},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tt.slowBody {
					w.WriteHeader(http.StatusOK)
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
				time.Sleep(100 * time.Millisecond)
				_, _ = io.WriteString(w, `{"inputTokens":1}`)
			}))
			defer srv.Close()
			counter := newCounter(counterTestCreds(), "us-east-1", srv.URL)
			ctx, cancel := tt.configure(counter)
			defer cancel()
			_, err := counter.CountContext(ctx, counterRequest(counterModelID))
			if tt.wantState != "" {
				var stateErr *CounterStateError
				if !errors.As(err, &stateErr) || stateErr.Reason != tt.wantState {
					t.Fatalf("error = %v, want state %q", err, tt.wantState)
				}
				return
			}
			if tt.wantNetwork {
				var networkErr *failure.NetworkError
				if !errors.As(err, &networkErr) {
					t.Fatalf("error = %T, want *failure.NetworkError", err)
				}
			}
			if !errors.Is(err, tt.wantCause) {
				t.Errorf("error = %v, want cause %v", err, tt.wantCause)
			}
		})
	}
}

func TestCounterReadAndNetworkFailures(t *testing.T) {
	t.Parallel()

	errBoom := errors.New("boom")
	tests := []struct {
		name      string
		doer      requestDoer
		wantCause error
	}{
		{name: "network failure", doer: doerFunc(func(*http.Request) (*http.Response, error) { return nil, errBoom }), wantCause: errBoom},
		{name: "response read failure", doer: doerFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: &errorReadCloser{err: errBoom}}, nil
		}), wantCause: errBoom},
		{name: "API response read failure", doer: doerFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusBadRequest, Body: &errorReadCloser{err: errBoom}}, nil
		}), wantCause: errBoom},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			counter := newCounter(counterTestCreds(), "us-east-1", "https://bedrock-runtime.us-east-1.amazonaws.com")
			counter.hc = tt.doer
			_, err := counter.CountContext(context.Background(), counterRequest(counterModelID))
			var networkErr *failure.NetworkError
			if !errors.As(err, &networkErr) {
				t.Fatalf("error = %T, want *failure.NetworkError", err)
			}
			if !errors.Is(err, tt.wantCause) {
				t.Errorf("error = %v, want cause", err)
			}
		})
	}
}

func TestCounterStateIsFailClosed(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		counter    *Counter
		wantReason CounterStateReason
	}{
		{name: "nil receiver", counter: nil, wantReason: CounterStateNilReceiver},
		{name: "zero value", counter: &Counter{}, wantReason: CounterStateMissingEndpoint},
		{name: "missing region", counter: &Counter{endpoint: "https://example.com", signer: auth.SigV4(counterTestCreds(), "us-east-1", bedrockService), hc: http.DefaultClient, timeout: time.Second}, wantReason: CounterStateMissingRegion},
		{name: "missing signer", counter: &Counter{endpoint: "https://example.com", region: "us-east-1", hc: http.DefaultClient, timeout: time.Second}, wantReason: CounterStateMissingAuthenticator},
		{name: "missing HTTP doer", counter: &Counter{endpoint: "https://example.com", region: "us-east-1", signer: auth.SigV4(counterTestCreds(), "us-east-1", bedrockService), timeout: time.Second}, wantReason: CounterStateMissingHTTPDoer},
		{name: "invalid timeout", counter: &Counter{endpoint: "https://example.com", region: "us-east-1", signer: auth.SigV4(counterTestCreds(), "us-east-1", bedrockService), hc: http.DefaultClient}, wantReason: CounterStateInvalidTimeout},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := tt.counter.CountContext(context.Background(), counterRequest(counterModelID))
			var stateErr *CounterStateError
			if !errors.As(err, &stateErr) || stateErr.Reason != tt.wantReason {
				t.Fatalf("error = %v, want state %q", err, tt.wantReason)
			}
			if got := tt.counter.CounterCapability(); got != (contextcount.CounterCapability{}) {
				t.Errorf("capability = %+v, want zero for invalid state", got)
			}
		})
	}
}

func TestCounterCapability(t *testing.T) {
	t.Parallel()

	base := newCounter(counterTestCreds(), "us-east-1", "https://bedrock-runtime.us-east-1.amazonaws.com")
	sameWithSecret := newCounter(auth.SigV4Credentials{AccessKeyID: "different", SecretAccessKey: "different", SessionToken: "different"}, "us-east-1", "HTTPS://BEDROCK-RUNTIME.US-EAST-1.AMAZONAWS.COM:443/")
	differentRegion := newCounter(counterTestCreds(), "us-west-2", "https://bedrock-runtime.us-west-2.amazonaws.com")
	tests := []struct {
		name      string
		counter   *Counter
		wantEqual *Counter
		wantDiff  *Counter
	}{
		{name: "valid exact same-endpoint metadata", counter: base},
		{name: "credentials do not affect identity", counter: sameWithSecret, wantEqual: base},
		{name: "region endpoint changes identity", counter: differentRegion, wantDiff: base},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.counter.CounterCapability()
			if err := got.Validate(); err != nil {
				t.Fatalf("capability Validate() error = %v", err)
			}
			if got.Provider != contextcount.ProviderID(llm.ProviderBedrock) || got.Transport != contextcount.CounterTransportSameEndpoint || got.Retention != contextcount.RetentionLogged || got.Quality != contextcount.CountQualityExactProvider || got.TokenizerRev == "" {
				t.Errorf("capability = %+v, want Bedrock/same-endpoint/logged/exact/pinned", got)
			}
			if tt.wantEqual != nil && got != tt.wantEqual.CounterCapability() {
				t.Errorf("equivalent endpoints/credentials changed capability")
			}
			if tt.wantDiff != nil && got.SecurityIdentity == tt.wantDiff.CounterCapability().SecurityIdentity {
				t.Errorf("different endpoint region has same security identity")
			}
			text := fmt.Sprintf("%+v", got)
			for _, secret := range []string{counterTestCreds().SecretAccessKey, counterTestCreds().AccessKeyID, "different"} {
				if strings.Contains(text, secret) {
					t.Errorf("capability leaked credential %q", secret)
				}
			}
		})
	}
}

func TestCounterConstructionAndClientSeparation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		creds       auth.SigV4Credentials
		region      string
		wantErr     bool
		wantCounter bool
	}{
		{name: "valid", creds: counterTestCreds(), region: "us-east-1", wantCounter: true},
		{name: "empty region", creds: counterTestCreds(), wantErr: true},
		{name: "uppercase region is rejected", creds: counterTestCreds(), region: "US-EAST-1", wantErr: true},
		{name: "region with path delimiter is rejected", creds: counterTestCreds(), region: "us-east-1/credential", wantErr: true},
		{name: "empty access key", creds: auth.SigV4Credentials{SecretAccessKey: "secret"}, region: "us-east-1", wantErr: true},
		{name: "empty secret key", creds: auth.SigV4Credentials{AccessKeyID: "key"}, region: "us-east-1", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := NewCounter(tt.creds, tt.region)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewCounter() error = %v, wantErr %v", err, tt.wantErr)
			}
			if (got != nil) != tt.wantCounter {
				t.Errorf("counter present = %v, want %v", got != nil, tt.wantCounter)
			}
			if tt.wantCounter {
				counter, ok := got.(*Counter)
				if !ok {
					t.Fatalf("NewCounter() result = %T, want *Counter", got)
				}
				if counter.timeout != 60*time.Second {
					t.Errorf("default timeout = %v, want 60s", counter.timeout)
				}
			}
			if tt.wantErr {
				var configErr *ConfigError
				if !errors.As(err, &configErr) {
					t.Fatalf("error = %T, want *ConfigError", err)
				}
			}
			client, clientErr := New(counterTestCreds(), "us-east-1")
			if clientErr != nil {
				t.Fatalf("New() error = %v", clientErr)
			}
			if _, ok := client.(contextcount.ContextCounter); ok {
				t.Fatal("ordinary Client unexpectedly implements ContextCounter")
			}
		})
	}
}

func TestCounterEndpointValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		endpoint   string
		want       string
		wantReason CounterEndpointReason
	}{
		{name: "canonical HTTPS", endpoint: "HTTPS://BEDROCK-RUNTIME.US-EAST-1.AMAZONAWS.COM:443/", want: "https://bedrock-runtime.us-east-1.amazonaws.com"},
		{name: "loopback HTTP", endpoint: "http://127.0.0.1:8080", want: "http://127.0.0.1:8080"},
		{name: "non-loopback plaintext", endpoint: "http://example.com", wantReason: CounterEndpointInsecureTransport},
		{name: "credentials", endpoint: "https://user:secret@example.com", wantReason: CounterEndpointCredentials},
		{name: "query", endpoint: "https://example.com?secret=1", wantReason: CounterEndpointUnexpectedComponent},
		{name: "fragment", endpoint: "https://example.com#secret", wantReason: CounterEndpointUnexpectedComponent},
		{name: "path", endpoint: "https://example.com/v1", wantReason: CounterEndpointUnexpectedComponent},
		{name: "unsupported scheme", endpoint: "ftp://example.com", wantReason: CounterEndpointUnsupportedScheme},
		{name: "missing host", endpoint: "https://", wantReason: CounterEndpointMissingHost},
		{name: "unicode host", endpoint: "https://exämple.com", wantReason: CounterEndpointNonASCIIHost},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := canonicalCounterEndpoint(tt.endpoint)
			if tt.wantReason == "" {
				if err != nil {
					t.Fatalf("canonicalCounterEndpoint() error = %v", err)
				}
				if got != tt.want {
					t.Errorf("endpoint = %q, want %q", got, tt.want)
				}
				return
			}
			if err == nil || err.Reason != tt.wantReason {
				t.Fatalf("error = %v, want reason %q", err, tt.wantReason)
			}
			if strings.Contains(err.Error(), tt.endpoint) || strings.Contains(err.Error(), "secret") {
				t.Errorf("error leaked rejected endpoint: %v", err)
			}
		})
	}
}

func FuzzCountResponse(f *testing.F) {
	for _, seed := range []string{`{"inputTokens":0}`, `{"inputTokens":1,"InputTokens":2}`, `null`, `{`} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw string) {
		_, err := decodeCountResponse([]byte(raw))
		if err != nil && strings.Contains(err.Error(), raw) && len(raw) > 8 {
			t.Errorf("error leaked raw response")
		}
	})
}

func FuzzCounterEndpoint(f *testing.F) {
	for _, seed := range []string{"https://bedrock-runtime.us-east-1.amazonaws.com", "http://127.0.0.1:8080", "https://user:secret@example.com", "://"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw string) {
		endpoint, err := canonicalCounterEndpoint(raw)
		if err != nil {
			if strings.Contains(err.Error(), raw) && len(raw) > 8 {
				t.Errorf("error leaked endpoint")
			}
			return
		}
		if endpoint == "" {
			t.Error("successful canonicalization returned empty endpoint")
		}
	})
}

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

type authenticatorFunc func(context.Context, *http.Request) error

func (f authenticatorFunc) Authorize(ctx context.Context, req *http.Request) error {
	return f(ctx, req)
}

type errorReadCloser struct{ err error }

func (e *errorReadCloser) Read([]byte) (int, error) { return 0, e.err }
func (e *errorReadCloser) Close() error             { return nil }

func richCounterRequest(modelID string) inference.Request {
	req := counterRequest(modelID)
	req.System = "You are exact."
	req.Messages = append(req.Messages,
		&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{&content.TextBlock{Text: "I can help."}}}},
		&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "Count all fields."}}}},
	)
	req.Override = &model.Sampling{MaxTokens: counterIntPtr(321)}
	return req
}

func counterRequest(name string) inference.Request {
	return inference.Request{
		Model: model.CustomModel(
			model.ProviderName(llm.ProviderBedrock),
			model.APIFormatAnthropic,
			"",
			name,
			model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}),
			model.WithTools(),
			model.WithImages(),
		),
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{
				Role: content.RoleUser,
				Blocks: []content.Block{
					&content.TextBlock{Text: "hello"},
				},
			}},
		},
	}
}

func counterTestCreds() auth.SigV4Credentials {
	return auth.SigV4Credentials{
		AccessKeyID:     "AKIDEXAMPLE",
		SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
	}
}

func counterIntPtr(value int) *int { return &value }
