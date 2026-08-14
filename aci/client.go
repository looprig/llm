package aci

// This file implements the Dstack ACI ("aci/1") confidential-inference client —
// the HEART of the package (Task 5.2). Client.Invoke runs the buffer-until-
// verified flow:
//
//	attest(cached) -> openaiapi encode -> ordered body -> sealRequest ->
//	POST /v1/chat/completions -> read FULL response + x-receipt-id ->
//	openResponse -> VerifyReceipt -> ONLY THEN openaiapi decode -> *inference.Response
//
// FAIL CLOSED. Every failure — attestation, encode, seal, transport, a non-2xx
// status, response open, receipt fetch, or receipt verification — returns a typed
// error and a NIL *inference.Response. No partial response ever escapes: the decode (the
// only step that produces a *Response) runs ONLY after VerifyReceipt passes. The
// API key is sent as a Bearer token but never appears in any returned error.
//
// The mutating/network steps are seam-injected so offline tests drive the whole
// flow against a fake gateway: httpDoer (the HTTP transport), now (the clock),
// quoteVerifier (the DCAP seam), attestFunc (the attest seam), and newNonce (the
// attestation nonce source). The composition root (New) wires the live defaults.

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"time"

	secp256k1 "github.com/decred/dcrd/dcrec/secp256k1/v4"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	failure "github.com/looprig/inference/failure"
	stream "github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/sse"
)

// Endpoint paths (relative to baseURL). They are protocol-pinned; the gateway
// serves exactly these routes (design doc §Endpoints).
const (
	pathChatCompletions = "/v1/chat/completions"
	pathAttestation     = "/v1/aci/attestation"
	pathReceiptsPrefix  = "/v1/aci/receipts/"

	// methodPost / endpointChat are the receipt-binding values: the receipt's
	// method/endpoint must match the request we made (ReceiptExpect).
	endpointChat = pathChatCompletions
	methodPost   = http.MethodPost

	// receiptIDHeader carries the receipt id of the just-served response; the
	// client fetches the matching receipt by this id.
	receiptIDHeader = "x-receipt-id"
	// contentTypeJSON is the POST body content type.
	contentTypeJSON = "application/json"
)

// HTTP client timeout budget. These bound every phase of a request so a slow or
// hung gateway can never block Invoke indefinitely (CLAUDE.md: no naked client,
// TLS >= 1.2, context deadline). The overall Timeout caps the whole exchange;
// the transport timeouts cap the handshake and header phases individually.
const (
	defaultClientTimeout         = 120 * time.Second
	defaultTLSHandshakeTimeout   = 10 * time.Second
	defaultResponseHeaderTimeout = 60 * time.Second
	defaultExpectContinueTimeout = 1 * time.Second
	defaultIdleConnTimeout       = 90 * time.Second
	defaultDialTimeout           = 10 * time.Second
	defaultMaxResponseBodyBytes  = 32 << 20 // 32 MiB cap on a buffered response.
	attestationNonceBytes        = 32       // 32 crypto/rand bytes -> 64 hex chars.
)

// httpDoer is the minimal HTTP seam the client depends on: just Do. The live
// implementation is an *http.Client with explicit timeouts (newHTTPClient);
// tests inject a fake gateway. Depending on this narrow interface (not the
// concrete *http.Client) keeps the client testable and least-privileged.
type httpDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// Client is the ACI confidential-inference client. It attests each model
// (cached), seals requests E2EE to the attested model key, POSTs them, opens the
// sealed response, and verifies the signed receipt before decoding — returning a
// provider-neutral *inference.Response only when every check passes.
//
// It implements inference.Client. It is safe for concurrent use: the only mutable shared
// state is the session cache, which is internally synchronized.
//
// Connection-binding note (fail-safe asymmetry): unlike the generic
// transport.Client — which binds one Endpoint and rejects a request whose
// Model.Provider/BaseURL differs with a pre-I/O *failure.ModelMismatchError — this
// client binds its gateway endpoint at construction (New's baseURL) and enforces
// model identity per request via TEE attestation. A provider/endpoint mismatch
// therefore surfaces as an *AttestationError (attestation cannot succeed against
// the wrong model/gateway), not a *failure.ModelMismatchError. This is fail-safe:
// the request is never sent when the check fails; only the error type differs.
type Client struct {
	baseURL string
	apiKey  string
	policy  Policy

	http  httpDoer
	now   func() time.Time
	cache *sessionCache

	// quoteVerifier is the DCAP seam used by the production attest path; tests
	// override the whole attest path via WithAttestFunc, so this is only read on
	// the live wiring.
	quoteVerifier quoteVerifier
	// newNonce produces the per-attestation report_data nonce.
	newNonce func() string
}

// Option configures a Client at construction. Only the knobs production or tests
// actually use are exposed; nothing speculative.
type Option func(*Client)

// WithHTTPDoer sets the HTTP transport (the gateway seam). Tests inject a fake
// gateway; production uses the default timed *http.Client.
func WithHTTPDoer(d httpDoer) Option {
	return func(c *Client) { c.http = d }
}

// WithNow sets the wall clock used for seal/open timestamps and the session-cache
// TTL. Tests inject a fixed clock for determinism.
func WithNow(now func() time.Time) Option {
	return func(c *Client) { c.now = now }
}

// WithQuoteVerifier overrides the DCAP quote verifier seam used by the live
// attest path (offline tests of the production attest wiring).
func WithQuoteVerifier(v quoteVerifier) Option {
	return func(c *Client) { c.quoteVerifier = v }
}

// WithAttestFunc overrides the per-model attestation the session cache wraps.
// Tests supply a fake that returns a synthetic *VerifiedReport, bypassing the
// real attestation chain (already covered by Phase-2 tests).
func WithAttestFunc(attest attestFunc) Option {
	return func(c *Client) { c.cache = newSessionCache(attest, defaultSessionTTL, c.now) }
}

// WithNonceFunc overrides the attestation nonce source (the report_data binding
// nonce). Production draws 32 crypto/rand bytes; tests can pin it.
func WithNonceFunc(newNonce func() string) Option {
	return func(c *Client) { c.newNonce = newNonce }
}

// New builds a Client and returns it as the inference.Client it implements. baseURL is the
// gateway origin (e.g. https://gateway.example); apiKey is the bearer token;
// policy is the attestation acceptance allow-list. Defaults — a timed
// *http.Client (TLS >= 1.2), time.Now, the live DCAP quote verifier, a
// crypto/rand nonce source, and a session cache wrapping the production attest —
// are applied first, then options override. Order matters: the cache binds the
// clock, so WithNow (when supplied) must take effect before the default cache is
// built; we therefore apply options that may need the clock after seeding the
// defaults, and (re)build the default cache only if no WithAttestFunc replaced it.
//
// New FAILS CLOSED on an unpinned policy: if policy pins no acceptance set and did
// not explicitly opt out via UnpinnedPolicy(), New returns (nil,
// *UnpinnedPolicyError) BEFORE any network object is constructed, so attestation
// can never be silently accepted against an empty allow-list.
func New(baseURL, apiKey string, policy Policy, opts ...Option) (inference.Client, error) {
	if err := policy.requireAcceptable(); err != nil {
		return nil, err
	}
	c := &Client{
		baseURL:       baseURL,
		apiKey:        apiKey,
		policy:        policy,
		http:          newHTTPClient(),
		now:           time.Now,
		quoteVerifier: defaultQuoteVerifier,
		newNonce:      defaultNonce,
	}

	// Apply caller options first so the clock (WithNow) and attest seam
	// (WithAttestFunc) are in place before the default cache is seeded.
	for _, opt := range opts {
		opt(c)
	}

	// If no option installed a cache (WithAttestFunc), wrap the production attest
	// with the now-final clock. This keeps the cache's TTL bound to the same clock
	// the seal/open path uses.
	if c.cache == nil {
		c.cache = newSessionCache(c.attestModel, defaultSessionTTL, c.now)
	}

	return c, nil
}

// newHTTPClient builds the production *http.Client with an explicit timeout
// budget and a TLS >= 1.2 transport (CLAUDE.md: no naked client, no insecure
// TLS). Every phase — dial, handshake, response headers — is bounded, and the
// overall Timeout caps the whole exchange.
func newHTTPClient() *http.Client {
	dialer := &net.Dialer{Timeout: defaultDialTimeout}
	transport := &http.Transport{
		DialContext:           dialer.DialContext,
		TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12},
		TLSHandshakeTimeout:   defaultTLSHandshakeTimeout,
		ResponseHeaderTimeout: defaultResponseHeaderTimeout,
		ExpectContinueTimeout: defaultExpectContinueTimeout,
		IdleConnTimeout:       defaultIdleConnTimeout,
		ForceAttemptHTTP2:     true,
	}
	return &http.Client{
		Transport: transport,
		Timeout:   defaultClientTimeout,
	}
}

// defaultNonce returns a fresh 32-byte crypto/rand nonce as 64 lowercase hex
// chars — the attestation report_data binding nonce. A crypto/rand read failure
// is reported as an empty string, which the attest path treats as a (typed)
// attestation failure rather than silently proceeding with a weak nonce.
func defaultNonce() string {
	var b [attestationNonceBytes]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return hex.EncodeToString(b[:])
}

// Invoke runs the buffer-until-verified flow for a single non-streaming chat
// request and returns the decoded *inference.Response, or a typed error and a NIL
// response on ANY failure. See the file header for the ordered stages; the decode
// runs ONLY after VerifyReceipt passes, so a verification failure never yields a
// partial response.
func (c *Client) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	if err := req.Model.Validate(); err != nil {
		return nil, err
	}
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return nil, err
	}
	model := req.Model.Name

	// 1. Attest (cached). A failure here is already a typed *llm.AttestationError.
	verified, err := c.cache.get(ctx, model)
	if err != nil {
		return nil, err
	}

	// 2. Encode to the OpenAI body, then re-parse into the ORDERED body the sealer
	//    mutates in place.
	body, err := openaiapi.EncodeRequest(req, false)
	if err != nil {
		return nil, &encodeError{Err: err}
	}
	parsed, err := ParseBodyValue(body)
	if err != nil {
		return nil, &encodeError{Err: err}
	}
	orderedBody, ok := parsed.(*Object)
	if !ok {
		return nil, &encodeError{Err: &bodyShapeError{}}
	}

	// 3. Capture the CLEARTEXT compact bytes NOW, BEFORE sealing — sealRequest
	//    mutates orderedBody in place, so this is the only chance to snapshot the
	//    receipt body_hash preimage.
	cleartextReqBody, err := CompactJSON(orderedBody)
	if err != nil {
		return nil, &encodeError{Err: err}
	}

	// 4. Seal the request E2EE to the attested model key (in place).
	sealed, err := sealRequest(orderedBody, verified, c.now)
	if err != nil {
		return nil, err
	}

	// 5. POST the sealed body; read the FULL response and the receipt id.
	wireBytes, receiptID, err := c.postInference(ctx, sealed)
	if err != nil {
		return nil, err
	}

	// 6. Open the sealed response with the per-request client key.
	opened, err := openResponse(wireBytes, sealed.ClientPriv, model, sealed.Nonce, sealed.Timestamp, c.now())
	if err != nil {
		return nil, err
	}

	// 7. Fetch + verify the receipt. The receipt binds the cleartext request body,
	//    the opened response cleartext, and the raw sealed wire bytes.
	receiptJSON, err := c.fetchReceipt(ctx, receiptID)
	if err != nil {
		return nil, err
	}
	expect := ReceiptExpect{
		Endpoint: endpointChat,
		Method:   methodPost,
		// Vendor and ModelID are intentionally EMPTY: confirmed live at Task 6.1
		// that upstream.verified carries the gateway-RESOLVED upstream identity
		// (e.g. provider "chutes", model_id "zai-org/GLM-5.2-TEE") for a requested
		// "z-ai/glm-5.2", and it varies by the attested gateway's routing. The
		// requested model is already bound via request.received.body_hash, so the
		// upstream binding is just result == "verified" on the attested gateway.
		Vendor:            "",
		ModelID:           "",
		ReqBody:           cleartextReqBody,
		RespBodyCleartext: opened,
		RespWireBytes:     wireBytes,
	}
	if err := VerifyReceipt(receiptJSON, verified, expect); err != nil {
		return nil, err
	}

	// 8. ONLY NOW — every check passed — decode the cleartext response.
	resp, err := openaiapi.DecodeResponse(opened)
	if err != nil {
		return nil, &decodeError{Err: err}
	}
	return resp, nil
}

// Stream runs the buffer-until-verified flow for a streaming chat request. It
// buffers the FULL sealed SSE response, opens + verifies it (receipt signature +
// request body_hash + wire_hash + upstream + the E2EE-authenticated open of every
// delta), and ONLY THEN returns a *stream.StreamReader replaying the already-verified
// opened deltas. On ANY failure it returns a typed error and a NIL reader, so the
// caller observes ZERO chunks — no unverified delta is ever observable.
//
// Streaming SKIPS the receipt cleartext_hash check (RespBodyCleartext is nil): the
// client only sees the sealed WIRE bytes, not the raw upstream SSE framing the
// gateway hashed for cleartext_hash, so it cannot reconstruct that preimage.
// wire_hash (over the exact wire bytes) plus the per-delta AEAD open — each delta
// AAD-bound to the attested gateway — carry content authenticity instead.
func (c *Client) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	if err := req.Model.Validate(); err != nil {
		return nil, err
	}
	if err := inference.ValidateRequestFeatures(req); err != nil {
		return nil, err
	}
	model := req.Model.Name

	// 1. Attest (cached).
	verified, err := c.cache.get(ctx, model)
	if err != nil {
		return nil, err
	}

	// 2. Encode with stream=true, re-parse into the ORDERED body, snapshot the
	//    cleartext compact bytes (the receipt body_hash preimage) BEFORE sealing.
	body, err := openaiapi.EncodeRequest(req, true)
	if err != nil {
		return nil, &encodeError{Err: err}
	}
	parsed, err := ParseBodyValue(body)
	if err != nil {
		return nil, &encodeError{Err: err}
	}
	orderedBody, ok := parsed.(*Object)
	if !ok {
		return nil, &encodeError{Err: &bodyShapeError{}}
	}
	cleartextReqBody, err := CompactJSON(orderedBody)
	if err != nil {
		return nil, &encodeError{Err: err}
	}

	// 3. Seal the request E2EE to the attested model key (in place).
	sealed, err := sealRequest(orderedBody, verified, c.now)
	if err != nil {
		return nil, err
	}

	// 4. POST the sealed body; buffer the FULL SSE wire response and the receipt id.
	wireBytes, receiptID, err := c.postInference(ctx, sealed)
	if err != nil {
		return nil, err
	}

	// 5. Parse the buffered SSE and OPEN every sealed delta in order. Any open
	//    failure is fail-closed e2ee_failed; no chunk escapes.
	chunks, err := openStreamDeltas(wireBytes, sealed.ClientPriv, model, sealed.Nonce, sealed.Timestamp)
	if err != nil {
		return nil, err
	}

	// 6. Fetch + verify the receipt. Streaming binds request body_hash, wire_hash,
	//    and the upstream event; cleartext_hash is SKIPPED (RespBodyCleartext nil).
	receiptJSON, err := c.fetchReceipt(ctx, receiptID)
	if err != nil {
		return nil, err
	}
	expect := ReceiptExpect{
		Endpoint: endpointChat,
		Method:   methodPost,
		// Vendor and ModelID intentionally EMPTY (see Invoke): upstream.verified
		// carries the gateway-resolved upstream identity, which varies by routing;
		// the requested model is bound via body_hash, so bind on result=="verified".
		Vendor:            "",
		ModelID:           "",
		ReqBody:           cleartextReqBody,
		RespBodyCleartext: nil, // streaming: skip cleartext_hash (wire_hash + E2EE cover it)
		RespWireBytes:     wireBytes,
	}
	if err := VerifyReceipt(receiptJSON, verified, expect); err != nil {
		return nil, err
	}
	result, hasResult, err := collectStreamResult(wireBytes)
	if err != nil {
		return nil, err
	}

	// 7. ONLY NOW — every check passed — hand back a reader replaying the verified
	//    deltas. Nothing produced this reader before verification, so a failure
	//    above yields no reader and zero observable chunks.
	return newReplayReader(chunks, result, hasResult), nil
}

// newReplayReader builds a *stream.StreamReader that replays pre-built,
// already-verified chunks and publishes their codec result at EOF. Next returns
// each chunk in order and (nil, io.EOF) once exhausted; Close is a no-op (the wire
// response was fully buffered and closed during Stream, so there is no underlying
// connection to release). The cursor is closed over locally, so a fresh reader is
// independent.
func newReplayReader(chunks []content.Chunk, result stream.StreamResult, hasResult bool) *stream.StreamReader[content.Chunk] {
	i := 0
	next := func() (content.Chunk, error) {
		if i >= len(chunks) {
			return nil, io.EOF
		}
		chunk := chunks[i]
		i++
		return chunk, nil
	}
	produceResult := func() (stream.StreamResult, bool, error) {
		return result, hasResult, nil
	}
	return stream.NewStreamReaderWithResult(next, nil, produceResult)
}

// bodyKeyDelta is the streaming chunk's per-choice sealed-field container: the
// gateway seals choices[].delta.content / choices[].delta.reasoning_content (the
// streaming analogue of the non-streaming choices[].message.* fields).
const bodyKeyDelta = "delta"

// doneSentinel is the gateway's terminal SSE payload. inference/wire/sse is a raw
// event framer that emits it as an ordinary frame (it does not interpret API
// sentinels); this package owns treating it as end-of-stream.
const doneSentinel = "[DONE]"

// openStreamDeltas parses the buffered SSE wire bytes and opens every sealed
// delta in order, returning the cleartext deltas as content.Chunks: a sealed
// choices[].delta.content maps to a *content.TextChunk, reasoning_content to a
// *content.ThinkingChunk. Each field is opened with the client key under the SAME
// chatResponseAAD used non-streaming, bound to that CHUNK's own id and the
// choice index. The [DONE] event terminates the stream. Any SSE read failure, a
// non-object chunk payload, or any AEAD open failure is fail-closed (open
// failures surface as e2ee_failed) so NO chunk escapes an unverified stream.
func openStreamDeltas(wireBytes []byte, clientPriv *secp256k1.PrivateKey, model, nonce string, ts uint64) ([]content.Chunk, error) {
	if clientPriv == nil {
		return nil, attestErr(reasonE2EEFailed, &e2eeOpenError{Stage: "nil client key"})
	}

	// inference/wire/sse de-frames the buffered bytes into per-event data payloads
	// (blank-line delimited, as the gateway emits: "data: {json}\n\n"). It owns the
	// NopCloser body; on an early framer error there is nothing to close.
	reader, err := sse.DecodeStreamFrames(io.NopCloser(bytes.NewReader(wireBytes)))
	if err != nil {
		return nil, &streamParseError{reason: "SSE read failed", cause: err}
	}
	defer reader.Close()

	var chunks []content.Chunk
	for {
		frame, err := reader.Next()
		// errors.Is, never ==: a framer that wraps its EOF (or is layered behind
		// one that does) would otherwise fall through to the failure branch and
		// report a cleanly finished stream as an SSE read failure, discarding every
		// chunk already decoded from it. The fail-closed rule above is about chunks
		// that were never verified; end of stream is not that.
		if errors.Is(err, io.EOF) {
			return chunks, nil
		}
		if err != nil {
			return nil, &streamParseError{reason: "SSE read failed", cause: err}
		}
		// The framer does not interpret the terminal sentinel; [DONE] ends the stream.
		if string(frame.Data) == doneSentinel {
			return chunks, nil
		}

		eventChunks, err := openStreamEvent(frame.Data, clientPriv, model, nonce, ts)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, eventChunks...)
	}
}

// collectStreamResult delegates terminal metadata decoding to the same OpenAI
// codec used by ordinary provider streams. The wire bytes remain internal until
// the receipt has been verified; only then does Stream attach this snapshot to
// the verified replay reader.
func collectStreamResult(wireBytes []byte) (stream.StreamResult, bool, error) {
	reader := openaiapi.NewStream(io.NopCloser(bytes.NewReader(wireBytes)))
	for {
		_, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			closeErr := reader.Close()
			if closeErr != nil {
				return stream.StreamResult{}, false, &streamParseError{reason: "usage stream decode and close failed", cause: errors.Join(err, closeErr)}
			}
			return stream.StreamResult{}, false, err
		}
	}
	result, ok := reader.Result()
	if err := reader.Close(); err != nil {
		return stream.StreamResult{}, false, &streamParseError{reason: "usage stream close failed", cause: err}
	}
	return result, ok, nil
}

// openStreamEvent opens one SSE chunk event: it order-preservingly parses the
// chunk JSON, reads the chunk id (the AAD {id}) and its choices, and opens each
// choice's sealed delta.content / delta.reasoning_content in order. A non-object
// payload is a malformed stream (fail-closed streamParseError); an open failure
// propagates as the fail-closed e2ee_failed AttestationError.
func openStreamEvent(payload []byte, clientPriv *secp256k1.PrivateKey, model, nonce string, ts uint64) ([]content.Chunk, error) {
	parsed, err := ParseBodyValue(payload)
	if err != nil {
		return nil, &streamParseError{reason: "chunk is not valid JSON", cause: err}
	}
	chunkObj, ok := parsed.(*Object)
	if !ok {
		return nil, &streamParseError{reason: "chunk is not a JSON object"}
	}

	chunkID := responseBodyID(chunkObj)
	choicesV, ok := objectGet(chunkObj, bodyKeyChoices)
	if !ok {
		return nil, nil // e.g. a keep-alive / usage-only chunk: no deltas.
	}
	choices, ok := choicesV.(Array)
	if !ok {
		return nil, nil
	}

	var out []content.Chunk
	for i := range choices {
		choice, ok := choices[i].(*Object)
		if !ok {
			continue
		}
		ci := itemIndex(choice, i)
		deltaV, ok := objectGet(choice, bodyKeyDelta)
		if !ok {
			continue
		}
		delta, ok := deltaV.(*Object)
		if !ok {
			continue
		}

		// content -> TextChunk. Open in place, then read the cleartext back.
		if err := openStringField(delta, respFieldContent, clientPriv,
			chatResponseAAD(e2eeRequestAlgo, model, chunkID, ci, respFieldContent, nonce, ts)); err != nil {
			return nil, attestErr(reasonE2EEFailed, err)
		}
		if s, ok := stringField(delta, respFieldContent); ok {
			out = append(out, &content.TextChunk{Text: s})
		}

		// reasoning_content -> ThinkingChunk.
		if err := openStringField(delta, respFieldReasoningContent, clientPriv,
			chatResponseAAD(e2eeRequestAlgo, model, chunkID, ci, respFieldReasoningContent, nonce, ts)); err != nil {
			return nil, attestErr(reasonE2EEFailed, err)
		}
		if s, ok := stringField(delta, respFieldReasoningContent); ok {
			// Index stays zero for the same reason as the shared Chat codec:
			// reasoning_content is one delta stream per choice with no
			// per-block index on the wire, so every fragment belongs to a
			// single reasoning block.
			out = append(out, &content.ThinkingChunk{Thinking: s})
		}
	}
	return out, nil
}

// stringField returns the opened string value of a delta field (content /
// reasoning_content) after openStringField has replaced the sealed blob with the
// cleartext. It returns ok false when the field is absent or not a string, so a
// delta that carried no sealed field produces no chunk.
func stringField(obj *Object, field string) (string, bool) {
	v, ok := objectGet(obj, field)
	if !ok {
		return "", false
	}
	s, ok := v.(String)
	if !ok {
		return "", false
	}
	return string(s), true
}

// postInference POSTs the sealed body to /v1/chat/completions with the E2EE
// headers, the bearer token, and the JSON content type, then reads the FULL
// response body (the receipt wire-hash preimage) and the x-receipt-id header. A
// transport failure returns *failure.NetworkError; a non-2xx status returns
// *failure.APIError; neither carries the API key.
func (c *Client) postInference(ctx context.Context, sealed *sealedRequest) ([]byte, string, error) {
	endpoint, err := c.endpointURL(pathChatCompletions, nil)
	if err != nil {
		return nil, "", err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(sealed.Body))
	if err != nil {
		return nil, "", &failure.NetworkError{Err: err}
	}
	for k, v := range sealed.Headers {
		httpReq.Header.Set(k, v)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", contentTypeJSON)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, "", &failure.NetworkError{Err: err}
	}
	defer resp.Body.Close()

	body, err := readAllLimited(resp.Body)
	if err != nil {
		return nil, "", &failure.NetworkError{Err: err}
	}
	if resp.StatusCode/100 != 2 {
		return nil, "", failure.APIErrorFromResponse(resp.StatusCode, body, resp.Header, 0)
	}
	return body, resp.Header.Get(receiptIDHeader), nil
}

// fetchReceipt GETs the receipt by id from /v1/aci/receipts/{id} with the bearer
// token and returns the raw receipt JSON. A transport failure returns
// *failure.NetworkError; a non-2xx status returns *failure.APIError.
func (c *Client) fetchReceipt(ctx context.Context, receiptID string) ([]byte, error) {
	if receiptID == "" {
		return nil, &receiptFetchError{reason: "response carried no x-receipt-id header"}
	}
	endpoint, err := c.endpointURL(pathReceiptsPrefix+url.PathEscape(receiptID), nil)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	defer resp.Body.Close()

	body, err := readAllLimited(resp.Body)
	if err != nil {
		return nil, &failure.NetworkError{Err: err}
	}
	if resp.StatusCode/100 != 2 {
		return nil, failure.APIErrorFromResponse(resp.StatusCode, body, resp.Header, 0)
	}
	return body, nil
}

// attestModel is the PRODUCTION per-model attestation the session cache wraps: it
// draws a fresh nonce, GETs the model's attestation report, and runs the full
// verifyReport chain (seam-based so offline tests of the live wiring can inject a
// fake quote verifier). It returns a typed *llm.AttestationError on any failure.
func (c *Client) attestModel(ctx context.Context, model string) (*VerifiedReport, error) {
	nonce := c.newNonce()
	if nonce == "" {
		return nil, attestErr(reasonQuoteInvalid, &attestNonceError{})
	}

	query := url.Values{}
	query.Set("model", model)
	query.Set("nonce", nonce)
	endpoint, err := c.endpointURL(pathAttestation, query)
	if err != nil {
		return nil, attestErr(reasonQuoteInvalid, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, attestErr(reasonQuoteInvalid, &failure.NetworkError{Err: err})
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, attestErr(reasonQuoteInvalid, &failure.NetworkError{Err: err})
	}
	defer resp.Body.Close()

	reportJSON, err := readAllLimited(resp.Body)
	if err != nil {
		return nil, attestErr(reasonQuoteInvalid, &failure.NetworkError{Err: err})
	}
	if resp.StatusCode/100 != 2 {
		return nil, attestErr(reasonQuoteInvalid, failure.APIErrorFromResponse(resp.StatusCode, reportJSON, resp.Header, 0))
	}

	return verifyReport(reportJSON, &nonce, c.now(), c.policy, c.quoteVerifier)
}

// endpointURL joins baseURL with path (and optional query), validating the result
// is a well-formed absolute URL. It returns a typed *invalidURLError on a bad
// baseURL so a misconfiguration fails closed before any I/O.
func (c *Client) endpointURL(path string, query url.Values) (string, error) {
	base, err := url.Parse(c.baseURL)
	if err != nil {
		return "", &invalidURLError{cause: err}
	}
	ref, err := url.Parse(path)
	if err != nil {
		return "", &invalidURLError{cause: err}
	}
	u := base.ResolveReference(ref)
	if query != nil {
		u.RawQuery = query.Encode()
	}
	return u.String(), nil
}

// readAllLimited reads the full body up to defaultMaxResponseBodyBytes, so a
// hostile or runaway gateway cannot exhaust memory with an unbounded body. A body
// that hits the cap is treated as a read error by the caller (it cannot be a
// valid sealed response or receipt).
func readAllLimited(r io.Reader) ([]byte, error) {
	limited := io.LimitReader(r, defaultMaxResponseBodyBytes+1)
	b, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if len(b) > defaultMaxResponseBodyBytes {
		return nil, &responseTooLargeError{limit: defaultMaxResponseBodyBytes}
	}
	return b, nil
}

// =============================================================================
// Typed client errors.
//
// Per CLAUDE.md every distinct failure mode is a concrete error struct callers
// can errors.As to; none returns a bare errors.New/fmt.Errorf. None of these
// carries the API key, plaintext, or key material — only structural facts and
// (where present) a chained cause.
// =============================================================================

// encodeError reports a failure turning the provider-neutral request into the
// ordered, sealable body: an openaiapi encode error, a body parse error, or a
// CompactJSON error. It chains the underlying cause via Unwrap.
type encodeError struct {
	Err error
}

func (e *encodeError) Error() string { return "aci/client: encode request: " + e.Err.Error() }
func (e *encodeError) Unwrap() error { return e.Err }

// decodeError reports a failure decoding the VERIFIED cleartext response into a
// provider-neutral *inference.Response. It can only occur AFTER every E2EE/receipt
// check passed, so it is a malformed-but-attested response. It chains the cause.
type decodeError struct {
	Err error
}

func (e *decodeError) Error() string { return "aci/client: decode response: " + e.Err.Error() }
func (e *decodeError) Unwrap() error { return e.Err }

// bodyShapeError reports that the encoded request body was not a JSON object —
// the only shape the sealer accepts. It carries no payload (the only fact is
// "top-level was not an object").
type bodyShapeError struct{}

func (e *bodyShapeError) Error() string {
	return "aci/client: encoded request body is not a JSON object"
}

// receiptFetchError reports a failure obtaining the receipt that is NOT a
// transport/status failure — chiefly a missing x-receipt-id header on the POST
// response, which leaves the client with no receipt id to fetch. reason is a
// fixed, code-defined label, never external data.
type receiptFetchError struct {
	reason string
}

func (e *receiptFetchError) Error() string { return "aci/client: receipt fetch: " + e.reason }

// attestNonceError reports that the attestation nonce source failed (an empty
// nonce from a crypto/rand read failure). It carries no payload; the only fact is
// "the nonce source produced nothing".
type attestNonceError struct{}

func (e *attestNonceError) Error() string {
	return "aci/client: attestation nonce generation failed"
}

// invalidURLError reports a baseURL or path that does not parse into a valid
// endpoint URL — a misconfiguration that fails closed before any I/O. It chains
// the underlying url.Parse error.
type invalidURLError struct {
	cause error
}

func (e *invalidURLError) Error() string {
	return "aci/client: invalid endpoint URL: " + e.cause.Error()
}
func (e *invalidURLError) Unwrap() error { return e.cause }

// responseTooLargeError reports a response body that exceeded the buffered-read
// cap. It carries the limit (a fixed byte count, never payload) so the caller can
// see the bound that was hit.
type responseTooLargeError struct {
	limit int
}

func (e *responseTooLargeError) Error() string {
	return "aci/client: response body exceeded the maximum buffered size of " +
		strconv.Itoa(e.limit) + " bytes"
}

// streamParseError reports a malformed buffered SSE stream: a data event whose
// payload is not a JSON object (the only chunk shape the gateway emits), or an
// SSE read failure. It is fail-closed — a malformed stream yields no reader — and
// carries no payload beyond a fixed reason label and any chained cause.
type streamParseError struct {
	reason string
	cause  error
}

func (e *streamParseError) Error() string {
	msg := "aci/client: stream parse: " + e.reason
	if e.cause != nil {
		return msg + ": " + e.cause.Error()
	}
	return msg
}

func (e *streamParseError) Unwrap() error { return e.cause }
