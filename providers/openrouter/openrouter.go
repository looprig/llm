// Package openrouter provides the OpenRouter-specific construction and request
// options for the OpenAI-compatible Chat Completions API.
package openrouter

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/openaiapi"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/transport"

	"github.com/looprig/llm"
)

const (
	defaultBaseURL                   = "https://openrouter.ai/api/v1"
	openRouterReasoningDetailsFormat = "openrouter-reasoning-details"
)

// ReasoningOptions controls OpenRouter's provider-specific reasoning object.
// Pointer fields preserve the difference between an omitted value and an
// explicitly supplied false or zero.
type ReasoningOptions struct {
	Effort    string `json:"effort,omitempty"`
	MaxTokens *int   `json:"max_tokens,omitempty"`
	Exclude   *bool  `json:"exclude,omitempty"`
	Enabled   *bool  `json:"enabled,omitempty"`
	Context   string `json:"context,omitempty"`
	Mode      string `json:"mode,omitempty"`
}

// ProviderRoutingOptions controls OpenRouter's provider-selection policy.
// Pointer fields preserve the difference between an omitted value and an
// explicitly supplied false.
type ProviderRoutingOptions struct {
	Order             []string `json:"order,omitempty"`
	AllowFallbacks    *bool    `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool    `json:"require_parameters,omitempty"`
	DataCollection    string   `json:"data_collection,omitempty"`
	ZDR               *bool    `json:"zdr,omitempty"`
}

type config struct {
	headers         http.Header
	usage           *bool
	reasoning       *ReasoningOptions
	promptCacheKey  string
	providerRouting *ProviderRoutingOptions
	tlsRootCAs      *x509.CertPool
}

// Option customizes an OpenRouter client at construction time.
type Option func(*config)

// WithTLSRootCAs installs a caller-owned verified certificate pool for tests
// and controlled clients; nil is rejected rather than silently using defaults.
func WithTLSRootCAs(roots *x509.CertPool) Option {
	if roots == nil {
		panic("openrouter: TLS roots must not be nil")
	}
	return func(c *config) { c.tlsRootCAs = roots }
}

// WithHTTPReferer adds OpenRouter's optional HTTP-Referer attribution header.
func WithHTTPReferer(value string) Option {
	return func(c *config) {
		if c.headers == nil {
			c.headers = make(http.Header)
		}
		if value == "" {
			c.headers.Del("HTTP-Referer")
			return
		}
		c.headers.Set("HTTP-Referer", value)
	}
}

// WithTitle adds OpenRouter's optional X-OpenRouter-Title attribution header.
func WithTitle(value string) Option {
	return func(c *config) {
		if c.headers == nil {
			c.headers = make(http.Header)
		}
		if value == "" {
			c.headers.Del("X-OpenRouter-Title")
			return
		}
		c.headers.Set("X-OpenRouter-Title", value)
	}
}

// WithUsage requests OpenRouter to include its usage metadata in the response.
func WithUsage(include bool) Option {
	return func(c *config) {
		c.usage = boolPtr(include)
	}
}

// WithReasoning adds OpenRouter's reasoning request object.
func WithReasoning(options ReasoningOptions) Option {
	return func(c *config) {
		c.reasoning = cloneReasoningOptions(options)
	}
}

// WithPromptCacheKey sets OpenRouter's stable prompt-cache key.
func WithPromptCacheKey(value string) Option {
	return func(c *config) {
		c.promptCacheKey = value
	}
}

// WithProviderRouting adds OpenRouter provider-routing preferences.
func WithProviderRouting(options ProviderRoutingOptions) Option {
	return func(c *config) {
		c.providerRouting = cloneProviderRoutingOptions(options)
	}
}

// New builds an OpenRouter inference client. The selected model must identify
// OpenRouter and the OpenAI API format; an empty API key is rejected before a
// client is constructed. An empty model base URL uses OpenRouter's canonical
// API root.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if err := llm.ValidateModel(selected); err != nil {
		return nil, err
	}
	if selected.Provider != model.ProviderName(llm.ProviderOpenRouter) {
		return nil, &model.ValidationError{
			Field:  "Provider",
			Reason: fmt.Sprintf("OpenRouter constructor requires provider %q", llm.ProviderOpenRouter),
		}
	}
	if key == "" {
		return nil, &llm.AuthRequiredError{Provider: llm.ProviderOpenRouter, Kind: auth.AuthAPIKey}
	}

	cfg := config{}
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	cfg = cloneConfig(cfg)

	baseURL := selected.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	transportOptions := []transport.Option{}
	if cfg.tlsRootCAs != nil {
		transportOptions = append(transportOptions, transport.WithTLSRootCAs(cfg.tlsRootCAs))
	}
	return transport.New(
		transport.Endpoint{
			BaseURL:   baseURL,
			Provider:  selected.Provider,
			APIFormat: selected.APIFormat,
		},
		chatRouter{headers: cfg.headers},
		requestCodec{config: cfg},
		auth.Key(key), transportOptions...,
	), nil
}

type chatRouter struct {
	headers http.Header
}

func (r chatRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	built, err := route.StaticChat("/chat/completions").BuildRoute(baseURL, req, mode)
	if err != nil {
		return route.Route{}, err
	}
	built.Header = r.headers.Clone()
	return built, nil
}

type requestCodec struct {
	config config
}

var _ codec.StreamingCodec = requestCodec{}

func (c requestCodec) EncodeRequest(req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (openaiapi.Codec{}).EncodeRequest(req, mode)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	if !c.config.hasBodyOptions() && !hasReplayableReasoningDetails(req) {
		return encoded, nil
	}

	raw, err := io.ReadAll(encoded.Body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: read encoded request: %w", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: decode encoded request: %w", err)
	}
	if c.config.usage != nil {
		body["usage"], err = json.Marshal(struct {
			Include bool `json:"include"`
		}{Include: *c.config.usage})
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode usage option: %w", err)
		}
	}
	if c.config.reasoning != nil {
		body["reasoning"], err = json.Marshal(c.config.reasoning)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode reasoning option: %w", err)
		}
		// The OpenRouter reasoning object is the explicit provider-specific
		// configuration. Do not send the legacy OpenAI reasoning_effort field
		// alongside it, since the two controls can disagree.
		delete(body, "reasoning_effort")
	}
	if c.config.promptCacheKey != "" {
		body["prompt_cache_key"], err = json.Marshal(c.config.promptCacheKey)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode prompt cache key: %w", err)
		}
	}
	if c.config.providerRouting != nil {
		body["provider"], err = json.Marshal(c.config.providerRouting)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode provider routing option: %w", err)
		}
	}
	if err := replayReasoningDetails(body, req); err != nil {
		return codec.EncodedRequest{}, err
	}

	patched, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("openrouter: encode extended request: %w", err)
	}
	return codec.EncodedRequest{
		Header: encoded.Header.Clone(),
		Body:   bytes.NewReader(patched),
	}, nil
}

// DecodeResponse layers OpenRouter's reasoning_details onto the shared OpenAI
// decoder.
//
// It deliberately does NOT patch usage. OpenRouter documents the same
// reasoning-token accounting as the format it speaks: reasoning tokens are
// "considered output tokens and charged accordingly"
// (openrouter.ai/docs/use-cases/reasoning-tokens), completion_tokens_details is
// a "Breakdown of completion tokens", and total_tokens is the "Sum of the above
// two fields", prompt and completion, with no reasoning addend
// (openrouter.ai/docs/api-reference/overview). A live HTTP 200 from
// nvidia/nemotron-3-ultra-550b-a55b:free nonetheless carried
// completion_tokens=216 with reasoning_tokens=226, which is OpenRouter
// disagreeing with itself rather than a dialect delta this boundary could
// absorb: there is no documented arithmetic relating the two, so adding them —
// the repair codec/geminiapi applies for Gemini, where the discovery document
// says thoughts are an addend — would be invented here. The counts pass through
// as reported and content.Usage.ReasoningWithinOutput makes the divergence
// visible to anything that prices or reports them.
func (requestCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	reasoningDetails := responseReasoningDetails(body)
	if normalized, err := normalizeOpenRouterReasoning(body); err == nil {
		body = normalized
	}
	response, err := (openaiapi.Codec{}).DecodeResponse(body)
	if err != nil || len(reasoningDetails) == 0 || response == nil || response.Message == nil {
		return response, err
	}
	for _, block := range response.Message.Blocks {
		if thinking, ok := block.(*content.ThinkingBlock); ok {
			thinking.ProviderState = append(json.RawMessage(nil), reasoningDetails...)
			thinking.ProviderStateFormat = openRouterReasoningDetailsFormat
			return response, nil
		}
	}
	response.Message.Blocks = append([]content.Block{
		content.NewThinkingBlock("", "", reasoningDetails, openRouterReasoningDetailsFormat),
	}, response.Message.Blocks...)
	return response, nil
}

func responseReasoningDetails(body []byte) json.RawMessage {
	var envelope struct {
		Choices []struct {
			Message struct {
				ReasoningDetails json.RawMessage `json:"reasoning_details"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &envelope) != nil || len(envelope.Choices) == 0 {
		return nil
	}
	raw := envelope.Choices[0].Message.ReasoningDetails
	if !isReasoningDetailsArray(raw) {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

// isReasoningDetailsArray reports whether a payload has the shape OpenRouter
// documents for reasoning_details: a non-empty array of records. It gates both
// ingest and replay, so state that could never be sent is never stored either.
func isReasoningDetailsArray(raw json.RawMessage) bool {
	if len(raw) == 0 || !json.Valid(raw) {
		return false
	}
	var details []json.RawMessage
	if json.Unmarshal(raw, &details) != nil || len(details) == 0 {
		return false
	}
	for _, detail := range details {
		var record map[string]json.RawMessage
		// A JSON null unmarshals into a map without error, so the nil check is
		// what rejects a null element.
		if json.Unmarshal(detail, &record) != nil || record == nil {
			return false
		}
		if !isReasoningDetailRecord(record) {
			return false
		}
	}
	return true
}

func isReasoningDetailRecord(record map[string]json.RawMessage) bool {
	var typ string
	if json.Unmarshal(record["type"], &typ) != nil || typ == "" {
		return false
	}
	if raw, ok := record["format"]; ok {
		var format string
		if json.Unmarshal(raw, &format) != nil || format == "" {
			return false
		}
	}
	if raw, ok := record["id"]; ok && string(raw) != "null" {
		var id string
		if json.Unmarshal(raw, &id) != nil {
			return false
		}
	}
	if raw, ok := record["index"]; ok {
		var index uint64
		if json.Unmarshal(raw, &index) != nil {
			return false
		}
	}

	var payload string
	switch typ {
	case "reasoning.summary":
		return json.Unmarshal(record["summary"], &payload) == nil
	case "reasoning.encrypted":
		return json.Unmarshal(record["data"], &payload) == nil
	case "reasoning.text":
		if json.Unmarshal(record["text"], &payload) != nil {
			return false
		}
		if raw, ok := record["signature"]; ok && string(raw) != "null" {
			var signature string
			if json.Unmarshal(raw, &signature) != nil {
				return false
			}
		}
		return true
	default:
		// OpenRouter owns this tagged union and may add variants. A non-empty
		// discriminator is enough to preserve an unknown record byte-for-byte;
		// applying a closed allowlist here would silently erase continuation
		// state from a newer provider response.
		return true
	}
}

// replayableReasoningState returns a block's OpenRouter reasoning state when it
// is both tagged with this dialect and shaped the way the API documents.
// ReplayableAs compares only the format tag, and a state reaches the encoder
// from wherever the message was rebuilt — a store round-trip, a compaction, a
// hand-assembled turn — so the payload is checked here as well. An unusable
// payload degrades to absent rather than failing the call: that matches how
// ingest drops details it cannot read and how a foreign format tag is skipped,
// and it keeps a bad cached state from costing a turn that OpenRouter would
// otherwise answer. Sending the payload anyway is the one option with no upside
// — OpenRouter rejects it with a 400.
func replayableReasoningState(block content.Block) (json.RawMessage, bool) {
	thinking, ok := block.(*content.ThinkingBlock)
	if !ok || !thinking.ReplayableAs(openRouterReasoningDetailsFormat) {
		return nil, false
	}
	if !isReasoningDetailsArray(thinking.ProviderState) {
		return nil, false
	}
	return append(json.RawMessage(nil), thinking.ProviderState...), true
}

func hasReplayableReasoningDetails(req inference.Request) bool {
	for _, message := range req.Messages {
		ai, ok := message.(*content.AIMessage)
		if !ok {
			continue
		}
		for _, block := range ai.Blocks {
			if _, ok := replayableReasoningState(block); ok {
				return true
			}
		}
	}
	return false
}

// replayReasoningDetails re-attaches stored reasoning_details to the assistant
// turns they came from. Placement is positional, which couples this to the
// shared OpenAI encoder emitting exactly one wire message per neutral message
// plus one optional leading system message. That cardinality is asserted rather
// than assumed: a reasoning sequence landing on the wrong assistant turn is an
// ordering violation OpenRouter cannot recover from, so drift in the shared
// encoder must fail the request instead of silently misplacing signatures.
func replayReasoningDetails(body map[string]json.RawMessage, req inference.Request) error {
	if !hasReplayableReasoningDetails(req) {
		return nil
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		return fmt.Errorf("openrouter: decode messages for reasoning replay: %w", err)
	}
	systemOffset := 0
	if req.System != "" {
		systemOffset = 1
	}
	if want := len(req.Messages) + systemOffset; len(messages) != want {
		return fmt.Errorf("openrouter: reasoning replay expects %d encoded messages for %d neutral messages, got %d: the shared OpenAI encoder no longer emits one wire message per neutral message", want, len(req.Messages), len(messages))
	}
	wireIndex := systemOffset
	for _, message := range req.Messages {
		if ai, ok := message.(*content.AIMessage); ok {
			for _, block := range ai.Blocks {
				if state, ok := replayableReasoningState(block); ok {
					messages[wireIndex]["reasoning_details"] = state
					break
				}
			}
		}
		wireIndex++
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("openrouter: encode reasoning replay: %w", err)
	}
	body["messages"] = encoded
	return nil
}

// DecodeStream layers OpenRouter's reasoning_details onto the shared OpenAI
// stream decoder. The details are provider state, not content: they travel on
// content.ThinkingChunk's ProviderState fields and never inside a wire content
// field. Encrypted-only details (the Claude tool-loop shape) carry no readable
// text, so the shared decoder produces no thinking chunk to carry them; this
// wrapper synthesizes a zero-text one instead, ahead of the text or tool-use
// chunks decoded from the same delta so reasoning still precedes what it
// decided. The shared dialect is untouched: it is spoken by every
// OpenAI-compatible provider and has no reasoning_details member to extend.
func (requestCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	transformer := &reasoningResponseBody{source: resp.Body}
	resp.Body = transformer
	decoded, err := (openaiapi.Codec{}).DecodeStream(resp)
	if err != nil {
		return nil, err
	}
	var deferred content.Chunk
	// deferredErr holds a terminal that arrived while this wrapper still owed
	// the stream a state carrier. The carrier goes out first with a nil error
	// and the terminal follows on the next call, so the terminal is delayed by
	// exactly one chunk and never swallowed.
	var deferredErr error
	next := func() (content.Chunk, error) {
		if deferred != nil {
			chunk := deferred
			deferred = nil
			return chunk, nil
		}
		if deferredErr != nil {
			err := deferredErr
			deferredErr = nil
			return nil, err
		}
		chunk, err := decoded.Next()
		if err != nil {
			// Details captured after the last thinking chunk — or with no
			// thinking chunk at all — would otherwise be lost at the end of the
			// stream. This rescue applies to EVERY terminal, not only io.EOF: a
			// stream that dies on a dropped connection is exactly the case where
			// the encrypted state matters most, because it is the only
			// continuation the next request has and nothing can reconstruct it
			// from the text. Restricting the rescue to a clean EOF discarded it
			// precisely when the turn was about to be truncated — and a
			// truncated turn now KEEPS a sealed thinking block, so the rescued
			// carrier survives instead of vanishing with the failed step.
			if state := transformer.takeUndeliveredState(); state != nil {
				deferredErr = err
				return state, nil
			}
			return nil, err
		}
		if thinking, ok := chunk.(*content.ThinkingChunk); ok {
			transformer.attachState(thinking)
			return chunk, nil
		}
		if state := transformer.takePendingState(); state != nil {
			deferred = chunk
			return state, nil
		}
		return chunk, nil
	}
	result := func() (stream.StreamResult, bool, error) {
		value, ok := decoded.Result()
		return value, ok, nil
	}
	return stream.NewStreamReaderWithResult(next, decoded.Close, result), nil
}

// normalizeOpenRouterReasoning translates OpenRouter's reasoning response
// aliases into the reasoning_content field understood by the shared OpenAI
// decoder. The neutral content model carries reasoning as text, so structured
// reasoning details contribute their text/summary fields when available.
func normalizeOpenRouterReasoning(body []byte) ([]byte, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, err
	}
	rawChoices, ok := envelope["choices"]
	if !ok {
		return body, nil
	}
	var choices []map[string]json.RawMessage
	if err := json.Unmarshal(rawChoices, &choices); err != nil {
		return body, nil
	}

	changed := false
	for _, choice := range choices {
		field := "message"
		rawMessage, ok := choice[field]
		if !ok {
			field = "delta"
			rawMessage, ok = choice[field]
		}
		if !ok {
			continue
		}
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil {
			continue
		}

		if rawReasoningContent, exists := message["reasoning_content"]; exists {
			var reasoningContent string
			if err := json.Unmarshal(rawReasoningContent, &reasoningContent); err == nil && reasoningContent != "" {
				continue
			}
		}

		reasoning := ""
		if rawReasoning, exists := message["reasoning"]; exists {
			_ = json.Unmarshal(rawReasoning, &reasoning)
		}
		if reasoning == "" {
			reasoning = reasoningDetailsText(message["reasoning_details"])
		}
		if reasoning == "" {
			continue
		}

		normalizedReasoning, err := json.Marshal(reasoning)
		if err != nil {
			return nil, err
		}
		message["reasoning_content"] = normalizedReasoning
		updatedMessage, err := json.Marshal(message)
		if err != nil {
			return nil, err
		}
		choice[field] = updatedMessage
		changed = true
	}
	if !changed {
		return body, nil
	}

	updatedChoices, err := json.Marshal(choices)
	if err != nil {
		return nil, err
	}
	envelope["choices"] = updatedChoices
	return json.Marshal(envelope)
}

func reasoningDetailsText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var details []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &details); err != nil {
		return ""
	}
	var parts []string
	for _, detail := range details {
		for _, field := range []string{"text", "summary"} {
			var value string
			if err := json.Unmarshal(detail[field], &value); err == nil && value != "" {
				parts = append(parts, value)
				break
			}
		}
	}
	return strings.Join(parts, "\n")
}

// reasoningResponseBody rewrites complete SSE data lines while preserving the
// underlying response body's ownership and streaming behavior.
type reasoningResponseBody struct {
	source  io.ReadCloser
	pending []byte
	output  bytes.Buffer
	done    bool
	err     error

	// stateMu guards the captured reasoning state, which the transform side
	// writes while the decode side reads it back onto chunks.
	stateMu sync.RWMutex
	// reasoningEntries accumulates every reasoning_details record in arrival
	// order; OpenRouter documents the sequence as unrearrangeable.
	reasoningEntries []json.RawMessage
	// statePending marks details captured from a delta that yields no thinking
	// chunk of its own, so this wrapper owes the stream a synthetic carrier.
	statePending bool
	// deliveredEntries is how many entries the most recent carrier held, so end
	// of stream can tell a complete delivery from a stale partial one.
	deliveredEntries int
}

func (b *reasoningResponseBody) Read(p []byte) (int, error) {
	for b.output.Len() == 0 {
		if b.err != nil {
			return 0, b.err
		}
		if b.done {
			return 0, io.EOF
		}

		buf := make([]byte, 32*1024)
		n, err := b.source.Read(buf)
		if n > 0 {
			b.pending = append(b.pending, buf[:n]...)
			b.processLines(false)
		}
		if err != nil {
			// errors.Is, not ==: a bare io.EOF is only what the body happens to
			// return today. Any reader that adds framing context on the way out
			// would make an equality test false, and the else branch below
			// records the terminal as a FAILURE — skipping the atEOF flush, so
			// the final unterminated event (routinely the one carrying
			// finish_reason) is left in the pending buffer and silently dropped.
			// A clean finish would become a discard.
			if errors.Is(err, io.EOF) {
				b.done = true
				b.processLines(true)
				if b.output.Len() == 0 {
					return 0, io.EOF
				}
				b.err = io.EOF
			} else {
				b.err = err
			}
		}
	}
	return b.output.Read(p)
}

func (b *reasoningResponseBody) Close() error {
	return b.source.Close()
}

func (b *reasoningResponseBody) processLines(atEOF bool) {
	for {
		line, rest, ok := splitSSELine(b.pending, atEOF)
		if !ok {
			return
		}
		b.output.Write(b.transformSSELine(line))
		b.pending = rest
	}
}

// attachState puts the reasoning details captured so far on a thinking chunk
// the shared decoder produced, which settles this wrapper's debt to the stream.
func (b *reasoningResponseBody) attachState(thinking *content.ThinkingChunk) {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	encoded := b.encodeEntriesLocked()
	if encoded == nil {
		return
	}
	thinking.ProviderState = encoded
	thinking.ProviderStateFormat = openRouterReasoningDetailsFormat
	b.statePending = false
	b.deliveredEntries = len(b.reasoningEntries)
}

// takePendingState returns a zero-text carrier for details that arrived on a
// delta the shared decoder had no thinking chunk for, or nil when nothing is
// owed.
func (b *reasoningResponseBody) takePendingState() *content.ThinkingChunk {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	if !b.statePending {
		return nil
	}
	return b.newCarrierLocked()
}

// takeUndeliveredState returns a carrier for captured details no chunk has
// carried in full, or nil when the stream already delivered every entry.
func (b *reasoningResponseBody) takeUndeliveredState() *content.ThinkingChunk {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	if b.deliveredEntries >= len(b.reasoningEntries) {
		return nil
	}
	return b.newCarrierLocked()
}

func (b *reasoningResponseBody) newCarrierLocked() *content.ThinkingChunk {
	encoded := b.encodeEntriesLocked()
	if encoded == nil {
		return nil
	}
	b.statePending = false
	b.deliveredEntries = len(b.reasoningEntries)
	// Index is deliberately left at zero, and every carrier must keep it there.
	// The opaque state is the whole ordered reasoning_details array, not one
	// record: OpenRouter requires the sequence be replayed intact, so the array
	// is the indivisible unit. Folding successive carriers into a single block
	// at index 0 is what preserves that; spreading them across indexes would
	// split one array across several blocks and reorder it on replay.
	return &content.ThinkingChunk{
		ProviderState:       encoded,
		ProviderStateFormat: openRouterReasoningDetailsFormat,
	}
}

func (b *reasoningResponseBody) encodeEntriesLocked() json.RawMessage {
	if len(b.reasoningEntries) == 0 {
		return nil
	}
	encoded, err := json.Marshal(b.reasoningEntries)
	if err != nil {
		return nil
	}
	return encoded
}

func splitSSELine(data []byte, atEOF bool) (line, rest []byte, ok bool) {
	for i, value := range data {
		switch value {
		case '\n':
			return data[:i+1], data[i+1:], true
		case '\r':
			if i+1 == len(data) && !atEOF {
				return nil, data, false
			}
			end := i + 1
			if i+1 < len(data) && data[i+1] == '\n' {
				end++
			}
			return data[:end], data[end:], true
		}
	}
	if atEOF && len(data) > 0 {
		return data, nil, true
	}
	return nil, data, false
}

func (b *reasoningResponseBody) transformSSELine(line []byte) []byte {
	coreEnd := len(line)
	if coreEnd > 0 && line[coreEnd-1] == '\n' {
		coreEnd--
		if coreEnd > 0 && line[coreEnd-1] == '\r' {
			coreEnd--
		}
	} else if coreEnd > 0 && line[coreEnd-1] == '\r' {
		coreEnd--
	}
	core := line[:coreEnd]
	if !bytes.HasPrefix(core, []byte("data:")) {
		return line
	}
	value := core[len("data:"):]
	space := len(value) > 0 && value[0] == ' '
	if space {
		value = value[1:]
	}
	captured := b.captureReasoningDetails(value)
	normalized, err := normalizeOpenRouterReasoning(value)
	if err != nil {
		normalized = value
	}
	// Details with no readable text (Claude's encrypted reasoning) leave the
	// shared decoder with nothing to emit for this delta, so the state needs a
	// carrier synthesized on the decode side rather than a marker written into
	// the body the decoder is about to read.
	if captured && !hasDecodableReasoning(normalized) {
		b.markStatePending()
	}
	if err != nil || bytes.Equal(normalized, value) {
		return line
	}

	result := make([]byte, 0, len(line)+len(normalized)-len(value))
	result = append(result, core[:len("data:")]...)
	if space {
		result = append(result, ' ')
	}
	result = append(result, normalized...)
	result = append(result, line[coreEnd:]...)
	return result
}

func hasDecodableReasoning(body []byte) bool {
	var envelope struct {
		Choices []struct {
			Delta struct {
				ReasoningContent json.RawMessage `json:"reasoning_content"`
				Reasoning        json.RawMessage `json:"reasoning"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &envelope) != nil || len(envelope.Choices) == 0 {
		return false
	}
	for _, raw := range []json.RawMessage{envelope.Choices[0].Delta.ReasoningContent, envelope.Choices[0].Delta.Reasoning} {
		var value string
		if json.Unmarshal(raw, &value) == nil && value != "" {
			return true
		}
	}
	return false
}

func (b *reasoningResponseBody) captureReasoningDetails(value []byte) bool {
	var envelope struct {
		Choices []struct {
			Delta struct {
				ReasoningDetails []json.RawMessage `json:"reasoning_details"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(value, &envelope) != nil || len(envelope.Choices) == 0 || len(envelope.Choices[0].Delta.ReasoningDetails) == 0 {
		return false
	}
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	for _, detail := range envelope.Choices[0].Delta.ReasoningDetails {
		b.reasoningEntries = append(b.reasoningEntries, append(json.RawMessage(nil), detail...))
	}
	return true
}

func (b *reasoningResponseBody) markStatePending() {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()
	b.statePending = true
}

func (c config) hasBodyOptions() bool {
	return c.usage != nil || c.reasoning != nil || c.promptCacheKey != "" || c.providerRouting != nil
}

func cloneConfig(in config) config {
	out := in
	out.headers = in.headers.Clone()
	if in.usage != nil {
		out.usage = boolPtr(*in.usage)
	}
	if in.reasoning != nil {
		out.reasoning = cloneReasoningOptions(*in.reasoning)
	}
	if in.providerRouting != nil {
		out.providerRouting = cloneProviderRoutingOptions(*in.providerRouting)
	}
	return out
}

func cloneReasoningOptions(in ReasoningOptions) *ReasoningOptions {
	out := in
	out.MaxTokens = intPtrValue(in.MaxTokens)
	out.Exclude = boolPtrValue(in.Exclude)
	out.Enabled = boolPtrValue(in.Enabled)
	return &out
}

func cloneProviderRoutingOptions(in ProviderRoutingOptions) *ProviderRoutingOptions {
	out := in
	out.Order = append([]string(nil), in.Order...)
	out.AllowFallbacks = boolPtrValue(in.AllowFallbacks)
	out.RequireParameters = boolPtrValue(in.RequireParameters)
	out.ZDR = boolPtrValue(in.ZDR)
	return &out
}

func boolPtr(value bool) *bool { return &value }

func boolPtrValue(value *bool) *bool {
	if value == nil {
		return nil
	}
	return boolPtr(*value)
}

func intPtrValue(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
