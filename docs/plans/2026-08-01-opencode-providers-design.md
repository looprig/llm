# OpenCode Priority Providers Design

**Date:** 2026-08-01

**Status:** Approved for implementation

## Goal

Add first-class OpenAI, Anthropic, and xAI providers to the `llm` module while
preserving the neutral `inference.Client`, `inference.Request`, and
`contextcount.ContextCounter` contracts already used by Gemini and OpenRouter.

## Verified protocol choice

OpenAI and xAI will use their native Responses APIs, not Chat Completions:

| Provider | Canonical base | Inference route | Counter route | Authentication |
| --- | --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `POST /responses` | `POST /responses/input_tokens` | `Authorization: Bearer <OPENAI_API_KEY>` |
| Anthropic | `https://api.anthropic.com/v1` | `POST /messages` | `POST /messages/count_tokens` | `x-api-key: <ANTHROPIC_API_KEY>` and `anthropic-version: 2023-06-01` |
| xAI | `https://api.x.ai/v1` | `POST /responses` | no full-request counter endpoint | `Authorization: Bearer <XAI_API_KEY>` |

OpenCode's current provider loader selects `sdk.responses(modelID)` for both
OpenAI and xAI. OpenAI documents the Responses streaming lifecycle and input
token endpoint. xAI explicitly describes Responses as the recommended API and
documents that Chat Completions does not return reasoning content. Anthropic's
Messages API is a separate native protocol with top-level system content,
content blocks, tool use/results, thinking, and named SSE events.

Sources checked during design:

- [OpenCode provider source](https://raw.githubusercontent.com/anomalyco/opencode/dev/packages/opencode/src/provider/provider.ts)
- [OpenCode provider documentation](https://opencode.ai/v2/docs/providers)
- [OpenAI Responses streaming reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)
- [OpenAI input-token reference](https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens)
- [xAI Responses comparison](https://docs.x.ai/developers/model-capabilities/text/comparison)
- [xAI inference reference](https://docs.x.ai/developers/rest-api-reference/inference)
- [Anthropic Messages reference](https://platform.claude.com/docs/en/api/messages)
- [Anthropic token-counting reference](https://platform.claude.com/docs/en/api/messages/count_tokens)
- [Anthropic API versioning](https://platform.claude.com/docs/en/api/versioning)

## Architecture

### Provider packages

Create separate packages:

- `providers/openai`: OpenAI constructor, options, Responses request/response
  adaptation, exact input-token counter, and deterministic tests.
- `providers/anthropic`: Anthropic constructor, options, Messages headers and
  body adaptation, exact Messages token counter, and deterministic tests.
- `providers/xai`: xAI constructor, options, Responses adaptation, explicit
  unsupported full-request counter classification, and deterministic tests.

The packages will follow the OpenRouter constructor pattern: validate the
selected model, require a non-empty API key, resolve the selected model's base
URL or the canonical default, then construct `transport.Client` with a static
route, codec, and `auth.Key`/`auth.Header` authenticator. Credentials never
enter model descriptors, options, logs, or errors.

### Shared Responses codec

Keep wire-dialect ownership in `inference`. The sibling repository already
contains the released `codec/openairesponses` package and the additive
`model.APIFormatOpenAIResponses` constant. `llm` will consume that codec and
vendor the package from the `inference` v0.6.0 line. The existing
`codec/openaiapi` Chat Completions package remains unchanged and continues to
serve OpenRouter, LMStudio, Chutes, Phala, and other Chat-compatible callers.

If the dependency audit finds a missing Responses fix, make that fix in
`inference` first, test and release it there, then update `llm`; do not fork a
second Responses codec in `llm`. The `inference` codec owns the shared
OpenAI/xAI Responses wire shape:

- Encode neutral system and conversation messages as Responses `instructions`
  and typed `input` items.
- Encode tools as function tools and required tool behavior as Responses
  `tool_choice`.
- Encode neutral structured output as Responses `text.format` JSON schema.
- Encode sampling limits and reasoning effort as Responses fields.
- Decode output message text, reasoning summary/content, and function calls into
  `TextBlock`, `ThinkingBlock`, and `ToolUseBlock`.
- Decode usage, cached input tokens, reasoning/output token details, and
  completed/incomplete/tool-call finish statuses into shared usage and finish
  reason values.
- Decode Responses SSE events (`response.created`, output item events, text and
  function-call deltas, reasoning deltas, and response completion/failure) into
  the existing stream chunks and terminal result.
- Reject malformed terminal responses and provider error events with the same
  bounded, typed failure behavior used by the existing codecs.

The `llm` update will be a minimal vendor synchronization for the released
Responses codec and its direct first-party domain dependencies, rather than a
bulk `go mod vendor` regeneration. Any source files copied from the newer
`inference` release will be listed and justified in the final handoff.

### Anthropic adaptation

Use the existing vendored `anthropicapi.Codec` for the neutral Messages mapping
and stream decoder, wrapping its encoded JSON to add stable provider options and
injecting the required `x-api-key`, `anthropic-version`, `Accept`, and optional
beta headers. The adapter will preserve native semantics instead of converting
Anthropic to OpenAI-shaped messages:

- top-level system prompt;
- alternating user/assistant content blocks;
- `tool_use` and `tool_result` blocks, including tool errors;
- thinking configuration and thinking blocks;
- cache-control content-block options;
- structured output and tool-choice fields where supported;
- named SSE events including text, thinking, tool-input JSON, message usage, and
  terminal stop reasons.

### Registry and auto selection

Extend the provider registry and validation truth table with:

- `ProviderOpenAI`, `ProviderAnthropic`, and `ProviderXAI`;
- their API-format support and empty-base-URL defaults;
- API-key authentication requirements;
- auto-constructor dispatch to the dedicated packages;
- exact counter dispatch for OpenAI and Anthropic and an explicit typed
  unavailable result for xAI;
- no changes to Gemini or existing OpenRouter behavior.

## Stable provider options

Expose only options that map directly to documented provider fields:

- OpenAI/xAI: reasoning effort/summary controls, service tier, metadata,
  prompt-cache key, and tool behavior where the provider accepts it.
- Anthropic: thinking controls, cache-control TTL, metadata, tool behavior,
  service tier, and explicitly requested beta headers.
- All three retain the shared neutral structured-output request and sampling
  fields rather than duplicating generic API surface.

Options will be immutable-by-convention through defensive copies, omit unset
fields, preserve explicit false/zero values where the provider distinguishes
them, and validate option values before network I/O.

## Counter behavior

- OpenAI: send the same encoded Responses input to `/responses/input_tokens`,
  decode `{object:"response.input_tokens",input_tokens:n}`, and report
  `CountQualityExactProvider`.
- Anthropic: send the Messages input shape to `/messages/count_tokens`, decode
  `{input_tokens:n}`, and report `CountQualityExactProvider`.
- xAI: do not pretend that `/v1/tokenize-text` is a complete request counter;
  it tokenizes a text string and does not account for the full neutral request,
  tools, images, or provider framing. Return the existing typed
  `CounterSupportError` and keep auto selection fail-closed.

Counters will include bounded response reads, context cancellation/timeouts,
model/provider binding checks, duplicate/malformed JSON rejection, API error
mapping, and capability metadata consistent with the existing exact counters.

## Tests

Each provider package will have deterministic `httptest` coverage for:

- constructor validation, canonical/default and selected base URLs;
- authentication, required and optional headers, and path/method checks;
- request JSON normalization for system/messages/content blocks/tools/tool
  results/structured output/provider options;
- JSON response text, reasoning, tool calls, usage, and finish reasons;
- native SSE stream framing, text/reasoning/tool deltas, terminal usage, and
  malformed/unknown events;
- malformed JSON, provider error status, missing fields, duplicate fields, and
  bounded body handling;
- exact counter request/response accounting for OpenAI and Anthropic and the
  explicit xAI unsupported-counter result;
- auto registry and dispatch coverage.

Tests will run against loopback-only `httptest` servers and will never use live
credentials or provider endpoints.

## Release and verification

Implementation occurs on `feat/opencode-providers` in the isolated `llm`
worktree. After all tests and requested security checks pass, review the diff
for unrelated changes, commit the feature, merge it into local `main`, bump the
feature release from `v0.5.0` to `v0.6.0`, create annotated tag `v0.6.0`, and
push only `main` and that tag to the configured `origin` remote. The pre-existing
main-worktree edit to vendored transport remains untouched throughout.
