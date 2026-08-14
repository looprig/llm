# CLAUDE.md — LLM Provider Development Guidelines

This module owns **providers**: the routing, auth, and per-vendor behaviour layered on
top of the shared API-format codecs in `inference/codec`. The workspace `AGENTS.md`
remains canonical for the dependency graph and release policy, and
`inference/AGENTS.md` for codec and schema rules — read that first, since everything
here builds on it.

## An API format is not a guarantee of identical behaviour

Roughly fifty providers here speak `openai` Chat, but the label describes an
*envelope*, not a contract. Providers under one format genuinely disagree:

- `max_tokens` is deprecated by OpenAI and rejected outright by its reasoning models,
  while most OpenAI-compatible servers accept only `max_tokens`. Snowflake Cortex
  requires `max_completion_tokens` and nothing else.
- OpenRouter returns errors with **HTTP 200** and adds `reasoning_details`.
- Azure emits an empty first streaming chunk carrying `prompt_filter_results`, and
  surfaces content filtering through `incomplete_details.reason` and `innererror`.

So: **treat the format as the baseline and the provider as a delta.** Never assume a
sibling provider behaves like its format, and never generalize one provider's quirk
into the shared dialect.

## Where a flavour belongs

Handle divergence **at the provider boundary**, leaving the shared codec untouched.
`providers/snowflake-cortex` is the pattern to follow: a documented `WithBodyPatch`
that renames one field, with a comment naming the vendor behaviour it accommodates.

The anti-pattern is equally instructive. `providers/azure` grew a private copy of the
Responses stream collector, so a fix that landed in the shared codec — malformed SSE
becoming an error — never reached it, and the defect survived behind a test that
asserted it. Before duplicating decode logic, ask whether the shared codec can expose
a seam instead; a private fork silently opts out of every future fix.

When you must fork, say so in a doc comment: what the vendor does differently, which
primary source documents it, and what would let the fork be deleted.

## Provider-specific rules

**Validate the encoded request.** `inference/codec/conformance` holds every encode
path against the format's official request schema. Wire `MustValidateRequest` into
every test that produces a request body, before its structural assertions. A gate that
is never called proves nothing — verify coverage by inverting the gate's condition and
watching tests fail, not by assuming the handler ran.

**Fixtures are only evidence if they are legal.** Every fixture is schema-validated on
every run, before any decoder sees it, so a wrong fixture cannot manufacture a wrong
conclusion. The gate walks the fixture directories, so new files are covered
automatically; keep it that way.

**Be explicit about what is schema-backed.** Vendor extensions such as OpenRouter's
`reasoning_details` pass validation only because the base schema does not close
`additionalProperties` — they are not checked. Label decode-only assertions as such
rather than implying schema backing.

**Reject locally what the provider cannot accept.** Bedrock's InvokeModel body takes
inline base64 images only, while first-party Anthropic accepts URLs — and both go
through the same encoder. Where a shared encoder can emit something a provider
forbids, catch it at the provider boundary, at the single chokepoint every caller
shares (`toBedrockBody` covers both inference and CountTokens; a check in one caller
would have left the other leaking).

**Cross-provider replay is a real path.** Tool ids minted by one provider are replayed
to another when a session switches models: Converse mints ids containing `.` and `:`,
which Anthropic's `^[a-zA-Z0-9_-]+$` rejects; OpenAI's `temperature` reaches 2 where
Anthropic bounds it to 1. Validate against the destination's constraints, not the
origin's.

**Never forward provider-opaque state across dialects.** Check `ReplayableAs` with the
provider's own format tag before replaying reasoning state, and validate its shape on
the way out as well as in — a payload restored from a store or a compaction can be
structurally valid and still wrong.

**Auth and routing are provider concerns**, not format concerns: base URLs, API
versions, header names, and credential types belong here. Normalize user-supplied base
URLs — a documented trailing slash should not fail every request.

## Testing

Run the affected provider packages together with `inference`; a codec change routinely
moves fixtures and byte counts here. Prefer real captured request bodies over
hand-written expectations, and never weaken a fixture or a gate to make a test pass —
if the gate rejects, the encoder is wrong until proven otherwise.
