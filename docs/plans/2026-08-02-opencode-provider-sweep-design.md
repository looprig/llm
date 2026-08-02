# OpenCode Provider Sweep Design

## Goal

Extend `llm` to cover the remaining explicitly documented OpenCode provider integrations while preserving the neutral `inference` contract and the existing provider/API-format policy model.

The scope is the provider directory in OpenCode's current documentation. Providers already implemented in this module are retained and not reimplemented: OpenAI, Anthropic, xAI, Google Gemini, Amazon Bedrock, Azure OpenAI, OpenRouter, and LM Studio. The broader Models.dev catalog and arbitrary custom providers remain out of scope because they are data-driven aliases rather than a stable set of provider APIs.

## Provider coverage

The missing documented integrations are:

- 302.AI, Atomic Chat, Azure Cognitive Services, Baseten, Cerebras.
- Cloudflare AI Gateway, Cloudflare Workers AI, Cortecs, DeepSeek, Deep Infra, DigitalOcean.
- FrogBot, Fireworks AI, GitLab Duo, GitHub Copilot, GMI Cloud, Google Vertex AI.
- Groq, Hugging Face, Helicone, llama.cpp, IO.NET, Moonshot AI, MiniMax.
- NVIDIA, Nebius Token Factory, Ollama, Ollama Cloud, OpenCode Zen, LLM Gateway.
- SAP AI Core, STACKIT, OVHcloud AI Endpoints, Scaleway, Snowflake Cortex.
- Together AI, Venice AI, Vercel AI Gateway, Z.AI, and ZenMux.

The exact current OpenCode documentation and local OpenCode source are the catalog authority. Each provider's own API documentation is the wire/authentication authority; OpenCode source is used to confirm provider IDs, default URLs, SDK choice, headers, and option translation.

## Architecture

`inference` remains provider-neutral. Existing codecs are reused when the provider is genuinely compatible:

- OpenAI Chat Completions: `inference/codec/openaiapi`.
- OpenAI Responses: `inference/codec/openairesponses`.
- Anthropic Messages: `inference/codec/anthropicapi`.
- Gemini generateContent: `inference/codec/geminiapi`.
- Bedrock InvokeModel/Converse: existing Bedrock provider codecs.

The `llm` module gains a small internal compatibility layer for common HTTP construction, API-key/bearer headers, provider metadata, generic unsupported-counter errors, and shared deterministic test helpers. Every public provider remains a separate package with its own constructor, default endpoint/auth policy, option type, counter constructor, errors, and tests. Thin packages may delegate to the compatibility layer, but provider identity and validation remain explicit and fail-closed.

New inference codecs are added only for a documented provider whose wire protocol cannot be represented by an existing codec. Provider-local wrappers are preferred when the divergence is limited to headers, route shape, or provider-specific response fields. No provider is labeled OpenAI-compatible solely because its SDK package happens to use an OpenAI-compatible adapter.

## Auth and construction

Constructors accept explicit credentials, never hard-code secrets, and follow the existing `auth` conventions. Environment variables are used only for provider-documented default discovery (for example resource names, project/region, or provider tokens), with explicit constructor/options values taking precedence. Providers requiring credentials unavailable to `auto.New`—OAuth, GCP credentials, AWS signing, or multi-field configuration—are directly constructible and return typed direct-construction errors from auto selection rather than silently using incomplete auth.

Provider defaults and options are limited to stable documented controls: base URL/resource or deployment, API key/bearer token, reasoning effort or budget, response format, tool behavior, prompt caching, service tier, metadata, documented custom headers, and beta/feature headers. Undocumented SDK internals and broad arbitrary option maps are not exposed as public API.

## Request and response behavior

Each provider must preserve the shared message model, including:

- system/developer/user messages and multimodal content;
- assistant tool calls and provider tool-call IDs;
- tool results, including multiple results and structured content;
- reasoning/thinking content and provider state needed for continuation;
- structured-output requests and response-format validation;
- usage accounting and finish-reason normalization;
- JSON responses, SSE streams, malformed frames, and bounded non-2xx errors.

Streaming decoders must tolerate provider keep-alives and documented terminal events while rejecting or reporting malformed payloads according to existing codec conventions. Usage is accumulated from terminal usage events or response metadata without double-counting streamed deltas.

Exact remote context counters are implemented only where the provider documents a matching endpoint and request semantics. Other providers expose the existing typed `CounterSupportError` boundary; they do not return an undocumented estimator.

## Testing and delivery

Implementation is test-first. Shared contract tests cover JSON, SSE, malformed responses, HTTP errors, headers/options, reasoning, tool calls/results, structured output, finish reasons, and usage. Provider-specific tests cover endpoint/default/auth behavior and any native deviations. Registry, model validation, auto construction, and counter matrices are updated in lockstep.

The feature is developed in an isolated worktree in small provider/API batches. After implementation, run the repository's full race test, formatting/vendor checks, module verification, lint, and vulnerability scan. Review the complete diff for deletions or unrelated edits, obtain the requested medium review, then commit the feature, merge it into local `main`, bump the next appropriate semantic version, create an annotated tag, and push `main` plus the tag only after all checks pass.
