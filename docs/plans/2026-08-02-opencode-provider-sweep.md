# OpenCode Provider Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the remaining explicitly documented OpenCode provider integrations to llm with explicit provider policy, stable options, shared semantic normalization, deterministic HTTP tests, and release integration.

**Architecture:** Keep inference neutral and reuse its OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini, and Bedrock codecs. Add a small internal providers/internal/compat layer for common OpenAI-compatible construction, route/header composition, provider-local body overlays, errors, and unsupported exact-counter results. Add separate public provider packages so provider names, defaults, auth, options, and tests remain explicit and fail-closed. Add new inference codecs only for genuinely native wire semantics that cannot be represented by the existing codecs.

**Tech Stack:** Go, inference codecs and transport, httptest, SSE fixtures, go test -race, go vet/staticcheck/gosec, govulncheck, vendored dependencies.

---

### Task 1: Freeze the provider/API/auth matrix

**Files:**

- Modify: llm.go
- Modify: provider.go
- Modify: auto/auto.go
- Modify: auto/counter.go
- Test: provider_test.go, auto/auto_test.go, auto/counter_test.go
- Reference: docs/plans/2026-08-02-opencode-provider-sweep-design.md

**Step 1: Record current OpenCode source facts**

Use the checked-out OpenCode providers.mdx, provider loader source, and Models.dev definitions to record each provider ID, endpoint, API format, auth method, and documented options. Verify each endpoint and auth method against that provider's official API documentation before its implementation batch.

**Step 2: Write failing registry tests**

Add table-driven cases for every new provider's canonical Provider, supported API format, empty-base policy, required auth, and auto/counter construction boundary. Include wrong-provider, wrong-format, unknown-provider, and unavailable-credential cases.

**Step 3: Run the focused tests and verify red**

Run: GOWORK=off go test ./... -run 'Test(Provider|Auto|Counter)'

Expected: failures identify the new provider constants and unclassified policy entries.

**Step 4: Implement minimal registry classification**

Add constants and fail-closed policy cases. Do not wire a constructor until its package exists; unsupported branches return typed validation/direct-construction errors.

**Step 5: Run focused tests and commit**

Run the focused suite again, then commit the registry classification as feat: register OpenCode provider identities.

---

### Task 2: Build and test the shared OpenAI-compatible provider core

**Files:**

- Create: providers/internal/compat/client.go
- Create: providers/internal/compat/options.go
- Create: providers/internal/compat/errors.go
- Create: providers/internal/compat/counter.go
- Create: providers/internal/compat/client_test.go
- Create: providers/internal/compat/options_test.go
- Create: providers/internal/compat/counter_test.go

**Step 1: Write failing shared contract tests**

Cover construction binding/auth, default and explicit base URLs, JSON requests/responses, SSE deltas and terminal usage, malformed JSON/SSE, bounded non-2xx errors, reasoning, tool calls/results, structured output, finish reasons, header overlays, and body options.

**Step 2: Run the tests to verify red**

Run: GOWORK=off go test ./providers/internal/compat -count=1

Expected: package/build failures because the compatibility implementation is absent.

**Step 3: Implement the minimal shared core**

Wrap openaiapi.Codec with a route that adds documented headers and a request codec that applies only explicit stable body overlays. Preserve existing normalization and stream behavior. Return a typed CounterSupportError for providers without an exact token endpoint.

**Step 4: Run tests and commit**

Run the package tests and race tests. Keep provider identity in the transport endpoint so cross-provider requests fail before I/O. Commit as feat: add shared OpenAI-compatible provider core.

---

### Task 3: Add documented OpenAI-compatible providers

**Files:** Create one package per provider, each with client.go, options.go, counter.go, errors.go, and deterministic tests:

- providers/p302ai
- providers/atomicchat
- providers/azurecognitive
- providers/baseten
- providers/cerebras
- providers/cloudflare
- providers/cloudflareworkers
- providers/cortecs
- providers/deepseek
- providers/deepinfra
- providers/digitalocean
- providers/frogbot
- providers/fireworks
- providers/gmicloud
- providers/groq
- providers/huggingface
- providers/helicone
- providers/llamacpp
- providers/ionet
- providers/moonshot
- providers/minimax
- providers/nebius
- providers/ollama
- providers/ollamacloud
- providers/opencode
- providers/stackit
- providers/ovhcloud
- providers/scaleway
- providers/together
- providers/venice
- providers/zai
- providers/zenmux

**Step 1: Write each provider's focused failing test**

Assert provider ID, canonical base URL/path, auth header, documented headers/body fields, request/response normalization, SSE behavior, and typed unsupported-counter result. Use local httptest servers only.

**Step 2: Run each focused test to verify red**

Run the package-specific test before implementation. The failure must identify missing behavior, not a malformed fixture.

**Step 3: Implement each thin provider package**

Use the shared core only for facts verified to be OpenAI-compatible. Expose only documented reasoning, response format, cache, service tier, gateway metadata, or beta-header options. Do not return undocumented token estimators.

**Step 4: Run small batches and commit**

After each batch, run GOWORK=off go test ./providers/... -count=1 and the relevant race tests. Commit coherent core, gateway, local, and hosted batches separately.

---

### Task 4: Add provider-specific gateway and metadata behavior

**Files:**

- Modify: relevant packages from Task 3
- Create/Modify: providers/internal/compat/headers.go
- Create/Modify: providers/internal/compat/body.go
- Test: provider-specific option and stream fixtures

**Step 1: Write failing tests from official docs**

Cover Cloudflare AI Gateway route/metadata, Cloudflare Workers AI model-in-path routing, Helicone observability headers, LLM Gateway provider/model routing, SAP AI Core deployment headers, Vercel AI Gateway headers, Venice options, and ZenMux routing.

**Step 2: Run focused tests to verify red**

Use deterministic request capture and assert credentials never appear in errors or logs.

**Step 3: Implement route/header/body overlays**

Keep provider-specific behavior in local wrappers or the compatibility layer. Do not broaden inference for gateway metadata that is not shared semantic content.

**Step 4: Run focused tests and commit**

Run the relevant provider packages and commit as feat: support OpenCode gateway provider options.

---

### Task 5: Add native and authentication-special providers

**Files:**

- Create: providers/vertex/*
- Create: providers/githubcopilot/*
- Create: providers/gitlab/*
- Create: providers/snowflake/*
- Create/Modify: providers/sap/* if native deployment semantics require it
- Modify: inference/codec/* only when an existing codec cannot represent the wire protocol
- Test: provider-local JSON/SSE/auth/error/usage suites

**Step 1: Write failing native-semantics tests**

Cover Vertex Gemini and Vertex-hosted Anthropic selection/auth, GitHub Copilot OAuth/bearer and model routing, GitLab Duo proxy headers and Anthropic semantics, Snowflake Cortex token/base URL construction, and provider-specific stream envelopes.

**Step 2: Run focused tests to verify red**

Use injected local endpoints and explicit test credentials; no test may depend on external credentials.

**Step 3: Implement native clients/codecs and direct-construction boundaries**

Reuse geminiapi or anthropicapi when semantics match. Add provider-specific authenticators/options for OAuth, GCP, or JWT only in llm. auto.New returns typed direct-construction errors when model plus API key cannot provide required credentials.

**Step 4: Run native tests and commit**

Run native provider packages and auto tests, then commit as feat: add native OpenCode provider integrations.

---

### Task 6: Complete registry, auto selection, and counter behavior

**Files:**

- Modify: auto/auto.go
- Modify: auto/counter.go
- Modify: provider.go
- Modify: llm.go
- Test: auto/auto_test.go, auto/counter_test.go, provider_test.go

**Step 1: Add failing matrix cases**

For every provider, assert validation, auth, default base behavior, constructor type, unsupported/direct-construction errors, and counter routing. Add positive auto cases only when credentials are sufficient.

**Step 2: Implement dispatch and fail-closed boundaries**

Do not make auto.New read arbitrary environment variables or synthesize OAuth/GCP/AWS credentials. Preserve Bedrock/Phala-style direct-construction guidance.

**Step 3: Run the matrix**

Run: GOWORK=off go test ./... -count=1 and GOWORK=off go test -race ./... -count=1

**Step 4: Commit registry completion**

Commit as feat: wire OpenCode providers into llm policy.

---

### Task 7: Documentation, formatting, and independent review

**Files:**

- Modify: provider package docs and relevant root documentation
- Modify: docs/plans/2026-08-02-opencode-provider-sweep.md if implementation decisions change

**Step 1: Reconcile every provider with source and official docs**

Check endpoint, auth header, route, stream framing, options, and counter boundary. Record any provider intentionally excluded from auto construction or exact counting.

**Step 2: Run formatting and diff checks**

Run gofmt on changed Go files, git diff --check, git status --short, and git diff --stat. Review for accidental deletions, vendor changes, generated files, or unrelated edits.

**Step 3: Request and address the medium review**

Obtain the requested GPT Sol medium review. Inspect every finding against source/docs/tests, implement verified fixes, and rerun affected tests.

**Step 4: Commit documentation/review fixes**

Commit as docs: finalize OpenCode provider coverage.

---

### Task 8: Full verification and release integration

**Step 1: Run every required check fresh**

Run:

    GOWORK=off go test -race -count=1 ./...
    make fmt-check vendor-check
    go mod verify
    make lint
    go tool govulncheck ./...

Use temporary cache directories outside the repository when needed. Do not claim completion or release readiness until every command exits successfully.

**Step 2: Review the final diff**

Confirm the feature worktree contains only the provider sweep, docs, and tests. Keep the pre-existing vendor/github.com/looprig/inference/transport/client.go edit on main untouched.

**Step 3: Merge and bump semantic version**

Commit version metadata in the feature branch, merge into local main without overwriting unrelated work, and bump the next appropriate semantic version.

**Step 4: Tag and push after verification**

Create an annotated semantic-version tag, push main and the tag, then verify remote refs and final worktree status.

