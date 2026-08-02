# OpenCode Provider Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct the provider identity, protocol, resilience, attribution, and contract-test gaps found by the GPT Sol medium review of the OpenCode provider sweep.

**Architecture:** Keep the neutral inference API and existing shared Chat, Responses, Anthropic, and Gemini codecs. Add only provider-specific adapters where the official wire contract requires them: a separate hosted Meta Llama package, OpenAI-compatible GMI routing, GitLab model mapping, SAP model-parameter patching, and Snowflake response normalization. Preserve the documented local `llama.cpp` provider under a distinct identity.

**Tech Stack:** Go, `httptest`, existing `providers/internal/compat` and `providers/internal/simple` adapters, OpenCode source fixtures, official provider API contracts.

---

### Task 1: Correct Llama identities and GMI protocol

**Files:**
- Modify: `llm.go`, `provider.go`, `auto/auto.go`, `auto/counter.go`, provider matrix tests
- Create: `providers/llama/client.go`, `providers/llama/client_test.go`, `providers/llama/counter.go`, `providers/llama/errors.go`, `providers/llama/options.go`
- Modify: `providers/llamacpp/*`, `providers/gmicloud/*`

**Step 1: Write failing tests**

Add tests proving hosted `llama` uses `https://api.llama.com/compat/v1`, `LLAMA_API_KEY`, bearer auth, and Chat Completions; local `llama.cpp` uses `http://127.0.0.1:8080/v1` with no auth; GMI accepts OpenAI Chat, uses bearer auth, and routes to `/chat/completions`.

**Step 2: Run focused tests and verify failure**

Run: `GOWORK=off go test ./providers/llama ./providers/llamacpp ./providers/gmicloud ./...`

Expected: failures showing the current identity and GMI Anthropic route mismatch.

**Step 3: Implement the minimum correction**

Introduce `ProviderLlamaCPP = "llama.cpp"`; make `ProviderLlama = "llama"` an authenticated hosted provider; dispatch both identities separately. Keep `llamacpp.New` temporarily compatible with legacy model identity construction while auto-selection uses the explicit local identity. Convert GMI to OpenAI Chat with bearer authentication. Keep options limited to documented headers/reasoning controls.

**Step 4: Run focused tests**

Run: `GOWORK=off go test -count=1 ./providers/llama ./providers/llamacpp ./providers/gmicloud ./auto`

Expected: PASS.

### Task 2: Add GitLab upstream model mapping

**Files:**
- Modify: `providers/gitlab/client.go`, `providers/gitlab/client_test.go`, `providers/gitlab/auth.go`
- Create or modify: `providers/gitlab/model_mapping.go`, `providers/gitlab/model_mapping_test.go`

**Step 1: Write failing mapping tests**

Cover the current upstream mappings for `duo-chat-opus-4-6`, `duo-chat-sonnet-4-6`, `duo-chat-haiku-4-5`, GPT chat models, GPT Codex Responses models, and an unknown alias. Assert the outbound model field and required API format.

**Step 2: Run tests and verify failure**

Run: `GOWORK=off go test ./providers/gitlab -run 'Model|Alias'`

Expected: current aliases remain unchanged or route-family validation is absent.

**Step 3: Implement mapping and bounded auth refresh**

Use a maintained internal table based on the current GitLab provider package. Rewrite only the outbound model field, reject an API-format mismatch, and invalidate the cached direct-access token after an inference 401 without retrying unboundedly.

**Step 4: Run focused tests**

Run: `GOWORK=off go test -count=1 ./providers/gitlab`

Expected: PASS, including exchange caching and one-time refresh coverage.

### Task 3: Correct SAP semantics and retry behavior

**Files:**
- Modify: `providers/sap-ai-core/client.go`, `providers/sap-ai-core/client_test.go`, `providers/sap-ai-core/errors.go`

**Step 1: Write failing tests**

Assert documented provider-specific model parameters are preserved for SAP requests, deployment discovery selects the documented deployment, and a canceled/transient discovery request can be retried successfully.

**Step 2: Run tests and verify failure**

Run: `GOWORK=off go test ./providers/sap-ai-core -run 'ModelParams|Retry|Discovery'`

Expected: current request lacks SAP-specific model parameters or permanently reuses the first error.

**Step 3: Implement the minimal adapter**

Preserve SAP’s documented Harmonized `/v2/chat` message envelope. Add narrowly scoped provider options/body patching for documented model parameters, and cache only successful resolution or immutable configuration errors. Never cache context cancellation, transport errors, 401/429, or 5xx failures. Do not replace the valid Harmonized endpoint with an inferred native orchestration protocol.

**Step 4: Run focused tests**

Run: `GOWORK=off go test -count=1 ./providers/sap-ai-core`

Expected: PASS.

### Task 4: Add Snowflake normalization

**Files:**
- Modify: `providers/snowflake-cortex/client.go`, `providers/snowflake-cortex/client_test.go`, `providers/snowflake-cortex/options_test.go`
- Modify: `providers/internal/compat/normalize.go` only if a reusable line-oriented SSE transform is needed

**Step 1: Write failing fixtures**

Cover an SSE chunk containing `"role":""` and a 400 JSON error containing `conversation complete`, expecting a normal stopped response/stream rather than a decode or API error.

**Step 2: Run tests and verify failure**

Run: `GOWORK=off go test ./providers/snowflake-cortex -run 'Normalize|Conversation'`

Expected: current strict OpenAI decoding fails or returns the 400 error.

**Step 3: Implement response/stream normalization**

Use `compat.Definition.NormalizeResponse` and `NormalizeStream` to match OpenCode’s documented transforms without changing other OpenAI-compatible providers; add a transport-level status normalization hook only if response decoding cannot safely preserve the original non-2xx response.

**Step 4: Run focused tests**

Run: `GOWORK=off go test -count=1 ./providers/snowflake-cortex`

Expected: PASS.

### Task 5: Strengthen shared contracts and remove foreign attribution defaults

**Files:**
- Modify: `providers/internal/contracttest/provider.go` and contract callers
- Modify: `providers/llmgateway/client.go`, `providers/zenmux/client.go`, `providers/vercel/client.go`, `providers/cerebras/client.go`
- Modify: relevant tests and `providers/opencode/client.go` identity checks

**Step 1: Write failing contract tests**

Add terminal usage assertions for Chat SSE, Responses tool/structured/reasoning fixtures, malformed/truncated SSE checks for wrapped providers, and constructor identity rejection for `opencode` versus `opencode-go`.

**Step 2: Run tests and verify failure**

Run: `GOWORK=off go test ./providers/internal/compat ./providers/internal/contracttest ./providers/llmgateway ./providers/zenmux ./providers/vercel ./providers/cerebras ./providers/opencode ./providers/opencode-go`

Expected: missing coverage or foreign attribution headers fail the assertions.

**Step 3: Implement corrections**

Make library attribution absent by default; retain explicit header options. Keep provider-specific required headers only. Add the missing contract cases and restrict constructors to their own provider identity.

**Step 4: Run focused tests**

Run: `GOWORK=off go test -count=1 ./providers/internal/... ./providers/llmgateway ./providers/zenmux ./providers/vercel ./providers/cerebras ./providers/opencode ./providers/opencode-go`

Expected: PASS.

### Task 6: Final verification and release

**Step 1: Run the full required checks**

Run:

- `GOWORK=off go test -race -count=1 ./...`
- `GOWORK=off make fmt-check vendor-check`
- `GOWORK=off go mod verify`
- `GOWORK=off make lint`
- `GOWORK=off go tool govulncheck ./...`

Expected: all PASS, with no `go.mod`, `go.sum`, or vendor changes.

**Step 2: Review and commit**

Review `git diff --check`, staged name/status for deletions or unrelated paths, then commit the corrective feature.

**Step 3: Integrate and release**

Merge into local `main`, create the next major release after the breaking Llama/GMI identity corrections as annotated `v1.0.0`, push only `main` and `v1.0.0`, and verify remote refs. Preserve the pre-existing unrelated vendor edit.
