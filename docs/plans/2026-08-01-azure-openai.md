# Azure OpenAI Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenCode-compatible Azure OpenAI Responses API support to the `llm` module with API-key authentication, resource-name endpoint resolution, shared request/response semantics, and explicit unsupported counter behavior.

**Architecture:** Add a dedicated `providers/azure` package that wraps the existing `inference/codec/openairesponses` codec and transport. Register `ProviderAzure` in the llm policy and auto composition roots. Resolve an explicit model base URL first, then a resource option, then `AZURE_RESOURCE_NAME` into `https://<resource>.openai.azure.com/openai/v1`; use `api-key` authentication. Do not add an inference codec or Azure Cognitive Services mode.

**Tech Stack:** Go, existing inference Responses codec/transport, `httptest`, race-enabled package tests, existing module/vendor workflow.

---

### Task 1: Add failing provider-policy and auto-registration tests

**Files:**
- Modify: `llm.go`
- Modify: `provider.go`
- Modify: `validate_test.go`
- Modify: `provider_test.go`
- Modify: `auto/auto.go`
- Modify: `auto/auto_test.go`
- Modify: `auto/counter.go`
- Modify: `auto/counter_test.go`

**Step 1: Write the failing tests**

Add fixtures and assertions for:

- `ProviderAzure` requiring `auth.AuthAPIKey`.
- Azure accepting only `APIFormatOpenAIResponses` and allowing an empty base URL for provider resolution.
- `auto.New` dispatching a valid Azure model with an explicit base URL and rejecting an empty key.
- `auto.NewCounter` returning `*llm.CounterSupportError` for Azure rather than falling through to an unclassified-provider error.

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
GOWORK=off go test ./... -run 'TestProvider|TestValidateModel|TestNew|TestNewCounter'
```

Expected: compilation or assertion failures because the Azure provider constant and registry cases do not exist yet.

**Step 3: Implement the minimal registry wiring**

Add the provider constant and policy cases, import the Azure package in `auto`, dispatch `auto.New` to `azure.New`, and classify Azure as exact-counter-unavailable in `auto.NewCounter`.

**Step 4: Run the focused tests to verify they pass**

Run the same focused command and expect PASS.

**Step 5: Commit**

```bash
git add llm.go provider.go validate_test.go provider_test.go auto/auto.go auto/auto_test.go auto/counter.go auto/counter_test.go
git commit -m "feat(llm): register Azure OpenAI provider"
```

### Task 2: Add failing Azure constructor, endpoint, and authentication tests

**Files:**
- Create: `providers/azure/client_test.go`
- Create: `providers/azure/options_test.go`
- Create: `providers/azure/errors_test.go` if a dedicated configuration error is introduced

**Step 1: Write the failing tests**

Use deterministic `httptest.Server` endpoints and a valid Azure Responses model to assert:

- the request path is `/openai/v1/responses` for a resource-derived base URL;
- `Model.BaseURL` overrides resource-derived configuration;
- `WithResourceName` overrides `AZURE_RESOURCE_NAME`;
- missing resource configuration returns a typed, secret-free error when `Model.BaseURL` is empty;
- the request carries `api-key: azure-test-key` and does not carry `Authorization`;
- empty API keys and wrong provider/API-format models fail before a client is returned;
- a JSON Responses response decodes through the shared codec with text, tool calls, reasoning, usage, and finish reason preserved;
- an SSE Responses stream decodes through the shared codec;
- malformed JSON/SSE and non-2xx responses return typed errors.

Keep environment-mutating tests sequential and use `t.Setenv`; do not run them in parallel.

**Step 2: Run the package tests to verify they fail**

Run:

```bash
GOWORK=off go test ./providers/azure
```

Expected: package/files or constructor symbols are missing.

### Task 3: Implement Azure endpoint resolution and Responses transport

**Files:**
- Create: `providers/azure/client.go`
- Create: `providers/azure/options.go`
- Create: `providers/azure/errors.go` if needed

**Step 1: Implement the smallest constructor**

Implement `New(selected model.Model, key auth.APIKey, options ...Option)` with this order:

1. validate the model through `llm.ValidateModel`;
2. require `ProviderAzure` and `APIFormatOpenAIResponses` through policy validation;
3. reject an empty API key with `llm.AuthRequiredError`;
4. clone options;
5. use `selected.BaseURL` when non-empty;
6. otherwise use the explicit resource option, then `AZURE_RESOURCE_NAME`;
7. validate the resource label and build `https://<resource>.openai.azure.com/openai/v1`;
8. construct `transport.New` with `route.StaticChat("/responses")`, a codec that delegates common behavior to `responses.Codec` while normalizing Azure reasoning/incomplete variants, and `auth.Header(key, "api-key")`.

`WithResourceName` must copy/normalize only its own string state. No credentials may be read from source or logged.

**Step 2: Run the constructor tests to verify they pass**

```bash
GOWORK=off go test ./providers/azure
```

Expected: PASS for endpoint, header, response, stream, and error cases.

**Step 3: Refactor only after green**

Deduplicate request-codec delegation if needed, preserving shared Responses behavior and the tested URL/header contract.

**Step 4: Commit**

```bash
git add providers/azure
git commit -m "feat(llm): add Azure OpenAI Responses client"
```

### Task 4: Add explicit unsupported counter coverage

**Files:**
- Create: `providers/azure/counter.go`
- Create: `providers/azure/counter_test.go`

**Step 1: Write the failing test**

Assert `NewCounter` returns no counter and an error that `errors.As` matches `*llm.CounterSupportError` with `ProviderAzure`, `CounterSupportExactUnavailable`, and `APIFormatOpenAIResponses`.

**Step 2: Run the test to verify it fails**

```bash
GOWORK=off go test ./providers/azure -run TestNewCounter
```

Expected: missing `NewCounter` or wrong error classification.

**Step 3: Implement the explicit boundary**

Return the shared typed unsupported-counter error. Do not estimate tokens or call OpenAI's `/responses/input_tokens` endpoint because Azure does not currently expose that route.

**Step 4: Run the test to verify it passes**

```bash
GOWORK=off go test ./providers/azure -run TestNewCounter
```

**Step 5: Commit**

```bash
git add providers/azure/counter.go providers/azure/counter_test.go
git commit -m "feat(llm): classify Azure token counter support"
```

### Task 5: Run module verification and review the complete diff

**Files:**
- No planned source changes; only fixes discovered by tests/review.

**Step 1: Run focused and package-wide tests**

```bash
GOWORK=off go test -race ./providers/azure ./auto ./...
```

**Step 2: Run required checks**

```bash
GOWORK=off go test -race -count=1 ./...
GOWORK=off make fmt-check vendor-check
GOWORK=off go mod verify
GOWORK=off make lint
GOWORK=off go tool govulncheck ./...
```

Core/inference do not define `vendor-check`; for this llm module run the combined target exactly as shown.

**Step 3: Review accidental changes**

Run `git diff --check`, `git diff --name-status main...HEAD`, and inspect the full diff for deletions, credential leaks, unrelated edits, and unnecessary vendor changes.

**Step 4: Request GPT Sol medium review**

Ask for a whole-worktree review focused on Azure endpoint/auth semantics, Responses parity, provider registration, counter classification, error/stream handling, and test completeness. Resolve all Critical/Important findings and rerun the affected tests.

### Task 6: Release and preserve unrelated user changes

**Files:**
- No additional planned source files.

**Step 1: Merge the feature branch into local llm `main`**

Before merging, stash only the pre-existing `vendor/github.com/looprig/inference/transport/client.go` edit. Fast-forward `main` to the feature branch, run the final checks on the actual main worktree, and restore the stash without overwriting the user’s timeout changes.

**Step 2: Bump and tag**

Create annotated `v0.8.0` for the Azure feature release.

**Step 3: Push only verified refs**

Push `main` and `v0.8.0` to the configured origin after every relevant check passes.

**Step 4: Final audit**

Confirm local main equals the tag and remote, no release diff contains deletions, the unrelated transport edit is restored, and the parent workspace remains untouched.
