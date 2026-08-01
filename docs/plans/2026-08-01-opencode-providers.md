# OpenCode Priority Providers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native OpenAI Responses, Anthropic Messages, and xAI Responses providers to \`llm\` while retaining the existing Chat Completions providers and shared public API.

**Architecture:** Consume the released \`github.com/looprig/inference/codec/openairesponses\` codec for OpenAI and xAI, keep \`openaiapi\` unchanged for Chat Completions, and wrap the existing native \`anthropicapi\` codec for Anthropic. Each new \`llm/providers/<provider>\` package owns endpoint defaults, authentication, provider options, counters, and tests; \`llm/auto\` and the provider registry only compose those packages.

**Tech Stack:** Go 1.26, \`inference.Client\`, \`inference/codec\`, \`transport.Client\`, \`httptest\`, race tests, vendored first-party dependencies, \`go vet\`, Staticcheck, Gosec, and Govulncheck.

---

### Task 1: Synchronize the released Responses codec without touching Chat support

**Files:**
- Modify: \`go.mod\`
- Modify: \`vendor/modules.txt\`
- Add/update: \`vendor/github.com/looprig/inference/codec/openairesponses/\`
- Modify: \`vendor/github.com/looprig/inference/codec/contracts.go\`
- Modify: \`vendor/github.com/looprig/inference/model/apiformat.go\`
- Modify: \`vendor/github.com/looprig/core/content/block.go\`
- Add/update: \`vendor/github.com/looprig/inference/internal/jsonstrict/\`

**Step 1: Write the dependency/format guard test**

Add a small registry test in \`provider_test.go\` that constructs a model using \`model.APIFormatOpenAIResponses\`, proving the vendored public constant is available and distinct from \`model.APIFormatOpenAI\`.

**Step 2: Run the focused test to verify it fails**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./...\`

Expected: FAIL because the current vendored inference snapshot has no \`APIFormatOpenAIResponses\` or \`codec/openairesponses\` package.

**Step 3: Synchronize only the released first-party source needed by the client codec**

Update the module requirement to \`github.com/looprig/inference v0.6.0\` and copy the released inference Responses package plus its direct supporting contracts, API-format constant, provider-state block support, and strict JSON helper into \`vendor/\`. Do not run a bulk vendor regeneration. Preserve the existing user modification in \`vendor/github.com/looprig/inference/transport/client.go\`.

Keep \`vendor/github.com/looprig/inference/codec/openaiapi\` unchanged and verify its Chat Completions tests still compile through existing providers.

**Step 4: Run the focused package tests to verify the dependency sync**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./...\`.

Expected: the Responses package compiles and the pre-existing Chat providers remain green.

**Step 5: Commit**

Run: \`git add go.mod vendor/modules.txt vendor/github.com/looprig/inference vendor/github.com/looprig/core/content/block.go && git commit -m "build(llm): consume inference Responses codec"\`.

### Task 2: Add provider registry entries and auto tests first

**Files:**
- Modify: \`provider.go\`
- Modify: \`validate.go\` only if provider validation needs a new explicit rule
- Modify: \`auto/auto.go\`
- Modify: \`auto/counter.go\`
- Test: \`provider_test.go\`
- Test: \`auto/auto_test.go\`
- Test: \`auto/apiformat_e2e_test.go\`

**Step 1: Write failing registry and auto-dispatch tests**

Cover \`ProviderOpenAI\`, \`ProviderAnthropic\`, and \`ProviderXAI\`; the Responses format for OpenAI/xAI; Anthropic Messages format; canonical empty-base rules; API-key requirements; and dispatch to each dedicated constructor. Add tests that assert the existing OpenRouter/LMStudio Chat Completions path remains unchanged.

**Step 2: Run focused tests to verify failure**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./auto ./... -run 'TestProvider|Test.*Auto|TestModelAPIFormat'\`.

Expected: FAIL because the new provider labels and format pairs are not yet registered.

**Step 3: Implement the fail-closed registry and dispatch**

Add provider constants and classify each provider in \`supportsAPIFormat\`, \`allowsEmptyBaseURL\`, and \`RequiredAuth\`. Dispatch only validated provider/format pairs in \`auto.New\`; call dedicated constructors for OpenAI, Anthropic, and xAI. Add their exact/unsupported counter classifications in \`auto.NewCounter\` without changing existing providers.

**Step 4: Run focused tests to verify the registry**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./auto ./... -run 'TestProvider|Test.*Auto|TestModelAPIFormat'\`.

Expected: PASS.

**Step 5: Commit**

Run: \`git add provider.go validate.go auto/auto.go auto/counter.go provider_test.go auto/auto_test.go auto/apiformat_e2e_test.go && git commit -m "feat(auto): register OpenAI Anthropic and xAI providers"\`.

### Task 3: Implement OpenAI Responses provider and exact counter

**Files:**
- Create: \`providers/openai/client.go\`
- Create: \`providers/openai/options.go\`
- Create: \`providers/openai/counter.go\`
- Test: \`providers/openai/client_test.go\`
- Test: \`providers/openai/counter_test.go\`

**Step 1: Write failing constructor/request/response tests**

Use \`httptest.Server\` to assert:

- default \`https://api.openai.com/v1\` and selected base URL normalization;
- \`POST /responses\`, \`Authorization: Bearer\`, JSON content type, and no secret in errors;
- neutral system/user/assistant/tool-result messages become \`instructions\`/typed \`input\` items;
- tools, required tool choice, structured output, max output tokens, reasoning, metadata, service tier, and prompt-cache key are encoded only when selected;
- text, reasoning summary, function calls, usage/cache/reasoning token details, and completed/incomplete/tool-call finish statuses decode correctly;
- malformed JSON and HTTP error responses are typed failures.

**Step 2: Run focused tests to verify failure**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/openai -run 'Test'\`.

Expected: FAIL because the package does not exist.

**Step 3: Implement the OpenAI constructor and Responses wrapper**

Construct \`transport.New\` with \`route.StaticChat("/responses")\`, \`openairesponses.Codec{}\`, and \`auth.Key\`. Wrap the codec's encoded JSON to apply only validated OpenAI options, preserving neutral fields and omitting unset options. Inject optional headers such as service-tier controls only when documented; never expose arbitrary credential-bearing header options.

**Step 4: Run the focused client tests**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/openai -run 'Test'\`.

Expected: PASS.

**Step 5: Write failing exact counter tests**

Assert \`POST /responses/input_tokens\`, the Responses input body without output/stream-only fields, bearer authentication, \`{object:"response.input_tokens",input_tokens:n}\` decoding, model binding, malformed/duplicate JSON rejection, HTTP errors, and \`CountQualityExactProvider\` accounting.

**Step 6: Implement the OpenAI counter**

Use a separately constructed counter with the same canonical endpoint identity, bounded reads, context timeout, and typed provider errors as the existing Gemini/Bedrock counters. Encode the full neutral input using the Responses codec, remove only fields unsupported by the count endpoint, and report the provider-returned count without estimating.

**Step 7: Run counter tests and commit**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/openai -run 'Test.*Counter|TestCounter'\`.

Expected: PASS.

Run: \`git add providers/openai && git commit -m "feat(llm): add OpenAI Responses provider"\`.

### Task 4: Implement Anthropic Messages provider and exact counter

**Files:**
- Create: \`providers/anthropic/client.go\`
- Create: \`providers/anthropic/options.go\`
- Create: \`providers/anthropic/counter.go\`
- Test: \`providers/anthropic/client_test.go\`
- Test: \`providers/anthropic/counter_test.go\`

**Step 1: Write failing native Messages tests**

Assert default/selected base URLs, \`POST /messages\`, \`x-api-key\`, \`anthropic-version: 2023-06-01\`, optional beta headers, top-level system, content blocks, tool use/results, thinking, cache-control, structured output, and service/tool behavior options. Exercise JSON responses with text, thinking, tool use, usage cache fields, and every supported stop-reason mapping.

**Step 2: Add failing native SSE tests**

Serve named \`message_start\`, \`content_block_start\`, text/thinking/input JSON delta, \`message_delta\`, and \`message_stop\` events. Assert neutral text, thinking, tool chunks, terminal usage, and finish reason. Include malformed and unknown events and ensure they fail/skip according to the shared codec contract.

**Step 3: Run focused tests to verify failure**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/anthropic -run 'Test'\`.

Expected: FAIL because the package does not exist.

**Step 4: Implement the constructor and native Messages option wrapper**

Construct \`transport.New\` with \`route.StaticChat("/messages")\`, the existing \`anthropicapi.Codec{}\`, and an \`auth.Header\` authenticator for \`x-api-key\`. Always inject the required API version header. Merge beta values defensively, and patch only documented provider fields into the codec's native body. Do not translate through \`openaiapi\`.

**Step 5: Run client and stream tests**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/anthropic -run 'Test'\`.

Expected: PASS.

**Step 6: Write and implement the Anthropic counter**

Send the Messages input shape to \`/messages/count_tokens\`, decode \`{input_tokens:n}\`, report exact-provider quality, and cover system/messages/tools/thinking fields, auth, model binding, malformed/duplicate JSON, bounded body, and API failures.

**Step 7: Commit**

Run: \`git add providers/anthropic && git commit -m "feat(llm): add Anthropic Messages provider"\`.

### Task 5: Implement xAI Responses provider and explicit counter posture

**Files:**
- Create: \`providers/xai/client.go\`
- Create: \`providers/xai/options.go\`
- Create: \`providers/xai/counter.go\`
- Test: \`providers/xai/client_test.go\`
- Test: \`providers/xai/counter_test.go\`

**Step 1: Write failing xAI Responses tests**

Assert default \`https://api.x.ai/v1\`, selected base URL, \`POST /responses\`, \`Authorization: Bearer\`, request normalization, reasoning controls, structured output, tools/tool results, service tier, metadata, and prompt-cache key. Reuse the same deterministic Responses JSON/SSE fixtures as OpenAI while asserting xAI-specific options and response usage behavior.

**Step 2: Run focused tests to verify failure**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/xai -run 'Test'\`.

Expected: FAIL because the package does not exist.

**Step 3: Implement the xAI constructor and options wrapper**

Use \`openairesponses.Codec{}\` and the same \`/responses\` route, but keep xAI options and defaults in the xAI package. Do not send OpenAI-only options unless xAI's current documentation confirms them. Preserve malformed response and provider error handling through the shared transport/codec contracts.

**Step 4: Implement the explicit unsupported counter contract**

\`xai.NewCounter\` must validate the API key and return the existing typed \`llm.CounterSupportError\` for full neutral-request counting. Do not mislabel xAI's text-only \`/v1/tokenize-text\` endpoint as an exact context counter. Cover the error ordering and auto behavior in tests.

**Step 5: Run tests and commit**

Run: \`GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test ./providers/xai ./auto\`.

Expected: PASS.

Run: \`git add providers/xai auto && git commit -m "feat(llm): add xAI Responses provider"\`.

### Task 6: Documentation, formatting, and full regression tests

**Files:**
- Modify: provider package documentation/comments as needed
- Modify: \`repositories.mk\` only if the parent release inventory must reflect the new llm tag
- Test: all existing and new Go tests

**Step 1: Review public API and option docs**

Ensure constructors, option names, counter behavior, default URLs, environment variable conventions (\`OPENAI_API_KEY\`, \`ANTHROPIC_API_KEY\`, \`XAI_API_KEY\`), and Chat-vs-Responses selection are documented without credentials or unsupported claims.

**Step 2: Run the complete requested checks**

Run from the \`llm\` worktree, using the task-local cache where needed:

~~~text
GOWORK=off GOCACHE=/private/tmp/looprig-opencode-gocache go test -race -count=1 ./...
make fmt-check vendor-check
go mod verify
make lint
go tool govulncheck ./...
~~~

Expected: every command exits zero. If network or loopback sandboxing blocks a command, rerun only that command with the required permission and record why.

**Step 3: Review the final diff and status**

Run:

~~~text
git diff --check
git diff --stat main...HEAD
git status --short --branch
git diff main...HEAD --name-status
~~~

Expected: no accidental deletions, no changes to unrelated existing files, no credentials, and the pre-existing dirty transport edit remains outside the feature commits.

**Step 4: Request a code review checkpoint**

Use \`@superpowers:requesting-code-review\` and review every provider, option, counter, codec boundary, and auto-dispatch change against the design and tests.

### Task 7: Release the feature from local main

**Files:**
- No source files beyond the approved parent release metadata

**Step 1: Finish the development branch**

Use \`@superpowers:finishing-a-development-branch\` after all checks and review are clean. Merge \`feat/opencode-providers\` into the nested \`llm\` repository's local \`main\` without overwriting the unrelated working-tree transport edit.

**Step 2: Bump the semantic version**

This is a feature release from \`v0.5.0\` to \`v0.6.0\`. Update only the release inventory/version metadata required by this repository's established workflow.

**Step 3: Create the annotated tag**

Run: \`git tag -a v0.6.0 -m "llm v0.6.0: add OpenAI Anthropic and xAI providers"\`.

**Step 4: Push only verified main and tag**

After final verification succeeds, push the nested \`llm\` \`main\` branch and \`v0.6.0\` tag to its configured remote. Do not push unrelated parent-worktree changes or the dirty vendor transport edit.

