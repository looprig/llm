# llm

`github.com/looprig/llm` is Looprig's batteries-included provider SDK, built on
the neutral [`inference`](https://github.com/looprig/inference) contract. It
holds the provider policy that `inference` deliberately leaves out:

- the known-provider registry (`llm.Provider`);
- which provider speaks which API format;
- each provider's auth requirements and default endpoint;
- the fail-closed model validation preset (`llm.ValidateModel`).

It has one client package per provider, plus `auto`, which picks and wires the
right `inference.Client` for a validated `model.Model`. It also holds the
provider-security machinery for confidential inference (`aci`, `e2e`, `tee`) and
the SigV4 authenticator (`llm/auth`). `inference` never imports `llm`.

## Status

Released. Every provider below is implemented and returns an
`inference.Client`. Wire encoding is delegated to the shared codecs in
`inference/codec`. A provider package handles only its own routing, auth and
documented deviations from the format.

Providers (`llm.Provider` value, package under `providers/`, accepted
`model.APIFormat`s, auth kind):

| Provider | Package | API formats | Auth |
|---|---|---|---|
| `anthropic` | `anthropic` | anthropic | API key |
| `openai` | `openai` | openai, openai-responses | API key |
| `google` | `gemini` | gemini | API key |
| `xai` | `xai` | openai, openai-responses | API key |
| `bedrock` | `bedrock` | anthropic, bedrock-converse | SigV4 |
| `azure` | `azure` | openai-responses | API key |
| `azure-cognitive-services` | `azure-cognitive-services` | openai, openai-responses, anthropic | API key |
| `google-vertex`, `google-vertex-anthropic` | `google-vertex` | gemini, anthropic (`-anthropic`: anthropic only) | GCP |
| `github-copilot` | `github-copilot` | openai, openai-responses, anthropic | OAuth |
| `gitlab` | `gitlab` | openai, openai-responses, anthropic | OAuth |
| `sap-ai-core` | `sap-ai-core` | openai | service key |
| `snowflake-cortex` | `snowflake-cortex` | openai | token |
| `openrouter` | `openrouter` | openai | API key |
| `opencode`, `opencode-go` | `opencode`, `opencode-go` | openai, openai-responses, anthropic | API key |
| `vercel` | `vercel` | openai, openai-responses, anthropic | API key |
| `cloudflare-ai-gateway` | `cloudflare-ai-gateway` | openai, openai-responses, anthropic | API key |
| `zenmux` | `zenmux` | openai, openai-responses, anthropic | API key |
| `venice` | `venice` | openai, openai-responses | API key |
| `deepinfra` | `deepinfra` | openai, anthropic | API key |
| `llmgateway` | `llmgateway` | openai, anthropic | API key |
| `minimax` | `minimax` | anthropic | API key |
| `chutes` | `chutes` | openai (end-to-end encrypted, TEE-attested) | API key |
| `phala` | `phala` | openai (attested confidential inference via `aci`) | API key |
| `302ai`, `baseten`, `cerebras`, `cloudflare-workers-ai`, `cortecs`, `deepseek`, `digitalocean`, `fireworks-ai`, `frogbot`, `gmicloud`, `groq`, `helicone`, `huggingface`, `io-net`, `llama`, `moonshotai`, `nebius`, `nvidia`, `ollama-cloud`, `ovhcloud`, `scaleway`, `stackit`, `synthetics`, `togetherai`, `zai` | `p302ai`, `baseten`, `cerebras`, `cloudflare-workers-ai`, `cortecs`, `deepseek`, `digitalocean`, `fireworks`, `frogbot`, `gmicloud`, `groq`, `helicone`, `huggingface`, `ionet`, `llama`, `moonshot`, `nebius`, `nvidia`, `ollamacloud`, `ovhcloud`, `scaleway`, `stackit`, `synthetic`, `together`, `zai` | openai | API key |
| `ollama`, `llama.cpp`, `atomic-chat` | `ollama`, `llamacpp`, `atomic-chat` | openai | none (local) |
| `lmstudio` | none (built by `auto` over the generic transport) | openai, anthropic | none (local) |

Known limits and behaviour a caller should know:

- **Some providers cannot be built by `auto.New`/`auto.NewWithAuth`, because
  `auto` only takes a model and a key or credential source.** Each returns a
  typed error that names the constructor to call instead:
  - `bedrock` needs SigV4 credentials and a region: `SigV4NotConstructibleError`, use `bedrock.New`.
  - `phala` needs a caller-verified attestation `aci.Policy`: `PolicyNotConstructibleError`, use `phala.New`.
  - OAuth, GCP, service-key and token providers given no credential return `CredentialNotConstructibleError`.
- `auto` never reads the environment. Some direct constructors fall back to
  environment variables when an argument is empty, for example
  `GITLAB_TOKEN`, `SNOWFLAKE_CORTEX_TOKEN`/`SNOWFLAKE_CORTEX_PAT`, and the Azure
  resource name and SAP AI Core settings.
- `providers/anthropic/subscription` and `providers/openai/subscription` only
  refuse third-party subscription registration. They contain no credential or
  transport implementation.
- Exact provider context counters (`auto.NewCounter`) exist for `google`,
  `openai`, `anthropic` and `xai`. `bedrock` needs `bedrock.NewCounter`. Other
  providers return `*llm.CounterSupportError`.
- `Request.SessionID` is forwarded only by `opencode` and `opencode-go`, as the
  `x-opencode-session` header. An explicit non-empty `WithHeader` for that
  header wins. Every other provider ignores it.

## Install

```sh
go get github.com/looprig/llm@latest
```

## Packages

| Package | Purpose |
|---|---|
| `llm` | Provider registry, provider/format table, auth policy (`AuthPolicyForModel`), `ValidateModel`, typed errors. |
| `llm/auto` | Composition root: `New` (static API key), `NewWithAuth` (`credentials.Source`), `NewCounter`. |
| `llm/providers/...` | One client package per provider (see the table above). |
| `llm/auth` | AWS SigV4 authenticator used by Bedrock. |
| `llm/aci` | Client for the Dstack private-ai-gateway "aci/1" confidential-inference protocol (attestation, E2EE, signed receipts). |
| `llm/e2e` | ML-KEM-768 / HKDF / ChaCha20-Poly1305 envelope primitives used by `chutes`. |
| `llm/tee` | Intel TDX quote and NVIDIA GPU evidence verification shared by attested providers. |

## Usage

```go
selected := model.CustomModel(
	model.ProviderName(llm.ProviderOpenAI),
	model.APIFormatOpenAIResponses,
	"https://api.openai.com/v1",
	"gpt-example",
)

client, err := auto.New(selected, auth.APIKey(key)) // key from your own config
if err != nil {
	return err
}
resp, err := client.Invoke(ctx, inference.Request{Model: selected /* , System, Messages ... */})
```

For credential-backed construction, where each request acquires a fresh lease,
pass a `credentials.Source` to `auto.NewWithAuth`. Runnable programs:
`examples/auto` and `examples/credentials`.

## Where it sits

Tier 3 in the Looprig workspace. Direct Looprig dependencies: `core`,
`credentials`, `inference`, and `secrets` (tests only). It is consumed by
products and tools that need real provider clients, such as `carbon`, `tests`
and `pluto/cmd/pluto`.

## Development

The Go baseline is 1.26.8. The module does not vendor. Verify it standalone:

```sh
GOWORK=off go test ./...
make check   # gofmt check, vet, staticcheck, gosec, govulncheck, race tests, build
```

Other targets: `fmt`, `fmt-check`, `vet`, `test`, `lint`, `vuln`, `secure`,
`check-staticcheck`, `check-gosec`, `check-vuln`, `build`. The lint and
security tools are pinned by `tool` directives in `go.mod`.

The default test run makes no network calls. Two opt-in suites reach real,
paid endpoints:

```sh
# Live provider conformance probes. Keys come from a carbon model catalogue
# (path overridable with LOOPRIG_LIVE_MODELS). Missing keys skip.
go test -tags live -v -run TestLive ./providers/internal/livetest/...

# Live Phala aci integration. Skips without PHALA_API_KEY.
PHALA_API_KEY=... go test -tags integration -race -run Live -v ./aci
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
