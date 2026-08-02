# Azure OpenAI Provider Design

## Goal

Add the Azure OpenAI provider documented by OpenCode as a first-class `llm`
provider, using Azure's current OpenAI-compatible Responses API while leaving
the existing OpenAI, Anthropic, xAI, Gemini, and Bedrock providers unchanged.

## Scope

This iteration implements OpenCode's Azure OpenAI entry only. Azure Cognitive
Services is deliberately deferred as a separate provider mode. Authentication
uses the API-key flow exposed by OpenCode; Microsoft Entra authentication can
be added later without changing the request/response codec boundary.

## Architecture

The provider will live in `providers/azure` and reuse
`inference/codec/openairesponses`. No new inference codec or API format is
needed. The provider validates `ProviderAzure` with
`APIFormatOpenAIResponses`, routes requests to `/responses`, and authenticates
with the Azure-specific `api-key` header.

When `Model.BaseURL` is set, it is used verbatim as the Azure OpenAI v1 API
root, allowing proxies and deterministic `httptest` servers. When it is empty,
the provider resolves `AZURE_RESOURCE_NAME` and builds
`https://<resource>.openai.azure.com/openai/v1`; missing resource configuration
fails closed with a typed validation/configuration error. The request's
existing `model` field carries the Azure deployment name, matching OpenCode's
documented requirement that the deployment name match the model name.

The modern v1 Responses endpoint does not expose the OpenAI
`/responses/input_tokens` counter route on Azure. `NewCounter` therefore returns
the shared typed `CounterSupportError` rather than silently estimating tokens;
normal response and stream usage remains decoded through the shared codec.

## Public API

Add:

- `llm.ProviderAzure`.
- `providers/azure.New(model.Model, auth.APIKey, ...Option)`.
- `providers/azure.WithResourceName(string)` for explicit configuration and
  tests when `Model.BaseURL` is empty.
- `providers/azure.NewCounter(auth.APIKey)` returning the typed unsupported
  counter error.

The constructor will accept an explicit `Model.BaseURL` before consulting the
resource option/environment. API keys remain constructor inputs and are never
read from or embedded in source code.

## Error and stream behavior

The shared Responses codec remains authoritative for message normalization,
tool calls/results, reasoning content, structured output, usage, finish
reasons, malformed JSON, SSE framing, and provider HTTP error decoding. Azure
only owns endpoint resolution, `api-key` authentication, provider validation,
and its explicit counter boundary.

## Testing

Deterministic `httptest` coverage will verify:

- provider validation and missing API key/resource errors;
- default resource-derived URL, explicit base URL, and resource option
  precedence;
- `api-key` header and absence of bearer authentication;
- Responses JSON and SSE request/response behavior, including tools,
  reasoning, structured output, usage, and finish reasons;
- malformed responses, malformed streams, and HTTP errors;
- unsupported counter classification;
- auto-provider dispatch and shared model-policy registration.

## Alternatives considered

1. Reuse the OpenAI provider directly with a custom header. This minimizes
   code, but loses Azure's provider identity, resource-name resolution, and
   explicit counter semantics.
2. Implement Azure's legacy deployment Chat Completions route. This would need
   a separate route/codec path and would not match OpenCode's current
   `/openai/v1` Azure OpenAI entry.
3. Add the modern Azure v1 Responses provider (chosen). It reuses the tested
   Responses dialect, matches current Microsoft and OpenCode documentation,
   and keeps Azure-specific policy isolated in one package.
