package auto

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/looprig/credentials"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"

	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/internal/credentialclient"
	"github.com/looprig/llm/providers/chutes"
	geminiprovider "github.com/looprig/llm/providers/gemini"
	"github.com/looprig/llm/providers/openrouter"
)

// The helpers below stand in for the deleted model catalogue: each returns a valid
// Model (OriginCustom) via model.CustomModel, used purely as a test fixture. They
// keep the repeated model rows DRY across this file's dispatch tables.
func testTLSRoots(srv *httptest.Server) *x509.CertPool {
	roots := x509.NewCertPool()
	roots.AddCert(srv.Certificate())
	return roots
}

func TestDynamicSupportMatrixIsExplicit(t *testing.T) {
	want := map[llm.Provider]map[model.APIFormat]struct{}{
		llm.ProviderOpenAI:     {model.APIFormatOpenAI: {}, model.APIFormatOpenAIResponses: {}},
		llm.ProviderOpenRouter: {model.APIFormatOpenAI: {}},
		llm.ProviderAnthropic:  {model.APIFormatAnthropic: {}},
		llm.ProviderLMStudio:   {model.APIFormatOpenAI: {}, model.APIFormatAnthropic: {}},
	}
	if !reflect.DeepEqual(dynamicSupport, want) {
		t.Fatalf("dynamic support matrix = %#v, want %#v", dynamicSupport, want)
	}
	if dynamicPolicySupported(model.Model{Provider: "future", APIFormat: model.APIFormatOpenAI}) {
		t.Fatal("unknown provider dynamically supported")
	}
}
func chutesKimiK2Model() model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderChutes), model.APIFormatOpenAI, "https://api.chutes.ai", "moonshotai/Kimi-K2.6-TEE", model.WithContextLimits(model.ContextLimits{WindowTokens: 128_000}), model.WithTools(), model.WithThinking())
}

func openRouterModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, "https://openrouter.ai/api/v1", name, model.WithTools())
}

func geminiFlashModel() model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderGoogle), model.APIFormatGemini, "https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash", model.WithContextLimits(model.ContextLimits{WindowTokens: 1_000_000}), model.WithTools(), model.WithImages(), model.WithThinking())
}

func lmStudioLocalModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderLMStudio), model.APIFormatOpenAI, "http://localhost:1234/v1", name, model.WithTools())
}

func openAIResponsesModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAIResponses, "https://api.openai.com/v1", name, model.WithTools(), model.WithThinking())
}

func anthropicMessagesModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderAnthropic), model.APIFormatAnthropic, "https://api.anthropic.com/v1", name, model.WithTools(), model.WithThinking())
}

func xAIResponsesModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderXAI), model.APIFormatOpenAIResponses, "https://api.x.ai/v1", name, model.WithTools(), model.WithThinking())
}

func azureResponsesModel(name string) model.Model {
	return model.CustomModel(model.ProviderName(llm.ProviderAzure), model.APIFormatOpenAIResponses, "https://resource.openai.azure.com/openai/v1", name, model.WithTools(), model.WithThinking())
}

// TestNew exercises the dispatch + fail-closed auth contract: valid models build a
// non-nil client, an unknown/self-contradictory model is rejected before dispatch
// with a *model.ValidationError, and a key-requiring provider given no key fails
// closed with a *llm.AuthRequiredError. LM Studio (AuthNone) succeeds with no key.
func TestNew(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		// model is built from catalog rows / CustomModel so validation passes on the
		// happy cases; the error cases deliberately fail an earlier ordered guard.
		model       model.Model
		key         auth.APIKey
		wantErr     bool
		wantAuthReq bool   // when wantErr: expect *llm.AuthRequiredError, else *model.ValidationError
		wantField   string // when set (ValidationError path): assert ValidationError.Field
	}{
		{name: "chutes with key", model: chutesKimiK2Model(), key: "k"},
		{name: "openrouter with key", model: openRouterModel("x"), key: "sk-or-key"},
		{name: "openai with key", model: openAIResponsesModel("gpt-5"), key: "sk-openai-key"},
		{name: "anthropic with key", model: anthropicMessagesModel("claude-sonnet-4-6"), key: "sk-ant-key"},
		{name: "xai with key", model: xAIResponsesModel("grok-4-5"), key: "xai-key"},
		{name: "azure with key", model: azureResponsesModel("gpt-4.1"), key: "azure-key"},
		{name: "google with key", model: geminiFlashModel(), key: "AIza-k"},
		{name: "lmstudio without key (AuthNone)", model: lmStudioLocalModel("qwen"), key: ""},
		{name: "lmstudio ignores a supplied key", model: lmStudioLocalModel("qwen"), key: "k"},
		{name: "phala empty key fails closed", model: model.CustomModel(model.ProviderName(llm.ProviderPhala), model.APIFormatOpenAI, "https://api.phala.network/v1", "zai-org/GLM-4.6", model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}), model.WithTools(), model.WithThinking()), key: "", wantErr: true, wantAuthReq: true},
		{name: "chutes empty key fails closed", model: chutesKimiK2Model(), key: "", wantErr: true, wantAuthReq: true},
		{name: "openrouter empty key fails closed", model: openRouterModel("x"), key: "", wantErr: true, wantAuthReq: true},
		{name: "openai empty key fails closed", model: openAIResponsesModel("gpt-5"), key: "", wantErr: true, wantAuthReq: true},
		{name: "anthropic empty key fails closed", model: anthropicMessagesModel("claude-sonnet-4-6"), key: "", wantErr: true, wantAuthReq: true},
		{name: "xai empty key fails closed", model: xAIResponsesModel("grok-4-5"), key: "", wantErr: true, wantAuthReq: true},
		{name: "azure empty key fails closed", model: azureResponsesModel("gpt-4.1"), key: "", wantErr: true, wantAuthReq: true},
		{name: "google empty key fails closed", model: geminiFlashModel(), key: "", wantErr: true, wantAuthReq: true},
		{
			name:    "unknown provider rejected before dispatch",
			model:   model.Model{Provider: "nope", APIFormat: model.APIFormatOpenAI, BaseURL: "https://x.example.test", Name: "m"},
			key:     "k",
			wantErr: true,
		},
		{name: "empty model rejected", model: model.Model{}, key: "k", wantErr: true},
		{
			name:    "self-contradictory model rejected before dispatch",
			model:   model.CustomModel(model.ProviderName(llm.ProviderPhala), model.APIFormatAnthropic, "https://api.phala.network/v1", "m"),
			key:     "k",
			wantErr: true,
		},
		{
			// lmstudio legitimately supports the anthropic dialect (validation passes).
			// Phase 1 fail-closed here (no anthropic codec); Phase 2 wired the
			// anthropicapi codec into codecFor, so this now resolves a real codec and
			// succeeds instead of erroring.
			name:  "lmstudio+anthropic now succeeds (anthropic codec wired)",
			model: model.CustomModel(model.ProviderName(llm.ProviderLMStudio), model.APIFormatAnthropic, "http://localhost:1234", "m"),
			key:   "",
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := New(tt.model, tt.key)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New() err = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				if got != nil {
					t.Fatalf("New() returned non-nil client (%T) alongside an error", got)
				}
				if tt.wantAuthReq {
					var are *llm.AuthRequiredError
					if !errors.As(err, &are) {
						t.Fatalf("err = %T, want *llm.AuthRequiredError", err)
					}
					if are.Provider != llm.Provider(tt.model.Provider) {
						t.Errorf("AuthRequiredError.Provider = %q, want %q", are.Provider, tt.model.Provider)
					}
					if are.Kind != auth.AuthAPIKey {
						t.Errorf("AuthRequiredError.Kind = %q, want %q", are.Kind, auth.AuthAPIKey)
					}
					return
				}
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Fatalf("err = %T, want *model.ValidationError", err)
				}
				if tt.wantField != "" && ve.Field != tt.wantField {
					t.Errorf("ValidationError.Field = %q, want %q", ve.Field, tt.wantField)
				}
				return
			}
			if got == nil {
				t.Fatal("New() returned nil client, want non-nil client")
			}
		})
	}
}

func TestNewSpecialAuthDoesNotDiscoverEnvironmentCredentials(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		provider llm.Provider
		format   model.APIFormat
		use      string
	}{
		{name: "gitlab oauth", provider: llm.ProviderGitLab, format: model.APIFormatOpenAI, use: "gitlab.New"},
		{name: "github copilot oauth", provider: llm.ProviderGitHubCopilot, format: model.APIFormatOpenAI, use: "githubcopilot.New"},
		{name: "vertex gcp", provider: llm.ProviderGoogleVertex, format: model.APIFormatGemini, use: "vertex.New"},
		{name: "sap service key", provider: llm.ProviderSAP, format: model.APIFormatOpenAI, use: "sapcore.New"},
		{name: "snowflake token", provider: llm.ProviderSnowflakeCortex, format: model.APIFormatOpenAI, use: "snowflake.New"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			selected := model.CustomModel(model.ProviderName(tt.provider), tt.format, "https://example.test/v1", "model")
			client, err := New(selected, "")
			if client != nil {
				t.Fatalf("New() returned %T with no explicit credential", client)
			}
			var directErr *CredentialNotConstructibleError
			if !errors.As(err, &directErr) {
				t.Fatalf("New() error = %T %v, want CredentialNotConstructibleError", err, err)
			}
			if directErr.Provider != tt.provider || directErr.Use != tt.use {
				t.Errorf("direct error = %+v, want provider %q/use %q", directErr, tt.provider, tt.use)
			}
		})
	}
}

// TestNewBedrockDirectsToConstructor confirms the SigV4 dispatch decision: a
// Bedrock model reaches New's dispatch (its RequiredAuth is AuthSigV4, so the
// empty-APIKey guard does NOT fire — no AuthRequiredError confusion) and returns a
// *SigV4NotConstructibleError directing the caller to bedrock.New, with no client.
// auto.New's only credential is an auth.APIKey, which cannot carry SigV4 creds.
func TestNewBedrockDirectsToConstructor(t *testing.T) {
	t.Parallel()

	// An empty key must NOT surface as an AuthRequiredError here: bedrock's auth
	// kind is SigV4, not APIKey, so the Phase-1 empty-APIKey guard is skipped.
	got, err := New(model.CustomModel(model.ProviderName(llm.ProviderBedrock), model.APIFormatAnthropic, "", "anthropic.claude-3-5-sonnet-20241022-v2:0", model.WithContextLimits(model.ContextLimits{WindowTokens: 200_000}), model.WithTools(), model.WithImages()), "")
	if got != nil {
		t.Fatalf("New() returned non-nil client (%T) for a SigV4 provider", got)
	}
	var sigErr *SigV4NotConstructibleError
	if !errors.As(err, &sigErr) {
		t.Fatalf("err = %T, want *SigV4NotConstructibleError", err)
	}
	if sigErr.Provider != llm.ProviderBedrock {
		t.Errorf("SigV4NotConstructibleError.Provider = %q, want %q", sigErr.Provider, llm.ProviderBedrock)
	}
	if sigErr.Use != "bedrock.New" {
		t.Errorf("SigV4NotConstructibleError.Use = %q, want %q", sigErr.Use, "bedrock.New")
	}

	// It must specifically NOT be an AuthRequiredError (the empty-APIKey path).
	var are *llm.AuthRequiredError
	if errors.As(err, &are) {
		t.Error("bedrock empty-key returned *llm.AuthRequiredError; SigV4 providers must skip the empty-APIKey guard")
	}
}

// TestNewPhalaNotConstructible confirms the Policy dispatch decision: a Phala model
// reaches New's dispatch with a present key (its RequiredAuth is AuthAPIKey, so the
// empty-APIKey guard does NOT fire) and returns a *PolicyNotConstructibleError
// directing the caller to phala.New, with no client. auto.New's inputs are
// (model, key) only — it carries no attestation acceptance Policy — so a Phala client
// cannot be built here; a defaulted policy would fail open.
func TestNewPhalaNotConstructible(t *testing.T) {
	t.Parallel()
	m := model.CustomModel(model.ProviderName(llm.ProviderPhala), model.APIFormatOpenAI, "https://inference.phala.com", "zai-org/GLM-4.6")
	got, err := New(m, "sk-live")
	if got != nil {
		t.Fatalf("New() returned non-nil client (%T) for a policy-requiring provider", got)
	}
	var pne *PolicyNotConstructibleError
	if !errors.As(err, &pne) {
		t.Fatalf("want *PolicyNotConstructibleError, got %v", err)
	}
	if pne.Provider != llm.ProviderPhala {
		t.Errorf("PolicyNotConstructibleError.Provider = %q, want %q", pne.Provider, llm.ProviderPhala)
	}
	if pne.Use != "phala.New" {
		t.Errorf("PolicyNotConstructibleError.Use = %q, want %q", pne.Use, "phala.New")
	}

	// It must specifically NOT be an AuthRequiredError: auth is checked before the
	// construct-directly dispatch, and a present key means the empty-APIKey guard is
	// skipped, so the policy dispatch — not an auth failure — is what surfaces here.
	var are *llm.AuthRequiredError
	if errors.As(err, &are) {
		t.Error("phala present-key returned *llm.AuthRequiredError; the policy construct-directly dispatch must surface, not an auth failure")
	}
}

// TestNewConcreteTypes pins bespoke legacy providers to their concrete client
// and verifies transport-backed providers pass through the credential adapter.
// The adapter is now the observable compatibility boundary for clients that
// support additive call-scoped authorization methods.
func TestNewConcreteTypes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		model model.Model
		key   auth.APIKey
		is    func(inference.Client) bool
		want  string
	}{
		{
			name:  "chutes wires the chutes client",
			model: chutesKimiK2Model(), key: "k",
			is:   func(l inference.Client) bool { _, ok := l.(*chutes.Client); return ok },
			want: "*chutes.Client",
		},
		{
			name:  "lmstudio wires the generic transport client",
			model: lmStudioLocalModel("qwen"), key: "",
			is:   func(l inference.Client) bool { _, ok := l.(*credentialclient.Client); return ok },
			want: "*credentialclient.Client",
		},
		{
			name:  "openrouter wires the generic transport client",
			model: openRouterModel("x"), key: "sk-or-key",
			is:   func(l inference.Client) bool { _, ok := l.(*credentialclient.Client); return ok },
			want: "*credentialclient.Client",
		},
		{
			name:  "google wires the bespoke gemini client",
			model: geminiFlashModel(), key: "AIza-k",
			is:   func(l inference.Client) bool { _, ok := l.(*geminiprovider.Client); return ok },
			want: "*geminiprovider.Client",
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := New(tt.model, tt.key)
			if err != nil {
				t.Fatalf("New() err = %v, want nil", err)
			}
			if !tt.is(got) {
				t.Fatalf("New() client = %T, want %s", got, tt.want)
			}
		})
	}
}

// TestCodecFor pins the codec-selection registry: each wire dialect auto can encode
// resolves to its concrete codec, and a format with no codec yet fails closed with a
// *model.ValidationError (Field "APIFormat") rather than silently mis-encoding.
// This is the internal seam that makes lmstudio+anthropic and OpenRouter+openai work.
func TestCodecFor(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		format  model.APIFormat
		is      func(codec.Codec) bool
		want    string
		wantErr bool
	}{
		{
			name:   "openai",
			format: model.APIFormatOpenAI,
			is:     func(c codec.Codec) bool { _, ok := c.(openaiapi.Codec); return ok },
			want:   "openaiapi.Codec",
		},
		{
			name:   "anthropic",
			format: model.APIFormatAnthropic,
			is:     func(c codec.Codec) bool { _, ok := c.(anthropicapi.Codec); return ok },
			want:   "anthropicapi.Codec",
		},
		{
			name:   "openai responses",
			format: model.APIFormatOpenAIResponses,
			is:     func(c codec.Codec) bool { _, ok := c.(openairesponses.Codec); return ok },
			want:   "openairesponses.Codec",
		},
		{
			name:   "gemini",
			format: model.APIFormatGemini,
			is:     func(c codec.Codec) bool { _, ok := c.(geminiapi.Codec); return ok },
			want:   "geminiapi.Codec",
		},
		{name: "bedrock-converse requires direct SigV4 construction", format: llm.APIFormatBedrockConverse, wantErr: true},
		{name: "unknown format fails closed", format: model.APIFormat("bogus"), wantErr: true},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := codecFor(tt.format)
			if (err != nil) != tt.wantErr {
				t.Fatalf("codecFor(%q) err = %v, wantErr %v", tt.format, err, tt.wantErr)
			}
			if tt.wantErr {
				if got != nil {
					t.Fatalf("codecFor(%q) returned non-nil codec (%T) alongside an error", tt.format, got)
				}
				var ve *model.ValidationError
				if !errors.As(err, &ve) {
					t.Fatalf("codecFor err = %T, want *model.ValidationError", err)
				}
				if ve.Field != "APIFormat" {
					t.Errorf("ValidationError.Field = %q, want %q", ve.Field, "APIFormat")
				}
				return
			}
			if !tt.is(got) {
				t.Fatalf("codecFor(%q) = %T, want %s", tt.format, got, tt.want)
			}
		})
	}
}

// TestNewLMStudioDefaultEndpoint is the explicit assertion that the dissolved
// lmstudio package's default loopback endpoint now works via a CustomModel row + the
// generic client, with no credentials.
func TestNewLMStudioDefaultEndpoint(t *testing.T) {
	t.Parallel()
	got, err := New(lmStudioLocalModel("m"), "")
	if err != nil {
		t.Fatalf("New(lmStudioLocalModel, \"\") err = %v, want nil", err)
	}
	if got == nil {
		t.Fatal("New(lmStudioLocalModel, \"\") = nil, want non-nil client")
	}
}

func TestNewWithOpenRouterOptions(t *testing.T) {
	t.Parallel()

	bodyCh := make(chan []byte, 1)
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodyCh <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"response-id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := New(selected, "sk-or-test", WithTLSRootCAs(testTLSRoots(srv)), WithOpenRouterOptions(openrouter.WithUsage(false)))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(<-bodyCh, &body); err != nil {
		t.Fatalf("request body JSON error = %v", err)
	}
	var usage struct {
		Include bool `json:"include"`
	}
	if err := json.Unmarshal(body["usage"], &usage); err != nil {
		t.Fatalf("usage JSON error = %v", err)
	}
	if usage.Include {
		t.Error("usage.include = true, want explicit false")
	}
}

func TestNewWithOpenRouterDelegatesStaticAPIKey(t *testing.T) {
	t.Parallel()

	requestHeaders := make(chan http.Header, 1)
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestHeaders <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"response-id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenRouter), model.APIFormatOpenAI, srv.URL, "model")
	client, err := New(selected, "sk-or-static", WithTLSRootCAs(testTLSRoots(srv)))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := (<-requestHeaders).Get("Authorization"); got != "Bearer sk-or-static" {
		t.Fatalf("Authorization = %q, want static API key delegation", got)
	}
}

func TestNewWithAuthUsesLeaseAuthorizer(t *testing.T) {
	t.Parallel()

	requestHeaders := make(chan http.Header, 1)
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestHeaders <- r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"response-id","model":"model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	selected := model.CustomModel(model.ProviderName(llm.ProviderOpenAI), model.APIFormatOpenAI, srv.URL, "model")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, err := policy.Accepted[0].Descriptor()
	if err != nil {
		t.Fatal(err)
	}
	generation, err := credentials.NewGeneration("auto-test")
	if err != nil {
		t.Fatal(err)
	}
	source := &testSource{descriptor: descriptor, lease: testLease{
		descriptor: descriptor,
		generation: generation,
		authorizer: auth.Key("lease-key"),
	}}
	client, err := NewWithAuth(selected, source, WithTLSRootCAs(testTLSRoots(srv)))
	if err != nil {
		t.Fatalf("NewWithAuth() error = %v", err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{Model: selected}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := (<-requestHeaders).Get("Authorization"); got != "Bearer lease-key" {
		t.Fatalf("Authorization = %q, want lease authorizer", got)
	}
}

func TestNewWithAuthRejectsNilAndRemoteNoneSource(t *testing.T) {
	t.Parallel()

	local := lmStudioLocalModel("local")
	if client, err := NewWithAuth(local, nil); client != nil || err == nil {
		t.Fatalf("NewWithAuth(local, nil) = (%T, %v), want construction error", client, err)
	}

	remote := openAIResponsesModel("remote")
	remotePolicy, err := llm.AuthPolicyForModel(remote)
	if err != nil {
		t.Fatal(err)
	}
	remoteNoneDescriptor, err := credentials.NewDescriptor(
		remotePolicy.Accepted[0].Provider,
		remotePolicy.Accepted[0].Transport,
		credentials.SchemeNone,
		credentials.UsageLocal,
		"",
		"",
		"",
	)
	if err != nil {
		t.Fatal(err)
	}
	remoteNone, err := credentials.NewNoneSource(remoteNoneDescriptor)
	if err != nil {
		t.Fatal(err)
	}
	if client, err := NewWithAuth(remote, remoteNone); client != nil || err == nil {
		t.Fatalf("NewWithAuth(remote, NoneSource) = (%T, %v), want exact-policy mismatch", client, err)
	}
}

func TestNewWithAuthAcceptsExplicitLocalNoneSource(t *testing.T) {
	t.Parallel()

	selected := lmStudioLocalModel("local")
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, err := policy.Accepted[0].Descriptor()
	if err != nil {
		t.Fatal(err)
	}
	source, err := credentials.NewNoneSource(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	client, err := NewWithAuth(selected, source)
	if err != nil || client == nil {
		t.Fatalf("NewWithAuth(local, NoneSource) = (%T, %v), want client", client, err)
	}
}

func TestStaticAPIKeyAuthorizerPreservesProviderHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		provider llm.Provider
		format   model.APIFormat
		header   string
	}{
		{name: "anthropic", provider: llm.ProviderAnthropic, format: model.APIFormatAnthropic, header: "x-api-key"},
		{name: "azure", provider: llm.ProviderAzure, format: model.APIFormatOpenAIResponses, header: "api-key"},
		{name: "azure anthropic", provider: llm.ProviderAzureCognitiveServices, format: model.APIFormatAnthropic, header: "x-api-key"},
		{name: "deepinfra anthropic", provider: llm.ProviderDeepInfra, format: model.APIFormatAnthropic, header: "x-api-key"},
		{name: "openai", provider: llm.ProviderOpenAI, format: model.APIFormatOpenAI, header: "Authorization"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			requestURL, err := url.Parse("https://provider.example/v1")
			if err != nil {
				t.Fatal(err)
			}
			req := &http.Request{Method: http.MethodPost, URL: requestURL, Header: make(http.Header)}
			if err := staticAPIKeyAuthorizer(tt.provider, tt.format, "key").Authorize(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			if got := req.Header.Get(tt.header); got == "" {
				t.Fatalf("%s header is empty", tt.header)
			}
		})
	}
}

type testSource struct {
	descriptor credentials.Descriptor
	lease      credentials.Lease
}

func (s *testSource) Reference() credentials.Reference   { return credentials.Reference{} }
func (s *testSource) Descriptor() credentials.Descriptor { return s.descriptor }
func (s *testSource) Acquire(context.Context) (credentials.Lease, error) {
	return s.lease, nil
}
func (s *testSource) Invalidate(context.Context, credentials.Generation, credentials.Failure) error {
	return nil
}
func (s *testSource) Close() error { return nil }

type testLease struct {
	descriptor credentials.Descriptor
	generation credentials.Generation
	authorizer httpauth.Authorizer
}

func (l testLease) Generation() credentials.Generation { return l.generation }
func (l testLease) Descriptor() credentials.Descriptor { return l.descriptor }
func (l testLease) ExpiresAt() time.Time               { return time.Time{} }
func (l testLease) Authorizer() httpauth.Authorizer    { return l.authorizer }

func TestDefaultGenericBaseURL(t *testing.T) {
	tests := []struct {
		name string
		p    llm.Provider
		want string
	}{
		{"openrouter", llm.ProviderOpenRouter, "https://openrouter.ai/api/v1"},
		{"lmstudio", llm.ProviderLMStudio, "http://localhost:1234/v1"},
		{"no default for others", llm.ProviderChutes, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := defaultGenericBaseURL(tt.p); got != tt.want {
				t.Errorf("defaultGenericBaseURL(%s) = %q, want %q", tt.p, got, tt.want)
			}
		})
	}
}
