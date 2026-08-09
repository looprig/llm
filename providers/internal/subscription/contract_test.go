package subscription

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looprig/credentials"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	model "github.com/looprig/inference/model"
	"github.com/looprig/inference/route"
	"github.com/looprig/inference/transport"
	"github.com/looprig/llm"
	"github.com/looprig/llm/internal/credentialclient"
	"github.com/looprig/secrets"
)

func TestRunRejectsAnUnconfiguredContract(t *testing.T) {
	t.Parallel()

	server := NewServer(t)
	defer server.Close()

	err := ValidateContract(Contract{})
	if err == nil {
		t.Fatal("ValidateContract(Contract{}) = nil, want configuration error")
	}
}

func TestCertificationWitnessRequiresAnExplicitConstructor(t *testing.T) {
	t.Parallel()

	if _, err := NewWitness(Contract{}); err == nil {
		t.Fatal("NewWitness(Contract{}) = nil error, want fail-closed certification error")
	}

	contract := fixtureContract()
	witness, err := NewWitness(contract)
	if err != nil {
		t.Fatalf("NewWitness(fixtureContract()) error = %v", err)
	}
	if err := witness.Validate(); err != nil {
		t.Fatalf("Witness.Validate() error = %v", err)
	}
}

func TestContractMatrixInvokesEveryIngressCodec(t *testing.T) {
	server := NewServer(t)
	defer server.Close()

	var constructorCalls atomic.Int32
	contract := fixtureContract()
	original := contract.Constructor
	contract.Constructor = func(selected model.Model, source credentials.Source, fixture *Server) (inference.Client, error) {
		constructorCalls.Add(1)
		return original(selected, source, fixture)
	}

	if err := Execute(contract, server); err != nil {
		t.Fatal(err)
	}
	if got, want := constructorCalls.Load(), int32(len(contract.Formats)*7); got != want {
		t.Fatalf("constructor calls = %d, want %d (normal/error/redirect/lifecycle/recovery/concurrent per format)", got, want)
	}
}

func fixtureContract() Contract {
	return Contract{
		Provider: "fixture-provider",
		// Use a transport identity already classified by the credential adapter's
		// closed auth-failure table; the runner itself remains provider-neutral.
		Transport: "responses",
		Issuer:    "https://issuer.fixture.invalid",
		Scheme:    credentials.SchemeOAuth,
		Usage:     credentials.UsageSubscription,
		Formats: []model.APIFormat{
			model.APIFormatAnthropic,
			model.APIFormatOpenAI,
			model.APIFormatOpenAIResponses,
		},
		NewSource: func(descriptor credentials.Descriptor) (credentials.Source, error) {
			return newFixtureSource(descriptor), nil
		},
		Constructor: fixtureConstructor,
	}
}

type fixtureSource struct {
	mu         sync.Mutex
	descriptor credentials.Descriptor
	generation credentials.Generation
	token      string
	closed     bool
}

func newFixtureSource(descriptor credentials.Descriptor) *fixtureSource {
	generation, _ := credentials.NewGeneration("fixture-generation-1")
	return &fixtureSource{
		descriptor: descriptor,
		generation: generation,
		token:      "fixture-token-1",
	}
}

func (s *fixtureSource) Reference() credentials.Reference   { return credentials.Reference{} }
func (s *fixtureSource) Descriptor() credentials.Descriptor { return s.descriptor }
func (s *fixtureSource) Acquire(ctx context.Context) (credentials.Lease, error) {
	if ctx == nil {
		return nil, errors.New("fixture: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, &credentials.SourceClosedError{}
	}
	secret, err := secrets.New([]byte(s.token))
	if err != nil {
		return nil, err
	}
	authorizer, err := httpauth.Bearer(secret)
	if err != nil {
		return nil, err
	}
	return fixtureLease{descriptor: s.descriptor, generation: s.generation, authorizer: authorizer}, nil
}
func (s *fixtureSource) Invalidate(ctx context.Context, generation credentials.Generation, failure credentials.Failure) error {
	if ctx == nil {
		return errors.New("fixture: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := generation.Validate(); err != nil {
		return err
	}
	if err := failure.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return &credentials.SourceClosedError{}
	}
	if generation != s.generation {
		return nil
	}
	s.generation, _ = credentials.NewGeneration("fixture-generation-2")
	s.token = "fixture-token-2"
	return nil
}
func (s *fixtureSource) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	return nil
}
func (s *fixtureSource) CanRecover(credentials.Failure) bool { return true }

type fixtureLease struct {
	descriptor credentials.Descriptor
	generation credentials.Generation
	authorizer httpauth.Authorizer
}

func (l fixtureLease) Generation() credentials.Generation { return l.generation }
func (l fixtureLease) Descriptor() credentials.Descriptor { return l.descriptor }
func (l fixtureLease) ExpiresAt() time.Time               { return time.Time{} }
func (l fixtureLease) Authorizer() httpauth.Authorizer    { return l.authorizer }

func fixtureConstructor(selected model.Model, source credentials.Source, fixture *Server) (inference.Client, error) {
	descriptor := source.Descriptor()
	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{{
		Provider: descriptor.Provider, Transport: descriptor.Transport,
		Scheme: descriptor.Scheme, Usage: descriptor.Usage,
		Issuer: descriptor.Issuer, Audience: descriptor.Audience,
	}}}
	var (
		inner  inference.Client
		router route.Router
		cdc    codec.Codec
	)
	switch selected.APIFormat {
	case model.APIFormatAnthropic:
		router = fixtureMessagesRouter{}
		cdc = anthropicapi.Codec{}
	case model.APIFormatOpenAI:
		router = route.StaticChat("/chat/completions")
		cdc = openaiapi.Codec{}
	case model.APIFormatOpenAIResponses:
		router = route.StaticChat("/responses")
		cdc = openairesponses.Codec{}
	default:
		return nil, fmt.Errorf("fixture: unsupported API format %q", selected.APIFormat)
	}
	inner = transport.NewWithAuth(transport.Endpoint{
		BaseURL: fixture.URL(), Provider: selected.Provider, APIFormat: selected.APIFormat,
	}, router, cdc, transport.WithTLSRootCAs(fixture.RootCAs()))
	return credentialclient.New(inner, source, policy)
}

type fixtureMessagesRouter struct{}

func (fixtureMessagesRouter) BuildRoute(baseURL string, req inference.Request, mode codec.RequestMode) (route.Route, error) {
	built, err := route.StaticChat("/messages").BuildRoute(baseURL, req, mode)
	if err != nil {
		return route.Route{}, err
	}
	if built.Header == nil {
		built.Header = make(http.Header)
	}
	built.Header.Set("anthropic-version", "2023-06-01")
	return built, nil
}
