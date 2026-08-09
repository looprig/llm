package credentialclient

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/stream"
	"github.com/looprig/llm"
)

func TestNewRejectsNilSource(t *testing.T) {
	t.Parallel()

	inner := &scopedClient{}
	policy := testPolicy()
	if client, err := New(inner, nil, policy); client != nil || err == nil {
		t.Fatalf("New(nil source) = (%T, %v), want nil client and error", client, err)
	}
}

func TestClientRechecksLeaseBindingBeforeAuthorization(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	matching := testDescriptor(policy.Accepted[0])
	mismatch := matching
	mismatch.Audience = "api://wrong"
	source := &fakeSource{
		descriptor: matching,
		leases:     []credentials.Lease{fakeLease{descriptor: mismatch, generation: mustGeneration(t, "one")}},
	}
	inner := &scopedClient{}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err == nil {
		t.Fatal("Invoke() = nil error for a mismatched acquired lease")
	}
	if got := inner.invokeCount(); got != 0 {
		t.Fatalf("inner InvokeWithAuth calls = %d, want zero", got)
	}
}

func TestClientInvalidatesAndReacquiresOnceForRefreshableAuthFailure(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")}
	second := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "two")}
	source := &fakeSource{descriptor: descriptor, leases: []credentials.Lease{first, second}}
	inner := &scopedClient{invokeErrs: []error{
		&failure.APIError{Status: 401, Code: "unauthorized"},
		nil,
	}}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err != nil {
		t.Fatalf("Invoke() error = %v, want recovery success", err)
	}
	if got := inner.invokeCount(); got != 2 {
		t.Fatalf("wire attempts = %d, want initial + one recovery", got)
	}
	if got := source.invalidateCount(); got != 1 {
		t.Fatalf("invalidations = %d, want one", got)
	}
	if got := source.invalidatedGeneration(); got != first.generation {
		t.Fatalf("invalidated generation = %v, want %v", got, first.generation)
	}
}

func TestClientDoesNotResetOuterBudgetAfterRecoveryFailure(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")}
	second := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "two")}
	source := &fakeSource{descriptor: descriptor, leases: []credentials.Lease{first, second}}
	authErr := &failure.APIError{Status: 401, Code: "unauthorized"}
	inner := &scopedClient{invokeErrs: []error{authErr, authErr, nil}}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); !errors.Is(err, authErr) {
		t.Fatalf("Invoke() error = %v, want second auth error", err)
	}
	if got := inner.invokeCount(); got != 2 {
		t.Fatalf("wire attempts = %d, want two (no outer retry-budget reset)", got)
	}
	if got := source.invalidateCount(); got != 1 {
		t.Fatalf("invalidations = %d, want one", got)
	}
}

func TestClientIsSafeForConcurrentInvoke(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	source := &fakeSource{descriptor: descriptor, leases: []credentials.Lease{fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")}}}
	inner := &scopedClient{}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	const calls = 32
	var wg sync.WaitGroup
	for i := 0; i < calls; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := client.Invoke(context.Background(), inference.Request{}); err != nil {
				t.Errorf("Invoke() error = %v", err)
			}
		}()
	}
	wg.Wait()
	if got := inner.invokeCount(); got != calls {
		t.Fatalf("wire attempts = %d, want %d", got, calls)
	}
}

func TestClientRejectsLegacyOnlyInnerClient(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	source := &fakeSource{descriptor: testDescriptor(policy.Accepted[0]), leases: []credentials.Lease{fakeLease{descriptor: testDescriptor(policy.Accepted[0]), generation: mustGeneration(t, "one")}}}
	if client, err := New(&legacyClient{}, source, policy); client != nil || err == nil {
		t.Fatalf("New(legacy-only inner) = (%T, %v), want fail-closed error", client, err)
	}
}

func testPolicy() llm.AuthPolicy {
	return llm.AuthPolicy{Accepted: []llm.AuthBinding{{
		Provider:  "openai",
		Transport: "responses",
		Scheme:    credentials.SchemeAPIKey,
		Usage:     credentials.UsageMeteredAPI,
		Issuer:    "https://api.openai.com",
		Audience:  "api://openai",
	}}}
}

func testDescriptor(binding llm.AuthBinding) credentials.Descriptor {
	descriptor, err := credentials.NewDescriptor(binding.Provider, binding.Transport, binding.Scheme, binding.Usage, binding.Issuer, binding.Audience, "")
	if err != nil {
		panic(err)
	}
	return descriptor
}

func mustGeneration(t *testing.T, value string) credentials.Generation {
	t.Helper()
	generation, err := credentials.NewGeneration(value)
	if err != nil {
		t.Fatal(err)
	}
	return generation
}

type fakeSource struct {
	mu          sync.Mutex
	descriptor  credentials.Descriptor
	leases      []credentials.Lease
	acquires    int
	invalidates int
	invalidated credentials.Generation
}

func (s *fakeSource) Reference() credentials.Reference   { return credentials.Reference{} }
func (s *fakeSource) Descriptor() credentials.Descriptor { return s.descriptor }
func (s *fakeSource) Acquire(context.Context) (credentials.Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.acquires++
	if len(s.leases) == 0 {
		return nil, errors.New("no lease")
	}
	lease := s.leases[0]
	if len(s.leases) > 1 {
		s.leases = s.leases[1:]
	}
	return lease, nil
}
func (s *fakeSource) Invalidate(_ context.Context, generation credentials.Generation, _ credentials.Failure) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.invalidates++
	s.invalidated = generation
	return nil
}
func (s *fakeSource) Close() error         { return nil }
func (s *fakeSource) invalidateCount() int { s.mu.Lock(); defer s.mu.Unlock(); return s.invalidates }
func (s *fakeSource) invalidatedGeneration() credentials.Generation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.invalidated
}

type fakeLease struct {
	descriptor credentials.Descriptor
	generation credentials.Generation
}

func (l fakeLease) Generation() credentials.Generation { return l.generation }
func (l fakeLease) Descriptor() credentials.Descriptor { return l.descriptor }
func (l fakeLease) ExpiresAt() time.Time               { return time.Time{} }
func (l fakeLease) Authorizer() httpauth.Authorizer    { return httpauth.None() }

type scopedClient struct {
	mu         sync.Mutex
	invokeErrs []error
	invokes    int
}

func (c *scopedClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return nil, errors.New("legacy Invoke should not be called")
}
func (c *scopedClient) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("legacy Stream should not be called")
}
func (c *scopedClient) InvokeWithAuth(context.Context, inference.Request, httpauth.Authorizer) (*inference.Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.invokes++
	if len(c.invokeErrs) == 0 {
		return &inference.Response{}, nil
	}
	err := c.invokeErrs[0]
	c.invokeErrs = c.invokeErrs[1:]
	return &inference.Response{}, err
}
func (c *scopedClient) StreamWithAuth(context.Context, inference.Request, httpauth.Authorizer) (*stream.StreamReader[content.Chunk], error) {
	return nil, io.EOF
}
func (c *scopedClient) invokeCount() int { c.mu.Lock(); defer c.mu.Unlock(); return c.invokes }

type legacyClient struct{}

func (*legacyClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return &inference.Response{}, nil
}
func (*legacyClient) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, io.EOF
}
