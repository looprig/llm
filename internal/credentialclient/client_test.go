package credentialclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/failure"
	"github.com/looprig/inference/retry"
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

func TestClientOriginGuardsConcreteRequestBeforeCredentialAttach(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	authorizer := &countingAuthorizer{}
	source := &fakeSource{descriptor: descriptor, leases: []credentials.Lease{fakeLease{descriptor: descriptor, generation: mustGeneration(t, "origin"), authorizer: authorizer}}}
	inner := &originCheckingClient{url: "https://evil.example/v1/chat/completions"}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err == nil {
		t.Fatal("Invoke() = nil error for cross-origin concrete request")
	}
	if got := authorizer.calls; got != 0 {
		t.Fatalf("credential authorizer calls = %d, want zero", got)
	}
}

func TestClientUsesLeaseMatchedBindingForOrigin(t *testing.T) {
	first := llm.AuthBinding{Provider: "openai", Transport: "chat", Scheme: credentials.SchemeAPIKey, Usage: credentials.UsageMeteredAPI, Issuer: "https://api.openai.com", Audience: "https://api.openai.com"}
	second := llm.AuthBinding{Provider: "openai", Transport: "responses", Scheme: credentials.SchemeAPIKey, Usage: credentials.UsageMeteredAPI, Issuer: "https://api.openai.com", Audience: "https://proxy.example.test"}
	policy := llm.AuthPolicy{Accepted: []llm.AuthBinding{first, second}}
	descriptor := testDescriptor(second)
	source := &fakeSource{descriptor: descriptor, recoverable: true, leases: []credentials.Lease{fakeLease{descriptor: descriptor, generation: mustGeneration(t, "matched")}}}
	client, err := New(&originCheckingClient{url: "https://proxy.example.test/v1/chat/completions"}, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err != nil {
		t.Fatalf("matched second binding rejected: %v", err)
	}
}

func TestClientInvalidatesAndReacquiresOnceForRefreshableAuthFailure(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")}
	second := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "two")}
	source := &fakeSource{descriptor: descriptor, recoverable: true, leases: []credentials.Lease{first, second}}
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

func TestInvalidateOnceCanceledLeaderIsRemovedForLaterRetry(t *testing.T) {
	t.Parallel()

	generation := mustGeneration(t, "cancel-leader")
	source := newCoordinationSource(testDescriptor(testPolicy().Accepted[0]))
	started := make(chan struct{})
	source.wait[generation] = started
	source.started[generation] = make(chan struct{})
	client := &Client{source: source}
	ctx, cancel := context.WithCancel(context.Background())
	leaderErr := make(chan error, 1)
	go func() {
		leaderErr <- client.invalidateOnce(ctx, generation, credentials.FailureAuthExpired)
	}()
	<-source.started[generation]
	cancel()
	if err := <-leaderErr; err == nil {
		t.Fatal("canceled leader error = nil, want cancellation")
	}
	delete(source.wait, generation)
	if err := client.invalidateOnce(context.Background(), generation, credentials.FailureAuthExpired); err != nil {
		t.Fatalf("later healthy invalidation = %v, want success", err)
	}
	if got := source.invalidateCount(generation); got != 2 {
		t.Fatalf("generation invalidation count = %d, want failed leader plus later retry", got)
	}
}

func TestInvalidateOnceTransientFailureThenSuccess(t *testing.T) {
	t.Parallel()

	generation := mustGeneration(t, "transient")
	source := newCoordinationSource(testDescriptor(testPolicy().Accepted[0]))
	source.errs[generation] = []error{errors.New("temporary fixture failure"), nil}
	client := &Client{source: source}
	if err := client.invalidateOnce(context.Background(), generation, credentials.FailureAuthExpired); err == nil {
		t.Fatal("first invalidation error = nil, want transient failure")
	}
	if err := client.invalidateOnce(context.Background(), generation, credentials.FailureAuthExpired); err != nil {
		t.Fatalf("second invalidation = %v, want success", err)
	}
	if got := source.invalidateCount(generation); got != 2 {
		t.Fatalf("generation invalidation count = %d, want 2", got)
	}
}

func TestInvalidateOnceAllowsOverlappingGenerationsAndPrunesAfterAcquire(t *testing.T) {
	t.Parallel()

	first := mustGeneration(t, "overlap-one")
	second := mustGeneration(t, "overlap-two")
	newer := mustGeneration(t, "overlap-new")
	source := newCoordinationSource(testDescriptor(testPolicy().Accepted[0]))
	firstRelease := make(chan struct{})
	secondRelease := make(chan struct{})
	source.wait[first] = firstRelease
	source.wait[second] = secondRelease
	source.started[first] = make(chan struct{})
	source.started[second] = make(chan struct{})
	client := &Client{source: source}
	results := make(chan error, 2)
	go func() { results <- client.invalidateOnce(context.Background(), first, credentials.FailureAuthExpired) }()
	go func() { results <- client.invalidateOnce(context.Background(), second, credentials.FailureAuthExpired) }()
	<-source.started[first]
	<-source.started[second]
	close(firstRelease)
	close(secondRelease)
	if err := <-results; err != nil {
		t.Fatalf("first overlapping invalidation = %v", err)
	}
	if err := <-results; err != nil {
		t.Fatalf("second overlapping invalidation = %v", err)
	}
	client.observeGeneration(newer)
	client.recoveryMu.Lock()
	defer client.recoveryMu.Unlock()
	if len(client.recovery) != 0 {
		t.Fatalf("successful recovery entries after newer generation = %d, want 0", len(client.recovery))
	}
}

func TestClientDoesNotResetOuterBudgetAfterRecoveryFailure(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")}
	second := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "two")}
	source := &fakeSource{descriptor: descriptor, recoverable: true, leases: []credentials.Lease{first, second}}
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

func TestClientComposesWithInferenceRetry(t *testing.T) {
	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	source := &fakeSource{descriptor: descriptor, recoverable: true, leases: []credentials.Lease{
		fakeLease{descriptor: descriptor, generation: mustGeneration(t, "one")},
		fakeLease{descriptor: descriptor, generation: mustGeneration(t, "two")},
		fakeLease{descriptor: descriptor, generation: mustGeneration(t, "three")},
	}}
	inner := &scopedClient{invokeErrs: []error{
		&failure.APIError{Status: 401, Code: "unauthorized"},
		&failure.APIError{Status: 500, Code: "server_error"},
		nil,
	}}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	outer, err := retry.New(client, retry.Policy{StableRetries: 0, StableDelay: time.Nanosecond, MaxAttempts: 2, MaxDelay: time.Nanosecond})
	if err != nil {
		t.Fatal(err)
	}
	response, err := outer.Invoke(context.Background(), inference.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if response.Attempts != 2 {
		t.Fatalf("outer attempts = %d, want 2", response.Attempts)
	}
	if got := inner.invokeCount(); got != 3 {
		t.Fatalf("wire attempts = %d, want 3", got)
	}
	if got := source.invalidateCount(); got != 1 {
		t.Fatalf("invalidations = %d, want 1", got)
	}
}

func TestClientRequiresRenewableSourceForRecovery(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "non-renewable")}
	source := &fakeSource{descriptor: descriptor, leases: []credentials.Lease{first}}
	inner := &scopedClient{invokeErrs: []error{&failure.APIError{Status: 401, Code: "unauthorized"}}}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err == nil {
		t.Fatal("Invoke() = nil error, want original auth error")
	}
	if got := inner.invokeCount(); got != 1 {
		t.Fatalf("wire attempts = %d, want one", got)
	}
	if got := source.invalidateCount(); got != 0 {
		t.Fatalf("invalidations = %d, want zero", got)
	}
}

func TestClientRequiresExplicitSafeAuthCodeForRecovery(t *testing.T) {
	t.Parallel()

	policy := testPolicy()
	descriptor := testDescriptor(policy.Accepted[0])
	first := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "blank-code")}
	second := fakeLease{descriptor: descriptor, generation: mustGeneration(t, "unused")}
	source := &fakeSource{descriptor: descriptor, recoverable: true, leases: []credentials.Lease{first, second}}
	inner := &scopedClient{invokeErrs: []error{&failure.APIError{Status: 401}}}
	client, err := New(inner, source, policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Invoke(context.Background(), inference.Request{}); err == nil {
		t.Fatal("Invoke() = nil error, want original auth error")
	}
	if got := inner.invokeCount(); got != 1 {
		t.Fatalf("wire attempts = %d, want one", got)
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
		Audience:  "https://api.openai.com",
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
	recoverable bool
	leases      []credentials.Lease
	acquires    int
	invalidates int
	invalidated credentials.Generation
}

type coordinationSource struct {
	mu          sync.Mutex
	descriptor  credentials.Descriptor
	wait        map[credentials.Generation]chan struct{}
	started     map[credentials.Generation]chan struct{}
	startedDone map[credentials.Generation]bool
	errs        map[credentials.Generation][]error
	invalidates map[credentials.Generation]int
}

func newCoordinationSource(descriptor credentials.Descriptor) *coordinationSource {
	return &coordinationSource{
		descriptor:  descriptor,
		wait:        make(map[credentials.Generation]chan struct{}),
		started:     make(map[credentials.Generation]chan struct{}),
		startedDone: make(map[credentials.Generation]bool),
		errs:        make(map[credentials.Generation][]error),
		invalidates: make(map[credentials.Generation]int),
	}
}

func (s *coordinationSource) Reference() credentials.Reference   { return credentials.Reference{} }
func (s *coordinationSource) Descriptor() credentials.Descriptor { return s.descriptor }
func (s *coordinationSource) Acquire(context.Context) (credentials.Lease, error) {
	return nil, errors.New("coordination source does not acquire")
}
func (s *coordinationSource) Invalidate(ctx context.Context, generation credentials.Generation, _ credentials.Failure) error {
	s.mu.Lock()
	s.invalidates[generation]++
	started := s.started[generation]
	if started == nil {
		started = make(chan struct{})
		s.started[generation] = started
	}
	if !s.startedDone[generation] {
		close(started)
		s.startedDone[generation] = true
	}
	wait := s.wait[generation]
	var err error
	if queued := s.errs[generation]; len(queued) > 0 {
		err = queued[0]
		s.errs[generation] = queued[1:]
	}
	s.mu.Unlock()
	if wait != nil {
		select {
		case <-wait:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return err
}
func (s *coordinationSource) Close() error { return nil }
func (s *coordinationSource) invalidateCount(generation credentials.Generation) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.invalidates[generation]
}

func (s *fakeSource) CanRecover(credentials.Failure) bool { return s.recoverable }

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
	authorizer httpauth.Authorizer
}

func (l fakeLease) Generation() credentials.Generation { return l.generation }
func (l fakeLease) Descriptor() credentials.Descriptor { return l.descriptor }
func (l fakeLease) ExpiresAt() time.Time               { return time.Time{} }
func (l fakeLease) Authorizer() httpauth.Authorizer {
	if l.authorizer != nil {
		return l.authorizer
	}
	return httpauth.None()
}

type countingAuthorizer struct{ calls int }

func (a *countingAuthorizer) Authorize(context.Context, *http.Request) error {
	a.calls++
	return nil
}

type originCheckingClient struct{ url string }

func (*originCheckingClient) Invoke(context.Context, inference.Request) (*inference.Response, error) {
	return nil, errors.New("legacy Invoke should not be called")
}
func (*originCheckingClient) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("legacy Stream should not be called")
}
func (c *originCheckingClient) InvokeWithAuth(ctx context.Context, _ inference.Request, authorizer httpauth.Authorizer) (*inference.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, nil)
	if err != nil {
		return nil, err
	}
	if err := authorizer.Authorize(ctx, request); err != nil {
		return nil, err
	}
	return &inference.Response{}, nil
}
func (*originCheckingClient) StreamWithAuth(context.Context, inference.Request, httpauth.Authorizer) (*stream.StreamReader[content.Chunk], error) {
	return nil, io.EOF
}

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
