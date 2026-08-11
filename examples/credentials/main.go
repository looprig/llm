package main

import (
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/credentials"
	"github.com/looprig/credentials/httpauth"
	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	"github.com/looprig/inference/model"
	"github.com/looprig/llm"
	"github.com/looprig/llm/auto"
)

func main() {
	var requests atomic.Int64
	var matched atomic.Int64
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requestNumber := requests.Add(1)
		expected := fmt.Sprintf("Bearer lease-key-%d", requestNumber)
		if request.Header.Get("Authorization") == expected {
			matched.Add(1)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"id":"response-id","model":"fake-model","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	selected := model.CustomModel(
		model.ProviderName(llm.ProviderOpenAI),
		model.APIFormatOpenAI,
		server.URL,
		"fake-model",
	)
	policy, err := llm.AuthPolicyForModel(selected)
	if err != nil {
		panic(err)
	}
	descriptor, err := policy.Accepted[0].Descriptor()
	if err != nil {
		panic(err)
	}
	source := &rotatingSource{descriptor: descriptor}
	for index := 1; index <= 2; index++ {
		generation, err := credentials.NewGeneration(fmt.Sprintf("example-v%d", index))
		if err != nil {
			panic(err)
		}
		source.leases = append(source.leases, fixedLease{
			descriptor: descriptor,
			generation: generation,
			authorizer: auth.Key(auth.APIKey(fmt.Sprintf("lease-key-%d", index))),
		})
	}
	defer source.Close()

	roots := x509.NewCertPool()
	roots.AddCert(server.Certificate())
	client, err := auto.NewWithAuth(selected, source, auto.WithTLSRootCAs(roots))
	if err != nil {
		panic(err)
	}
	var response *inference.Response
	for range 2 {
		response, err = client.Invoke(context.Background(), inference.Request{Model: selected})
		if err != nil {
			panic(err)
		}
	}

	fmt.Printf("acquires=%d rotated-authorizations=%d\n", source.acquires.Load(), matched.Load())
	text, ok := response.Message.Blocks[0].(*content.TextBlock)
	if !ok {
		panic("provider returned a non-text response")
	}
	fmt.Printf("provider-response=%s\n", text.Text)
}

type rotatingSource struct {
	descriptor credentials.Descriptor
	leases     []credentials.Lease
	acquires   atomic.Int64
}

func (source *rotatingSource) Reference() credentials.Reference { return credentials.Reference{} }

func (source *rotatingSource) Descriptor() credentials.Descriptor { return source.descriptor }

func (source *rotatingSource) Acquire(context.Context) (credentials.Lease, error) {
	index := source.acquires.Add(1) - 1
	if index < 0 || index >= int64(len(source.leases)) {
		return nil, errors.New("example credential source exhausted")
	}
	return source.leases[index], nil
}

func (*rotatingSource) Invalidate(context.Context, credentials.Generation, credentials.Failure) error {
	return nil
}

func (*rotatingSource) Close() error { return nil }

type fixedLease struct {
	descriptor credentials.Descriptor
	generation credentials.Generation
	authorizer httpauth.Authorizer
}

func (lease fixedLease) Generation() credentials.Generation { return lease.generation }

func (lease fixedLease) Descriptor() credentials.Descriptor { return lease.descriptor }

func (fixedLease) ExpiresAt() time.Time { return time.Time{} }

func (lease fixedLease) Authorizer() httpauth.Authorizer { return lease.authorizer }
