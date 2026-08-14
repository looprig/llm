//go:build live

package livetest

import (
	"context"
	"errors"
	"time"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	failure "github.com/looprig/inference/failure"
	"github.com/looprig/inference/stream"
)

// Capacity failures are not conformance results. A 429 from a free tier and a
// 503 from an overloaded gateway say nothing whatever about the body we sent —
// the server never looked at it — and letting them land as test failures would
// bury the findings this suite exists to produce under noise that changes run to
// run. They are retried a bounded number of times and, if they persist,
// classified as an ENVIRONMENT outcome rather than an encoder one.
var transientStatuses = map[int]bool{
	408: true, // request timeout
	429: true, // rate limited / quota exhausted
	500: true, // upstream fault
	502: true, // bad gateway
	503: true, // unavailable / overloaded
	504: true, // gateway timeout
}

// isTransient reports whether err is a capacity or availability failure rather
// than a rejection of the request body.
//
// A transport fault counts. If the connection timed out or was dropped, no
// server ever formed an opinion about our body — there is no verdict to record,
// which is exactly the same epistemic position as a 503. This matters most for
// providers/chutes, whose requests cross an attested TEE tunnel that
// occasionally stalls; reporting that as "the gateway rejected our tool_choice"
// would be a fabricated conformance result.
func isTransient(err error) bool {
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) {
		return transientStatuses[apiErr.Status]
	}
	var netErr *failure.NetworkError
	if errors.As(err, &netErr) {
		return true
	}
	return errors.Is(err, context.DeadlineExceeded)
}

func transientStatus(err error) int {
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) {
		return apiErr.Status
	}
	return 0
}

// retryTransient wraps a client so capacity failures are retried with a simple
// escalating backoff before they reach a probe.
//
// It is deliberately a DECORATOR over inference.Client rather than logic inside
// each probe: a probe that retried its own call would have to distinguish
// "retry" from "this is the second turn of the conversation", and the retried
// request must be byte-identical to the first for the retry to mean anything.
// Wrapping the client keeps every probe written as a single linear
// conversation, which is what makes them readable as use cases.
//
// It honours a server-advertised Retry-After when there is one, because a free
// tier that names a wait is the one source that actually knows.
type retryTransient struct {
	inner    inference.Client
	attempts int
	backoff  time.Duration
}

var _ inference.Client = retryTransient{}

func withRetries(inner inference.Client) inference.Client {
	return retryTransient{inner: inner, attempts: 4, backoff: 2 * time.Second}
}

func (r retryTransient) Invoke(ctx context.Context, req inference.Request) (*inference.Response, error) {
	var lastErr error
	for attempt := 1; attempt <= r.attempts; attempt++ {
		resp, err := r.inner.Invoke(ctx, req)
		if err == nil {
			if resp != nil {
				resp.Attempts = attempt
			}
			return resp, nil
		}
		lastErr = err
		if !isTransient(err) || attempt == r.attempts {
			return nil, err
		}
		if !sleepCtx(ctx, r.waitFor(err, attempt)) {
			return nil, err
		}
	}
	return nil, lastErr
}

func (r retryTransient) Stream(ctx context.Context, req inference.Request) (*stream.StreamReader[content.Chunk], error) {
	var lastErr error
	for attempt := 1; attempt <= r.attempts; attempt++ {
		reader, err := r.inner.Stream(ctx, req)
		if err == nil {
			return reader, nil
		}
		lastErr = err
		if !isTransient(err) || attempt == r.attempts {
			return nil, err
		}
		if !sleepCtx(ctx, r.waitFor(err, attempt)) {
			return nil, err
		}
	}
	return nil, lastErr
}

// waitFor prefers the server's own Retry-After and otherwise escalates
// linearly. It caps the wait so one exhausted daily quota cannot stall a run
// for the whole probe timeout.
func (r retryTransient) waitFor(err error, attempt int) time.Duration {
	const maxWait = 20 * time.Second
	var apiErr *failure.APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		if apiErr.RetryAfter < maxWait {
			return apiErr.RetryAfter
		}
		return maxWait
	}
	wait := time.Duration(attempt) * r.backoff
	if wait > maxWait {
		return maxWait
	}
	return wait
}

// sleepCtx waits for d, reporting false if the context ended first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
