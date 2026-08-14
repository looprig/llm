//go:build live

package livetest

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
)

// maxCapturedBody bounds what the recorder retains from one exchange. It matches
// transport.MaxErrorResponseBodyBytes so a captured error body is never shorter
// than what the transport itself saw.
const maxCapturedBody = 64 << 10

// exchange is one recorded request/response pair. Headers are deliberately NOT
// retained: they carry the credential, and nothing a conformance probe needs to
// report lives in them.
type exchange struct {
	Method       string
	Path         string
	RequestBody  []byte
	Status       int
	ContentType  string
	ResponseBody []byte // captured for non-2xx only; a 2xx body is the decoder's job
	Streamed     bool
}

// recorder is a loopback reverse proxy that forwards our client's requests
// verbatim to the real upstream and keeps a transcript of what crossed it.
//
// Why a proxy rather than a RoundTripper. Only some provider constructors expose
// WithRoundTripper (providers/openai does; providers/anthropic does not), and a
// probe suite that reached into different injection seams per provider would be
// testing the seams rather than the wire. Every provider constructor accepts a
// Model.BaseURL, and llm.ValidateModel permits plaintext http for a loopback
// host, so a loopback base URL is the one injection point all of them share.
// (llm/auto.New is deliberately stricter — requestOriginForModel demands HTTPS
// for any authenticated provider — so these probes call the same provider
// constructors auto.New dispatches to, one layer below it.)
//
// The proxy is byte-transparent: it copies method, path suffix, headers and body
// through unchanged, so what the upstream receives is exactly what our encoder
// produced. It adds no X-Forwarded-* headers (ReverseProxy.Rewrite only adds
// them when SetXForwarded is called), and FlushInterval -1 forwards SSE frames
// as they arrive so streaming behaves as it does direct.
type recorder struct {
	server   *httptest.Server
	upstream *url.URL

	mu        sync.Mutex
	exchanges []exchange
}

// newRecorder starts a loopback proxy in front of upstreamBase (an absolute
// https URL, path prefix included) and stops it when the test ends.
func newRecorder(t *testing.T, upstreamBase string) *recorder {
	t.Helper()
	parsed, err := url.Parse(strings.TrimRight(upstreamBase, "/"))
	if err != nil {
		t.Fatalf("livetest: upstream base %q is not a URL: %v", upstreamBase, err)
	}
	if parsed.Scheme != "https" || parsed.Host == "" {
		t.Fatalf("livetest: upstream base %q must be an absolute https URL", upstreamBase)
	}

	rec := &recorder{upstream: parsed}
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.Out.URL.Scheme = parsed.Scheme
			pr.Out.URL.Host = parsed.Host
			pr.Out.URL.Path = parsed.Path + pr.In.URL.Path
			pr.Out.URL.RawQuery = pr.In.URL.RawQuery
			pr.Out.Host = parsed.Host
		},
		FlushInterval:  -1,
		ModifyResponse: rec.captureResponse,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			// Surface an upstream transport failure as a distinguishable status
			// so a probe never reports a network fault as a provider rejection.
			http.Error(w, "livetest proxy upstream error: "+scrub(err.Error()), http.StatusBadGateway)
		},
	}

	rec.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.captureRequest(r)
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(rec.server.Close)
	return rec
}

// baseURL is the loopback origin to hand a provider constructor as Model.BaseURL.
func (r *recorder) baseURL() string { return r.server.URL }

func (r *recorder) captureRequest(req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		body = []byte("<livetest: request body unreadable: " + err.Error() + ">")
	}
	_ = req.Body.Close()
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	captured := body
	if len(captured) > maxCapturedBody {
		captured = captured[:maxCapturedBody]
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.exchanges = append(r.exchanges, exchange{
		Method:      req.Method,
		Path:        req.URL.Path,
		RequestBody: append([]byte(nil), captured...),
	})
}

func (r *recorder) captureResponse(resp *http.Response) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.exchanges) == 0 {
		return nil
	}
	current := &r.exchanges[len(r.exchanges)-1]
	current.Status = resp.StatusCode
	current.ContentType = resp.Header.Get("Content-Type")
	current.Streamed = strings.HasPrefix(current.ContentType, "text/event-stream")

	// A 2xx body is left untouched: the decoder under test must read it, and an
	// SSE body must not be buffered. Only an error body is captured, because a
	// sanitized failure.APIError cannot carry it and it is the single most
	// valuable output of a live conformance probe.
	if resp.StatusCode/100 == 2 {
		return nil
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxCapturedBody))
	_ = resp.Body.Close()
	if err != nil {
		body = []byte("<livetest: error body unreadable: " + err.Error() + ">")
	}
	// Forward the bytes onward untouched, but keep a READABLE copy for the
	// report. An edge (Cloudflare, in front of at least one gateway here)
	// gzips its error pages, and a compressed blob in the log defeats the
	// whole purpose of capturing the error body.
	current.ResponseBody = decodedForReport(body, resp.Header.Get("Content-Encoding"))
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	resp.Header.Del("Content-Length")
	return nil
}

// decodedForReport gunzips a captured body when the response says it is gzipped,
// returning the original bytes when it is not or when decompression fails.
func decodedForReport(body []byte, contentEncoding string) []byte {
	if !strings.EqualFold(strings.TrimSpace(contentEncoding), "gzip") {
		return body
	}
	reader, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return body
	}
	defer func() { _ = reader.Close() }()
	plain, err := io.ReadAll(io.LimitReader(reader, maxCapturedBody))
	if err != nil || len(plain) == 0 {
		return body
	}
	return plain
}

// snapshot returns a copy of the transcript so far. A nil recorder is the
// legitimate state for a provider whose client cannot be pointed at a loopback
// origin (providers/chutes binds several gateway hosts of its own), so every
// reporting method tolerates it and simply has nothing to add.
func (r *recorder) snapshot() []exchange {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]exchange(nil), r.exchanges...)
}

// dumpEnabled reports whether LOOPRIG_LIVE_DUMP asks every probe to print its
// full scrubbed transcript, not just the failing ones. A pass is often as
// interesting as a failure here — it is the only way to see WHICH body a
// permissive gateway accepted — but printing every body by default would bury
// the result, so it is opt-in.
func dumpEnabled() bool {
	value := os.Getenv("LOOPRIG_LIVE_DUMP")
	return value != "" && value != "0" && !strings.EqualFold(value, "false")
}

// recentExchanges bounds what a failure path prints. One recorder serves every
// subtest of a target, so by the last case its transcript holds dozens of
// unrelated bodies; printing all of them buries the two that matter. A failing
// probe sends at most a handful of requests, so the tail is the failure.
const recentExchanges = 3

// dump writes the WHOLE transcript when LOOPRIG_LIVE_DUMP is set. A pass is
// often as interesting as a failure here, but only on request.
func (r *recorder) dump(t *testing.T) {
	t.Helper()
	if dumpEnabled() {
		r.reportAll(t)
	}
}

// report writes the tail of the transcript to the test log, scrubbed. Call it
// from a failure path so a rejected request can be diagnosed against the exact
// bytes that produced it.
func (r *recorder) report(t *testing.T) {
	t.Helper()
	r.reportTail(t, recentExchanges)
}

// reportAll writes every recorded exchange.
func (r *recorder) reportAll(t *testing.T) {
	t.Helper()
	r.reportTail(t, 0)
}

func (r *recorder) reportTail(t *testing.T, limit int) {
	t.Helper()
	if r == nil {
		t.Logf("no proxy transcript: this provider's client is not routed through the loopback recorder")
		return
	}
	all := r.snapshot()
	start := 0
	if limit > 0 && len(all) > limit {
		start = len(all) - limit
		t.Logf("transcript: showing the last %d of %d exchanges", limit, len(all))
	}
	for i := start; i < len(all); i++ {
		ex := all[i]
		t.Logf("exchange %d: %s %s -> %d %s", i+1, ex.Method, ex.Path, ex.Status, ex.ContentType)
		t.Logf("  request  : %s", scrubBytes(ex.RequestBody))
		if len(ex.ResponseBody) > 0 {
			t.Logf("  response : %s", scrubBytes(ex.ResponseBody))
		}
	}
}

// lastErrorBody returns the scrubbed error body of the MOST RECENT exchange, and
// only if that exchange itself failed.
//
// Scoping to the final exchange rather than scanning backwards for the newest
// failure is the whole point. One recorder serves every subtest of a target, so
// a backwards scan happily returns a 404 from three subtests ago to explain a
// request that actually died of a transport timeout — attributing one probe's
// server message to another's failure, which is worse than having no message at
// all. If the last request did not produce an error body (a network fault, a
// cancelled context), the honest answer is nothing.
func (r *recorder) lastErrorBody() string {
	all := r.snapshot()
	if len(all) == 0 {
		return ""
	}
	last := all[len(all)-1]
	if last.Status == 0 || last.Status/100 == 2 {
		return ""
	}
	return scrubBytes(last.ResponseBody)
}
