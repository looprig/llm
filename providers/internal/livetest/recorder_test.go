//go:build live

package livetest

import (
	"bytes"
	"io"
	"net/http/httptest"
	"testing"
)

func TestCaptureRequestBoundsTranscriptWithoutTruncatingForwardedBody(t *testing.T) {
	want := bytes.Repeat([]byte("request-payload-"), maxCapturedBody/8)
	req := httptest.NewRequest("POST", "https://example.test/v1/messages", bytes.NewReader(want))
	rec := &recorder{}

	rec.captureRequest(req)
	forwarded, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("ReadAll(forwarded body) error = %v", err)
	}
	if !bytes.Equal(forwarded, want) {
		t.Fatalf("forwarded body length = %d, want unchanged %d", len(forwarded), len(want))
	}
	got := rec.snapshot()
	if len(got) != 1 || len(got[0].RequestBody) != maxCapturedBody {
		t.Fatalf("captured body length = %d, want bounded %d", len(got[0].RequestBody), maxCapturedBody)
	}
}
