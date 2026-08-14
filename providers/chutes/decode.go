package chutes

import (
	"crypto/mlkem"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sync/atomic"

	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/llm/e2e"
)

// decodeResponse opens the sealed /e2e/invoke response: respMlkemCT(1088) is
// decapsulated with the ephemeral response key to recover the shared secret,
// then e2e.Open derives the e2e-resp-v1 key, AEAD-opens, and gunzips the
// OpenAI JSON. The plaintext is then parsed into a provider-neutral
// *inference.Response.
func decodeResponse(body []byte, respDK *mlkem.DecapsulationKey768) (*inference.Response, error) {
	if len(body) < e2e.MLKEMCTSize {
		return nil, &e2e.Error{Op: "open response", Err: e2e.ErrShortBlob}
	}
	respCT := body[:e2e.MLKEMCTSize]
	blob := body[e2e.MLKEMCTSize:]
	shared, err := respDK.Decapsulate(respCT)
	if err != nil {
		return nil, &e2e.Error{Op: "decapsulate response", Err: err}
	}
	plaintext, err := e2e.Open(shared, respCT, blob, []byte("e2e-resp-v1"), true)
	if err != nil {
		return nil, err
	}
	return openaiapi.DecodeResponse(plaintext)
}

// tryDecryptErrorBody peels chutes' two-layer error wrapping so apiError
// surfaces something readable. HTTP body is plaintext JSON
// `{"detail": "<upstream body, sometimes base64 of binary>"}`. When the inner
// payload happens to be an e2e envelope sealed to our response key (rare in
// practice — chutes-api just relays response.text per its source), we open it.
// When it's opaque binary (the common case, because the upstream model layer
// emits its own un-keyed binary blob — verified live), we substitute a
// synthetic body whose detail names the most likely cause, so callers do not
// see ~2KB of base64 garbage in chat. Returns the body to hand to apiError.
func tryDecryptErrorBody(body []byte, respDK *mlkem.DecapsulationKey768) []byte {
	if respDK == nil {
		return body
	}
	if plaintext := tryDecryptJSONWrappedDetail(body, respDK); plaintext != nil {
		return plaintext
	}
	if plaintext := tryDecryptRawEnvelope(body, respDK); plaintext != nil {
		return plaintext
	}
	// Couldn't decrypt. Three sub-cases, and the first two must not be
	// confused — "we could not decrypt this" is not the same claim as "this
	// was encrypted":
	//   - body is {"detail":"<opaque base64 of binary>"} — genuinely sealed
	//     and lost to us. Substitute a friendly synthetic detail so chat shows
	//     actionable text, and keep the bytes for forensics.
	//   - body is readable text, e.g. the gateway's plaintext 429
	//     {"detail":"Instance is at maximum capacity, try again later"} —
	//     never encrypted in the first place. Nothing failed, the operator can
	//     already read it, and there is nothing to analyse. Pass it through
	//     silently.
	//   - body is opaque bytes in some other shape — leave the bytes alone so
	//     apiError can still try its extractors, but capture them, because an
	//     envelope we do not recognize is exactly what the dump is for.
	if synthetic := synthesizeOpaqueDetail(body); synthetic != nil {
		return synthetic
	}
	if isOpaque(body) {
		dumpUndecryptableBody(body)
	}
	return body
}

// synthesizeOpaqueDetail recognizes {"detail":"<base64 of binary>"} bodies
// and returns a substitute JSON body whose detail is a human-readable
// explanation. Returns nil if body does not match that shape (so the caller
// preserves the original bytes for the next layer to inspect).
func synthesizeOpaqueDetail(body []byte) []byte {
	var env struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(body, &env); err != nil || env.Detail == "" {
		return nil
	}
	decoded, derr := base64.StdEncoding.DecodeString(env.Detail)
	if derr != nil || !isOpaque(decoded) {
		return nil // not base64, too short, or text after all: keep the original
	}
	// The real message is sealed and unavailable to us, so it is unavailable to
	// the operator too. Name the dump in the substitute detail: that file is
	// the only remaining route to the original bytes, and an operator who reads
	// this error in chat should not have to know the dump exists to find it.
	msg := fmt.Sprintf(
		"chutes returned an opaque encrypted error (%d bytes, client cannot decrypt). "+
			"Most common cause: prompt exceeded the model's context window. "+
			"Run a smaller prompt or check the model's context_length on /v1/models.",
		len(decoded),
	)
	if path := dumpUndecryptableBody(body); path != "" {
		msg += " Raw bytes: " + path
	}
	out, err := json.Marshal(map[string]string{"detail": msg})
	if err != nil {
		return nil
	}
	return out
}

// isOpaque reports whether b looks like binary rather than text, and is the
// single test for "there is something here a maintainer would need the raw
// bytes to understand". 32 bytes is enough to distinguish: binary entropy
// leaves well under 75% of them printable, while text or JSON is at ~100%.
// Anything shorter than that is too small to judge and is treated as text.
func isOpaque(b []byte) bool {
	if len(b) < 32 {
		return false
	}
	printable := 0
	for _, c := range b[:32] {
		if (c >= 0x20 && c < 0x7f) || c == '\n' || c == '\r' || c == '\t' {
			printable++
		}
	}
	return printable < 24
}

// maxDumpBodySize caps how much of an undecryptable body we write to disk.
// Prevents an adversarially large response from filling the temp filesystem.
const maxDumpBodySize = 1 << 20 // 1 MiB

// maxDumps caps how many undecryptable bodies one process leaves in the temp
// directory. A size limit alone bounds each file, not the count: a long-lived
// agent that keeps hitting a sealed 4xx would otherwise accumulate a new pair of
// files per request, forever, with nothing to clean them up. A handful of
// samples is all the forensics this is for — they are all the same bytes after
// the first.
const maxDumps = 8

// dumpCount tracks dumps against maxDumps for the life of the process.
var dumpCount atomic.Int64

// dumpUndecryptableBody persists a sealed body we couldn't open to a unique
// temp file and returns its path (empty if nothing was written). Cheap
// forensics: lets a maintainer compare the real wire format against our assumed
// envelope. Best-effort; any IO failure is silently ignored (we still surface
// the body to the caller).
//
// Callers must have established that the body really is opaque — see isOpaque.
// Nothing here deletes these files, so both dimensions are bounded up front:
// maxDumpBodySize per file, maxDumps per process.
func dumpUndecryptableBody(body []byte) string {
	if len(body) > maxDumpBodySize {
		slog.Warn("chutes: error body decryption failed; body too large to dump",
			"size", len(body),
			"limit", maxDumpBodySize,
		)
		return ""
	}
	if n := dumpCount.Add(1); n > maxDumps {
		if n == maxDumps+1 {
			slog.Warn("chutes: error body decryption failed; dump limit reached, not writing more",
				"limit", maxDumps,
			)
		}
		return ""
	}
	f, err := os.CreateTemp("", "chutes-undecryptable-*.bin")
	if err != nil {
		return ""
	}
	_, _ = f.Write(body)
	_ = f.Close()
	slog.Warn("chutes: error body decryption failed; dumped raw bytes for analysis",
		"path", f.Name(),
		"size", len(body),
	)
	dumpDecodedDetail(body)
	return f.Name()
}

// dumpDecodedDetail writes the base64-decoded `detail` of a JSON-wrapped body
// alongside the raw dump, so inspecting it does not mean chaining
// `jq .detail | base64 -d`. It shares the caller's slot against maxDumps.
func dumpDecodedDetail(body []byte) {
	var env struct {
		Detail string `json:"detail"`
	}
	if json.Unmarshal(body, &env) != nil || env.Detail == "" {
		return
	}
	decoded, derr := base64.StdEncoding.DecodeString(env.Detail)
	if derr != nil {
		return
	}
	g, err := os.CreateTemp("", "chutes-undecryptable-detail-*.bin")
	if err != nil {
		return
	}
	_, _ = g.Write(decoded)
	_ = g.Close()
	slog.Warn("chutes: also dumped decoded detail bytes",
		"path", g.Name(),
		"size", len(decoded),
	)
}

// tryDecryptJSONWrappedDetail handles {"detail":"<base64-of-e2e-envelope>"}.
// Returns nil (NOT body) on any failure so the caller can fall through.
func tryDecryptJSONWrappedDetail(body []byte, respDK *mlkem.DecapsulationKey768) []byte {
	var env struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(body, &env); err != nil || env.Detail == "" {
		return nil
	}
	decoded, err := base64.StdEncoding.DecodeString(env.Detail)
	if err != nil {
		return nil
	}
	return tryDecryptRawEnvelope(decoded, respDK)
}

// tryDecryptRawEnvelope handles a bare mlkem_ct || nonce || ct || tag blob.
// Returns nil on any failure (length too small, decap fail, open fail).
func tryDecryptRawEnvelope(blob []byte, respDK *mlkem.DecapsulationKey768) []byte {
	if len(blob) < e2e.MLKEMCTSize+e2e.NonceSize+e2e.TagSize {
		return nil
	}
	respCT := blob[:e2e.MLKEMCTSize]
	sealed := blob[e2e.MLKEMCTSize:]
	shared, err := respDK.Decapsulate(respCT)
	if err != nil {
		return nil
	}
	if plaintext, err := e2e.Open(shared, respCT, sealed, []byte("e2e-resp-v1"), true); err == nil {
		return plaintext
	}
	if plaintext, err := e2e.Open(shared, respCT, sealed, []byte("e2e-resp-v1"), false); err == nil {
		return plaintext
	}
	return nil
}
