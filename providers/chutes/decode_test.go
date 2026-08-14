package chutes

import (
	"crypto/mlkem"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// opaqueDetailBody builds {"detail":"<base64 of binary>"} — the shape a
// genuinely sealed error the client cannot open arrives in. The payload is
// deterministic (bytes 0..255) so the printability heuristic never flakes.
func opaqueDetailBody(t *testing.T) []byte {
	t.Helper()
	raw := make([]byte, 256)
	for i := range raw {
		raw[i] = byte(i)
	}
	body, err := json.Marshal(map[string]string{"detail": base64.StdEncoding.EncodeToString(raw)})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return body
}

func tempDumpKey(t *testing.T) *mlkem.DecapsulationKey768 {
	t.Helper()
	t.Setenv("TMPDIR", t.TempDir())
	dumpCount.Store(0)
	t.Cleanup(func() { dumpCount.Store(0) })
	respDK, err := mlkem.GenerateKey768()
	if err != nil {
		t.Fatalf("GenerateKey768() error = %v", err)
	}
	return respDK
}

func dumpedFiles(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(os.TempDir())
	if err != nil {
		t.Fatalf("ReadDir(%s) error = %v", os.TempDir(), err)
	}
	var names []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "chutes-undecryptable-") {
			names = append(names, e.Name())
		}
	}
	return names
}

// TestTryDecryptErrorBodyDoesNotDumpPlaintext pins the distinction the dumper
// conflated: a body that was never encrypted is not a decryption failure. The
// gateway's plaintext capacity error is readable as it stands, so it must reach
// the caller untouched and leave nothing behind on disk. (103 of the 107 files
// found in a real temp directory were this one 61-byte body.)
func TestTryDecryptErrorBodyDoesNotDumpPlaintext(t *testing.T) {
	respDK := tempDumpKey(t)

	body := []byte(`{"detail":"Instance is at maximum capacity, try again later"}`)
	got := tryDecryptErrorBody(body, respDK)

	if string(got) != string(body) {
		t.Errorf("tryDecryptErrorBody() = %s, want the body unchanged", got)
	}
	if files := dumpedFiles(t); len(files) != 0 {
		t.Errorf("plaintext error body dumped %d file(s) %v, want none", len(files), files)
	}
}

// TestTryDecryptErrorBodyNamesTheDump covers the other half of the conflation: a
// genuinely sealed body the client cannot open. The operator loses the real
// message, so the substitute must at least say where the bytes went.
func TestTryDecryptErrorBodyNamesTheDump(t *testing.T) {
	respDK := tempDumpKey(t)

	got := tryDecryptErrorBody(opaqueDetailBody(t), respDK)

	var env struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(got, &env); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", got, err)
	}
	files := dumpedFiles(t)
	if len(files) == 0 {
		t.Fatal("opaque error body dumped no files, want the raw bytes captured")
	}
	if !strings.Contains(env.Detail, os.TempDir()) {
		t.Errorf("detail = %q, want it to name the dump path under %s", env.Detail, os.TempDir())
	}
}

// TestDumpUndecryptableBodyIsBounded pins the cap. Without one, every 4xx in a
// long-lived process leaves another pair of files in the temp directory
// forever.
func TestDumpUndecryptableBodyIsBounded(t *testing.T) {
	respDK := tempDumpKey(t)

	body := opaqueDetailBody(t)
	for i := 0; i < maxDumps+5; i++ {
		_ = tryDecryptErrorBody(body, respDK)
	}

	// Each dump writes at most two files: the body and its decoded detail.
	if files := dumpedFiles(t); len(files) > 2*maxDumps {
		t.Errorf("dumped %d files, want at most %d", len(files), 2*maxDumps)
	}
}
