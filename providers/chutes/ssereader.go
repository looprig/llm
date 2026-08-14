package chutes

import (
	"bufio"
	"errors"
	"io"
	"strings"
)

// errSSEDone is the terminal signal returned by sseEventReader.next when it
// reads the literal `data: [DONE]` payload. It is deliberately distinct from
// io.EOF: [DONE] is a terminal the gateway sent, while io.EOF only says the
// body stopped arriving — which happens both on a legitimate completion the
// gateway did not punctuate (the Chutes capture cut at max_tokens ends that
// way, its final sealed chunk carrying finish_reason "length") and on a
// connection dropped mid-generation. stream.go must not collapse the two; see
// relayTerminal and finishUnterminated.
var errSSEDone = errors.New("sse: [DONE]")

// sseEventReader is an SSE reader that yields the full accumulated `data:`
// payload of each event (multiple `data:` lines in one event are joined with
// "\n", per the SSE spec). Comment/keepalive lines (starting with ':') and
// other fields (`event:`, `id:`, `retry:`) are skipped.
//
// The Chutes e2e stream does NOT use SSE `event:` names; it encodes the event
// type as a JSON key inside the data payload (`{"e2e_init":...}`,
// `{"e2e":...}`, `{"usage":...}`, `{"e2e_error":...}`). That is confirmed by
// the captured fixture testdata/stream.sse and by the reference transport
// (chutesai/chutes-e2ee-transport src/chutes_e2ee/transport.py, which parses
// `data: ` lines and switches on the decoded JSON key). So this reader only
// needs to surface the data payload; stream.go inspects the JSON key.
type sseEventReader struct {
	r  io.ReadCloser
	sc *bufio.Scanner
}

func newSSEEventReader(r io.ReadCloser) *sseEventReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 1 MiB max line
	return &sseEventReader{r: r, sc: sc}
}

// next returns the next event's data payload. It returns errSSEDone for a
// `data: [DONE]` terminal and io.EOF at a clean end of input. A pending partial
// event (data lines with no trailing blank line) is flushed at EOF.
func (s *sseEventReader) next() (string, error) {
	var data strings.Builder
	have := false
	for s.sc.Scan() {
		line := s.sc.Text()
		if line == "" {
			if have {
				return finishEvent(data.String())
			}
			continue // blank line with no accumulated data: skip
		}
		if strings.HasPrefix(line, ":") {
			continue // comment / keepalive
		}
		if strings.HasPrefix(line, "data:") {
			payload := strings.TrimPrefix(line, "data:")
			payload = strings.TrimPrefix(payload, " ")
			if have {
				data.WriteByte('\n')
			}
			data.WriteString(payload)
			have = true
			continue
		}
		// event:, id:, retry: and any other field are ignored.
	}
	// bufio.Scanner suppresses only a BARE io.EOF: Scanner.Err compares with ==,
	// so a body that reports its end of input wrapped surfaces here as a scan
	// error rather than as nothing. A genuine mid-stream fault still short-
	// circuits — flushing a partial event under one would hand the decoder
	// truncated JSON and report a dropped connection as a parse failure — but an
	// EOF, however it is wrapped, must not, because it is the ordinary way a
	// pending partial event ends and that event is routinely the last one,
	// carrying finish_reason. Checking the error before the flush discarded it
	// and turned a completed generation into a truncated one.
	//
	// The wrapped error is returned as-is rather than flattened to io.EOF: it is
	// still a terminal, its framing context may matter to a caller, and pump
	// classifies it with errors.Is precisely so it does not have to be flattened
	// here.
	err := s.sc.Err()
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	if have {
		return finishEvent(data.String())
	}
	if err != nil {
		return "", err
	}
	return "", io.EOF
}

func finishEvent(data string) (string, error) {
	if data == "[DONE]" {
		return "", errSSEDone
	}
	return data, nil
}

func (s *sseEventReader) Close() error {
	return s.r.Close()
}
