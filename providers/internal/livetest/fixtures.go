//go:build live

package livetest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"strconv"
	"sync"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

// The multimodal probes need a payload with ONE unambiguous, verifiable answer.
// A photograph would test the model's vision rather than our encoding, and a
// wrong answer would be indistinguishable from a dropped image. A solid colour
// and a short nonce string are things any multimodal model reads correctly if —
// and only if — the bytes arrived and were understood as media.
const (
	// imageColourName is the answer a model that received the image must give.
	imageColourName = "red"
	// documentNonce is the token embedded in every document fixture. It is
	// deliberately not a word: a model that never saw the document cannot
	// produce it, so echoing it is proof of delivery rather than of guessing.
	documentNonce = "ZQ7431"
)

var (
	pngOnce  sync.Once
	pngBytes []byte
	pdfOnce  sync.Once
	pdfBytes []byte
)

// redPNG is a small solid-red PNG, generated rather than vendored so no binary
// fixture enters the tree and the bytes are provably what they claim to be.
// 64x64 rather than 1x1: several providers reject or silently downscale
// degenerate images, and a rejection over image DIMENSIONS would be
// indistinguishable from a rejection of our encoding.
func redPNG() []byte {
	pngOnce.Do(func() {
		img := image.NewRGBA(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				img.Set(x, y, color.RGBA{R: 220, G: 20, B: 20, A: 255})
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			panic("livetest: encode fixture png: " + err.Error())
		}
		pngBytes = buf.Bytes()
	})
	return pngBytes
}

// noncePDF is a minimal, valid, uncompressed one-page PDF whose only visible
// content is documentNonce.
//
// It is BUILT rather than embedded because a PDF's cross-reference table holds
// byte offsets into its own body: hand-writing one and then editing a single
// character silently corrupts it, and a corrupt PDF rejected by a provider
// would be reported here as an encoder defect. Computing the offsets from the
// bytes actually emitted makes that class of false finding unreachable.
func noncePDF() []byte {
	pdfOnce.Do(func() {
		objects := []string{
			"<< /Type /Catalog /Pages 2 0 R >>",
			"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
			"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 120] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
			"", // content stream, filled below
			"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
		}
		streamText := "BT /F1 24 Tf 24 56 Td (" + documentNonce + ") Tj ET\n"
		objects[3] = "<< /Length " + strconv.Itoa(len(streamText)) + " >>\nstream\n" + streamText + "endstream"

		var buf bytes.Buffer
		buf.WriteString("%PDF-1.4\n")
		offsets := make([]int, len(objects))
		for i, body := range objects {
			offsets[i] = buf.Len()
			fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", i+1, body)
		}
		xref := buf.Len()
		fmt.Fprintf(&buf, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
		for _, offset := range offsets {
			fmt.Fprintf(&buf, "%010d 00000 n \n", offset)
		}
		fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
		pdfBytes = buf.Bytes()
	})
	return pdfBytes
}

// imageMessage is a user turn carrying inline image bytes plus the question.
// Inline base64 is the only multimodal source every one of these providers
// accepts without a prior upload, and it is the source this session added.
func imageMessage(prompt string) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{
		Role: content.RoleUser,
		Blocks: []content.Block{
			&content.ImageBlock{
				MediaType: content.MediaTypeImagePNG,
				Source:    content.ImageSource{Data: redPNG()},
			},
			&content.TextBlock{Text: prompt},
		},
	}}
}

// documentMessage is a user turn carrying one document block plus the question.
func documentMessage(doc *content.DocumentBlock, prompt string) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{
		Role:   content.RoleUser,
		Blocks: []content.Block{doc, &content.TextBlock{Text: prompt}},
	}}
}

// pdfDocument and textDocument are the two document shapes the neutral block can
// actually reach on the wire. They are separate probes because the dialects
// disagree about which is legal: anthropicapi's documentSourceOf maps binary
// data ONLY to Base64PDFSource and text ONLY to PlainTextSource (both media
// types are wire consts), while geminiapi sends bytes as an inlineData Blob and
// text as a plain Part.text, and openaiapi sends either as one base64 `file`
// part. A provider can therefore accept one and reject the other, and reporting
// them together would hide which.
func pdfDocument() *content.DocumentBlock {
	return &content.DocumentBlock{
		MediaType: content.MediaTypeDocumentPDF,
		Name:      "nonce.pdf",
		Data:      noncePDF(),
	}
}

func textDocument() *content.DocumentBlock {
	return &content.DocumentBlock{
		MediaType: content.MediaTypeDocumentText,
		Name:      "nonce.txt",
		Text:      "Internal reference sheet.\nThe verification code is " + documentNonce + ".\nEnd of sheet.\n",
	}
}

// timeTool is the SECOND tool the tool-choice probes need. A named-choice probe
// with one declared tool proves nothing: "the model called the only tool it
// had" and "the server honoured our tool_choice" produce identical transcripts.
// With two tools and a prompt that points at the OTHER one, only an honoured
// choice can produce the observed call.
func timeTool() inference.Tool {
	return inference.Tool{
		Name:        "get_local_time",
		Description: "Return the current local time for one city.",
		Schema: json.RawMessage(`{
			"type": "object",
			"properties": {"city": {"type": "string", "description": "City name"}},
			"required": ["city"]
		}`),
	}
}

// weatherOutputSchema is the structured-output contract. It satisfies the
// portable subset inference.ValidateOutputSchema enforces: every object node
// declares additionalProperties:false and lists every property in `required`,
// which is also exactly what OpenAI's strict mode and Anthropic's output_config
// format demand.
func weatherOutputSchema() *inference.OutputSchema {
	return &inference.OutputSchema{
		Name:        "city_report",
		Description: "One city and its temperature in Celsius.",
		Strict:      true,
		Schema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"city": {"type": "string"},
				"temperature_c": {"type": "integer"}
			},
			"required": ["city", "temperature_c"],
			"additionalProperties": false
		}`),
	}
}

// cityReport is the decode target for weatherOutputSchema. Decoding into a
// typed value rather than a map is the point: inference.DecodeOutput is strict,
// so an extra or missing member fails here rather than passing as a map with
// surprising keys.
type cityReport struct {
	City         string `json:"city"`
	TemperatureC int    `json:"temperature_c"`
}
