package bedrock

import "encoding/json"

// Bedrock InvokeModel body constants for Anthropic-on-Bedrock.
const (
	// fieldModel is the top-level key the Anthropic Messages body carries the model
	// id in. Bedrock takes the model id in the request URL path instead, so this key
	// is removed from the body.
	fieldModel = "model"
	// fieldAnthropicVersion is the key Bedrock requires in place of "model" to select
	// the Anthropic wire contract.
	fieldAnthropicVersion = "anthropic_version"
	// anthropicVersionBedrock is the pinned Bedrock Anthropic wire-contract version.
	anthropicVersionBedrock = "bedrock-2023-05-31"

	// fieldMessages, blockTypeImage, and imageSourceBase64 name the parts of the
	// Anthropic body the image-source guard reads. imageSourceBase64 is the only
	// source form Bedrock accepts, so the guard is an allowlist: anything else
	// (today Anthropic's "url") is rejected rather than passed through.
	fieldMessages     = "messages"
	blockTypeImage    = "image"
	imageSourceBase64 = "base64"
)

// scannedBlock is the read-only projection of an Anthropic content block that the
// image-source guard needs: the block discriminator, the image `source`
// discriminator, and the nested `content` a tool_result block carries (which can
// itself hold image blocks). Everything else is deliberately absent — this shape
// is never re-encoded, so it cannot perturb the byte-identical rewrite.
type scannedBlock struct {
	Type   string `json:"type"`
	Source *struct {
		Type string `json:"type"`
	} `json:"source"`
	Content json.RawMessage `json:"content"`
}

// toBedrockBody rewrites an Anthropic Messages request body into the Bedrock
// InvokeModel body for Anthropic models: it removes the top-level "model" field
// (Bedrock takes the model id in the URL path, not the body) and adds
// "anthropic_version":"bedrock-2023-05-31". The transform is a decode/rewrite/
// re-encode over the JSON object at the serialization boundary — using
// map[string]json.RawMessage keeps every other field byte-identical (no lossy
// round-trip through a typed struct that could drop a field the codec added), and
// leaves domain typing to the codec that produced the input.
func toBedrockBody(anthropicBody []byte) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(anthropicBody, &fields); err != nil {
		return nil, &BodyTransformError{Err: err}
	}

	if err := checkImageSources(fields[fieldMessages]); err != nil {
		return nil, err
	}

	delete(fields, fieldModel)

	version, err := json.Marshal(anthropicVersionBedrock)
	if err != nil {
		return nil, &BodyTransformError{Err: err}
	}
	fields[fieldAnthropicVersion] = version

	out, err := json.Marshal(fields)
	if err != nil {
		return nil, &BodyTransformError{Err: err}
	}
	return out, nil
}

// checkImageSources rejects any image block whose source Bedrock cannot fetch.
// The Anthropic encoder this body comes from emits a remote {"type":"url"} source
// verbatim because Anthropic's own API accepts one; Bedrock does not, so without
// this guard the URL reaches InvokeModel and comes back as an opaque HTTP 400.
// The check lives here rather than in invokeAnthropic because toBedrockBody is the
// single chokepoint every Anthropic-dialect InvokeModel body passes through —
// Invoke and the CountTokens counter both call it, so one guard covers both.
//
// The scan is read-only and tolerant: a `messages` value that is absent or not the
// codec's array-of-blocks shape is left alone rather than rejected, so the guard
// can only ever refuse a body it positively recognizes as carrying a bad source.
func checkImageSources(messages json.RawMessage) error {
	blocks, ok := decodeBlocks(messages)
	if !ok {
		return nil
	}
	for _, message := range blocks {
		if err := checkBlockImageSources(message.Content); err != nil {
			return err
		}
	}
	return nil
}

// checkBlockImageSources walks one `content` array, recursing into the nested
// content of a tool_result block (a tool result can return an image).
func checkBlockImageSources(content json.RawMessage) error {
	blocks, ok := decodeBlocks(content)
	if !ok {
		return nil
	}
	for _, block := range blocks {
		if block.Type == blockTypeImage && block.Source != nil && block.Source.Type != imageSourceBase64 {
			return &UnsupportedImageSourceError{SourceType: block.Source.Type}
		}
		if err := checkBlockImageSources(block.Content); err != nil {
			return err
		}
	}
	return nil
}

// decodeBlocks projects a raw `messages`/`content` array into scannedBlocks. It
// reports false (not an error) for an absent or differently shaped value, keeping
// the guard from turning an unrecognized-but-valid body into a local failure.
func decodeBlocks(raw json.RawMessage) ([]scannedBlock, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	var blocks []scannedBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, false
	}
	return blocks, true
}
