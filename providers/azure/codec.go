package azure

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	responses "github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/stream"
	"github.com/looprig/inference/wire/sse"
)

const (
	eventResponseCompleted     = "response.completed"
	eventResponseFailed        = "response.failed"
	eventResponseIncomplete    = "response.incomplete"
	eventReasoningTextDelta    = "response.reasoning_text.delta"
	eventReasoningDelta        = "response.reasoning.delta"
	eventReasoningSummaryDelta = "response.reasoning_summary.delta"

	contentTypeReasoningText = "reasoning_text"
	incompleteReasonFilter   = "content_filter"
)

type responseMetadata struct {
	Status            string             `json:"status"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

type requestCodec struct {
	config config
}

var _ codec.StreamingCodec = requestCodec{}

func (c requestCodec) EncodeRequest(req inference.Request, mode codec.RequestMode) (codec.EncodedRequest, error) {
	encoded, err := (responses.Codec{}).EncodeRequest(req, mode)
	if err != nil {
		return codec.EncodedRequest{}, err
	}
	if !c.config.hasBodyOptions() {
		return encoded, nil
	}

	raw, err := io.ReadAll(encoded.Body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("azure: read encoded request: %w", err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("azure: decode encoded request: %w", err)
	}
	if c.config.reasoning != nil {
		body["reasoning"], err = json.Marshal(c.config.reasoning)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("azure: encode reasoning option: %w", err)
		}
	}
	if c.config.metadata != nil {
		body["metadata"], err = json.Marshal(c.config.metadata)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("azure: encode metadata option: %w", err)
		}
	}
	if c.config.promptCacheKey != "" {
		body["prompt_cache_key"], err = json.Marshal(c.config.promptCacheKey)
		if err != nil {
			return codec.EncodedRequest{}, fmt.Errorf("azure: encode prompt cache key option: %w", err)
		}
	}

	patched, err := json.Marshal(body)
	if err != nil {
		return codec.EncodedRequest{}, fmt.Errorf("azure: encode extended request: %w", err)
	}
	return codec.EncodedRequest{Header: encoded.Header.Clone(), Body: bytes.NewReader(patched)}, nil
}

func (requestCodec) DecodeResponse(body []byte) (*inference.Response, error) {
	return decodeResponse(body)
}

func (requestCodec) DecodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	return decodeStream(resp)
}

// decodeResponse adds the Azure Responses variants that the shared OpenAI
// Responses codec intentionally does not need to model: direct reasoning-text
// content and content-filter termination.
func decodeResponse(body []byte) (*inference.Response, error) {
	normalized, metadata, err := normalizeResponseBody(body)
	if err != nil {
		return nil, err
	}
	decoded, err := (responses.Codec{}).DecodeResponse(normalized)
	if err != nil {
		return nil, err
	}
	if decoded.FinishReason != stream.FinishReasonToolUse && metadata.Status == "incomplete" && metadata.IncompleteDetails != nil && metadata.IncompleteDetails.Reason == incompleteReasonFilter {
		decoded.FinishReason = stream.FinishReasonContentFilter
	}
	return decoded, nil
}

// normalizeResponseBody translates Azure's direct reasoning_text content into
// the Responses summary representation understood by the shared codec. The
// original fields remain otherwise untouched, so encrypted reasoning state and
// ordinary output items continue through the shared decoder unchanged.
func normalizeResponseBody(body []byte) ([]byte, responseMetadata, error) {
	var metadata responseMetadata
	if err := json.Unmarshal(body, &metadata); err != nil {
		return nil, responseMetadata{}, err
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, responseMetadata{}, err
	}
	rawOutput, ok := root["output"]
	if !ok || string(rawOutput) == "null" {
		return body, metadata, nil
	}
	var output []json.RawMessage
	if err := json.Unmarshal(rawOutput, &output); err != nil {
		return nil, responseMetadata{}, err
	}

	changed := false
	for index, rawItem := range output {
		var item map[string]json.RawMessage
		if err := json.Unmarshal(rawItem, &item); err != nil {
			return nil, responseMetadata{}, err
		}
		var itemType string
		if rawType, exists := item["type"]; exists {
			if err := json.Unmarshal(rawType, &itemType); err != nil {
				return nil, responseMetadata{}, err
			}
		}
		if itemType != "reasoning" {
			continue
		}

		var contentParts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if rawContent, exists := item["content"]; exists && string(rawContent) != "null" {
			if err := json.Unmarshal(rawContent, &contentParts); err != nil {
				return nil, responseMetadata{}, err
			}
		}

		var summary []json.RawMessage
		if rawSummary, exists := item["summary"]; exists && string(rawSummary) != "null" {
			if err := json.Unmarshal(rawSummary, &summary); err != nil {
				return nil, responseMetadata{}, err
			}
		}
		itemChanged := false
		for _, part := range contentParts {
			if part.Type != contentTypeReasoningText || part.Text == "" {
				continue
			}
			encoded, err := json.Marshal(map[string]string{"type": "summary_text", "text": part.Text})
			if err != nil {
				return nil, responseMetadata{}, err
			}
			summary = append(summary, encoded)
			itemChanged = true
			changed = true
		}
		if itemChanged {
			encoded, err := json.Marshal(summary)
			if err != nil {
				return nil, responseMetadata{}, err
			}
			item["summary"] = encoded
			output[index], err = json.Marshal(item)
			if err != nil {
				return nil, responseMetadata{}, err
			}
		}
	}
	if !changed {
		return body, metadata, nil
	}
	encodedOutput, err := json.Marshal(output)
	if err != nil {
		return nil, responseMetadata{}, err
	}
	root["output"] = encodedOutput
	normalized, err := json.Marshal(root)
	if err != nil {
		return nil, responseMetadata{}, err
	}
	return normalized, metadata, nil
}

type streamEvent struct {
	Type     string          `json:"type"`
	Delta    json.RawMessage `json:"delta"`
	Response json.RawMessage `json:"response"`
}

type streamCollector struct {
	terminalSeen bool
	result       stream.StreamResult
}

func decodeStream(resp *http.Response) (*stream.StreamReader[content.Chunk], error) {
	frames, err := sse.DecodeStreamFrames(resp.Body)
	if err != nil {
		return nil, err
	}
	collector := &streamCollector{}
	return stream.FramesToChunksWithResult(frames, collector.mapFrame, collector.resultValue), nil
}

func (c *streamCollector) mapFrame(frame stream.StreamFrame) ([]content.Chunk, error) {
	var event streamEvent
	if err := json.Unmarshal(frame.Data, &event); err != nil {
		return nil, nil
	}

	switch event.Type {
	case eventResponseFailed:
		var response struct {
			Error *struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(event.Response, &response); err != nil {
			return nil, nil
		}
		streamErr := &responses.StreamAPIError{}
		if response.Error != nil {
			streamErr.Code = response.Error.Code
			streamErr.Message = response.Error.Message
		}
		return nil, streamErr
	case eventResponseCompleted, eventResponseIncomplete:
		c.terminalSeen = true
		if len(event.Response) == 0 || string(event.Response) == "null" {
			return nil, nil
		}
		decoded, err := decodeResponse(event.Response)
		if err != nil {
			return nil, err
		}
		c.result = stream.StreamResult{
			Usage:        decoded.Usage,
			Model:        decoded.Model,
			FinishReason: decoded.FinishReason,
		}
		return nil, nil
	case eventReasoningTextDelta, eventReasoningDelta, eventReasoningSummaryDelta:
		if delta, ok := stringDelta(event.Delta); ok && delta != "" {
			return []content.Chunk{&content.ThinkingChunk{Thinking: delta}}, nil
		}
		return nil, nil
	default:
		return (responses.Codec{}).DecodeEvent(frame.Data)
	}
}

func (c *streamCollector) resultValue() (stream.StreamResult, bool, error) {
	if !c.terminalSeen {
		return stream.StreamResult{}, false, nil
	}
	return c.result, true, nil
}

func stringDelta(raw json.RawMessage) (string, bool) {
	var delta string
	if err := json.Unmarshal(raw, &delta); err == nil {
		return delta, true
	}
	var object struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return "", false
	}
	return object.Text, object.Text != ""
}
