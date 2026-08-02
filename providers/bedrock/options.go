package bedrock

import (
	"encoding/json"
	"fmt"
)

// ReasoningOptions controls the model-specific reasoning object carried inside
// Bedrock's additionalModelRequestFields. The native Converse envelope keeps
// this field model-specific; these fields match the documented Anthropic
// thinking shape used by Claude models on Bedrock.
type ReasoningOptions struct {
	Type         string
	BudgetTokens *int
}

// ServiceTier selects Bedrock's documented request service tier.
type ServiceTier string

const (
	ServiceTierDefault  ServiceTier = "default"
	ServiceTierPriority ServiceTier = "priority"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierReserved ServiceTier = "reserved"
)

// PerformanceLatency selects Bedrock's performanceConfig latency mode.
type PerformanceLatency string

const (
	PerformanceLatencyStandard  PerformanceLatency = "standard"
	PerformanceLatencyOptimized PerformanceLatency = "optimized"
)

// GuardrailOptions configures the native Converse guardrailConfig object.
type GuardrailOptions struct {
	Identifier           string
	Version              string
	Trace                string
	StreamProcessingMode string
}

// CachePointOptions selects a native Converse cachePoint content block.
type CachePointOptions struct {
	Type string
	TTL  string
}

const (
	CachePointTTL5m = "5m"
	CachePointTTL1h = "1h"
)

type config struct {
	reasoning                    *ReasoningOptions
	additionalModelRequestFields json.RawMessage
	additionalResponseFieldPaths []string
	guardrail                    *GuardrailOptions
	performanceLatency           PerformanceLatency
	serviceTier                  ServiceTier
	requestMetadata              map[string]string
	cachePoint                   *CachePointOptions
}

// Option customizes a Bedrock client. Options affect native Converse requests;
// the existing Anthropic-on-Bedrock InvokeModel path remains byte-compatible.
type Option func(*config)

// WithReasoning configures native model reasoning controls.
func WithReasoning(options ReasoningOptions) Option {
	return func(c *config) {
		copy := options
		if options.BudgetTokens != nil {
			budget := *options.BudgetTokens
			copy.BudgetTokens = &budget
		}
		c.reasoning = &copy
	}
}

// WithAdditionalModelRequestFields adds documented provider-specific Converse
// request fields. The value must be a JSON object; it is merged with the
// reasoning object when WithReasoning is also present.
func WithAdditionalModelRequestFields(fields json.RawMessage) Option {
	return func(c *config) {
		c.additionalModelRequestFields = append(json.RawMessage(nil), fields...)
	}
}

// WithAdditionalModelResponseFieldPaths asks Bedrock for documented additional
// response fields.
func WithAdditionalModelResponseFieldPaths(paths ...string) Option {
	return func(c *config) { c.additionalResponseFieldPaths = append([]string(nil), paths...) }
}

// WithGuardrail configures the native Converse guardrail request object.
func WithGuardrail(options GuardrailOptions) Option {
	return func(c *config) {
		copy := options
		c.guardrail = &copy
	}
}

// WithPerformanceLatency sets the native performanceConfig latency value.
func WithPerformanceLatency(latency PerformanceLatency) Option {
	return func(c *config) { c.performanceLatency = latency }
}

// WithServiceTier sets the native serviceTier type.
func WithServiceTier(tier ServiceTier) Option {
	return func(c *config) { c.serviceTier = tier }
}

// WithRequestMetadata attaches the native requestMetadata map. The map is
// copied when the option is applied.
func WithRequestMetadata(metadata map[string]string) Option {
	return func(c *config) {
		if metadata == nil {
			c.requestMetadata = nil
			return
		}
		c.requestMetadata = make(map[string]string, len(metadata))
		for key, value := range metadata {
			c.requestMetadata[key] = value
		}
	}
}

// WithPromptCachePoint adds a native cachePoint block at the end of the system
// content when present, or at the end of the final message otherwise.
func WithPromptCachePoint(options CachePointOptions) Option {
	return func(c *config) {
		copy := options
		c.cachePoint = &copy
	}
}

func (c config) clone() config {
	clone := c
	if c.reasoning != nil {
		reasoning := *c.reasoning
		if c.reasoning.BudgetTokens != nil {
			budget := *c.reasoning.BudgetTokens
			reasoning.BudgetTokens = &budget
		}
		clone.reasoning = &reasoning
	}
	clone.additionalModelRequestFields = append(json.RawMessage(nil), c.additionalModelRequestFields...)
	clone.additionalResponseFieldPaths = append([]string(nil), c.additionalResponseFieldPaths...)
	if c.guardrail != nil {
		guardrail := *c.guardrail
		clone.guardrail = &guardrail
	}
	if c.requestMetadata != nil {
		clone.requestMetadata = make(map[string]string, len(c.requestMetadata))
		for key, value := range c.requestMetadata {
			clone.requestMetadata[key] = value
		}
	}
	if c.cachePoint != nil {
		cachePoint := *c.cachePoint
		clone.cachePoint = &cachePoint
	}
	return clone
}

func (c config) hasConverseOptions() bool {
	return c.reasoning != nil || len(c.additionalModelRequestFields) > 0 || len(c.additionalResponseFieldPaths) > 0 || c.guardrail != nil || c.performanceLatency != "" || c.serviceTier != "" || c.requestMetadata != nil || c.cachePoint != nil
}

func (c config) applyConverse(body []byte, streaming bool) ([]byte, error) {
	if !c.hasConverseOptions() {
		return body, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, &OptionError{Reason: "decode Converse request", Err: err}
	}
	if fields == nil {
		return nil, &OptionError{Reason: "Converse request is not an object"}
	}

	if err := c.applyAdditionalModelRequestFields(fields); err != nil {
		return nil, err
	}
	if len(c.additionalResponseFieldPaths) > 0 {
		encoded, err := json.Marshal(c.additionalResponseFieldPaths)
		if err != nil {
			return nil, &OptionError{Reason: "encode additionalModelResponseFieldPaths", Err: err}
		}
		fields["additionalModelResponseFieldPaths"] = encoded
	}
	if c.guardrail != nil {
		guardrail := map[string]json.RawMessage{}
		if c.guardrail.Identifier != "" {
			guardrail["guardrailIdentifier"], _ = json.Marshal(c.guardrail.Identifier)
		}
		if c.guardrail.Version != "" {
			guardrail["guardrailVersion"], _ = json.Marshal(c.guardrail.Version)
		}
		if c.guardrail.Trace != "" {
			guardrail["trace"], _ = json.Marshal(c.guardrail.Trace)
		}
		if streaming && c.guardrail.StreamProcessingMode != "" {
			guardrail["streamProcessingMode"], _ = json.Marshal(c.guardrail.StreamProcessingMode)
		}
		encoded, err := json.Marshal(guardrail)
		if err != nil {
			return nil, &OptionError{Reason: "encode guardrailConfig", Err: err}
		}
		fields["guardrailConfig"] = encoded
	}
	if c.performanceLatency != "" {
		encoded, err := json.Marshal(map[string]string{"latency": string(c.performanceLatency)})
		if err != nil {
			return nil, &OptionError{Reason: "encode performanceConfig", Err: err}
		}
		fields["performanceConfig"] = encoded
	}
	if c.serviceTier != "" {
		encoded, err := json.Marshal(map[string]string{"type": string(c.serviceTier)})
		if err != nil {
			return nil, &OptionError{Reason: "encode serviceTier", Err: err}
		}
		fields["serviceTier"] = encoded
	}
	if c.requestMetadata != nil {
		encoded, err := json.Marshal(c.requestMetadata)
		if err != nil {
			return nil, &OptionError{Reason: "encode requestMetadata", Err: err}
		}
		fields["requestMetadata"] = encoded
	}
	if c.cachePoint != nil {
		if err := applyCachePoint(fields, *c.cachePoint); err != nil {
			return nil, err
		}
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		return nil, &OptionError{Reason: "encode Converse request", Err: err}
	}
	return encoded, nil
}

func (c config) applyConverseCountTokens(body []byte) ([]byte, error) {
	if c.reasoning == nil && len(c.additionalModelRequestFields) == 0 && c.cachePoint == nil {
		return body, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, &OptionError{Reason: "decode CountTokens request", Err: err}
	}
	if fields == nil {
		return nil, &OptionError{Reason: "CountTokens request is not an object"}
	}
	if err := c.applyAdditionalModelRequestFields(fields); err != nil {
		return nil, err
	}
	if c.cachePoint != nil {
		if err := applyCachePoint(fields, *c.cachePoint); err != nil {
			return nil, err
		}
	}
	encoded, err := json.Marshal(fields)
	if err != nil {
		return nil, &OptionError{Reason: "encode CountTokens request", Err: err}
	}
	return encoded, nil
}

func (c config) applyAdditionalModelRequestFields(fields map[string]json.RawMessage) error {
	if c.reasoning == nil && len(c.additionalModelRequestFields) == 0 {
		return nil
	}
	additional := make(map[string]json.RawMessage)
	if len(c.additionalModelRequestFields) > 0 {
		if err := json.Unmarshal(c.additionalModelRequestFields, &additional); err != nil || additional == nil {
			return &OptionError{Reason: "additionalModelRequestFields must be a JSON object", Err: err}
		}
	}
	if c.reasoning != nil {
		thinking := make(map[string]json.RawMessage)
		thinkingType := c.reasoning.Type
		if thinkingType == "" {
			thinkingType = "enabled"
		}
		thinking["type"], _ = json.Marshal(thinkingType)
		if c.reasoning.BudgetTokens != nil {
			thinking["budget_tokens"], _ = json.Marshal(*c.reasoning.BudgetTokens)
		}
		encodedThinking, err := json.Marshal(thinking)
		if err != nil {
			return &OptionError{Reason: "encode reasoning option", Err: err}
		}
		additional["thinking"] = encodedThinking
	}
	encodedAdditional, err := json.Marshal(additional)
	if err != nil {
		return &OptionError{Reason: "encode additionalModelRequestFields", Err: err}
	}
	fields["additionalModelRequestFields"] = encodedAdditional
	return nil
}

func applyCachePoint(fields map[string]json.RawMessage, options CachePointOptions) error {
	typ := options.Type
	if typ == "" {
		typ = "default"
	}
	cachePointFields := map[string]string{"type": typ}
	if options.TTL != "" {
		cachePointFields["ttl"] = options.TTL
	}
	cachePoint, err := json.Marshal(cachePointFields)
	if err != nil {
		return &OptionError{Reason: "encode cachePoint", Err: err}
	}
	block, err := json.Marshal(map[string]json.RawMessage{"cachePoint": cachePoint})
	if err != nil {
		return &OptionError{Reason: "encode cachePoint block", Err: err}
	}
	if rawSystem, ok := fields["system"]; ok {
		var system []json.RawMessage
		if err := json.Unmarshal(rawSystem, &system); err != nil {
			return &OptionError{Reason: "decode system for cachePoint", Err: err}
		}
		system = append(system, block)
		fields["system"], err = json.Marshal(system)
		if err != nil {
			return &OptionError{Reason: "encode system cachePoint", Err: err}
		}
		return nil
	}
	rawMessages, ok := fields["messages"]
	if !ok {
		return &OptionError{Reason: "cachePoint requires system or messages"}
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil || len(messages) == 0 {
		return &OptionError{Reason: "cachePoint requires a non-empty messages array", Err: err}
	}
	var content []json.RawMessage
	if err := json.Unmarshal(messages[len(messages)-1]["content"], &content); err != nil {
		return &OptionError{Reason: "decode final message for cachePoint", Err: err}
	}
	content = append(content, block)
	messages[len(messages)-1]["content"], err = json.Marshal(content)
	if err != nil {
		return &OptionError{Reason: "encode final message cachePoint", Err: err}
	}
	fields["messages"], err = json.Marshal(messages)
	if err != nil {
		return &OptionError{Reason: "encode messages cachePoint", Err: err}
	}
	return nil
}

// OptionError reports a local provider-option encoding failure without retaining
// request bytes or credential material.
type OptionError struct {
	Reason string
	Err    error
}

func (e *OptionError) Error() string {
	if e.Err != nil {
		return "bedrock: " + e.Reason + ": " + e.Err.Error()
	}
	return fmt.Sprintf("bedrock: %s", e.Reason)
}

func (e *OptionError) Unwrap() error { return e.Err }
