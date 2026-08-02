// Package snowflake provides Snowflake Cortex's documented OpenAI-compatible
// Chat Completions endpoint.
package snowflake

import (
	"encoding/json"
	"net/url"
	"os"
	"strings"

	"github.com/looprig/inference"
	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	"github.com/looprig/llm/providers/internal/simple"
)

const accountEnvironment = "SNOWFLAKE_ACCOUNT"

type Option func(*config)

type config struct {
	account string
	options []simple.Option
}

// WithAccount sets the Snowflake account identifier used when Model.BaseURL is
// empty. It accepts the account locator/org form used in the Cortex URL.
func WithAccount(account string) Option {
	return func(c *config) { c.account = strings.TrimSpace(account) }
}

func WithHeader(name, value string) Option {
	return func(c *config) { c.options = append(c.options, simple.WithHeader(name, value)) }
}

func WithReasoningEnabled(enabled bool) Option {
	return func(c *config) { c.options = append(c.options, simple.WithReasoningEnabled(enabled)) }
}

func WithServiceTier(value string) Option {
	return func(c *config) { c.options = append(c.options, simple.WithServiceTier(value)) }
}

// New constructs a Snowflake Cortex client. Snowflake's Chat Completions API
// names the output limit max_completion_tokens; the adapter translates the
// shared OpenAI request field without changing the public inference contract.
func New(selected model.Model, key auth.APIKey, options ...Option) (inference.Client, error) {
	if key == "" {
		key = auth.APIKey(strings.TrimSpace(os.Getenv("SNOWFLAKE_CORTEX_TOKEN")))
		if key == "" {
			key = auth.APIKey(strings.TrimSpace(os.Getenv("SNOWFLAKE_CORTEX_PAT")))
		}
	}
	var cfg config
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	baseURL := strings.TrimRight(strings.TrimSpace(selected.BaseURL), "/")
	if baseURL == "" {
		account := cfg.account
		if account == "" {
			account = strings.TrimSpace(os.Getenv(accountEnvironment))
		}
		if account == "" {
			return nil, &ConfigurationError{Reason: AccountMissing}
		}
		if !validAccount(account) {
			return nil, &ConfigurationError{Reason: AccountInvalid}
		}
		baseURL = (&url.URL{
			Scheme: "https",
			Host:   account + ".snowflakecomputing.com",
			Path:   "/api/v2/cortex/v1",
		}).String()
	}
	selected.BaseURL = baseURL
	defaults := []simple.Option{
		simple.WithBodyPatch(func(body map[string]json.RawMessage) error {
			if value, ok := body["max_tokens"]; ok {
				if _, alreadySet := body["max_completion_tokens"]; !alreadySet {
					body["max_completion_tokens"] = append(json.RawMessage(nil), value...)
				}
				delete(body, "max_tokens")
			}
			return nil
		}),
	}
	defaults = append(defaults, cfg.options...)
	return simple.New(selected, key, simple.Definition{
		Provider:       llm.ProviderSnowflakeCortex,
		DefaultBaseURL: baseURL,
		DefaultPath:    "/chat/completions",
		Authentication: auth.AuthAPIKey,
	}, defaults...)
}

func validAccount(account string) bool {
	if len(account) == 0 || len(account) > 255 || account[0] == '.' || account[0] == '-' || account[0] == '_' {
		return false
	}
	last := account[len(account)-1]
	if last == '.' || last == '-' || last == '_' {
		return false
	}
	previousDot := false
	for _, ch := range account {
		switch {
		case ch >= 'a' && ch <= 'z', ch >= 'A' && ch <= 'Z', ch >= '0' && ch <= '9', ch == '-', ch == '_':
			previousDot = false
		case ch == '.':
			if previousDot {
				return false
			}
			previousDot = true
		default:
			return false
		}
	}
	return true
}
