//go:build live

package livetest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"
)

// catalogEnv overrides the catalogue location for a developer whose carbon
// config lives elsewhere. The default is the carbon config path.
const catalogEnv = "LOOPRIG_LIVE_MODELS"

// defaultCatalogPath is relative to the user's home directory. The catalogue is
// the user's own configuration and lives OUTSIDE every repository on purpose:
// no key value may ever be copied into the workspace.
var defaultCatalogPath = filepath.Join(".looprig", "carbon", "models.json")

// catalogEntry is the subset of a carbon model row these probes need. Fields the
// probes do not use are ignored rather than modelled, so a catalogue schema
// change cannot break the loader.
type catalogEntry struct {
	Alias     string `json:"alias"`
	Provider  string `json:"provider"`
	APIFormat string `json:"api_format"`
	BaseURL   string `json:"base_url"`
	Model     string `json:"model"`
	APIKey    string `json:"api_key"`

	ContextLimits struct {
		WindowTokens    int `json:"window_tokens"`
		MaxInputTokens  int `json:"max_input_tokens"`
		MaxOutputTokens int `json:"max_output_tokens"`
	} `json:"context_limits"`

	Capabilities struct {
		Tools    bool `json:"tools"`
		Thinking bool `json:"thinking"`
		Images   bool `json:"images"`
		// ThinkingDialect names which reasoning request shape the model takes
		// ("adaptive" or "budget"). Absent means undeclared, and the Anthropic
		// encoder fails closed rather than guessing — see
		// model.ThinkingDialect.
		ThinkingDialect string `json:"thinking_dialect"`
	} `json:"capabilities"`

	Efforts       []string `json:"efforts"`
	DefaultEffort string   `json:"default_effort"`
}

type catalog struct {
	Models []catalogEntry `json:"models"`
}

var (
	loadOnce   sync.Once
	loaded     *catalog
	loadReason string
)

// loadCatalog reads the developer's carbon catalogue once per test binary, then
// layers on the dotenv-authenticated targets the catalogue does not carry (see
// envcatalog.go). A missing or unreadable file is a skip reason for the
// catalogue's own rows, never a failure: these probes are opt-in by possession
// of credentials. The dotenv rows are still merged in that case, because the
// first-party origin endpoints are authenticated independently of the carbon
// catalogue and are the probes most worth running.
func loadCatalog(t *testing.T) *catalog {
	t.Helper()
	loadOnce.Do(func() {
		var parsed catalog
		path := os.Getenv(catalogEnv)
		if path == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				loadReason = "cannot resolve home directory: " + err.Error()
				return
			}
			path = filepath.Join(home, defaultCatalogPath)
		}
		if raw, err := os.ReadFile(path); err != nil { //#nosec G304 -- developer-supplied local catalogue path, live-tagged test only
			loadReason = "no live model catalogue (" + path + "): " + err.Error()
		} else if err := json.Unmarshal(raw, &parsed); err != nil {
			loadReason = "live model catalogue is not valid JSON: " + err.Error()
			parsed = catalog{}
		}
		parsed.Models = append(parsed.Models, envRows()...)
		if len(parsed.Models) == 0 {
			if loadReason == "" {
				loadReason = "no catalogue rows and no dotenv credentials"
			}
			return
		}
		loaded = &parsed
		registerSecrets(&parsed)
	})
	if loaded == nil {
		t.Skip("livetest: " + loadReason)
	}
	return loaded
}

// entry returns the catalogue row for alias, skipping when the row or its key is
// absent. A row whose provider needs no key (a local LM Studio endpoint) is
// allowed through with an empty key; every other row must carry one.
func entry(t *testing.T, alias string) catalogEntry {
	t.Helper()
	cat := loadCatalog(t)
	for _, row := range cat.Models {
		if row.Alias != alias {
			continue
		}
		if row.Model == "" {
			t.Skipf("livetest: catalogue row %q has no model id", alias)
		}
		if row.APIKey == "" && row.Provider != "lmstudio" {
			t.Skipf("livetest: catalogue row %q has no API key", alias)
		}
		return row
	}
	t.Skipf("livetest: catalogue has no row %q", alias)
	return catalogEntry{}
}

// key returns the row's credential. It is returned as auth.APIKey and must never
// be printed; scrub redacts it from anything these probes emit.
func (e catalogEntry) key() auth.APIKey { return auth.APIKey(e.APIKey) }

// selectedModel builds the neutral model descriptor from the catalogue row,
// overriding the base URL with the loopback recorder's origin. Capabilities come
// from the catalogue because they are exactly the local gating data a real
// caller would carry; Thinking in particular selects max_completion_tokens over
// max_tokens in the Chat encoder and enables the Anthropic thinking block.
func (e catalogEntry) selectedModel(baseURL string, opts ...model.ModelOption) model.Model {
	base := []model.ModelOption{
		model.WithContextLimits(model.ContextLimits{
			WindowTokens:    tokenCount(e.ContextLimits.WindowTokens),
			MaxInputTokens:  tokenCount(e.ContextLimits.MaxInputTokens),
			MaxOutputTokens: tokenCount(e.ContextLimits.MaxOutputTokens),
		}),
	}
	if e.Capabilities.Tools {
		base = append(base, model.WithTools())
	}
	if e.Capabilities.Thinking {
		base = append(base, model.WithThinkingDialect(model.ThinkingDialect(e.Capabilities.ThinkingDialect)))
	}
	if e.Capabilities.Images {
		base = append(base, model.WithImages())
	}
	base = append(base, opts...)
	return model.CustomModel(
		model.ProviderName(e.Provider),
		model.APIFormat(e.APIFormat),
		baseURL,
		e.Model,
		base...,
	)
}

// supportsEffort reports whether the catalogue row advertises the named effort,
// so a thinking probe can skip rather than send a level the gateway rejects for
// reasons that have nothing to do with our encoder.
func (e catalogEntry) supportsEffort(effort string) bool {
	for _, candidate := range e.Efforts {
		if strings.EqualFold(candidate, effort) {
			return true
		}
	}
	return false
}
