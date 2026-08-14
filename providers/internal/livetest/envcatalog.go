//go:build live

package livetest

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"

	model "github.com/looprig/inference/model"
)

// envFileEnv overrides the dotenv location for a developer whose keys live
// somewhere other than the repository root.
const envFileEnv = "LOOPRIG_LIVE_ENV"

// dotenvName is the gitignored, mode-0600 file the repository root carries. It
// is READ at run time and never copied: no value from it is written to any
// tracked file, and every value it yields is registered with scrub before a
// probe can log anything.
const dotenvName = ".env"

// envRow describes one live target that the carbon catalogue does not contain
// but the repository's own dotenv can authenticate. The carbon catalogue is the
// developer's runtime configuration, so it holds only the models they actually
// run; the FIRST-PARTY origin APIs — api.anthropic.com and Google's
// generativelanguage endpoint — are exactly the endpoints a conformance probe
// most wants and a day-to-day catalogue least needs. Declaring them here keeps
// the two concerns apart: the catalogue stays the user's, and the probe suite
// names the origins it wants to be measured against.
type envRow struct {
	alias     string
	provider  string
	apiFormat model.APIFormat
	baseURL   string
	name      string
	// keyVar is the dotenv variable holding this endpoint's credential.
	keyVar string

	window    int
	maxOutput int

	tools    bool
	thinking bool
	images   bool
	// thinkingDialect is the reasoning request shape this model accepts. It is
	// a LIVE-MEASURED, per-model fact (TestLiveAnthropicThinkingDialect is what
	// measures it), recorded beside the endpoint rather than inferred from the
	// model id in the codec.
	thinkingDialect model.ThinkingDialect

	efforts       []string
	defaultEffort string
}

// envTargets is the fixed set of endpoints this suite wants that the catalogue
// does not carry. Model ids are pinned rather than derived: `gemini-2.5-flash`
// is retired for new projects, so the catalogue's own Gemini rows are dead and
// `gemini-flash-latest` is the reachable alias.
var envTargets = []envRow{
	{
		alias: "anthropic-haiku-4.5", provider: "anthropic", apiFormat: model.APIFormatAnthropic,
		baseURL: "https://api.anthropic.com/v1", name: "claude-haiku-4-5-20251001", keyVar: "ANTHROPIC_API_KEY",
		window: 200000, maxOutput: 64000,
		tools: true, thinking: true, images: true,
		// Measured 2026-08-13: api.anthropic.com answers
		// thinking:{"type":"adaptive"} on this model with HTTP 400
		// "adaptive thinking is not supported on this model".
		thinkingDialect: model.ThinkingDialectBudget,
		efforts:         []string{"none", "low", "medium", "high"}, defaultEffort: "medium",
	},
	{
		alias: "anthropic-sonnet-5", provider: "anthropic", apiFormat: model.APIFormatAnthropic,
		baseURL: "https://api.anthropic.com/v1", name: "claude-sonnet-5", keyVar: "ANTHROPIC_API_KEY",
		window: 200000, maxOutput: 64000,
		tools: true, thinking: true, images: true,
		// Measured 2026-08-13: api.anthropic.com answers
		// thinking:{"type":"enabled","budget_tokens":N} on this model with HTTP
		// 400 `"thinking.type.enabled" is not supported for this model. Use
		// "thinking.type.adaptive" and "output_config.effort"`.
		thinkingDialect: model.ThinkingDialectAdaptive,
		efforts:         []string{"none", "low", "medium", "high"}, defaultEffort: "medium",
	},
	{
		alias: "gemini-flash", provider: "google", apiFormat: model.APIFormatGemini,
		baseURL: "https://generativelanguage.googleapis.com/v1beta", name: "gemini-flash-latest", keyVar: "GEMINI_API_KEY",
		window: 1000000, maxOutput: 8192,
		tools: true, thinking: true, images: true,
		efforts: []string{"none", "low", "medium", "high"}, defaultEffort: "low",
	},
	{
		// The lite tier is the one that can actually carry a matrix: the free
		// quota on gemini-flash-latest is 20 requests PER DAY per model, which
		// a full use-case sweep exhausts in one run. Both are declared because
		// they are different models and a capability answer from one does not
		// transfer to the other.
		alias: "gemini-flash-lite", provider: "google", apiFormat: model.APIFormatGemini,
		baseURL: "https://generativelanguage.googleapis.com/v1beta", name: "gemini-flash-lite-latest", keyVar: "GEMINI_API_KEY",
		window: 1000000, maxOutput: 8192,
		tools: true, thinking: true, images: true,
		efforts: []string{"none", "low", "medium", "high"}, defaultEffort: "low",
	},
	{
		alias: "openrouter-gemma-4-free", provider: "openrouter", apiFormat: model.APIFormatOpenAI,
		baseURL: "https://openrouter.ai/api/v1", name: "google/gemma-4-26b-a4b-it:free", keyVar: "OPENROUTER_API_KEY",
		window: 32768, maxOutput: 4096,
		tools: true, images: true,
	},
	{
		alias: "openrouter-gpt-oss-20b-free", provider: "openrouter", apiFormat: model.APIFormatOpenAI,
		baseURL: "https://openrouter.ai/api/v1", name: "openai/gpt-oss-20b:free", keyVar: "OPENROUTER_API_KEY",
		window: 32768, maxOutput: 4096,
		tools: true, thinking: true,
		efforts: []string{"none", "low", "medium", "high"}, defaultEffort: "low",
	},
	{
		alias: "openrouter-nemotron-free", provider: "openrouter", apiFormat: model.APIFormatOpenAI,
		baseURL: "https://openrouter.ai/api/v1", name: "nvidia/nemotron-3-ultra-550b-a55b:free", keyVar: "OPENROUTER_API_KEY",
		window: 32768, maxOutput: 4096,
		tools: true,
	},
}

// entryFor materializes one declared target as a catalogue row, given the
// resolved credential. Shape-compatible with a carbon row on purpose: every
// probe then reads one row type and cannot tell (or care) which source it came
// from.
func (r envRow) entryFor(key string) catalogEntry {
	row := catalogEntry{
		Alias:         r.alias,
		Provider:      r.provider,
		APIFormat:     string(r.apiFormat),
		BaseURL:       r.baseURL,
		Model:         r.name,
		APIKey:        key,
		Efforts:       r.efforts,
		DefaultEffort: r.defaultEffort,
	}
	row.ContextLimits.WindowTokens = r.window
	row.ContextLimits.MaxInputTokens = r.window
	row.ContextLimits.MaxOutputTokens = r.maxOutput
	row.Capabilities.Tools = r.tools
	row.Capabilities.Thinking = r.thinking
	row.Capabilities.ThinkingDialect = string(r.thinkingDialect)
	row.Capabilities.Images = r.images
	return row
}

// envRows resolves every declared target whose credential is present. A target
// with no credential is simply absent, so entry() reports it as a missing row
// and the probe skips — the same no-op path as a missing catalogue.
func envRows() []catalogEntry {
	values := loadDotenv()
	rows := make([]catalogEntry, 0, len(envTargets))
	for _, target := range envTargets {
		key := values[target.keyVar]
		if key == "" {
			key = os.Getenv(target.keyVar)
		}
		if key == "" {
			continue
		}
		rows = append(rows, target.entryFor(key))
	}
	return rows
}

// loadDotenv reads KEY=VALUE pairs from the repository dotenv. A missing file is
// not an error: the process environment is then the only source, which is how
// this runs anywhere but the author's machine.
func loadDotenv() map[string]string {
	values := map[string]string{}
	path := os.Getenv(envFileEnv)
	if path == "" {
		found, ok := findUpwards(dotenvName)
		if !ok {
			return values
		}
		path = found
	}
	file, err := os.Open(path) //#nosec G304 -- developer-supplied local dotenv path, live-tagged test only
	if err != nil {
		return values
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key, value, ok := parseDotenvLine(scanner.Text())
		if !ok {
			continue
		}
		values[key] = value
	}
	return values
}

// parseDotenvLine splits one dotenv line, tolerating `export ` prefixes, blank
// lines, comments and quoted values.
func parseDotenvLine(line string) (string, string, bool) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || strings.HasPrefix(trimmed, "#") {
		return "", "", false
	}
	trimmed = strings.TrimPrefix(trimmed, "export ")
	key, value, found := strings.Cut(trimmed, "=")
	if !found {
		return "", "", false
	}
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	if len(value) >= 2 {
		if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
			value = value[1 : len(value)-1]
		}
	}
	if key == "" || value == "" {
		return "", "", false
	}
	return key, value, true
}

// findUpwards walks from the working directory towards the filesystem root
// looking for name. The test binary's working directory is the package
// directory, so the repository root is several levels up.
func findUpwards(name string) (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		candidate := filepath.Join(dir, name)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}
