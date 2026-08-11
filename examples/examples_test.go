package examples_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"
)

type docsManifest struct {
	SchemaVersion int `json:"schemaVersion"`
	Repository    string
	ProofSources  []struct {
		ID     string
		Type   string
		Path   string
		Symbol string
	} `json:"proofSources"`
	Examples []struct {
		ID             string
		SourcePath     string
		Availability   string
		OfflineCommand string
		WorkflowPath   string
		JobID          string `json:"jobId"`
		Assertion      string
		Cleanup        string
		LiveGate       any
		ProofIDs       []string `json:"proofIds"`
	} `json:"examples"`
}

func TestDocumentationManifestAndWorkflow(t *testing.T) {
	manifestBytes, err := os.ReadFile("../testdata/docs/examples.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest docsManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.SchemaVersion != 1 || manifest.Repository != "llm" {
		t.Fatalf("manifest identity = (%d, %q), want (1, llm)", manifest.SchemaVersion, manifest.Repository)
	}
	if len(manifest.Examples) != 2 {
		t.Fatalf("examples = %d, want 2", len(manifest.Examples))
	}
	workflow, err := os.ReadFile("../.github/workflows/docs-examples.yml")
	if err != nil {
		t.Fatal(err)
	}
	workflowText := string(workflow)
	for _, example := range manifest.Examples {
		if example.Availability != "source-workspace" || example.WorkflowPath != ".github/workflows/docs-examples.yml" || example.JobID != "docs-examples" {
			t.Fatalf("example %q has invalid source/workflow metadata", example.ID)
		}
		if example.Assertion == "" || example.Cleanup == "" || example.LiveGate != nil || len(example.ProofIDs) < 2 {
			t.Fatalf("example %q has incomplete verification metadata", example.ID)
		}
		if !strings.Contains(workflowText, "run: "+example.OfflineCommand) {
			t.Fatalf("workflow does not literally run %q", example.OfflineCommand)
		}
	}
	if !strings.Contains(workflowText, "GOWORK=off GOCACHE=/tmp/looprig-llm-docs-gocache go test -race ./...") {
		t.Fatal("workflow does not run the full standalone race suite")
	}
}

func TestRunnableDocumentationExamples(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		path string
		want string
	}{
		{
			name: "automatic provider and counter selection",
			path: "./auto",
			want: strings.Join([]string{
				"provider=openai client-ready=true",
				"counter-provider=openai exact=true separate-endpoint=true",
				"xai-exact-counter=false",
				"",
			}, "\n"),
		},
		{
			name: "credential source lease",
			path: "./credentials",
			want: strings.Join([]string{
				"acquires=2 rotated-authorizations=2",
				"provider-response=ok",
				"",
			}, "\n"),
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			command := exec.Command("go", "run", test.path)
			command.Env = append(command.Environ(), "GOWORK=off")
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("go run %s: %v\n%s", test.path, err, output)
			}
			if got := string(output); got != test.want {
				t.Fatalf("go run %s output = %q, want %q", test.path, got, test.want)
			}
		})
	}
}
