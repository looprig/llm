package vertex_test

import (
	"errors"
	"testing"

	"github.com/looprig/inference/auth"
	model "github.com/looprig/inference/model"

	"github.com/looprig/llm"
	vertex "github.com/looprig/llm/providers/google-vertex"
)

func TestRejectsLocationThatCouldRewriteTheVertexHost(t *testing.T) {
	selected := model.CustomModel(model.ProviderName(llm.ProviderGoogleVertex), model.APIFormatGemini, "", "gemini-2.5-flash")
	_, err := vertex.New(selected, auth.APIKey("access-token"), vertex.WithProject("project"), vertex.WithLocation("us-central1/attacker.example"))
	var configErr *vertex.ConfigurationError
	if !errors.As(err, &configErr) || configErr.Reason != vertex.ProjectOrLocationInvalid {
		t.Fatalf("New() error = %T %v, want invalid project/location configuration", err, err)
	}
}
