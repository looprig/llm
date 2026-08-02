package azure

import (
	"errors"
	"testing"
)

func TestResolveBaseURLPrecedence(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "environment-resource")

	tests := []struct {
		name       string
		baseURL    string
		resource   string
		want       string
		wantErr    bool
		wantReason ResourceConfigurationReason
	}{
		{name: "explicit base URL wins", baseURL: "https://proxy.example.test/openai/v1", resource: "option-resource", want: "https://proxy.example.test/openai/v1"},
		{name: "option wins over environment", resource: "option-resource", want: "https://option-resource.openai.azure.com/openai/v1"},
		{name: "environment fallback", want: "https://environment-resource.openai.azure.com/openai/v1"},
		{name: "missing resource", wantErr: true, wantReason: ResourceConfigurationMissing},
		{name: "invalid resource", resource: "not/a-resource", wantErr: true, wantReason: ResourceConfigurationInvalid},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "missing resource" {
				t.Setenv("AZURE_RESOURCE_NAME", "")
			}
			got, err := resolveBaseURL(tt.baseURL, tt.resource)
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolveBaseURL() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				var resourceErr *ResourceConfigurationError
				if !errors.As(err, &resourceErr) || resourceErr.Reason != tt.wantReason {
					t.Fatalf("resolveBaseURL() error = %T %v, want reason %q", err, err, tt.wantReason)
				}
				return
			}
			if got != tt.want {
				t.Errorf("resolveBaseURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWithResourceNameCopiesConfiguration(t *testing.T) {
	var cfg config
	WithResourceName("resource-a")(&cfg)
	if cfg.resourceName != "resource-a" {
		t.Fatalf("resourceName = %q, want resource-a", cfg.resourceName)
	}
}
