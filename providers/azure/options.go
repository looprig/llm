package azure

import (
	"os"
	"strings"
)

const resourceNameEnvironment = "AZURE_RESOURCE_NAME"

// ResourceConfigurationReason classifies Azure resource-name configuration
// failures without including provider-controlled values in the error text.
type ResourceConfigurationReason string

const (
	ResourceConfigurationMissing ResourceConfigurationReason = "resource name is missing"
	ResourceConfigurationInvalid ResourceConfigurationReason = "resource name is invalid"
)

// ResourceConfigurationError reports a missing or malformed Azure resource
// name. It is intentionally secret-free and inspectable with errors.As.
type ResourceConfigurationError struct {
	Reason ResourceConfigurationReason
}

func (e *ResourceConfigurationError) Error() string {
	return "azure: resource configuration: " + string(e.Reason)
}

type config struct {
	resourceName string
}

// Option customizes Azure OpenAI endpoint resolution.
type Option func(*config)

// WithResourceName sets the Azure resource name used when Model.BaseURL is
// empty. It takes precedence over AZURE_RESOURCE_NAME.
func WithResourceName(name string) Option {
	return func(c *config) { c.resourceName = strings.TrimSpace(name) }
}

func (c config) clone() config {
	c.resourceName = strings.TrimSpace(c.resourceName)
	return c
}

func resolveBaseURL(explicit, optionResource string) (string, error) {
	if baseURL := strings.TrimRight(strings.TrimSpace(explicit), "/"); baseURL != "" {
		return baseURL, nil
	}
	resource := strings.TrimSpace(optionResource)
	if resource == "" {
		resource = strings.TrimSpace(os.Getenv(resourceNameEnvironment))
	}
	if resource == "" {
		return "", &ResourceConfigurationError{Reason: ResourceConfigurationMissing}
	}
	if !validResourceName(resource) {
		return "", &ResourceConfigurationError{Reason: ResourceConfigurationInvalid}
	}
	return "https://" + resource + ".openai.azure.com/openai/v1", nil
}

func validResourceName(name string) bool {
	if len(name) == 0 || len(name) > 64 || name[0] == '-' || name[len(name)-1] == '-' {
		return false
	}
	for _, char := range name {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') && (char < '0' || char > '9') && char != '-' {
			return false
		}
	}
	return true
}
