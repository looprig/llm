package openaisubscription

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"unicode"

	"github.com/looprig/credentials"
	"github.com/looprig/credentials/refresh"
)

type AvailableModel struct {
	Slug        string `json:"slug"`
	DisplayName string `json:"display_name"`
}

// ListModels returns the current selected account's displayable models in
// server order. Catalog inclusion is not proof an inference request will succeed.
func ListModels(ctx context.Context, source credentials.Source, client *http.Client) ([]AvailableModel, error) {
	if source == nil {
		return nil, ErrModels
	}
	descriptor := source.Descriptor()
	if descriptor.Provider != "openai-subscription" || descriptor.Transport != "responses" || descriptor.Scheme != credentials.SchemeOAuth || descriptor.Usage != credentials.UsageSubscription || descriptor.Issuer != Issuer || descriptor.Audience != "https://api.openai.com" {
		return nil, ErrModels
	}
	lease, err := source.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	var catalog struct {
		Models []struct {
			Slug        string `json:"slug"`
			DisplayName string `json:"display_name"`
			Visibility  string `json:"visibility"`
		} `json:"models"`
	}
	err = requestJSON(ctx, client, http.MethodGet, BaseURL+"/models", nil, func(r *http.Request) error { return lease.Authorizer().Authorize(ctx, r) }, &catalog)
	if err != nil {
		return nil, ErrModels
	}
	result := make([]AvailableModel, 0, len(catalog.Models))
	seen := map[string]bool{}
	for _, entry := range catalog.Models {
		if entry.Visibility != "list" {
			continue
		}
		if entry.Slug == "" || len(entry.Slug) > 256 || len(entry.DisplayName) > 256 || strings.IndexFunc(entry.Slug, func(c rune) bool { return unicode.IsControl(c) || unicode.IsSpace(c) }) >= 0 || strings.IndexFunc(entry.DisplayName, unicode.IsControl) >= 0 || seen[entry.Slug] {
			return nil, ErrModels
		}
		seen[entry.Slug] = true
		result = append(result, AvailableModel{Slug: entry.Slug, DisplayName: entry.DisplayName})
	}
	return result, nil
}

// Revoke ends this renewable session using the issuer's discovery document.
// A local deletion is separate from confirmed remote revocation.
func Revoke(ctx context.Context, client *http.Client, state refresh.State) error {
	registration, err := RegistrationFromState(state)
	if err != nil {
		return ErrRevoke
	}
	var discovery struct {
		Issuer             string `json:"issuer"`
		RevocationEndpoint string `json:"revocation_endpoint"`
	}
	if requestJSON(ctx, client, http.MethodGet, DiscoveryURL, nil, nil, &discovery) != nil || discovery.Issuer != Issuer {
		return ErrRevoke
	}
	endpoint, err := url.Parse(discovery.RevocationEndpoint)
	if err != nil || endpoint.Scheme != "https" || endpoint.Host != "auth.openai.com" || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" || endpoint.Path == "" {
		return ErrRevoke
	}
	raw := state.RefreshToken.Bytes()
	defer clear(raw)
	if requestJSON(ctx, client, http.MethodPost, endpoint.String(), url.Values{"client_id": {registration.ClientID}, "token_type_hint": {"refresh_token"}, "token": {string(raw)}}, nil, nil) != nil {
		return ErrRevoke
	}
	return nil
}
