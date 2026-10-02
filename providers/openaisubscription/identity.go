package openaisubscription

import (
	"context"
	"crypto"
	"crypto/ecdh"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"strings"
	"time"
)

type identity struct{ Subject, Email string }
type jwk struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	Alg string `json:"alg"`
	Use string `json:"use"`
	N   string `json:"n"`
	E   string `json:"e"`
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

// verifyIdentity verifies only a small explicit set of asymmetric signature
// algorithms using the pinned issuer's JWKS. JWT-controlled URLs and keys are
// never consulted. A fresh JWKS read naturally handles issuer key rotation.
func verifyIdentity(ctx context.Context, client *http.Client, token, clientID, nonce string) (identity, error) {
	if len(token) > 32<<10 || !validClientID(clientID) || nonce == "" {
		return identity{}, ErrIdentity
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return identity{}, ErrIdentity
	}
	headerRaw, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return identity{}, ErrIdentity
	}
	var header struct {
		Alg  string   `json:"alg"`
		Kid  string   `json:"kid"`
		Crit []string `json:"crit"`
	}
	if json.Unmarshal(headerRaw, &header) != nil || header.Kid == "" || len(header.Crit) > 0 || header.Alg != "RS256" && header.Alg != "ES256" {
		return identity{}, ErrIdentity
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return identity{}, ErrIdentity
	}
	var keys struct {
		Keys []jwk `json:"keys"`
	}
	if requestJSON(ctx, client, http.MethodGet, JWKSURL, nil, nil, &keys) != nil || len(keys.Keys) > 100 {
		return identity{}, ErrIdentity
	}
	var selected *jwk
	for i := range keys.Keys {
		key := &keys.Keys[i]
		if key.Kid == header.Kid {
			if selected != nil {
				return identity{}, ErrIdentity
			}
			selected = key
		}
	}
	if selected == nil || selected.Alg != "" && selected.Alg != header.Alg || selected.Use != "" && selected.Use != "sig" {
		return identity{}, ErrIdentity
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	if !verifySignature(*selected, header.Alg, digest[:], signature) {
		return identity{}, ErrIdentity
	}
	claimsRaw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return identity{}, ErrIdentity
	}
	var claims struct {
		Issuer          string          `json:"iss"`
		Audience        json.RawMessage `json:"aud"`
		AuthorizedParty string          `json:"azp"`
		Subject         string          `json:"sub"`
		Email           string          `json:"email"`
		Nonce           string          `json:"nonce"`
		Exp             int64           `json:"exp"`
		Iat             int64           `json:"iat"`
		Nbf             int64           `json:"nbf"`
	}
	if json.Unmarshal(claimsRaw, &claims) != nil {
		return identity{}, ErrIdentity
	}
	now := time.Now().Unix()
	if claims.Issuer != Issuer || claims.Subject == "" || len(claims.Subject) > 1024 || claims.Nonce != nonce || claims.Exp <= now-5 || claims.Iat <= 0 || claims.Iat > now+5 || claims.Nbf > now+5 || claims.AuthorizedParty != "" && claims.AuthorizedParty != clientID {
		return identity{}, ErrIdentity
	}
	var audience string
	if json.Unmarshal(claims.Audience, &audience) == nil {
		if audience != clientID {
			return identity{}, ErrIdentity
		}
	} else {
		var audiences []string
		if json.Unmarshal(claims.Audience, &audiences) != nil || len(audiences) == 0 {
			return identity{}, ErrIdentity
		}
		found := false
		for _, aud := range audiences {
			found = found || aud == clientID
		}
		if !found || len(audiences) > 1 && claims.AuthorizedParty != clientID {
			return identity{}, ErrIdentity
		}
	}
	return identity{Subject: claims.Subject, Email: claims.Email}, nil
}
func verifySignature(key jwk, alg string, digest, signature []byte) bool {
	if alg == "RS256" && key.Kty == "RSA" {
		n, err := base64.RawURLEncoding.DecodeString(key.N)
		if err != nil || len(n) < 256 || len(n) > 1024 {
			return false
		}
		e, err := base64.RawURLEncoding.DecodeString(key.E)
		if err != nil || len(e) == 0 || len(e) > 4 {
			return false
		}
		exponent := new(big.Int).SetBytes(e).Int64()
		if exponent < 3 || exponent > 2147483647 || exponent%2 == 0 {
			return false
		}
		return rsa.VerifyPKCS1v15(&rsa.PublicKey{N: new(big.Int).SetBytes(n), E: int(exponent)}, crypto.SHA256, digest, signature) == nil
	}
	if alg == "ES256" && key.Kty == "EC" && key.Crv == "P-256" && len(signature) == 64 {
		x, err := base64.RawURLEncoding.DecodeString(key.X)
		if err != nil || len(x) != 32 {
			return false
		}
		y, err := base64.RawURLEncoding.DecodeString(key.Y)
		if err != nil || len(y) != 32 {
			return false
		}
		pointX, pointY := new(big.Int).SetBytes(x), new(big.Int).SetBytes(y)
		encodedPoint := append([]byte{4}, x...)
		encodedPoint = append(encodedPoint, y...)
		if _, err := ecdh.P256().NewPublicKey(encodedPoint); err != nil {
			return false
		}
		return ecdsa.Verify(&ecdsa.PublicKey{Curve: elliptic.P256(), X: pointX, Y: pointY}, digest, new(big.Int).SetBytes(signature[:32]), new(big.Int).SetBytes(signature[32:]))
	}
	return false
}
