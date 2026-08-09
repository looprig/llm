// Package subscription provides provider-neutral, fixture-only certification
// contracts for credential-backed subscription transports.
//
// An enabled OpenAI or Anthropic subscription package must add a package test
// that constructs a Witness with its explicit Constructor and Contract, then
// calls Witness.Run against NewServer(t). The witness is deliberately
// fail-closed: no constructor, provider endpoint, client ID, account selector,
// or model-discovery behavior can be inferred by this package. Registration
// gates that are unavailable must not fabricate a witness or provider client.
package subscription
