module github.com/looprig/llm

go 1.26.4

require (
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.1
	github.com/google/go-tdx-guest v0.3.1
	github.com/looprig/core v0.2.0
	github.com/looprig/inference v0.3.1-0.20260718005749-13e4d7f173b3
	golang.org/x/crypto v0.52.0
)

require (
	github.com/google/logger v1.1.1 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	google.golang.org/protobuf v1.31.0 // indirect
)

replace github.com/looprig/core => ../core

replace github.com/looprig/inference => ../inference
