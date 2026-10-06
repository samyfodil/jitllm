module github.com/samyfodil/jitllm/server

go 1.26.0

require (
	connectrpc.com/connect v1.21.0
	github.com/samyfodil/jitllm v0.0.0
	golang.org/x/net v0.59.0
	google.golang.org/protobuf v1.36.12
)

require (
	github.com/go-webgpu/goffi v0.6.3 // indirect
	golang.org/x/text v0.42.0 // indirect
)

// ★ THE SERVER IS ITS OWN MODULE FOR THE SAME REASON ui/ IS: so the ENGINE
// keeps one dependency. connect, protobuf and x/net are a serving concern and
// nothing on the inference path needs them, so carrying them in the root
// go.mod would make every consumer of the library pay for an RPC stack it does
// not import. AGENTS.md's "scope is jitllm's only defence" is about more than
// architectures.
replace github.com/samyfodil/jitllm => ../
