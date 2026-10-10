module github.com/jitllm/jitllm/common

go 1.26.0

require github.com/jitllm/jitllm v0.0.0

require (
	connectrpc.com/connect v1.21.0 // indirect
	golang.org/x/net v0.59.0 // indirect
	golang.org/x/text v0.42.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

require (
	github.com/go-webgpu/goffi v0.6.3 // indirect
	github.com/jitllm/jitllm/server v0.0.0
)

replace github.com/jitllm/jitllm => ../

replace github.com/jitllm/jitllm/server => ../server
