module github.com/jitllm/jitllm/ui

go 1.26.0

require (
	github.com/gogpu/gg v0.52.3
	github.com/gogpu/gogpu v0.53.0
	github.com/gogpu/ui v0.1.54
	github.com/jitllm/jitllm v0.0.0
	github.com/jitllm/jitllm/common v0.0.0
	github.com/jitllm/jitllm/server v0.0.0
	golang.org/x/image v0.45.0
	golang.org/x/sys v0.48.0
	golang.org/x/text v0.42.0
)

require (
	connectrpc.com/connect v1.21.0 // indirect
	golang.org/x/net v0.59.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

require (
	github.com/coregx/signals v0.1.1 // indirect
	github.com/go-webgpu/goffi v0.6.3 // indirect
	github.com/go-webgpu/webgpu v0.5.5 // indirect
	github.com/gogpu/gpucontext v0.28.0 // indirect
	github.com/gogpu/gputypes v0.5.2 // indirect
	github.com/gogpu/naga v0.18.0 // indirect
	github.com/gogpu/wgpu v0.31.4 // indirect
)

replace github.com/jitllm/jitllm => ../

replace github.com/jitllm/jitllm/common => ../common

replace github.com/jitllm/jitllm/server => ../server
