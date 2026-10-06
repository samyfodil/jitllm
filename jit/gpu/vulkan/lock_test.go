package vulkan

import (
	"testing"

	"github.com/samyfodil/jitllm/internal/testlock"
)

// gpuLock is jit/gpu/backend's cross-process test lock (internal/testlock).
//
// TestMemAgreesWithTheKernelDriver compares two samples of free VRAM, and
// `go test ./...` runs jit/gpu/backend in parallel, whose tests allocate
// gigabytes. The lock keeps them apart; widening the 10% slack instead would
// cost the test's sensitivity to a layout slip.
func gpuLock(t *testing.T) { testlock.GPU(t) }
