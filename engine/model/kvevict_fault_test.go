//go:build jitllmfault

package model

import (
	"testing"

	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// TestKVEvictionGateDiscriminates runs the eviction gate's squeezed arm with
// the streamed pass's upload skipped, so each pass reads whatever the free ids
// held, and demands the comparison fail.
func TestKVEvictionGateDiscriminates(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, false)
	tier.SetPagedFault("stream")
	defer tier.SetPagedFault("")
	got := eg.run(t, true)
	if got.st.KVStreamPasses == 0 {
		t.Fatal("no streamed pass ran: the violation was never exercised")
	}
	dev, at := worstStep(got.logits, free.logits)
	t.Logf("violation: worst logit NMSE %.3e at step %d", dev, at)
	if dev < evictNMSE {
		t.Fatalf("the gate passed a stream that uploaded nothing (NMSE %.3e)", dev)
	}
}
