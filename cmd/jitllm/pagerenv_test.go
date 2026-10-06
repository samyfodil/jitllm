package main

import "testing"

// pagerChunk must turn JITLLM_CHUNK_KIB into the byte count model.WithChunk
// takes, and refuse what File.SetChunk would: a value that is not a power of
// two, or under the 4096 alignment, would mean unaligned page reads, which the
// direct handle answers with EINVAL rather than data.
func TestTheChunkKnobReachesThePager(t *testing.T) {
	t.Setenv("JITLLM_CHUNK_KIB", "64")
	if c := pagerChunk(); c != 64<<10 {
		t.Fatalf("pagerChunk is %d for JITLLM_CHUNK_KIB=64, want %d", c, 64<<10)
	}
	for _, bad := range []string{"", "0", "3", "-8", "48"} {
		t.Setenv("JITLLM_CHUNK_KIB", bad)
		if c := pagerChunk(); c != 0 {
			t.Errorf("JITLLM_CHUNK_KIB=%q gave a chunk of %d", bad, c)
		}
	}
}
