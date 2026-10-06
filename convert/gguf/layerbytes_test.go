package gguf

import (
	"testing"
	"unsafe"
)

// TestLayerBytesMatchesLayerRange checks the slice accessor against the range
// it is built on, for every block of a committed model, and then checks that it
// never panics for an index no block has.
//
// The ring calls LayerBytes with derived indices that can leave the block
// range, so an out-of-range index must give nil, never a panic.
func TestLayerBytesMatchesLayerRange(t *testing.T) {
	f := open(t, stories260K)

	n := 0
	for {
		off, size, ok := f.LayerRange(n)
		if !ok {
			break
		}
		b := f.LayerBytes(n)
		if uint64(len(b)) != size {
			t.Fatalf("block %d: LayerBytes is %d bytes, LayerRange says %d", n, len(b), size)
		}
		// The same bytes, not merely the same length: compare the address
		// against DataStart+off, which is what Bytes would have produced.
		want := uintptr(unsafe.Pointer(&f.Data[f.DataStart+off]))
		got := uintptr(unsafe.Pointer(&b[0]))
		if got != want {
			t.Fatalf("block %d starts at %#x, want %#x", n, got, want)
		}
		// And it must lie inside the mapping, which is the bound the whole
		// accessor exists to get right.
		end := uintptr(unsafe.Pointer(&f.Data[0])) + uintptr(len(f.Data))
		if got+uintptr(len(b)) > end {
			t.Fatalf("block %d runs past the end of the mapping", n)
		}
		n++
	}
	if n == 0 {
		t.Fatal("no blocks found: this model has no blk.N tensors and the test proves nothing")
	}
	t.Logf("%d blocks", n)

	for _, bad := range []int{-1, -1 << 20, n, n + 1, 1 << 20, int(^uint(0) >> 1)} {
		if b := f.LayerBytes(bad); b != nil {
			t.Fatalf("LayerBytes(%d) returned %d bytes for a block that does not exist", bad, len(b))
		}
	}
}
