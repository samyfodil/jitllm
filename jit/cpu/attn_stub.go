//go:build !amd64 && !arm64

package cpu

import "fmt"

// EmitAttnScores and EmitAttnAcc have no implementation on architectures other
// than amd64 and arm64. The stubs keep callers building and report the absence
// as an error rather than returning a bad kernel.
func EmitAttnScores(hd, kvStride int, f16 bool) ([]byte, error) {
	return nil, fmt.Errorf("jit: no attention scores kernel on this architecture")
}

func EmitAttnAcc(hd, kvStride int, f16 bool) ([]byte, error) {
	return nil, fmt.Errorf("jit: no attention accumulate kernel on this architecture")
}

func EmitAttnAccInto(hd, kvStride int, f16 bool) ([]byte, error) {
	return nil, fmt.Errorf("jit: no attention accumulate kernel on this architecture")
}

// EmitRoPE likewise has no emitter here.
func EmitRoPE(hd, nrot int, neox bool) ([]byte, error) {
	return nil, fmt.Errorf("jit: no rotary kernel on this architecture")
}

// EmitRoPESplit likewise.
func EmitRoPESplit(hd, nrot int) ([]byte, error) {
	return nil, fmt.Errorf("jit: no rotary kernel on this architecture")
}
