package gguf

import "unsafe"

// ptr is the address of b's first byte. It is the one unsafe line in this
// package and it exists for exactly one caller: madvise measures in pages and
// a []byte cannot say where its pages begin without its address.
//
// The pointer is not retained, not converted to uintptr for arithmetic that
// outlives the expression, and not used to read anything -- pageAlignIn does
// the arithmetic on a plain uint64 copy and then reslices b, so the garbage
// collector's view of the slice is the only thing that ever indexes it.
func ptr(b []byte) unsafe.Pointer { return unsafe.Pointer(&b[0]) }
