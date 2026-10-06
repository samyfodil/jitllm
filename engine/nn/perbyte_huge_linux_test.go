//go:build jitllmbench && linux

package nn

import (
	"syscall"
	"unsafe"
)

// hugeCopy copies b into a fresh buffer that asked for transparent huge pages
// before its first touch, so the weights sit on 2 MiB pages.
func hugeCopy(b []byte) []byte {
	const huge = 2 << 20
	if len(b) == 0 {
		return b
	}
	raw := make([]byte, len(b)+huge)
	off := (huge - int(uintptr(unsafe.Pointer(&raw[0]))%huge)) % huge
	out := raw[off : off+len(b)]
	syscall.Madvise(out, 14) // MADV_HUGEPAGE
	copy(out, b)
	return out
}
