//go:build linux

package jlm

import (
	"os"
	"syscall"
	"unsafe"
)

// One copy, and it is ours. Buffered I/O leaves every byte in the kernel's
// page cache as well as in our frame, and cgroup v2 charges that cache to the
// same limit, so a budget of B wants 2B of the cgroup. mmap only moves the
// copy into the kernel and gives up residency as this process's decision.
// O_DIRECT reads straight into our buffer, at a read rate somewhat below
// buffered and with no page-cache shadow; that cost is paid once, where being
// one page short of the budget costs a page re-read every token.
//
// It is a second descriptor, not a flag on the first: O_DIRECT needs the
// offset, length and buffer address block-aligned. Page reads are (both
// alignUp'd to Align); the header, sections and dense region are not, and
// keep the buffered handle.
func openDirect(path string) *os.File {
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_DIRECT, 0)
	if err != nil {
		// Not every filesystem supports it (tmpfs does not). Buffered still
		// works; it just costs the second copy.
		return nil
	}
	return f
}

// alignedBuf returns n bytes whose first byte is Align-aligned, which O_DIRECT
// requires of the destination and Go's allocator does not promise.
func alignedBuf(n int) []byte {
	raw := make([]byte, n+Align)
	off := (Align - int(uintptr(unsafe.Pointer(&raw[0]))%Align)) % Align
	return raw[off : off+n : off+n]
}

// directOK reports whether this offset and length may go through the direct
// handle. Anything else falls back to the buffered one rather than failing.
func directOK(off int64, n int) bool { return off%Align == 0 && n%Align == 0 }

// aligned reports whether a slice's first byte is Align-aligned.
func aligned(b []byte) bool {
	return len(b) == 0 || uintptr(unsafe.Pointer(&b[0]))%Align == 0
}

func closeDirect(f *os.File) {
	if f != nil {
		f.Close()
	}
}
