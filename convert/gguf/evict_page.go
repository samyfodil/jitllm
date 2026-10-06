//go:build linux || darwin

package gguf

import "os"

// pageSize is what madvise measures in. Read once: os.Getpagesize is a cached
// value on Linux, but this is on the eviction path of every block and every
// expert.
var pageSize = uint64(os.Getpagesize())

// pageAlignIn narrows b to the whole pages it contains: the start rounded up,
// the end rounded down. It returns an empty slice when b spans no whole page.
//
// Inward, not outward: a tensor's first and last pages are shared with its
// neighbours, and rounding outward would evict part of a tensor that may be
// hot. Narrowing gives up at most two pages per range.
func pageAlignIn(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	base := uint64(uintptr(ptr(b)))
	lo := (base + pageSize - 1) &^ (pageSize - 1)
	hi := (base + uint64(len(b))) &^ (pageSize - 1)
	if hi <= lo {
		return nil
	}
	return b[lo-base : hi-base]
}
