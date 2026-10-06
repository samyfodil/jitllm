package gguf

import "sync"

// The registry of live mappings, and the reason Evict consults it.
//
// MADV_DONTNEED is safe on a read-only private file mapping (clean pages
// re-fault from disk) and destructive on anonymous memory (the Go heap: the
// next read silently returns zeros), and the two are the same []byte to a
// caller. Callers such as the packed arena are handed opaque slices, so the
// check lives here, where the mappings were made: a range outside one gets
// zero and no syscall. The cost is a mutex and a scan over the open files per
// evicted range.
var mapped struct {
	mu sync.Mutex
	rs []mappedRange
}

type mappedRange struct {
	lo, hi uintptr
}

func registerMapping(b []byte) {
	if len(b) == 0 {
		return
	}
	lo := uintptr(ptr(b))
	mapped.mu.Lock()
	mapped.rs = append(mapped.rs, mappedRange{lo: lo, hi: lo + uintptr(len(b))})
	mapped.mu.Unlock()
}

func unregisterMapping(b []byte) {
	if len(b) == 0 {
		return
	}
	lo := uintptr(ptr(b))
	mapped.mu.Lock()
	for i, r := range mapped.rs {
		if r.lo == lo {
			mapped.rs = append(mapped.rs[:i], mapped.rs[i+1:]...)
			break
		}
	}
	mapped.mu.Unlock()
}

// Mapped reports whether b lies entirely inside a file mapping this package has
// open. It is what makes Evict safe to offer a []byte of unknown provenance.
func Mapped(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	lo := uintptr(ptr(b))
	hi := lo + uintptr(len(b))
	mapped.mu.Lock()
	defer mapped.mu.Unlock()
	for _, r := range mapped.rs {
		if lo >= r.lo && hi <= r.hi {
			return true
		}
	}
	return false
}
