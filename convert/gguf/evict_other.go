//go:build !linux && !darwin

package gguf

// Evict is a no-op where Go's syscall package has no Madvise, and reports zero
// bytes handed back. Residency then falls back to the kernel's own policy.
func Evict(b []byte) int { return 0 }
