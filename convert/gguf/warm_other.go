//go:build !linux

package gguf

// warm is a no-op where Go's syscall package has no Madvise, and its stop
// function is a no-op too. See the linux sibling.
func warm(b []byte) func() { return func() {} }
