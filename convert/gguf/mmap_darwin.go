package gguf

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile maps the whole file read-only and private.
//
// The same three syscalls as the Linux sibling, for a different reason. Linux
// mmaps weights because read-only FILE mappings are exempt from RLIMIT_DATA
// (>= 4.7), which is what lets jitllm cap its anonymous memory hard while leaving
// weights unlimited. macOS has no such interaction; the reason here is simply
// that the alternative is reading half a gigabyte into the Go heap.
func mmapFile(path string, _ bool) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	n := st.Size()
	if n <= 0 {
		return nil, nil, fmt.Errorf("gguf: %s: empty file", path)
	}
	if n != int64(int(n)) {
		return nil, nil, fmt.Errorf("gguf: %s: %d bytes does not fit in int", path, n)
	}

	b, err := syscall.Mmap(int(f.Fd()), 0, int(n), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: mmap %s (%d bytes): %w", path, n, err)
	}
	return b, func() error { return syscall.Munmap(b) }, nil
}
