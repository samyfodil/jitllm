package gguf

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// mmapFile maps the whole file read-only and copy-on-write.
//
// Windows has no mmap syscall; the pair is CreateFileMapping to make a section
// object and MapViewOfFile to put it in the address space. PAGE_READONLY plus
// FILE_MAP_READ is the analogue of PROT_READ with MAP_PRIVATE: the view is
// backed by the file, never dirtied, and never written back.
//
// Mapping keeps the weights in page cache rather than anonymous memory, so they
// cost nothing against a process memory cap and are shared with other mappers.
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

	// A zero maximum size means "the whole file", but passing the size
	// explicitly keeps the mapping honest if the file grows under us.
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil,
		syscall.PAGE_READONLY, uint32(n>>32), uint32(n), nil)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: CreateFileMapping %s (%d bytes): %w", path, n, err)
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(n))
	if err != nil {
		syscall.CloseHandle(h)
		return nil, nil, fmt.Errorf("gguf: MapViewOfFile %s (%d bytes): %w", path, n, err)
	}
	b := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(n))
	return b, func() error {
		err := syscall.UnmapViewOfFile(addr)
		if cerr := syscall.CloseHandle(h); err == nil {
			err = cerr
		}
		return err
	}, nil
}
