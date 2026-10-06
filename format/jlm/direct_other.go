//go:build !linux

package jlm

import "os"

// O_DIRECT is Linux's spelling. Darwin has F_NOCACHE on the descriptor, which
// is a different contract (advice rather than a bypass) and Apple Silicon
// has unified memory where the trade differs; buffered there.
func openDirect(path string) *os.File { return nil }

func alignedBuf(n int) []byte { return make([]byte, n) }

func directOK(off int64, n int) bool { return false }

func aligned(b []byte) bool { return false }

func closeDirect(f *os.File) {}
