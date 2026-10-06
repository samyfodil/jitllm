//go:build unix

package testlock

import (
	"os"
	"syscall"
)

func lock(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_EX) }

// unlock's error is moot: closing the file releases the lock anyway.
func unlock(f *os.File) { syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }
