//go:build windows

package testlock

import (
	"os"
	"syscall"
	"unsafe"
)

// LockFileEx and UnlockFileEx are not in Go's syscall package on Windows, and
// the root module takes no dependency for a test lock, so they are called
// through kernel32 directly.
var (
	kernel32     = syscall.NewLazyDLL("kernel32.dll")
	lockFileEx   = kernel32.NewProc("LockFileEx")
	unlockFileEx = kernel32.NewProc("UnlockFileEx")
)

const lockfileExclusiveLock = 0x2

// lock takes an exclusive lock on the file's first byte, blocking until it is
// free: the wait flock(LOCK_EX) gives on Unix.
func lock(f *os.File) error {
	var ol syscall.Overlapped
	r, _, err := lockFileEx.Call(f.Fd(), lockfileExclusiveLock, 0, 1, 0, uintptr(unsafe.Pointer(&ol)))
	if r == 0 {
		return err
	}
	return nil
}

// unlock's result is moot: closing the handle releases the lock anyway.
func unlock(f *os.File) {
	var ol syscall.Overlapped
	unlockFileEx.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(&ol)))
}
