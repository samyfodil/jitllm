//go:build !unix && !windows

package testlock

import "os"

// lock is a no-op where there is no file lock to take.
func lock(f *os.File) error { return nil }

func unlock(f *os.File) {}
