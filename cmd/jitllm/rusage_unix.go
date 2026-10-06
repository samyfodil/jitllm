//go:build unix

package main

import (
	"runtime"
	"syscall"
)

// processUsage is the process's user+system CPU seconds and its peak resident
// bytes, from getrusage.
func processUsage() (cpu float64, maxRSS uint64) {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	cpu = float64(ru.Utime.Sec+ru.Stime.Sec) + float64(ru.Utime.Usec+ru.Stime.Usec)/1e6
	maxRSS = uint64(ru.Maxrss)
	// macOS reports ru_maxrss in bytes, Linux and the BSDs in KiB.
	if runtime.GOOS != "darwin" && runtime.GOOS != "ios" {
		maxRSS *= 1024
	}
	return cpu, maxRSS
}
