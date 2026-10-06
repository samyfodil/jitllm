//go:build !linux && !darwin

package sched

// offHeapPlatform: no anonymous mmap here, so the weights are on the heap.
const offHeapPlatform = false
