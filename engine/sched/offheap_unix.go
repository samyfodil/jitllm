//go:build linux || darwin

package sched

// offHeapPlatform: format/jlm maps its weight memory here (kernels.MapOffHeap).
const offHeapPlatform = true
