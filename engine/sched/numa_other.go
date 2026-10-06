//go:build !linux

package sched

import "unsafe"

// NUMANodes reports no nodes where the OS gives no topology: nothing to place.
func NUMANodes() []int { return nil }

// MemPolicyDefault is true where there is no policy to have chosen.
func MemPolicyDefault() bool { return true }

// NodeOf is -1: no node to report.
func NodeOf(unsafe.Pointer) int { return -1 }

func nodeCPUs() ([]int, [][]int) { return nil, nil }
func currentCPU() int            { return -1 }
