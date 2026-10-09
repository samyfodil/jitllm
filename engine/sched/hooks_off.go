//go:build !jitllmtest

package sched

// workerWake is the test hook a release build compiles out (hooks_on.go).
func workerWake(*Pool, int) {}
