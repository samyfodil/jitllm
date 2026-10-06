// Package testlock serialises GPU tests across processes.
//
// `go test ./...` runs packages in parallel, and packages that open a device
// contend for VRAM and the driver, failing in a different package each run. A
// file lock costs nothing when uncontended and turns the failure into a wait.
//
// Tests only: nothing in the engine takes it.
package testlock

import (
	"os"
	"path/filepath"
	"testing"
)

// GPU takes the cross-process GPU test lock and releases it when t ends. Every
// package that allocates on a card takes the same file, so two of them do not
// allocate on one card at the same time.
//
// A lock that cannot be taken is no lock rather than no test: the test runs.
func GPU(t testing.TB) {
	t.Helper()
	p := filepath.Join(os.TempDir(), "jitllm-gpu-test.lock")
	f, err := os.OpenFile(p, os.O_CREATE|os.O_RDWR, 0o666)
	if err != nil {
		return
	}
	if err := lock(f); err != nil {
		f.Close()
		return
	}
	t.Cleanup(func() {
		unlock(f)
		f.Close()
	})
}
