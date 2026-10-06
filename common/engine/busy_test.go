package engine

import (
	"testing"
)

// Busy must stay true while any counted job is queued or running: a reply
// running, a switch queued behind it, the reply ends, the switch starts.
// Against the violation (busyEnd posting false) the middle assertion fails.
func TestBusyCountsOverlappingJobs(t *testing.T) {
	sh := newTestShell()
	e := &Engine{sh: sh, st: sh.state()}

	e.busyBegin() // the reply
	e.busyBegin() // a switch queued behind it
	e.busyEnd()   // the reply finishes
	sh.DrainQueue(0)
	if !sh.Store.Busy.Get() {
		t.Fatal("Busy went false while a queued job was still to run")
	}
	e.busyEnd() // the switch finishes
	sh.DrainQueue(0)
	if sh.Store.Busy.Get() {
		t.Fatal("Busy stayed true with nothing queued or running")
	}
}
