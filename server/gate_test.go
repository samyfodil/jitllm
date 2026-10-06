package server

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// The gates on the gate. The serialisation is the one behaviour of this server
// that a caller cannot see from the outside, so it is the one that most needs
// a test that can fail.

// TestADeviceRunsOneSessionAtATime.
//
// VIOLATION SIGNATURE. Change newGate's width for a device from
// Config.DeviceConcurrency to 2 and this fails with
//
//	both sessions were inside the gate at once: overlap detected
//
// which is exactly the state the engine cannot survive -- two sessions sharing
// one scratch set on one device.
func TestADeviceRunsOneSessionAtATime(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	gs1 := e.gatesFor([]string{"cuda:0"})
	gs2 := e.gatesFor([]string{"cuda:0"})

	var mu sync.Mutex
	inside := 0
	overlap := false

	if _, _, err := gs1.acquire(context.Background(), "s1", 0); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	mu.Lock()
	inside++
	mu.Unlock()

	got := make(chan struct{})
	go func() {
		gs2.acquire(context.Background(), "s2", 0)
		mu.Lock()
		inside++
		if inside > 1 {
			overlap = true
		}
		mu.Unlock()
		close(got)
	}()

	select {
	case <-got:
		t.Fatal("the second session entered while the first held the device; " +
			"both sessions were inside the gate at once: overlap detected")
	case <-time.After(150 * time.Millisecond):
	}

	// The queue is visible while it is happening, which is the whole point.
	g := e.gate("cuda:0")
	queue, running, waiting := g.snapshot()
	if running != 1 || waiting != 1 {
		t.Fatalf("queue reports running=%d waiting=%d, want 1 and 1 (queue %v)", running, waiting, queue)
	}
	if g.mode != ExecutionSerialised {
		t.Fatalf("a device gate reports %v, want SERIALISED", g.mode)
	}

	mu.Lock()
	inside--
	mu.Unlock()
	gs1.release()

	select {
	case <-got:
	case <-time.After(2 * time.Second):
		t.Fatal("the second session never got in after the first released: it was refused, not queued")
	}
	if overlap {
		t.Fatal("both sessions were inside the gate at once: overlap detected")
	}
	gs2.release()
}

// TestASecondSessionIsQueuedAndNeverRefused: a second session must wait,
// never be refused.
//
// VIOLATION SIGNATURE. Make acquire return an error when the semaphore is full
// instead of blocking, and this fails with
//
//	the second session was refused rather than queued: server: ...
func TestASecondSessionIsQueuedAndNeverRefused(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	a := e.gatesFor([]string{"cuda:0"})
	b := e.gatesFor([]string{"cuda:0"})

	if _, _, err := a.acquire(context.Background(), "a", 0); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	done := make(chan error, 1)
	var waited time.Duration
	go func() {
		w, _, err := b.acquire(context.Background(), "b", 0)
		waited = w
		done <- err
	}()
	time.Sleep(120 * time.Millisecond)
	a.release()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the second session was refused rather than queued: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the second session never entered")
	}
	if waited < 50*time.Millisecond {
		t.Fatalf("the wait was reported as %v, which cannot be right for a gate held 120ms; "+
			"a queued_millis of ~0 tells a caller nothing happened when it did", waited)
	}
	b.release()
}

// TestDecliningToWaitIsNotARefusalOfTheSession.
//
// A queue timeout ends the WAIT, not the session: the caller may try again.
func TestDecliningToWaitIsNotARefusalOfTheSession(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	a := e.gatesFor([]string{"cuda:0"})
	b := e.gatesFor([]string{"cuda:0"})
	a.acquire(context.Background(), "a", 0)

	_, _, err := b.acquire(context.Background(), "b", 30*time.Millisecond)
	if !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("timed-out wait returned %v, want ErrQueueTimeout", err)
	}
	// It must have cleaned up after itself: a caller that gave up must not
	// still be counted as waiting, or the queue depth every later request
	// reports is permanently wrong.
	if _, _, waiting := e.gate("cuda:0").snapshot(); waiting != 0 {
		t.Fatalf("after a timed-out wait the gate still reports %d waiting; the giver-up "+
			"was never dequeued", waiting)
	}
	a.release()
	if _, _, err := b.acquire(context.Background(), "b", time.Second); err != nil {
		t.Fatalf("retry after a timeout failed: %v -- the timeout refused the SESSION, not the wait", err)
	}
	b.release()
}

// TestTwoSessionsAcrossTwoDevicesDoNotDeadlock.
//
// Two sessions each needing cuda:0 and vulkan:1 would deadlock if they took
// them in opposite orders; gatesFor sorts.
//
// VIOLATION SIGNATURE. Remove `sort.Strings(sorted)` from gatesFor and this
// hangs, then fails with
//
//	deadlocked: neither session completed within 2s
//
// (the test uses its own timeout rather than the package one so the failure is
// a message rather than a panic dump).
func TestTwoSessionsAcrossTwoDevicesDoNotDeadlock(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	done := make(chan struct{}, 2)
	for i, ids := range [][]string{{"cuda:0", "vulkan:1"}, {"vulkan:1", "cuda:0"}} {
		go func(i int, ids []string) {
			for n := 0; n < 40; n++ {
				gs := e.gatesFor(ids)
				if _, _, err := gs.acquire(context.Background(), "s", 0); err != nil {
					t.Errorf("session %d acquire: %v", i, err)
					break
				}
				time.Sleep(time.Millisecond)
				gs.release()
			}
			done <- struct{}{}
		}(i, ids)
	}
	for n := 0; n < 2; n++ {
		select {
		case <-done:
		case <-time.After(4 * time.Second):
			t.Fatal("deadlocked: neither session completed within 2s")
		}
	}
}

// TestPartialAcquireReleasesWhatItTook.
//
// A session that gets cuda:0 and then times out on vulkan:1 must not keep
// cuda:0: holding one card for a request that will not run stalls it for
// everyone.
//
// VIOLATION SIGNATURE. Delete the `gs.release()` on the error path in
// gateSet.acquire and this fails with
//
//	cuda:0 is still held after a failed acquire: running=1
func TestPartialAcquireReleasesWhatItTook(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	// Hold vulkan:1 so the second gate of the pair cannot be taken.
	blocker := e.gatesFor([]string{"vulkan:1"})
	blocker.acquire(context.Background(), "blocker", 0)

	gs := e.gatesFor([]string{"cuda:0", "vulkan:1"})
	if _, _, err := gs.acquire(context.Background(), "s", 40*time.Millisecond); !errors.Is(err, ErrQueueTimeout) {
		t.Fatalf("acquire returned %v, want ErrQueueTimeout", err)
	}
	if _, running, _ := e.gate("cuda:0").snapshot(); running != 0 {
		t.Fatalf("cuda:0 is still held after a failed acquire: running=%d", running)
	}
	blocker.release()
}

// TestTheHostIsSerialisedByDefault.
//
// Every JIT runs its own pool of decode cores, so two host sessions
// oversubscribe them. The default must be 1, and it must be a knob.
func TestTheHostIsSerialisedByDefault(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	if g := e.gate(HostGateID); g.mode != ExecutionSerialised {
		t.Fatalf("the host gate is %v by default, want SERIALISED", g.mode)
	}
	if note := e.gate(HostGateID).note; note == "" {
		t.Fatal("the host gate reports no reason; a limit a caller cannot see the reason for " +
			"reads as a slow server")
	}
	wide := New(Config{ModelDir: t.TempDir(), HostConcurrency: 4})
	if g := wide.gate(HostGateID); g.mode != ExecutionParallel {
		t.Fatalf("with HostConcurrency 4 the host gate is %v, want PARALLEL -- the limit is a "+
			"configuration, not a constant", g.mode)
	}
}
