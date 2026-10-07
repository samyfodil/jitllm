package server

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
)

// A gate records the sessions running on a device and never holds one back:
// a second session on the same card is recorded beside the first, and leaving
// takes it off the record.
func TestAGateRecordsAndNeverBlocks(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir(), Probe: noProbe})
	defer e.Close()
	a := e.gatesFor([]string{"cuda:0", HostGateID})
	b := e.gatesFor([]string{"cuda:0", HostGateID, "cuda:0"})
	if len(b.gates) != 2 {
		t.Fatalf("%d gates for two distinct ids: a repeated id must be one gate", len(b.gates))
	}
	if d := a.acquire("a"); d != 0 {
		t.Fatalf("the first session found %d already running", d)
	}
	done := make(chan int32, 1)
	go func() { done <- b.acquire("b") }()
	select {
	case d := <-done:
		if d != 1 {
			t.Fatalf("the second session found %d running, want 1", d)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a second session on the same card waited for the first")
	}
	q, running, waiting := e.gate("cuda:0").snapshot()
	if running != 2 || waiting != 0 || len(q) != 2 {
		t.Fatalf("the card records %v, running %d, waiting %d; want both running and none waiting", q, running, waiting)
	}
	a.release()
	b.release()
	if q, running, _ := e.gate("cuda:0").snapshot(); running != 0 || len(q) != 0 {
		t.Fatalf("the card still records %v (%d running) after both left", q, running)
	}
}

// Two sessions of one model generate at the same time, and each one's answer
// is the answer it gives alone. On the host they take turns on one shared pool
// a region at a time (sched.Shared); on a device, with batching off, the tier
// takes its scratch a step at a time. Interleaving is asserted from the token
// times: each session produced a token while the other was mid-reply, which a
// lock held across a whole generate makes impossible.
func TestTwoSessionsGenerateAtOnce(t *testing.T) {
	t.Run("host", func(t *testing.T) {
		e, _, _ := loadedEngine(t, smallModel, "m", LoadOptions{})
		twoAtOnce(t, e, "m")
	})
	t.Run("device", func(t *testing.T) {
		if n, err := backend.CUDACount(); (err != nil || n == 0) && !hasVulkan() {
			t.Skip("NO DEVICE: the device arm did not run")
		}
		path := modelPath(t, deviceModel)
		e := New(Config{Probe: oneCardProbe, Version: "test", MaxBatchRows: 1})
		t.Cleanup(e.Close)
		lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m", DeviceIDs: []string{"auto"}})
		if err != nil {
			t.Fatal(err)
		}
		if lm.gpu == nil {
			t.Skip("auto opened no device: the device arm did not run")
		}
		twoAtOnce(t, e, "m")
	})
}

func hasVulkan() bool {
	ds, err := backend.VulkanDevices()
	return err == nil && len(ds) > 0
}

func twoAtOnce(t *testing.T, e *Engine, modelID string) {
	t.Helper()
	prompts := [2]string{"Once upon a time", "The little dog ran to the park and"}
	// Inside the device model's 128-position context with either prompt.
	const n = 100
	type run struct {
		ids   []int32
		times []time.Time
		err   error
		fin   *Finished
	}
	gen := func(sid, prompt string) run {
		var r run
		r.err = e.Generate(context.Background(), GenerateOptions{
			SessionID: sid, Prompt: Prompt{Kind: PromptText, Text: prompt}, MaxTokens: n, IgnoreEOS: true,
		}, func(ev Event) error {
			if ev.Kind == EventFinished {
				r.fin = ev.Finished
			}
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				r.ids = append(r.ids, ev.Token.ID)
				r.times = append(r.times, time.Now())
			}
			return nil
		})
		return r
	}
	for i := range 2 {
		if _, err := e.CreateSession(SessionOptions{ModelID: modelID, SessionID: string(rune('a' + i)), MaxSeq: 256}); err != nil {
			t.Fatal(err)
		}
	}
	var solo [2]run
	for i := range 2 {
		if solo[i] = gen(string(rune('a'+i)), prompts[i]); solo[i].err != nil || len(solo[i].ids) != n {
			t.Fatalf("solo %d: %d tokens, %v, finished %+v", i, len(solo[i].ids), solo[i].err, solo[i].fin)
		}
	}

	var both [2]run
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range 2 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			both[i] = gen(string(rune('a'+i)), prompts[i])
		}(i)
	}
	finished := make(chan struct{})
	go func() { wg.Wait(); close(finished) }()
	close(start)
	select {
	case <-finished:
	case <-time.After(5 * time.Minute):
		t.Fatal("two generates at once did not finish")
	}

	for i := range 2 {
		if both[i].err != nil {
			t.Fatalf("session %d beside the other: %v", i, both[i].err)
		}
		if len(both[i].ids) != n {
			t.Fatalf("session %d beside the other produced %d tokens, want %d", i, len(both[i].ids), n)
		}
		for k := range n {
			if both[i].ids[k] != solo[i].ids[k] {
				t.Fatalf("session %d token %d is %d beside the other and %d alone: the sessions wrote each other's state",
					i, k, both[i].ids[k], solo[i].ids[k])
			}
		}
	}
	// Interleaved: each session produced a token strictly inside the other's
	// reply. Held for a whole generate, one reply would end before the other's
	// first token.
	inside := func(x, y run) bool {
		for _, at := range x.times {
			if at.After(y.times[0]) && at.Before(y.times[n-1]) {
				return true
			}
		}
		return false
	}
	if !inside(both[0], both[1]) || !inside(both[1], both[0]) {
		t.Fatalf("the two generates did not interleave: a [%v..%v], b [%v..%v]",
			both[0].times[0].Format(time.StampMicro), both[0].times[n-1].Format(time.StampMicro),
			both[1].times[0].Format(time.StampMicro), both[1].times[n-1].Format(time.StampMicro))
	}
}
