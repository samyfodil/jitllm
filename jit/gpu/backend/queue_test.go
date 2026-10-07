package backend_test

import (
	"encoding/binary"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
	"github.com/samyfodil/jitllm/jit/gpu/ir"
)

// TestQueuesRunSessionsAtOnce: two sessions, each on a queue of its own, run
// on one device at the same time. Each launches the SAME kernel with its own
// buffers and inputs, so a launch that took the other queue's arguments, or a
// read that was not ordered after its own queue's launch, answers the other
// queue's number. The kernel is one block of a long dependent chain, so two of
// them fit on the device together: on queues that overlap, both take about as
// long as one; serialised, twice as long.
func TestQueuesRunSessionsAtOnce(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		qd, ok := d.(backend.Queued)
		if !ok {
			continue
		}
		ran++
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) { queuesAtOnce(t, d, qd) })
	}
	if ran == 0 {
		t.Skip("no backend here runs sessions on queues")
	}
}

// spinTrip is the chain's length: integers to 2^24 are exact in f32, so x0 +
// spinTrip is the exact answer.
const spinTrip = 1 << 22

func queuesAtOnce(t *testing.T, d backend.Device, qd backend.Queued) {
	b := ir.New("spin", [3]int{32, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.TID(), b.Const(ir.U32, 31))
	x0 := b.Load(ir.F32, pIn, i, 0)
	b.Loop(spinTrip)
	x := b.Phi(ir.F32, x0)
	b.SetPhi(x, b.Fma(x, b.ConstF32(1), b.ConstF32(1)))
	b.EndLoop()
	b.Store(pOut, i, x, 0)
	kern, err := d.Compile(b.Done())
	if err != nil {
		t.Skipf("compile: %v", err)
	}
	defer kern.Close()

	type lane struct {
		q       backend.Queue
		in, out backend.Buf
	}
	var lanes [2]lane
	for k := range lanes {
		q, err := qd.NewQueue()
		if err != nil {
			t.Fatalf("NewQueue: %v", err)
		}
		defer q.Close()
		in, err := d.Alloc(32 * 4)
		if err != nil {
			t.Fatal(err)
		}
		defer in.Free()
		out, err := d.Alloc(32 * 4)
		if err != nil {
			t.Fatal(err)
		}
		defer out.Free()
		lanes[k] = lane{q, in, out}
	}
	const rounds = 12
	var spanMu sync.Mutex
	var spans [2][][2]time.Time
	timing := false
	run := func(k, round int) error {
		l := lanes[k]
		x0 := float32(1000*k + round)
		in := make([]byte, 32*4)
		for j := range 32 {
			binary.LittleEndian.PutUint32(in[j*4:], math.Float32bits(x0+float32(j)))
		}
		out := make([]byte, 32*4)
		var err error
		qd.SessionOn(l.q, func(s backend.Session) {
			t0 := time.Now()
			defer func() {
				if timing {
					spanMu.Lock()
					spans[k] = append(spans[k], [2]time.Time{t0, time.Now()})
					spanMu.Unlock()
				}
			}()
			if err = s.Write(l.in, in); err != nil {
				return
			}
			if err = s.Launch(kern, 1, 32, l.in, l.out); err != nil {
				return
			}
			err = s.Read(l.out, out)
		})
		if err != nil {
			return err
		}
		for j := range 32 {
			got := math.Float32frombits(binary.LittleEndian.Uint32(out[j*4:]))
			if want := x0 + float32(j) + spinTrip; got != want {
				t.Errorf("queue %d round %d lane %d: %v, want %v -- another queue's arguments or an unordered read", k, round, j, got, want)
				return nil
			}
		}
		return nil
	}

	aloneRun := func() time.Duration {
		start := time.Now()
		for r := range rounds {
			if err := run(0, r); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(start)
	}
	// Warm the clocks before anything is timed: a laptop GPU ramps for tens
	// of milliseconds, and the first arm would carry it.
	for range 3 {
		aloneRun()
	}
	a1 := aloneRun()

	// Each session body's span, to show they overlapped: serialised, no two do.
	timing = true
	start := time.Now()
	var wg sync.WaitGroup
	for k := range lanes {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			for r := range rounds {
				if err := run(k, r); err != nil {
					t.Error(err)
					return
				}
			}
		}(k)
	}
	wg.Wait()
	both := time.Since(start)
	timing = false
	alone := min(a1, aloneRun())
	t.Logf("%d rounds on one queue %v, on two at once %v (%.2fx)", rounds, alone, both, float64(both)/float64(alone))

	overlapped := 0
	for _, a := range spans[0] {
		for _, b := range spans[1] {
			if a[0].Before(b[1]) && b[0].Before(a[1]) {
				overlapped++
			}
		}
	}
	if overlapped == 0 {
		t.Fatal("no session on one queue overlapped one on the other: they ran one after another")
	}
	// Serialised, two queues' rounds take twice one's; overlapped, about one.
	if both > alone*3/2 {
		t.Fatalf("two queues took %v against one's %v: their sessions did not run at once", both, alone)
	}
}
