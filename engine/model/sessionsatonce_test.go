package model

import (
	"slices"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/internal/testmodels"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// Two States on one tier step on the device at the same time: each is its
// own session (tier.GPU.Attach), runs in a lane of its own and submits on a
// queue of its own (jit/gpu/tier/devsess.go), and the device's shared state
// waits only where it must (jit/gpu/tier/inflight.go). These gates hold the
// sessions to the answers each gives alone, and check the configuration was
// selected: their steps overlapped, and the device saw a submission start
// while another was in flight (tier.Stats.SubsBeside).

// atOnceGen is how many tokens each session decodes in a gate.
const atOnceGen = 48

// atOnceRun is one session's decode: its greedy ids, each step's logits, and
// each step's span on the wall clock.
type atOnceRun struct {
	ids    []int32
	logits [][]float32
	spans  [][2]time.Time
}

// atOncePrompts are the sessions' prompts, one each.
func atOncePrompts(m *Model) [2][]int32 {
	return [2][]int32{
		m.Vocab.Encode("Once upon a time, there was a little girl named", true),
		m.Vocab.Encode("The big dog ran to the park and", true),
	}
}

// atOnceTier opens spec with the tuners off, so every arm runs the same
// kernels in the same reduction order.
func atOnceTier(t *testing.T, spec string, cfg func(*tier.Config)) *tier.GPU {
	t.Helper()
	opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}
	if cfg != nil {
		opts = append(opts, tier.WithConfig(cfg))
	}
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	t.Cleanup(g.Close)
	return g
}

// atOnceState places a State wholly on g and prefills prompt, returning the
// first greedy token.
func atOnceState(t *testing.T, m *Model, g *tier.GPU, seq int, prompt []int32) (*State, int32) {
	t.Helper()
	st := m.NewState(seq)
	t.Cleanup(func() { st.Close() })
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("%d of %d blocks on the device, head there %v: %s", st.devCount(), m.Cfg.NLayer,
			st.HeadOnDevice(), g.Err())
	}
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	return st, Greedy(lg)
}

// decode runs n greedy steps from next into r.
func (r *atOnceRun) decode(st *State, next int32, n int) error {
	for range n {
		t0 := time.Now()
		lg, err := st.Forward(next)
		r.spans = append(r.spans, [2]time.Time{t0, time.Now()})
		if err != nil {
			return err
		}
		next = Greedy(lg)
		r.ids = append(r.ids, next)
		r.logits = append(r.logits, slices.Clone(lg))
	}
	return nil
}

// atOnce decodes every state from its first token, one goroutine each,
// started together.
func atOnce(t *testing.T, sts []*State, next []int32, n int) []atOnceRun {
	t.Helper()
	runs := make([]atOnceRun, len(sts))
	errs := make([]error, len(sts))
	var ready, wg sync.WaitGroup
	start := make(chan struct{})
	for k := range sts {
		ready.Add(1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			ready.Done()
			<-start
			errs[k] = runs[k].decode(sts[k], next[k], n)
		}()
	}
	ready.Wait()
	close(start)
	wg.Wait()
	for k, err := range errs {
		if err != nil {
			t.Fatalf("session %d: %v", k, err)
		}
	}
	return runs
}

// sameRun fails unless got is want to the bit: the same ids and the same
// logits at every step.
func sameRun(t *testing.T, what string, got, want atOnceRun) {
	t.Helper()
	if len(got.ids) != len(want.ids) {
		t.Fatalf("%s: %d steps, alone %d", what, len(got.ids), len(want.ids))
	}
	for i := range want.ids {
		if got.ids[i] != want.ids[i] {
			t.Fatalf("%s: step %d is token %d, alone %d\n  at once %v\n  alone   %v", what, i,
				got.ids[i], want.ids[i], got.ids, want.ids)
		}
		for j := range want.logits[i] {
			if got.logits[i][j] != want.logits[i][j] {
				t.Fatalf("%s: step %d logit %d is %v at once, %v alone", what, i, j,
					got.logits[i][j], want.logits[i][j])
			}
		}
	}
}

// overlaps is how many of a's steps overlap one of b's on the wall clock.
func overlaps(a, b atOnceRun) int {
	n := 0
	for _, x := range a.spans {
		for _, y := range b.spans {
			if x[0].Before(y[1]) && y[0].Before(x[1]) {
				n++
				break
			}
		}
	}
	return n
}

// TestSessionsStepAtOnce is the gate for sessions stepping at once: two
// States of stories15M, every block and the head on one device, decode from
// two goroutines together, and each produces exactly the ids and logits it
// produces alone on the same tier. Their steps overlap on the wall clock and
// the device started submissions beside one another (SubsBeside): run with a
// lock held across the submission, the ids still agree and SubsBeside stays
// at zero, so this fails.
//
// NVIDIA's Vulkan driver runs one submission's dispatches at a time across
// queues (backend.TestQueuesRunSessionsAtOnce), so there the gate holds the
// answers and the host-side overlap; how much the device itself overlaps is
// TestSessionsOverlapOnCUDA's to measure.
func TestSessionsStepAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := atOncePrompts(m)
	const seq = 256
	for _, spec := range []string{"cuda:0", "vulkan:0", "vulkan:1"} {
		t.Run(spec, func(t *testing.T) {
			g := atOnceTier(t, spec, nil)
			var alone [2]atOnceRun
			for k := range prompts {
				st, next := atOnceState(t, m, g, seq, prompts[k])
				if err := alone[k].decode(st, next, atOnceGen); err != nil {
					t.Fatal(err)
				}
				st.Close()
			}
			before := g.Stats()
			var sts []*State
			var next []int32
			for k := range prompts {
				st, n := atOnceState(t, m, g, seq, prompts[k])
				sts, next = append(sts, st), append(next, n)
			}
			runs := atOnce(t, sts, next, atOnceGen)
			after := g.Stats()
			for k := range runs {
				sameRun(t, []string{"session 0", "session 1"}[k], runs[k], alone[k])
			}
			beside := after.SubsBeside - before.SubsBeside
			ov := overlaps(runs[0], runs[1])
			t.Logf("%s: %d steps each, ids and logits identical to each alone; %d of session 0's steps "+
				"overlapped session 1's, %d submissions started beside another in flight; the second "+
				"session's lane is %d bytes of scratch (%d with it, %d without)",
				g.Name(), atOnceGen, ov, beside, int64(after.ScratchBytes)-int64(before.ScratchBytes),
				after.ScratchBytes, before.ScratchBytes)
			if ov == 0 {
				t.Fatal("no step of one session overlapped the other's: they ran one after another")
			}
			if beside == 0 {
				t.Fatal("no submission started while another was in flight: the sessions' device calls " +
					"were serialised")
			}
		})
	}
}

// TestSessionsOverlapOnCUDA measures what TestSessionsStepAtOnce only
// selects: two sessions decoding at once on CUDA take clearly less than twice
// the wall time of one alone. stories15M's decode leaves the card mostly idle
// between its launches, which two streams fill. The arms alternate round by
// round in one process, and the median of the per-round ratios is what is
// held (RULE 2); it is its own test, so a slow round fails nothing but this
// (RULE 4).
func TestSessionsOverlapOnCUDA(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := atOncePrompts(m)
	// stories15M's context is 128 positions, so every round places fresh
	// States and times their decode alone.
	const rounds, n, warm = 10, 96, 2
	g := atOnceTier(t, "cuda:0", nil)
	seq := max(len(prompts[0]), len(prompts[1])) + n + 2
	var ratios []float64
	for r := range rounds + warm {
		a, na := atOnceState(t, m, g, seq, prompts[0])
		var solo atOnceRun
		t0 := time.Now()
		if err := solo.decode(a, na, n); err != nil {
			t.Fatal(err)
		}
		one := time.Since(t0)
		a.Close()
		a, na = atOnceState(t, m, g, seq, prompts[0])
		b, nb := atOnceState(t, m, g, seq, prompts[1])
		// One step untimed: the second session's lane is built at its first
		// step beside the first, and given back when a session detaches.
		w := atOnce(t, []*State{a, b}, []int32{na, nb}, 1)
		na, nb = w[0].ids[0], w[1].ids[0]
		t0 = time.Now()
		atOnce(t, []*State{a, b}, []int32{na, nb}, n)
		both := time.Since(t0)
		a.Close()
		b.Close()
		if r >= warm {
			ratios = append(ratios, float64(both)/float64(one))
		}
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	q1, q3 := ratios[len(ratios)/4], ratios[3*len(ratios)/4]
	t.Logf("%s: %d tokens of two sessions at once against one alone, per round: median %.2fx "+
		"(IQR %.2f-%.2f) %v", g.Name(), n, med, q1, q3, ratios)
	// Serialised, two sessions' steps take about twice one's.
	if med >= 1.6 {
		t.Fatalf("two sessions at once took %.2fx one alone: their submissions did not overlap", med)
	}
}

// TestSessionsPageAtOnce: two sessions stepping at once on a device that
// pages -- every block streamed through the fewest slots the budget leaves,
// so blocks page out and in under the other session's submissions -- produce
// exactly the ids and logits each produces alone with every block resident.
// PageIns and PageOuts during the run at once are the selection check that the
// budget bound, and SubsBeside that the sessions stepped together.
func TestSessionsPageAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := atOncePrompts(m)
	const seq, gen = 256, 24
	for _, spec := range []string{"cuda:0", "vulkan:0"} {
		t.Run(spec, func(t *testing.T) {
			g := atOnceTier(t, spec, nil)
			for li := range m.Cfg.NLayer {
				g.Stream(li, true)
			}
			place := func(prompt []int32) (*State, int32) {
				st := m.NewState(seq)
				t.Cleanup(func() { st.Close() })
				if err := st.SetDeviceLayers(g, m.Cfg.NLayer); err != nil {
					t.Fatal(err)
				}
				// The head may stay on the host: under paging a slot is
				// worth more than the projection (tier PrepHead).
				if st.devCount() != m.Cfg.NLayer {
					t.Fatalf("%d of %d blocks on the device: %s", st.devCount(), m.Cfg.NLayer, g.Err())
				}
				lg, err := st.Prefill(prompt)
				if err != nil {
					t.Fatal(err)
				}
				return st, Greedy(lg)
			}
			// Alone, each with every block resident: paging must not change
			// a bit of it.
			var alone [2]atOnceRun
			for k := range prompts {
				st, next := place(prompts[k])
				if err := alone[k].decode(st, next, gen); err != nil {
					t.Fatal(err)
				}
				st.Close()
			}
			// At once: both placed and prefilled, one step together so the
			// second's lane is built and charged, then the tightest budget
			// that still leaves two slots: every block pages out and back in
			// every token, under the other session's submissions.
			a, na := place(prompts[0])
			b, nb := place(prompts[1])
			first := atOnce(t, []*State{a, b}, []int32{na, nb}, 1)
			hi := g.Stats().BudgetUsed
			if _, err := g.SetBudget(hi); err != nil || g.Stats().Slots < 2 {
				t.Fatalf("the budget the sessions charged (%d bytes) leaves %d slot(s): %v %s", hi,
					g.Stats().Slots, err, g.Err())
			}
			lo := uint64(0)
			for hi-lo > 4<<10 {
				mid := lo + (hi-lo)/2
				if _, err := g.SetBudget(mid); err == nil && g.Stats().Slots >= 2 {
					hi = mid
				} else {
					lo = mid
				}
			}
			if _, err := g.SetBudget(hi); err != nil {
				t.Fatal(err)
			}
			if s := g.Stats().Slots; s < 2 || s >= m.Cfg.NLayer {
				t.Fatalf("%d slot(s) for %d blocks: the budget does not page", s, m.Cfg.NLayer)
			}
			before := g.Stats()
			rest := atOnce(t, []*State{a, b}, []int32{first[0].ids[0], first[1].ids[0]}, gen-1)
			after := g.Stats()
			for k := range rest {
				got := first[k]
				got.ids = append(got.ids, rest[k].ids...)
				got.logits = append(got.logits, rest[k].logits...)
				sameRun(t, []string{"session 0", "session 1"}[k], got, alone[k])
			}
			ins, outs := after.PageIns-before.PageIns, after.PageOuts-before.PageOuts
			beside := after.SubsBeside - before.SubsBeside
			t.Logf("%s: %d slot(s) for %d blocks, %d page-in(s) and %d page-out(s) while the sessions "+
				"stepped at once, %d submissions beside another; ids and logits identical to each alone "+
				"with every block resident", g.Name(), after.Slots, m.Cfg.NLayer, ins, outs, beside)
			if ins == 0 || outs == 0 {
				t.Fatalf("%d page-ins and %d page-outs: nothing paged while the sessions ran", ins, outs)
			}
			if a.devCount() != m.Cfg.NLayer || b.devCount() != m.Cfg.NLayer {
				t.Fatalf("%d and %d of %d blocks still the device's: the budget sent blocks home, which "+
					"is not paging", a.devCount(), b.devCount(), m.Cfg.NLayer)
			}
		})
	}
}

// TestSessionsEvictAtOnce: two sessions stepping at once whose histories
// outgrow the KV budget, so their oldest pages go home (kvevict.go) while the
// other session steps, and come back through the pool's free pages for every
// attention. Each is teacher-forced through the ids it chose in a run alone
// with no budget, and its logits held to that run's within the eviction gate's
// band -- as are the same two sessions taking turns a step at a time on one
// goroutine, what two sessions on one device did before they stepped at once.
// Eviction under another session's step must neither lose a page nor hand one
// to the wrong sequence. KVEvictions and KVStreamPasses during the run at once
// are the selection check. The greedy ids that differ are logged, not held:
// a streamed pass's order flips a near tie (one in two hundred on CUDA, where
// one session evicting alone already reads 8e-3).
func TestSessionsEvictAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range []string{"cuda:0", "vulkan:0"} {
		t.Run(spec, func(t *testing.T) {
			turns := evictAtOnce(t, m, spec, false)
			both := evictAtOnce(t, m, spec, true)
			t.Logf("%s: taking turns %d page(s) home, %d streamed passes, worst NMSE %.3e, %d id(s) "+
				"different; at once %d, %d, %.3e, %d (against each alone with no budget)", spec,
				turns.ev, turns.passes, turns.worst, turns.flips, both.ev, both.passes, both.worst, both.flips)
			if both.ev == 0 || both.passes == 0 {
				t.Fatalf("%d evictions and %d streamed passes at once: the budget never bound", both.ev, both.passes)
			}
			// The eviction gate's own band (TestKVEvictionRunsPastTheBudget):
			// a streamed pass sums its keys in another order, by how many
			// pages are home and when they went, which is the schedule's --
			// taking turns and at once each send different pages home at
			// different steps. A page lost or read as another sequence's is
			// tenths, not thousandths.
			for _, a := range []evictArm{turns, both} {
				if !(a.worst < evictNMSE) {
					t.Fatalf("worst logit NMSE %.3e against the runs alone, bound %.0e: a page was lost or "+
						"read as another's", a.worst, evictNMSE)
				}
			}
		})
	}
}

// evictArm is one arm of TestSessionsEvictAtOnce: the pages sent home, the
// streamed passes, and the logits' worst NMSE and greedy ids against the runs
// alone with no budget.
type evictArm struct {
	ev, passes, flips int
	worst             float64
}

// evictAtOnce runs two sessions of m on a fresh tier over spec: each alone
// with no budget, then both teacher-forced through those runs' ids under a
// budget that makes their histories evict -- at once, or taking turns a step
// at a time.
func evictAtOnce(t *testing.T, m *Model, spec string, together bool) evictArm {
	t.Helper()
	prompts := atOncePrompts(m)
	const seq, gen = 128, 100
	g := atOnceTier(t, spec, func(c *tier.Config) {
		c.KVPage = 64 // the smallest page: a short history is two
		c.StagedDecode = true
	})
	var alone [2]atOnceRun
	var first [2]int32
	for k := range prompts {
		st, next := atOnceState(t, m, g, seq, prompts[k])
		first[k] = next
		if err := alone[k].decode(st, next, gen); err != nil {
			t.Fatal(err)
		}
		st.Close()
	}
	sts := make([]*State, len(prompts))
	for k := range prompts {
		sts[k], _ = atOnceState(t, m, g, seq, prompts[k])
	}
	got := make([][][]float32, len(sts))
	errs := make([]error, len(sts))
	// step forces session k through its step i.
	step := func(k, i int) {
		if errs[k] != nil {
			return
		}
		in := first[k]
		if i > 0 {
			in = alone[k].ids[i-1]
		}
		lg, err := sts[k].Forward(in)
		if err != nil {
			errs[k] = err
			return
		}
		got[k] = append(got[k], slices.Clone(lg))
	}
	forced := func(from, to int) {
		if together {
			var wg sync.WaitGroup
			for k := range sts {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := from; i < to; i++ {
						step(k, i)
					}
				}()
			}
			wg.Wait()
		} else {
			for i := from; i < to; i++ {
				for k := range sts {
					step(k, i)
				}
			}
		}
		for k, err := range errs {
			if err != nil {
				t.Fatalf("session %d: %v (%s)", k, err, g.Err())
			}
		}
	}
	// One step before the budget is cut, so a lane the second session runs
	// in at once is built and charged first; then a little slack past what
	// the histories hold: they can grow only by sending their oldest pages
	// home.
	forced(0, 1)
	if _, err := g.SetBudget(g.Stats().BudgetUsed + 256<<10); err != nil {
		t.Fatal(err)
	}
	before := g.Stats()
	forced(1, gen)
	after := g.Stats()
	for k, st := range sts {
		if st.devCount() != m.Cfg.NLayer {
			t.Fatalf("session %d holds %d of %d blocks on the device after the run: eviction must not "+
				"move blocks home (%s)", k, st.devCount(), m.Cfg.NLayer, g.Err())
		}
	}
	r := evictArm{ev: after.KVEvictions - before.KVEvictions, passes: after.KVStreamPasses - before.KVStreamPasses}
	for k := range got {
		for i, lg := range got[k] {
			if e := logitNMSE(lg, alone[k].logits[i]); !(e <= r.worst) {
				r.worst = e
			}
			if Greedy(lg) != alone[k].ids[i] {
				r.flips++
			}
		}
	}
	return r
}
