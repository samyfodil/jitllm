package server

import (
	"fmt"
	"math/rand"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/samyfodil/jitllm/jit/gpu/backend"
	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// The fairness level's gates (batchfair.go). The policy's arithmetic is gated
// on rows with no model: the allotment of a step's prompt budget, the queue's
// order and the victims a time slice picks. The time slice's answer and its
// bound on a wait are gated end to end, on the host and on each GPU backend.

// fairLoop is a loop for the policy's arithmetic alone: no model, no
// goroutine.
func fairLoop(level, width int) *stepLoop {
	return &stepLoop{fair: level, width: width}
}

func promptRow(n, fed int) *row { return &row{ids: make([]int32, n), fed: fed} }

// oldAllot is promptUnits' allotment before the fairness level: oldest first,
// as much of each prompt as the budget has left.
func oldAllot(rows []*row, budget int) []int {
	out := make([]int, len(rows))
	for i, r := range rows {
		if budget <= 0 {
			break
		}
		out[i] = min(budget, len(r.ids)-r.fed)
		budget -= out[i]
	}
	return out
}

func allotOf(lp *stepLoop, rows []*row, budget int) []int {
	lp.prompts = append(lp.prompts[:0], rows...)
	lp.alloc = make([]int, len(rows))
	lp.allot(budget)
	return slices.Clone(lp.alloc)
}

// TestFairnessZeroIsFirstComeFirstServed: at level 0 the allotment is the one
// before the level existed, to the token, over random prompts and budgets; the
// queue keeps its arrival order whatever the priorities; and no row is ever a
// time slice's victim, however long it has run.
func TestFairnessZeroIsFirstComeFirstServed(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	lp := fairLoop(0, 64)
	for range 2000 {
		rows := make([]*row, 1+rng.Intn(12))
		for i := range rows {
			n := 1 + rng.Intn(700)
			rows[i] = promptRow(n, rng.Intn(n))
		}
		budget := rng.Intn(600)
		if got, want := allotOf(lp, rows, budget), oldAllot(rows, budget); !slices.Equal(got, want) {
			t.Fatalf("budget %d: level 0 allots %v, first come first served %v", budget, got, want)
		}
	}
	t0 := time.Now()
	lp.waiting = []*row{{since: t0.Add(time.Hour)}, {since: t0, prio: 1}, {since: t0.Add(-time.Hour), parked: true}}
	before := slices.Clone(lp.waiting)
	lp.order()
	if !slices.Equal(lp.waiting, before) {
		t.Fatal("level 0 reordered the queue")
	}
	lp.rows = []*row{{ids: []int32{1}, fed: 1, slice: 1 << 20, maxTokens: 1 << 30, n: 1}}
	if v := lp.victims(&row{}, 0, 0, 0); v != nil {
		t.Fatalf("level 0 time-sliced %d row(s)", len(v))
	}
}

// TestFairFeedingAdvancesEveryPrompt: above level 0 every admitted prompt is
// fed at every step until it is in, however many there are and however small
// the budget, and the allotment never passes the budget but to give each a
// token. Against level 0, a step with more
// prompts than its budget covers leaves some at zero -- the count the gate
// reads must be able to fail.
func TestFairFeedingAdvancesEveryPrompt(t *testing.T) {
	run := func(level int) (fed, had int) {
		lp := fairLoop(level, 64)
		rows := make([]*row, 8)
		for i := range rows {
			rows[i] = promptRow(100+37*i, 0)
		}
		for step := 0; step < 400; step++ {
			var live []*row
			for _, r := range rows {
				if r.prompting() {
					live = append(live, r)
				}
			}
			if len(live) == 0 {
				break
			}
			budget := 32
			a := allotOf(lp, live, budget)
			sum := 0
			for i, k := range a {
				sum += k
				if k > 0 {
					fed++
				}
				live[i].fed += k
			}
			had += len(live)
			if sum > max(budget, len(live)) {
				t.Fatalf("level %d: a step allots %d tokens, budget %d", level, sum, budget)
			}
		}
		return fed, had
	}
	for _, level := range []int{1, DefaultFairness, 100} {
		fed, had := run(level)
		t.Logf("level %d: %d of %d prompting rows fed", level, fed, had)
		if fed != had {
			t.Fatalf("level %d: %d of %d prompting rows fed", level, fed, had)
		}
	}
	if fed, had := run(0); fed == had {
		t.Fatalf("VIOLATION level 0 fed every prompting row (%d of %d): the count cannot tell", fed, had)
	}
}

// TestFairFeedingEndsAShortPromptInOneChunk: a prompt whose rest fits its
// turn is fed whole, so its chunk ends at its last row and wants its logits.
func TestFairFeedingEndsAShortPromptInOneChunk(t *testing.T) {
	lp := fairLoop(100, 64)
	rows := []*row{promptRow(500, 0), promptRow(5, 0), promptRow(500, 0)}
	a := allotOf(lp, rows, 64)
	if a[1] != 5 {
		t.Fatalf("a 5-token prompt beside two long ones got %d tokens of a 64-token step: %v", a[1], a)
	}
}

// TestFairnessLevelIsMonotonic: the quantum falls and the even share rises
// with the level; priority's head start shrinks.
func TestFairnessLevelIsMonotonic(t *testing.T) {
	if fairQuantum(0) != 0 || fairQuantum(1) != fairQuantumMax || fairQuantum(100) != fairQuantumMin {
		t.Fatalf("quantum: %d at 0, %d at 1, %d at 100", fairQuantum(0), fairQuantum(1), fairQuantum(100))
	}
	for f := 2; f <= 100; f++ {
		if fairQuantum(f) > fairQuantum(f-1) || fairShare(f, 512) < fairShare(f-1, 512) || fairBoost(f) > fairBoost(f-1) {
			t.Fatalf("level %d: quantum %d (%d), share %d (%d), boost %v (%v)", f, fairQuantum(f), fairQuantum(f-1),
				fairShare(f, 512), fairShare(f-1, 512), fairBoost(f), fairBoost(f-1))
		}
	}
}

// TestFairQueueOrder: a parked row resumes before a request that arrived
// after it was parked, and after one that arrived before; a high-priority
// request goes ahead of normal ones that have waited less than its head start
// and behind those that have waited longer, so age bounds what priority can
// cost anyone. The violation, the queue in arrival order, fails each.
func TestFairQueueOrder(t *testing.T) {
	t0 := time.Now()
	boost := fairBoost(DefaultFairness)
	early := &row{since: t0.Add(-time.Second)}
	parked := &row{since: t0, parked: true}
	late := &row{since: t0.Add(time.Second)}
	oldNormal := &row{since: t0.Add(-2 * boost)}
	high := &row{since: t0.Add(2 * time.Second), prio: 1}
	arrival := []*row{late, parked, high, early, oldNormal}
	want := []*row{oldNormal, high, early, parked, late}
	check := func(lp *stepLoop) error {
		lp.waiting = slices.Clone(arrival)
		lp.order()
		if !slices.Equal(lp.waiting, want) {
			return fmt.Errorf("order %v, want %v", names(lp.waiting, arrival), names(want, arrival))
		}
		return nil
	}
	if err := check(fairLoop(DefaultFairness, 4)); err != nil {
		t.Fatal(err)
	}
	if err := check(fairLoop(0, 4)); err == nil {
		t.Fatal("VIOLATION arrival order passed the queue-order gate")
	}
}

func names(rs, of []*row) string {
	var s []string
	for _, r := range rs {
		s = append(s, strconv.Itoa(slices.Index(of, r)))
	}
	return strings.Join(s, ",")
}

// TestFairVictims: a time slice parks only decoding rows past their quantum,
// the lowest priority and the longest served first, and only as many as the
// waiting request needs; none when parking every one would not admit it.
func TestFairVictims(t *testing.T) {
	q := fairQuantum(100)
	dec := func(slice, prio int) *row {
		return &row{ids: []int32{1}, fed: 1, slice: slice, prio: prio, n: 1, maxTokens: 1 << 20}
	}
	young, old, older, vip, prompting := dec(q-1, 0), dec(q, 0), dec(3*q, 0), dec(5*q, 1), &row{ids: []int32{1, 2}, slice: 9 * q}
	lp := fairLoop(100, 4)
	lp.rows = []*row{young, old, older, vip, prompting}
	lp.width = 5
	v := lp.victims(&row{}, 0, 0, 0)
	if len(v) != 1 || v[0] != older {
		t.Fatalf("one free row wanted: victims %v", v)
	}
	lp.width = 3
	v = lp.victims(&row{}, 0, 0, 0)
	if len(v) != 3 || v[0] != older || v[1] != old || v[2] != vip {
		t.Fatalf("three free rows wanted: %d victims", len(v))
	}
	lp.width = 2
	if v = lp.victims(&row{}, 0, 0, 0); v != nil {
		t.Fatalf("four free rows wanted, three eligible: %d parked for nothing", len(v))
	}
}

// fairRequests is the load the end-to-end gates send: more greedy requests
// than rows, each long enough to outlast several quanta.
func fairRequests(n, maxTokens int) []*v1.GenerateRequest {
	var out []*v1.GenerateRequest
	for i := range n {
		out = append(out, &v1.GenerateRequest{ModelId: "dev", Prompt: text(hostPrompts[i%len(hostPrompts)]),
			MaxTokens: int32(maxTokens), IgnoreEos: true})
	}
	return out
}

// TestTimeSlicedRowsAnswerAsAlone: with two rows and six requests at level
// 100, rows are parked and resumed many times (the parks counted), and every
// request answers what it answers alone -- on the host and on each GPU
// backend, where a park sends the row's history home and a resume brings it
// back. Against a park that loses the logits the row held, rows part from
// their alone runs.
func TestTimeSlicedRowsAnswerAsAlone(t *testing.T) {
	arms := []struct{ name, dev string }{{"host", ""}, {"cuda", "cuda:0"}}
	if infos, err := backend.VulkanDevices(); err == nil {
		for _, in := range infos {
			if in.Compute && !in.Software && !in.Unified {
				arms = append(arms, struct{ name, dev string }{"vulkan", "vulkan:" + strconv.Itoa(in.Index)})
				break
			}
		}
	} else {
		t.Logf("no Vulkan (%v): the Vulkan arm did not run", err)
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			for _, violate := range []bool{false, true} {
				parks, parted := timeSliced(t, arm.dev, violate)
				t.Logf("%s violate=%v: %d parks, %d request(s) parted from alone", arm.name, violate, parks, parted)
				if parks == 0 {
					t.Fatalf("%s: no row was parked: the gate compared unsliced rows", arm.name)
				}
				if !violate && parted > 0 {
					t.Fatalf("%s: %d time-sliced request(s) parted from alone", arm.name, parted)
				}
				if violate && parted == 0 {
					t.Fatalf("VIOLATION %s: a park that loses the logits passed", arm.name)
				}
			}
		})
	}
}

func fairEngine(t *testing.T, dev string, cfg Config) (*Engine, *LoadedModel, clients) {
	t.Helper()
	if dev == "" {
		return hostBatchEngine(t, deviceModel, cfg)
	}
	path := modelPath(t, deviceModel)
	cfg.Version = "test"
	if cfg.DefaultMaxSeq == 0 {
		cfg.DefaultMaxSeq = 512
	}
	e := New(cfg)
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{dev}, Sessions: 8})
	if err != nil {
		t.Fatalf("loading %s onto %s: %v", deviceModel, dev, err)
	}
	if lm.loop == nil || lm.gpu == nil {
		t.Fatalf("%s on %s: no device step loop", deviceModel, dev)
	}
	requireWholeOnDevice(t, e, lm)
	return e, lm, serveEngine(t, e)
}

func timeSliced(t *testing.T, dev string, violate bool) (parks int64, apart int) {
	level := 100
	e, lm, c := fairEngine(t, dev, Config{MaxBatchRows: 2, Fairness: &level})
	lm.loop.parkDropsLogits = violate
	reqs := fairRequests(6, 3*fairQuantum(level))
	got := concurrently(t, e, lm, c, reqs)
	parks = lm.loop.stats.parks.Load()
	if lm.gpu != nil {
		home, freed := e.preempt.blocksHome.Load(), lm.loop.stats.parkFreed.Load()
		t.Logf("%s: parking brought %d block(s) home and freed %d positions of the card's room", dev, home, freed)
		if home == 0 || freed <= 0 {
			t.Fatalf("%s: %d parks moved %d block(s) home and freed %d positions: nothing left the card", dev, parks, home, freed)
		}
	}
	if lm.loop.stats.resumes.Load() != parks {
		t.Fatalf("%d parks, %d resumes: a parked row never came back", parks, lm.loop.stats.resumes.Load())
	}
	for i, r := range reqs {
		want := complete(c, r)
		if want.err != nil {
			t.Fatal(want.err)
		}
		if parted(t, lm, hostPrompts[i%len(hostPrompts)], got[i].ids, want.ids) >= 0 {
			apart++
		}
	}
	return parks, apart
}

// TestTimeSlicesBoundTheWait: six requests on two rows. At level 100 every
// request has its first token within the bound a round of quanta gives --
// the rows ahead of it each run one quantum (that a parked row resumes before
// a later arrival and after an earlier one is TestFairQueueOrder's). At level 0 the requests behind the first two wait for whole
// generations, past the bound: the violation the gate must catch.
func TestTimeSlicesBoundTheWait(t *testing.T) {
	const n, width, gen = 6, 2, 96
	run := func(level int) (worst int64, bound int64) {
		e, lm, c := fairEngine(t, "", Config{MaxBatchRows: width, Fairness: &level})
		var mu sync.Mutex
		var starts []int64
		lm.loop.onStarted = func(r *row) {
			mu.Lock()
			starts = append(starts, r.startStep)
			mu.Unlock()
		}
		concurrently(t, e, lm, c, fairRequests(n, gen))
		q := int64(fairQuantum(100))
		// A wave of width rows runs a quantum before the next wave is in, a
		// prompt a step or two each: (n/width - 1) waves ahead of the last.
		bound = int64(n/width-1)*(q+4) + 4
		return slices.Max(starts), bound
	}
	worst, bound := run(100)
	t.Logf("level 100: the last first token at step %d, bound %d", worst, bound)
	if worst > bound {
		t.Fatalf("level 100: a request's first token waited %d steps, bound %d", worst, bound)
	}
	if worst, _ := run(0); worst <= bound {
		t.Fatalf("VIOLATION level 0: the last first token at step %d, within the bound %d", worst, bound)
	}
}
