package backend

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/samyfodil/jitllm/jit/gpu/cuda"
	"github.com/samyfodil/jitllm/jit/gpu/ir"
	"github.com/samyfodil/jitllm/jit/gpu/ptx"
)

// cudaDev owns the device from one goroutine and serialises every call onto it.
//
// A CUDA context is current per OS thread, and goroutines migrate between
// threads (t.Run subtests, sched.Pool workers), which yields
// CUDA_ERROR_INVALID_CONTEXT. So one owner goroutine holds a locked thread with
// the context current, and other callers either post a closure to it or run
// inline with the context bound (see do and inline).
type cudaDev struct {
	allocCount
	d *cuda.Device
	// mu guards reqs against Close. do reads it and Close nils it, and without
	// this that is a data race whose benign-looking outcome is a send on a
	// closed channel -- a panic from the caller's goroutine, not an error.
	//
	// It is NOT held across the request: do waits for the owner goroutine to
	// finish, and holding a lock across that would serialise every device call
	// behind every other one and deadlock a request that itself calls do.
	mu   sync.Mutex
	reqs chan func()
	// owner is the OS thread the owner goroutine is locked to (-1 where the
	// platform cannot say); see do.
	owner int64
	// cap is the capture stream, created on first use and reused. A stream is
	// a resource rather than a mode: nothing reads it except the session that
	// is capturing, and that session runs on the owner goroutine.
	cap   *cuda.Stream
	noCap bool // stream creation failed; do not keep asking
	// tev is a ring of (before, after) event pairs around the last three
	// replays, and ti counts replays timed; see timeReplay.
	tev [3][2]*cuda.Event
	ti  int
	// lastMarks is the last timed replay's marks, settled at the next one.
	lastMarks []kernMark
	// run serialises inline calls (see do); in is the thread holding it, so a
	// call made from inside one runs straight through instead of waiting on
	// itself.
	run sync.Mutex
	in  atomic.Int64
	// posted routes every call through the owner goroutine instead of running
	// it inline; Opts.CUDAPosted, fixed at open.
	posted bool
	// reqFree is Session's free list; see Session.
	reqMu   sync.Mutex
	reqFree []*sessReq
}

// openCUDA is what backend.Open uses: the first card.
func openCUDA(o Opts) (Device, error) { return OpenCUDAWith(0, o) }

// CUDACount is how many CUDA devices the driver sees, without opening one.
// Opening costs a context, a locked OS thread and an owner goroutine, so a
// caller asks this first and opens only the ordinals it chose.
func CUDACount() (int, error) { return cuda.Count() }

// OpenCUDA opens one CUDA device by ordinal, with its own context and its own
// owner goroutine. One owner per device keeps each context current on its own
// thread without pushing or popping per call; only the driver binding is
// shared.
func OpenCUDA(ord int) (Device, error) { return OpenCUDAWith(ord, Opts{}) }

// OpenCUDAWith is OpenCUDA with this device's open choices.
func OpenCUDAWith(ord int, opts Opts) (Device, error) {
	reqs := make(chan func(), 1)
	type opened struct {
		d   *cuda.Device
		tid int64
		err error
	}
	out := make(chan opened, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		d, err := cuda.OpenDevice(ord)
		out <- opened{d, threadID(), err}
		if err != nil {
			return
		}
		for {
			f, ok := spinRecv(reqs, ownerSpin)
			if !ok {
				return
			}
			f()
		}
	}()
	o := <-out
	if o.err != nil {
		return nil, o.err
	}
	return &cudaDev{d: o.d, reqs: reqs, owner: o.tid, posted: opts.CUDAPosted}, nil
}

// Session batches a sequence of device calls into ONE ownership hop.
//
// The request do runs and the session handed to f come off a free list rather
// than a closure and a fresh cudaSession: both escape (do may post to the
// owner, f is the caller's), so building them per call was two heap objects
// on every decode token. Nested sessions take a request each.
func (c *cudaDev) Session(f func(Session)) {
	c.reqMu.Lock()
	var r *sessReq
	if n := len(c.reqFree); n > 0 {
		r, c.reqFree = c.reqFree[n-1], c.reqFree[:n-1]
	}
	c.reqMu.Unlock()
	if r == nil {
		r = &sessReq{c: c}
		r.run = r.call
	}
	r.f = f
	c.do(r.run)
	r.f = nil
	c.reqMu.Lock()
	c.reqFree = append(c.reqFree, r)
	c.reqMu.Unlock()
}

// sessReq is one Session call: the caller's body, the session it is handed,
// and the function do runs, built once per request.
type sessReq struct {
	c   *cudaDev
	f   func(Session)
	s   cudaSession
	run func()
}

func (r *sessReq) call() {
	r.s = cudaSession{c: r.c}
	r.f(&r.s)
}

// cudaSession's methods run on the owner goroutine already, so they call
// straight through. It is a distinct type rather than a flag on cudaDev because
// a flag would have to be read by a goroutine that is not the owner, and a
// misread would run a driver call on the wrong thread -- the exact bug this
// whole mechanism exists to prevent.
//
// It is a POINTER so that BeginRecord can redirect launches onto the capture
// stream for the length of a recording. That state lives here rather than on
// cudaDev for the reason above: the value is made, mutated and read inside one
// Session that is already running on the owner.
type cudaSession struct {
	c *cudaDev
	// stream is where launches go: zero is the legacy stream, non-zero only
	// while a capture is open.
	stream cuda.CUstream
	// marks collects a timed capture's per-launch events; nil otherwise.
	// It points at recMarks, which holds them between BeginRecord and
	// EndRecord.
	marks    *[]kernMark
	recMarks []kernMark
}

func (*cudaSession) Write(b Buf, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	return b.(*cudaBuf).b.Write(unsafe.Pointer(&p[0]), len(p))
}

func (*cudaSession) WriteAt(b Buf, off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	return b.(*cudaBuf).b.WriteAt(off, unsafe.Pointer(&p[0]), len(p))
}

func (*cudaSession) Read(b Buf, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	countRead(len(p))
	return b.(*cudaBuf).b.Read(unsafe.Pointer(&p[0]), len(p))
}

// Launch fills the module's own argument array in place, so a launch
// allocates nothing. The scratch is per-module state, safe because every call
// here is on one thread; see cuda.Module.args.
func (s *cudaSession) Launch(k Kernel, groups, width int, bufs ...Buf) error {
	ck := k.(*cudaKern)
	if err := checkWidth("ptx", ck.name, ck.group, width); err != nil {
		return err
	}
	m := ck.m
	args := m.Args(len(bufs))
	if args == nil {
		return fmt.Errorf("backend: cuda: %d kernel arguments is too many", len(bufs))
	}
	for i, b := range bufs {
		// A nil buffer is an error, not a panic: every caller can fall back to
		// the host, and a panic here would take a server down.
		cb, ok := b.(*cudaBuf)
		if !ok || cb == nil {
			return fmt.Errorf("backend: cuda: kernel argument %d of %d is nil", i, len(bufs))
		}
		args[i] = cb.b.Arg()
	}
	// No Sync: see the Session interface. cuMemcpyDtoH on the legacy stream
	// waits for prior work on it and on every blocking stream, so the readback
	// orders this whether it ran here or inside a replayed graph.
	if err := m.LaunchArgs(s.stream, groups, width, args); err != nil {
		return err
	}
	if s.marks != nil {
		// Inside a timed capture: an event NODE after the launch, so a replay
		// stamps each kernel's end with no host in the loop.
		if e, err := cuda.NewEvent(); err == nil && e.RecordNode(s.stream) == nil {
			*s.marks = append(*s.marks, kernMark{e, ck.name})
		}
	}
	return nil
}

// cudaKernTiming arms an event pair around every launch outside a capture;
// see SetCUDAKernelTiming.
var cudaKernTiming atomic.Bool

// SetCUDAKernelTiming arms per-kernel device-clock timing on CUDA: a capture
// records an event node after every launch, and each replay's kernels are read
// back one replay later -- the SAME measurement scripts/kshim.c takes of
// llama.cpp's graphs, so the two engines' kernels can be set side by side. It
// applies to graphs recorded after it is armed. A measurement instrument,
// default off: the event nodes cost device time of their own.
func SetCUDAKernelTiming(on bool) { cudaKernTiming.Store(on) }

// kernMark is one captured launch's end event: kernel i's device time is its
// event minus kernel i-1's (the capture's first mark is a bare start event).
type kernMark struct {
	ev   *cuda.Event
	name string
}

// settleMarks adds a completed replay's per-kernel times to kernTimes.
func settleMarks(m []kernMark) {
	for i := 1; i < len(m); i++ {
		if ms, err := cuda.Elapsed(m[i-1].ev, m[i].ev); err == nil {
			kernTimes.add(m[i].name, ms*1e3)
		}
	}
}

// kernTimes sums device microseconds and counts per kernel name.
var kernTimes kernTotals

type kernTotals struct {
	sync.Mutex
	us map[string]float64
	n  map[string]int
}

func (t *kernTotals) add(name string, us float32) {
	t.Lock()
	defer t.Unlock()
	if t.us == nil {
		t.us, t.n = map[string]float64{}, map[string]int{}
	}
	t.us[name] += float64(us)
	t.n[name]++
}

// CUDAKernelTimes returns, and clears, the per-kernel device microseconds and
// launch counts SetCUDAKernelTiming collected.
func CUDAKernelTimes() (us map[string]float64, n map[string]int) {
	kernTimes.Lock()
	defer kernTimes.Unlock()
	us, n = kernTimes.us, kernTimes.n
	kernTimes.us, kernTimes.n = nil, nil
	return us, n
}

func (*cudaSession) Sync() error { return cuda.Sync() }

// BeginRecord starts capturing this session's launches into a graph instead
// of running them. It redirects the launches rather than intercepting them, so
// a caller records with the same code it would otherwise run and the two
// cannot drift apart (see Record).
func (s *cudaSession) BeginRecord() error {
	if !cuda.GraphsAvailable() {
		return fmt.Errorf("backend: cuda: this driver has no stream capture")
	}
	if s.c.cap == nil {
		if s.c.noCap {
			return fmt.Errorf("backend: cuda: no capture stream")
		}
		st, err := s.c.d.NewStream()
		if err != nil {
			s.c.noCap = true
			return err
		}
		s.c.cap = st
	}
	if err := s.c.cap.BeginCapture(); err != nil {
		return err
	}
	s.stream = s.c.cap.Handle()
	s.recMarks = nil
	if cudaKernTiming.Load() && cuda.EventsAvailable() {
		if e, err := cuda.NewEvent(); err == nil && e.RecordNode(s.stream) == nil {
			s.recMarks = []kernMark{{e, ""}}
			s.marks = &s.recMarks
		}
	}
	return nil
}

// EndRecord closes the capture BeginRecord opened and returns its graph.
func (s *cudaSession) EndRecord() (Recording, error) {
	marks := s.recMarks
	s.stream, s.marks, s.recMarks = 0, nil, nil
	// EndCapture runs even when the launches failed: a stream left capturing
	// refuses every later launch on it, and its error is the informative one.
	g, err := s.c.cap.EndCapture()
	if err != nil {
		return nil, err
	}
	return &cudaGraph{g: g, dev: s.c, marks: marks}, nil
}

func (s *cudaSession) Replay(r Recording) error {
	cg := r.(*cudaGraph)
	if len(cg.marks) > 0 {
		// The previous timed replay is done: decode read its token back.
		settleMarks(s.c.lastMarks)
		s.c.lastMarks = cg.marks
	}
	if !cudaTiming.Load() || !cuda.EventsAvailable() {
		return cg.g.Launch(s.c.cap)
	}
	return s.c.timeReplay(cg.g)
}

// timeReplay is Replay between two events on the capture stream. Replay i's
// pair is read at replay i+1: its graph is done by then, because decode reads
// the token back between them (a legacy-stream copy waits for this stream),
// and so is replay i-1's after-event, which makes the gap between two graphs
// -- the device idle while the host takes the token and issues the next --
// a device-clock number too.
func (c *cudaDev) timeReplay(g *cuda.Graph) error {
	cur := &c.tev[c.ti%3]
	for i := range cur {
		if cur[i] == nil {
			e, err := cuda.NewEvent()
			if err != nil {
				return g.Launch(c.cap)
			}
			cur[i] = e
		}
	}
	if c.ti >= 1 {
		prev := c.tev[(c.ti-1)%3]
		busy, err := cuda.Elapsed(prev[0], prev[1])
		gap := float32(-1)
		if err == nil && c.ti >= 2 {
			if g2, e2 := cuda.Elapsed(c.tev[(c.ti-2)%3][1], prev[0]); e2 == nil {
				gap = g2
			}
		}
		if err == nil {
			graphTimes.add(busy, gap)
		}
	}
	if err := cur[0].Record(c.cap); err != nil {
		return err
	}
	if err := g.Launch(c.cap); err != nil {
		return err
	}
	c.ti++
	return cur[1].Record(c.cap)
}

// graphTimes holds the last replays' device times, in microseconds.
var graphTimes replayTimes

type replayTimes struct {
	sync.Mutex
	busy, gap []float32
}

func (t *replayTimes) add(busyMs, gapMs float32) {
	t.Lock()
	defer t.Unlock()
	if len(t.busy) >= 1<<16 {
		t.busy, t.gap = t.busy[:0], t.gap[:0]
	}
	t.busy = append(t.busy, busyMs*1e3)
	t.gap = append(t.gap, gapMs*1e3)
}

// CUDAGraphTimes returns, and clears, the device-clock time of every timed
// replay and the idle gap before it (-1 where there was no previous one), in
// microseconds. See SetCUDATiming.
func CUDAGraphTimes() (busy, gap []float32) {
	graphTimes.Lock()
	defer graphTimes.Unlock()
	busy, gap = graphTimes.busy, graphTimes.gap
	graphTimes.busy, graphTimes.gap = nil, nil
	return busy, gap
}

// cudaGraph is a Recording. Free hops to the owner goroutine the way cudaBuf's
// does, because destroying a graph is a driver call like any other while the
// caller releasing one is the tier, on whatever goroutine decode runs on.
type cudaGraph struct {
	g     *cuda.Graph
	dev   *cudaDev
	marks []kernMark // SetCUDAKernelTiming's event nodes, if it was armed
}

func (g *cudaGraph) Free() {
	g.dev.do(func() {
		g.g.Destroy()
		if len(g.marks) > 0 && len(g.dev.lastMarks) > 0 && &g.dev.lastMarks[0] == &g.marks[0] {
			settleMarks(g.dev.lastMarks) // freed while pending: its last replay is done by now
			g.dev.lastMarks = nil
		}
		for _, m := range g.marks {
			m.ev.Destroy()
		}
	})
}

// errCUDAClosed is what a call that returns something answers on a closed
// device. Each such call starts its error at this and lets f overwrite it, so
// the dropped call below is a refusal rather than a success with nothing done
// -- an Alloc that answered a buffer with no memory behind it, or a Read that
// left the caller's bytes as they were.
var errCUDAClosed = errors.New("backend: cuda: the device is closed")

// do runs f on the device-owning goroutine and waits for it.
//
// A closed device is a no-op, not a hang: Close nils c.reqs, and a send on a nil
// channel blocks forever. Dropping the work is correct for a release, because
// cuCtxDestroy already reclaimed every allocation, stream and module of the
// context; a call that answers something reports errCUDAClosed instead.
//
// A call from the owner's own thread (that is, from inside a Session) runs
// inline, because posting it would wait on itself forever. Such calls are
// counted (OwnerCalls) so gates can hold the tier to issuing none.
//
// A panic inside a posted request is re-raised on the caller, so it names the
// caller's stack and does not kill the owner.
func (c *cudaDev) do(f func()) {
	tid := threadID()
	if tid >= 0 && (tid == c.owner || tid == c.in.Load()) {
		ownerCalls.Add(1)
		f()
		return
	}
	if !c.posted && tid >= 0 && c.inline(f) {
		return
	}
	c.mu.Lock()
	reqs := c.reqs
	c.mu.Unlock()
	if reqs == nil {
		return
	}
	postedCalls.Add(1)
	done := make(chan struct{})
	var pv any
	// The send is on the SNAPSHOT, not c.reqs: Close cannot nil a local, and it
	// cannot close the channel either while its own lock below is held.
	sent := func() (ok bool) {
		defer func() {
			// A Close that won the race closed the channel between the snapshot
			// and the send. The work is moot (cuCtxDestroy reclaimed it), and a
			// panic out of a Free would kill the process.
			if recover() != nil {
				ok = false
			}
		}()
		reqs <- func() {
			defer func() { pv = recover(); close(done) }()
			f()
		}
		return true
	}()
	if !sent {
		return
	}
	spinWait(done, callerSpin)
	if pv != nil {
		panic(pv)
	}
}

// inline runs f on the calling thread with the device's context made current
// there, under run, and reports false only when the device is closed.
//
// Posting to the owner costs two goroutine wakeups per Session, a measurable
// share of a device decode token. A context can be current on any number of
// threads; what the owner provided was serialisation, and run gives that without
// moving the work. The thread is locked for the call so the context stays
// current.
func (c *cudaDev) inline(f func()) bool {
	c.run.Lock()
	defer c.run.Unlock()
	c.mu.Lock()
	open := c.reqs != nil
	c.mu.Unlock()
	if !open {
		return false
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tid := threadID()
	c.in.Store(tid)
	defer c.in.Store(-1)
	if err := c.d.Bind(); err != nil {
		return false
	}
	f()
	return true
}

// allocRefused counts allocations a device refused for want of memory.
var allocRefused atomic.Int64

// AllocRefused is how many device allocations were refused because the card
// was full -- by this process or, on a shared card, by another one. Placement
// declines around it; a gate that needs a model on the card reads it to tell
// "the card is too small here" from a failure.
func AllocRefused() int64 { return allocRefused.Load() }

// postedCalls counts device calls that took the hop to the owner goroutine.
var postedCalls atomic.Int64

// PostedCalls is how many CUDA calls were posted to an owner goroutine rather
// than run inline; a gate holds it still across a Session.
func PostedCalls() int64 { return postedCalls.Load() }

// ownerCalls counts device calls made from inside a Session, which do runs
// inline rather than deadlocking on; see OwnerCalls.
var ownerCalls atomic.Int64

// OwnerCalls is how many device calls were made from inside a Session. The
// backend completes them, so this is a hygiene count, not an error count.
func OwnerCalls() int64 { return ownerCalls.Load() }

// Both ends of the ownership hop poll before they park, because waking a
// parked goroutine costs tens of microseconds with the card idle meanwhile
// (CUDA's own synchronise spins for the same reason). The owner polls for its
// next request for ownerSpin, and a caller polls for its result for
// callerSpin, which is longer than a decode token.
//
// Each poll yields: the owner still needs a P, and a caller spinning without
// yielding starves it until preemption.
//
// The windows are fixed and cost one host core while a device call runs; make
// them options if a host needs that core back during device decode.
const (
	ownerSpin  = 2 * time.Millisecond
	callerSpin = 50 * time.Millisecond
)

func spinRecv(c chan func(), d time.Duration) (func(), bool) {
	t0 := time.Now()
	for time.Since(t0) < d {
		select {
		case f, ok := <-c:
			return f, ok
		default:
			runtime.Gosched()
		}
	}
	f, ok := <-c
	return f, ok
}

func spinWait(done chan struct{}, d time.Duration) {
	t0 := time.Now()
	for time.Since(t0) < d {
		select {
		case <-done:
			return
		default:
			runtime.Gosched()
		}
	}
	<-done
}

// Name is ordinal, card and PTX target together -- "#0 <card name> (sm_86)".
//
// Target alone is a capability, not an identity: tier.deviceKey hashes
// API()+"/"+Name() for its probe cache, and two cards share a target. The
// ordinal separates two identical cards, which share a name too.
func (c *cudaDev) Name() string { return c.d.Label() }
func (c *cudaDev) API() string  { return "ptx" }

// Ordinal is which CUDA device this is, for a caller placing work across
// several. It is an OPTIONAL interface for the same reason Mem is: the other
// two backends enumerate differently, and a caller that does not need it must
// not have to know.
func (c *cudaDev) Ordinal() int { return c.d.Ordinal() }

// Slots comes from the driver: SM count times max threads per SM.
func (c *cudaDev) Slots() int { return c.d.Slots() }

// Mem is free and total device memory. It is an optional interface rather than
// a Device method so a backend that cannot say need not invent a number.
// CUDA's figure is machine-wide; Vulkan's (VK_EXT_memory_budget) accounts for
// other processes; Metal's counts only this process and is a slice of host RAM.
//
// cuMemGetInfo answers for the current context, so cuda.Device.Mem binds this
// device's context before asking.
func (c *cudaDev) Mem() (free, total uint64, err error) {
	err = errCUDAClosed
	c.do(func() { free, total, err = c.d.Mem() })
	return
}

// Close destroys the context on the owner goroutine and only then releases it,
// so the OS thread that held the context outlives the context; see
// cuda.Device.Close.
func (c *cudaDev) Close() {
	// No inline call may be running on the context while it is destroyed.
	c.run.Lock()
	defer c.run.Unlock()
	c.mu.Lock()
	if c.reqs == nil {
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()
	done := make(chan struct{})
	c.reqs <- func() {
		// The stream goes before the context that owns it.
		c.cap.Destroy()
		c.d.Close()
		close(done)
	}
	<-done
	c.mu.Lock()
	close(c.reqs)
	c.reqs = nil
	c.mu.Unlock()
}

func (c *cudaDev) Alloc(n int) (Buf, error) {
	var b *cuda.Buffer
	err := errCUDAClosed
	c.do(func() { b, err = c.d.Alloc(n) })
	if err != nil {
		if errors.Is(err, cuda.ErrOutOfMemory) {
			allocRefused.Add(1)
		}
		return nil, err
	}
	return &cudaBuf{b: b, dev: c, acc: c.take(n, cudaFootprint(n))}, nil
}

// GuaranteedLanes: a CUDA warp is 32 threads on every architecture NVIDIA has
// shipped, WARP_SIZE is 32 in every CUDA toolkit, and ptx.Lower emits
// shfl.sync with a full member mask and maxLane 31 -- one segment, the whole
// warp. So 32 is guaranteed by the target and nothing else is offered: a width
// this backend cannot produce is refused rather than approximated.
//
// It is not read from the device (CU_DEVICE_ATTRIBUTE_WARP_SIZE): what matters
// is the width the lowerer assumed, a compile-time 32. A device reporting
// anything else would need a different lowering, not a different check.
func (c *cudaDev) GuaranteedLanes(w int) (bool, string) {
	if w == ir.SubgroupLanes {
		return true, ""
	}
	return false, fmt.Sprintf("a CUDA warp is %d lanes and ptx emits shfl.sync over exactly "+
		"those; %d cannot be asked for", ir.SubgroupLanes, w)
}

func (c *cudaDev) Compile(k *ir.Kernel) (Kernel, error) {
	src, err := ptx.Lower(k, c.d.Target)
	if err != nil {
		return nil, err
	}
	var m *cuda.Module
	err = errCUDAClosed
	c.do(func() { m, err = c.d.Load(src, k.Name) })
	if err != nil {
		return nil, err
	}
	return &cudaKern{m: m, dev: c, name: k.Name, group: k.Group}, nil
}

type cudaBuf struct {
	b   *cuda.Buffer
	dev *cudaDev
	acc *counted
}

// Write and Read are real device copies here, unlike the mapped backends. That
// asymmetry is why Buf is not a []byte.
func (b *cudaBuf) Write(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	err := errCUDAClosed
	b.dev.do(func() { err = b.b.Write(unsafe.Pointer(&p[0]), len(p)) })
	return err
}

func (b *cudaBuf) WriteAt(off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	err := errCUDAClosed
	b.dev.do(func() { err = b.b.WriteAt(off, unsafe.Pointer(&p[0]), len(p)) })
	return err
}

func (b *cudaBuf) Read(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	countRead(len(p))
	err := errCUDAClosed
	b.dev.do(func() { err = b.b.Read(unsafe.Pointer(&p[0]), len(p)) })
	return err
}

func (b *cudaBuf) Free() {
	b.dev.do(func() { b.b.Free() })
	b.acc.give()
}

// Copy is cuMemcpyDtoD on the legacy stream, which waits for every launch
// before it on that stream and on every blocking stream, followed by a
// context synchronise so the bytes have landed when it returns.
func (c *cudaDev) Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error {
	db, ok := dst.(*cudaBuf)
	sb, ok2 := src.(*cudaBuf)
	if !ok || !ok2 || db == nil || sb == nil || db.dev != c || sb.dev != c {
		return fmt.Errorf("backend: cuda: a copy between buffers this device did not allocate")
	}
	if err := copyRange(db.b.Len(), dstOff, sb.b.Len(), srcOff, n, db.b == sb.b); err != nil || n == 0 {
		return err
	}
	err := errCUDAClosed
	c.do(func() {
		if err = cuda.CopyDtoD(db.b, dstOff, sb.b, srcOff, n); err == nil {
			err = cuda.Sync()
		}
	})
	return err
}

type cudaKern struct {
	m    *cuda.Module
	dev  *cudaDev
	name string // ir.Kernel.Name, for SetCUDAKernelTiming and checkWidth
	// group is the declared workgroup. CUDA runs the launch's width, so a
	// mismatch computes here and not on Vulkan; checkWidth refuses it on both.
	group [3]int
}

// Launch synchronises before returning. cuLaunchKernel is asynchronous and the
// Vulkan and Metal arms block, so without this the interface would mean two
// things per backend, and timing a launch would measure only the enqueue.
func (k *cudaKern) Launch(groups, width int, bufs ...Buf) error {
	if err := checkWidth("ptx", k.name, k.group, width); err != nil {
		return err
	}
	cb := make([]*cuda.Buffer, len(bufs))
	for i, b := range bufs {
		cb[i] = b.(*cudaBuf).b
	}
	err := errCUDAClosed
	k.dev.do(func() {
		if err = k.m.Launch(groups, width, cb...); err == nil {
			err = cuda.Sync()
		}
	})
	return err
}

func (k *cudaKern) Close() { k.dev.do(func() { k.m.Unload() }) }
