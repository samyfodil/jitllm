package server

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/samyfodil/jitllm/engine/model"
	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// stepLoop is continuous batching for one loaded model on its devices.
//
// A generate on a session whose every block and head are on one device does
// not take the device's gate for itself. It becomes a ROW of the model's step
// loop: one goroutine that owns every live row's model.State, and each step
// gathers each row's next token and runs them through model.StepRuns -- one
// device step for all of them, every block's weights read once
// (nn.SessionStepper), each row's history its own session's. Each row then
// samples its own logits with its own sampler, exactly as the one-at-a-time
// path does, and its events go to an outbox its request goroutine drains, so a
// slow client never holds up a step.
//
// The loop holds the model's gates while it has rows, as one generate held
// them before; while another request waits on any of those gates it admits no
// one new, lets its rows finish and hands the gates over, so a session on the
// one-at-a-time path is delayed by a batch the way it was by a generate, and
// never starved by one.
//
// A joint step is not always the faster: for each row count the loop times it
// against running the rows one session after another, in situ, and runs the
// faster (batchchoose.go; Config.JointSteps forces either).
//
// A new row's prompt is fed a chunk at a time inside the same steps
// (promptUnits): its chunk rides beside the decoding rows' tokens, so
// admission never stalls them by more than the rows it adds to a step.
type stepLoop struct {
	e  *Engine
	lm *LoadedModel
	// id is what the loop is called in its gates' queues.
	id string

	width       int
	promptChunk int

	ctx  context.Context
	quit context.CancelFunc
	done chan struct{}

	mu      sync.Mutex
	waiting []*row
	// rows is every admitted row, prompting or decoding, in admission order.
	// Only the loop goroutine changes it, under mu, so the loop reads it
	// without the lock and telemetry reads it with.
	rows    []*row
	stopped bool
	wake    chan struct{}

	// shape counts admissions and retirements: a joint step the device
	// refused is not retried until the rows it was refused for change.
	shape      uint64
	refusedFor uint64
	refused    bool

	// choice says, per row count, whether a decode step runs joint or one
	// session after another (batchchoose.go).
	choice *jointChoice

	stats batchCounters
}

// batchCounters is what BatchStats reports; see telemetry.proto.
type batchCounters struct {
	steps, rowsStepped, jointSteps, jointRows, soloRows atomic.Int64
	maxRows, maxSessions                                atomic.Int32
	promptChunks, promptSteps                           atomic.Int64
	admissions, admitWait                               atomic.Int64
	refusals, separateSteps                             atomic.Int64
	lastRefusal                                         atomic.Pointer[string]
}

// errUnloaded ends a request still waiting for a row when its model goes.
var errUnloaded = fmt.Errorf("%w: the model was unloaded before this request got a row", ErrNotFound)

// batchWidth is the most requests the loop holds rows for: the configured
// width, bounded by the widest step a device runs across sessions
// (model.MaxStepRows rows, each decoding session taking one). The KV each row
// needs is not part of it: a session holds its own history from creation, and
// one the device has no room for is not wholly on the device and never
// becomes a row.
func batchWidth(cfg int) int {
	if cfg <= 0 || cfg > model.MaxStepRows {
		return model.MaxStepRows
	}
	return cfg
}

func newStepLoop(e *Engine, lm *LoadedModel) *stepLoop {
	ctx, quit := context.WithCancel(context.Background())
	lp := &stepLoop{
		e:           e,
		lm:          lm,
		id:          "batch:" + lm.id,
		width:       batchWidth(e.cfg.MaxBatchRows),
		promptChunk: e.cfg.PromptChunk,
		choice:      newJointChoice(e.cfg.JointSteps),
		ctx:         ctx,
		quit:        quit,
		done:        make(chan struct{}),
		wake:        make(chan struct{}, 1),
	}
	go lp.run()
	return lp
}

// row is one generate inside the loop.
type row struct {
	s         *Session
	ids       []int32
	reset     bool
	echo      bool
	ephemeral bool
	sampler   model.Sampler
	maxTokens int
	ignoreEOS bool
	stream    *streamText
	enqueued  time.Time
	depth     int32

	// admitted closes when the loop takes the row; done when it lets go of
	// the row for good, after its last event is in the outbox.
	admitted  chan struct{}
	done      chan struct{}
	cancelled atomic.Bool

	// The loop goroutine's own half, written before admitted or done closes
	// and read by the request goroutine only after.
	waited      time.Duration
	fed         int
	logits      []float32
	next        [1]int32
	out         []int32
	n           int
	promptStart time.Time
	prefill     time.Duration
	decodeStart time.Time
	result      rowResult

	omu    sync.Mutex
	events []Event
	notify chan struct{}
}

type rowResult struct {
	reason FinishReason
	stop   string
	n      int
	decode time.Duration
	err    error
}

func (r *row) prompting() bool { return r.fed < len(r.ids) }

func (r *row) push(ev Event) {
	r.omu.Lock()
	r.events = append(r.events, ev)
	r.omu.Unlock()
	select {
	case r.notify <- struct{}{}:
	default:
	}
}

func (r *row) take() []Event {
	r.omu.Lock()
	defer r.omu.Unlock()
	evs := r.events
	r.events = nil
	return evs
}

func (r *row) isDone() bool {
	select {
	case <-r.done:
		return true
	default:
		return false
	}
}

// joins says whether a generate on s runs as a row of the model's step loop.
// The caller holds s.mu. A session that is not wholly on one device runs alone
// on its gates as before. One that is, and whose history then outgrows the
// card, is refused the joint step (the tier cannot append its page) and runs
// that token alone, where State.Forward hands a block to the host
// (SetRelocate); from then on it is stepped alone.
func (e *Engine) joins(s *Session) *stepLoop {
	lp := s.lm.loop
	if lp == nil || !s.st.Steppable() {
		return nil
	}
	return lp
}

// submit queues r for a row, recording in r.depth how many requests were
// ahead -- under the lock, since the loop may admit r and read it at once.
func (lp *stepLoop) submit(r *row) error {
	lp.mu.Lock()
	if lp.stopped {
		lp.mu.Unlock()
		return errUnloaded
	}
	lp.waiting = append(lp.waiting, r)
	r.depth = int32(len(lp.waiting) - 1)
	r.s.queuePos.Store(r.depth + 1)
	lp.mu.Unlock()
	select {
	case lp.wake <- struct{}{}:
	default:
	}
	return nil
}

// withdraw takes r back out of the queue, reporting false when the loop has
// already admitted it.
func (lp *stepLoop) withdraw(r *row) bool {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	for i, w := range lp.waiting {
		if w == r {
			lp.waiting = append(lp.waiting[:i], lp.waiting[i+1:]...)
			lp.renumber()
			return true
		}
	}
	return false
}

// renumber republishes every waiting session's queue position. lp.mu held.
func (lp *stepLoop) renumber() {
	for i, w := range lp.waiting {
		w.s.queuePos.Store(int32(i + 1))
	}
}

// close stops the loop. The model's sessions are closed first (UnloadModel,
// Engine.Close), so no row is live; a request still waiting for one ends with
// errUnloaded.
func (lp *stepLoop) close() {
	lp.mu.Lock()
	lp.stopped = true
	waiting := lp.waiting
	lp.waiting = nil
	lp.mu.Unlock()
	lp.quit()
	<-lp.done
	for _, r := range waiting {
		r.result.err = errUnloaded
		close(r.done)
	}
}

func (lp *stepLoop) run() {
	defer close(lp.done)
	for lp.awaitWork() {
		gs := lp.e.gatesFor(lp.lm.gateIDs())
		if _, _, err := gs.acquire(lp.ctx, lp.id, 0); err != nil {
			return // only the loop's own context ends an untimed wait
		}
		lp.serve(gs)
		gs.release()
	}
}

// awaitWork blocks until a request is waiting, reporting false when the loop
// is stopping.
func (lp *stepLoop) awaitWork() bool {
	for {
		lp.mu.Lock()
		n := len(lp.waiting)
		lp.mu.Unlock()
		if n > 0 {
			return true
		}
		select {
		case <-lp.wake:
		case <-lp.ctx.Done():
			return false
		}
	}
}

// serve runs steps until no row is left. It admits only while nobody else is
// waiting for the gates it holds.
func (lp *stepLoop) serve(gs *gateSet) {
	for {
		if lp.ctx.Err() != nil {
			for len(lp.rows) > 0 {
				lp.finish(lp.rows[0], FinishCancelled, "", nil)
			}
			return
		}
		if !gs.contended() {
			lp.admit()
		}
		if len(lp.rows) == 0 {
			return
		}
		lp.iterate()
	}
}

// admit moves waiting requests into rows while there is room.
func (lp *stepLoop) admit() {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	if len(lp.waiting) == 0 || len(lp.rows) >= lp.width {
		return
	}
	for len(lp.waiting) > 0 && len(lp.rows) < lp.width {
		r := lp.waiting[0]
		lp.waiting = lp.waiting[1:]
		r.waited = time.Since(r.enqueued)
		r.s.queuePos.Store(0)
		lp.rows = append(lp.rows, r)
		lp.shape++
		lp.stats.admissions.Add(1)
		lp.stats.admitWait.Add(int64(r.waited))
		close(r.admitted)
	}
	lp.renumber()
}

// started queues a row's GenerateStarted (and its echo) once its prompt is in.
func (r *row) started() {
	lm := r.s.lm
	devBlocks := r.s.st.GPULayers()
	r.push(Event{Kind: EventStarted, Started: &Started{
		SessionID:    r.s.id,
		ModelID:      lm.id,
		Ephemeral:    r.ephemeral,
		PromptTokens: len(r.ids),
		QueuedFor:    r.waited,
		QueueDepth:   r.depth,
		DeviceBlocks: devBlocks,
		HostBlocks:   lm.m.Cfg.NLayer - devBlocks,
		DeviceIDs:    lm.deviceIDs,
		Prefill:      r.prefill,
		Execution:    ExecutionParallel,
		Batched:      true,
	}})
	if r.echo {
		r.push(Event{Kind: EventToken, Token: &Token{ID: -1, Text: lm.m.Vocab.Decode(r.ids), Index: -1}})
	}
	r.decodeStart = time.Now()
}

// unit is one row's part of a step: its next tokens -- the one it sampled, or
// a chunk of its prompt -- and whether the logits after them are wanted.
type unit struct {
	r      *row
	tokens []int32
	logits bool
	prompt bool
}

// iterate runs one step: every decoding row's next token, and as much of the
// admitted prompts as the step has room for, in one model.StepRuns.
func (lp *stepLoop) iterate() {
	units := lp.decodeUnits()
	decoding := len(units)
	units = append(units, lp.promptUnits(min(lp.promptChunk, model.MaxStepRows-decoding))...)
	if len(units) == 0 {
		return
	}
	rows := 0
	for _, u := range units {
		rows += len(u.tokens)
	}
	lp.stats.steps.Add(1)
	lp.stats.rowsStepped.Add(int64(rows))
	lp.stats.maxRows.Store(max(lp.stats.maxRows.Load(), int32(rows)))
	lp.stats.maxSessions.Store(max(lp.stats.maxSessions.Load(), int32(len(units))))
	if decoding > 0 && decoding < len(units) {
		lp.stats.promptSteps.Add(1)
	}
	lp.step(units)
}

// decodeUnits samples every decoding row's next token from the logits its
// last step left, retires the rows that end here, and returns the rest's
// tokens to step.
//
// The order per row is the one-at-a-time path's per token, so a batched
// generate stops on the same token, with the same position, as one alone: a
// cancel is honoured before sampling, max_tokens before sampling, an EOG
// token is never fed (unless the row ignores it, and then it is a token like
// any other), and a stop string ends the row before its token is fed.
func (lp *stepLoop) decodeUnits() []unit {
	vocab := lp.lm.m.Vocab
	var units []unit
	for _, r := range append([]*row(nil), lp.rows...) {
		if r.prompting() {
			continue
		}
		if r.cancelled.Load() {
			lp.finish(r, FinishCancelled, "", nil)
			continue
		}
		if r.n >= r.maxTokens {
			lp.finish(r, FinishMaxTokens, "", nil)
			continue
		}
		next := r.sampler.Sample(r.logits)
		r.sampler.Observe(next)
		if !r.ignoreEOS && vocab.IsEOG(next) {
			lp.finish(r, FinishEOS, "", nil)
			continue
		}
		r.out = append(r.out, next)
		chunk, hit, match := r.stream.push(r.out)
		r.push(Event{Kind: EventToken, Token: &Token{ID: next, Text: chunk, Index: r.n}})
		if hit {
			r.n++
			lp.finish(r, FinishStop, match, nil)
			continue
		}
		r.next[0] = next
		units = append(units, unit{r: r, tokens: r.next[:], logits: true})
	}
	return units
}

// promptUnits is admission: the admitted prompts' next chunks, oldest first,
// at most budget tokens in all. They ride in the step the decoding rows take
// (model.StepRuns), each chunk as rows of its own session at its next
// positions, so a new prompt never holds the decoding rows up by more than the
// rows it adds to one step. A chunk that ends its prompt wants its logits --
// the row samples from them at the next step; one that does not costs no head
// projection.
func (lp *stepLoop) promptUnits(budget int) []unit {
	var units []unit
	for _, r := range append([]*row(nil), lp.rows...) {
		if !r.prompting() {
			continue
		}
		// A request that left mid-prompt is not fed the rest: it ends here,
		// its history holding what was fed, as a cancel mid-decode leaves it.
		if r.cancelled.Load() {
			if r.fed > 0 {
				r.prefill = time.Since(r.promptStart)
			}
			r.started()
			lp.finish(r, FinishCancelled, "", nil)
			continue
		}
		if budget <= 0 {
			continue
		}
		if r.fed == 0 {
			if r.reset {
				r.s.st.Reset()
			}
			r.promptStart = time.Now()
		}
		k := min(budget, len(r.ids)-r.fed)
		budget -= k
		units = append(units, unit{r: r, tokens: r.ids[r.fed : r.fed+k], logits: r.fed+k == len(r.ids), prompt: true})
	}
	return units
}

// step runs the units: those of sessions wholly on the device as one
// model.StepRuns, any other alone, and moves each row on by what it ran.
func (lp *stepLoop) step(units []unit) {
	var joint, solo []unit
	for _, u := range units {
		if u.r.s.st.Steppable() {
			joint = append(joint, u)
		} else {
			solo = append(solo, u)
		}
	}
	if (lp.refused && lp.refusedFor == lp.shape) || len(joint) < 2 {
		solo, joint = append(solo, joint...), nil
	}
	// A decode step that could go joint asks the measured choice. A step
	// carrying a prompt chunk always goes joint: admission is what keeps a new
	// prompt from holding the decoding rows up, and alone its chunk would.
	chose, timed := true, len(joint) >= 2 && len(solo) == 0 && !anyPrompt(joint)
	if timed {
		if chose = lp.choice.pick(len(joint)); !chose {
			lp.stats.separateSteps.Add(1)
			solo, joint = joint, nil
		}
	}
	t0 := time.Now()
	if len(joint) > 0 && !lp.stepJoint(joint) {
		solo = append(solo, joint...)
	}
	for _, u := range solo {
		out, err := model.StepRuns([]model.Run{{State: u.r.s.st, Tokens: u.tokens, Logits: u.logits}})
		if err != nil {
			lp.finish(u.r, FinishError, "", err)
			continue
		}
		lp.stats.soloRows.Add(int64(len(u.tokens)))
		lp.ran(u, out[0])
	}
	// A step the device refused ran its rows alone, which is neither arm.
	if timed && !(lp.refused && lp.refusedFor == lp.shape) {
		lp.choice.observe(len(units), chose, time.Since(t0))
	}
}

func anyPrompt(units []unit) bool {
	for _, u := range units {
		if u.prompt {
			return true
		}
	}
	return false
}

// stepJoint runs units as one model.StepRuns. It reports false, with no row
// moved, when the device refused the step before running any of it; those
// units then run alone, and the refusal stands until the rows change.
func (lp *stepLoop) stepJoint(units []unit) bool {
	runs := make([]model.Run, len(units))
	rows := 0
	for i, u := range units {
		runs[i] = model.Run{State: u.r.s.st, Tokens: u.tokens, Logits: u.logits}
		rows += len(u.tokens)
	}
	gpu := lp.lm.gpu
	before := gpu.Stats().SessionRows
	out, err := model.StepRuns(runs)
	if err != nil {
		// A hybrid's linear block keeps a summary that a step advances in
		// place: once any advanced, running the tokens again would apply them
		// twice, so those rows end with the device's error.
		if gpu.RecSteps() > 0 {
			for _, u := range units {
				lp.finish(u.r, FinishError, "", err)
			}
			return true
		}
		why := err.Error()
		lp.stats.refusals.Add(1)
		lp.stats.lastRefusal.Store(&why)
		lp.refused, lp.refusedFor = true, lp.shape
		return false
	}
	// The tier's own count says whether the rows really shared a step:
	// StepRuns runs States one after another, correctly, whenever they cannot.
	if gpu.Stats().SessionRows-before == rows {
		lp.stats.jointSteps.Add(1)
		lp.stats.jointRows.Add(int64(rows))
	} else {
		lp.stats.soloRows.Add(int64(rows))
	}
	for i, u := range units {
		lp.ran(u, out[i])
	}
	return true
}

// ran moves u's row on by the tokens it just ran: a decoding row by one
// token, a prompting one by its chunk -- and a prompt that is in becomes a
// decoding row, its GenerateStarted queued.
func (lp *stepLoop) ran(u unit, logits []float32) {
	r := u.r
	if !u.prompt {
		r.logits = logits
		r.n++
		return
	}
	r.fed += len(u.tokens)
	lp.stats.promptChunks.Add(1)
	if r.prompting() {
		return
	}
	r.logits = logits
	r.prefill = time.Since(r.promptStart)
	r.s.prefilled.Add(int64(len(r.ids)))
	lp.lm.tokensPrefilled.Add(int64(len(r.ids)))
	r.started()
}

// finish retires r: its held-back text is flushed as the one-at-a-time path
// flushes it, and from here on the loop never touches its State again.
func (lp *stepLoop) finish(r *row, reason FinishReason, stop string, err error) {
	if err == nil && (reason == FinishMaxTokens || reason == FinishEOS || reason == FinishCancelled) {
		if tail := r.stream.flush(); tail != "" {
			r.push(Event{Kind: EventToken, Token: &Token{ID: -1, Text: tail, Index: r.n}})
		}
	}
	var decode time.Duration
	if !r.decodeStart.IsZero() {
		decode = time.Since(r.decodeStart)
	}
	r.result = rowResult{reason: reason, stop: stop, n: r.n, decode: decode, err: err}
	lp.mu.Lock()
	for i, x := range lp.rows {
		if x == r {
			lp.rows = append(lp.rows[:i], lp.rows[i+1:]...)
			break
		}
	}
	lp.shape++
	lp.mu.Unlock()
	close(r.done)
}

// generateBatched is Generate's arm for a session that runs as a row. The
// caller holds s.mu, so nothing but the loop touches s.st until r.done.
func (e *Engine) generateBatched(ctx context.Context, lp *stepLoop, s *Session, o GenerateOptions,
	ids []int32, ephemeral bool, emit func(Event) error) error {
	lm := s.lm
	sampler := s.sampling
	if o.Sampling != nil {
		sampler = *o.Sampling
	}
	maxTokens := o.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	r := &row{
		s:         s,
		ids:       ids,
		reset:     !o.Continue,
		echo:      o.Echo,
		ephemeral: ephemeral,
		sampler:   sampler,
		maxTokens: maxTokens,
		ignoreEOS: o.IgnoreEOS,
		stream:    newStreamText(lm.m.Vocab, o.Stop),
		enqueued:  time.Now(),
		admitted:  make(chan struct{}),
		done:      make(chan struct{}),
		notify:    make(chan struct{}, 1),
		out:       make([]int32, 0, maxTokens),
	}
	if err := lp.submit(r); err != nil {
		return err
	}
	defer s.queuePos.Store(0)

	// ---- the queue: until a row is free, the request can still leave.
	var timeout <-chan time.Time
	if o.QueueTimeout > 0 {
		t := time.NewTimer(o.QueueTimeout)
		defer t.Stop()
		timeout = t.C
	}
	select {
	case <-r.admitted:
	case <-r.done:
		return r.result.err
	case <-ctx.Done():
		if lp.withdraw(r) {
			return ctx.Err()
		}
		r.cancelled.Store(true)
	case <-timeout:
		if lp.withdraw(r) {
			return ErrQueueTimeout
		}
	}
	s.running.Store(true)
	defer s.running.Store(false)
	s.lastUsed.Store(time.Now().UnixMilli())

	// ---- the row: drain its outbox until the loop lets go of it. A cancel or
	// a failed write marks the row, and the loop retires it at its next step.
	var emitErr error
	cancelled := ctx.Done()
	for {
		fin := r.isDone()
		for _, ev := range r.take() {
			if emitErr != nil {
				break
			}
			if err := emit(ev); err != nil {
				emitErr = err
				r.cancelled.Store(true)
			}
		}
		if fin {
			break
		}
		select {
		case <-r.notify:
		case <-r.done:
		case <-cancelled:
			r.cancelled.Store(true)
			cancelled = nil
		}
	}

	res := r.result
	s.generated.Add(int64(res.n))
	lm.tokensGenerated.Add(int64(res.n))
	s.lastUsed.Store(time.Now().UnixMilli())
	s.refresh()
	if emitErr != nil {
		return emitErr
	}
	if res.err != nil {
		return res.err
	}
	rate := 0.0
	if res.decode > 0 && res.n > 0 {
		rate = float64(res.n) / res.decode.Seconds()
	}
	return emit(Event{Kind: EventFinished, Finished: &Finished{
		Reason:           res.reason,
		StopMatched:      res.stop,
		PromptTokens:     len(ids),
		CompletionTokens: res.n,
		Prefill:          r.prefill,
		Decode:           res.decode,
		TokensPerSecond:  rate,
		BytesPerToken:    lm.m.BytesPerToken(),
		Position:         s.st.Pos(),
	}})
}

// snapshot is the loop's queue as GetDeviceQueue reports it: the sessions in
// rows, then the ones waiting for one.
func (lp *stepLoop) snapshot() (live, waiting []string) {
	lp.mu.Lock()
	defer lp.mu.Unlock()
	for _, r := range lp.rows {
		live = append(live, r.s.id)
	}
	for _, r := range lp.waiting {
		waiting = append(waiting, r.s.id)
	}
	return live, waiting
}

// deviceQueue is a device's queue as GetDeviceQueue and ListDevices report it:
// the gate's holders and waiters, with a model's step loop -- one entry in the
// gate's own queue -- expanded into the sessions in its rows and the requests
// waiting for one. key is a gate key (Engine.gateKey), not a device id as
// written.
func (e *Engine) deviceQueue(key string) (g *gate, queue []string, running, waiting int) {
	g = e.gate(key)
	gq, running, waiting := g.snapshot()
	loops := map[string]*stepLoop{}
	for _, lm := range e.Models() {
		if lm.loop != nil && hasString(lm.gateIDs(), key) {
			loops[lm.loop.id] = lm.loop
		}
	}
	for _, x := range gq {
		lp, ok := loops[x]
		if !ok {
			queue = append(queue, x)
			continue
		}
		delete(loops, x)
		live, wait := lp.snapshot()
		queue = append(append(queue, live...), wait...)
		waiting += len(wait)
		if g.holds(x) {
			running += len(live) - 1
		} else {
			waiting-- // the loop itself, waiting for the gate on its requests' behalf
		}
	}
	// A loop between taking a request and asking for the gate is in no queue
	// yet; its requests are still waiting.
	for _, lp := range loops {
		_, wait := lp.snapshot()
		queue = append(queue, wait...)
		waiting += len(wait)
	}
	return g, queue, max(0, running), max(0, waiting)
}

// sessionMode is how a generate on s would run: as a row of a step it shares,
// or alone on gates that serialise it.
func sessionMode(s *Session, gs *gateSet) ExecutionMode {
	if s.snapBatched.Load() {
		return ExecutionParallel
	}
	return gs.mode()
}

// pb is the loop's BatchStats, or nil for a model with no loop.
func (lp *stepLoop) pb() *v1.BatchStats {
	if lp == nil {
		return nil
	}
	live, wait := lp.snapshot()
	c := &lp.stats
	out := &v1.BatchStats{
		Width:               int32(lp.width),
		LiveRows:            int32(len(live)),
		Waiting:             int32(len(wait)),
		Steps:               c.steps.Load(),
		RowsStepped:         c.rowsStepped.Load(),
		JointSteps:          c.jointSteps.Load(),
		JointRows:           c.jointRows.Load(),
		SoloRows:            c.soloRows.Load(),
		MaxRowsPerStep:      c.maxRows.Load(),
		MaxSessionsPerStep:  c.maxSessions.Load(),
		PromptChunks:        c.promptChunks.Load(),
		PromptSteps:         c.promptSteps.Load(),
		Admissions:          c.admissions.Load(),
		AdmissionWaitMillis: time.Duration(c.admitWait.Load()).Milliseconds(),
		JointRefusals:       c.refusals.Load(),
		SeparateSteps:       c.separateSteps.Load(),
	}
	for _, ch := range lp.choice.snapshot() {
		out.Choices = append(out.Choices, &v1.JointChoice{
			Rows:               int32(ch.rows),
			Settled:            ch.settled,
			Joint:              ch.joint,
			JointStepMillis:    float64(ch.jointMedian) / float64(time.Millisecond),
			SeparateStepMillis: float64(ch.separateMedian) / float64(time.Millisecond),
			Probes:             int32(ch.probes),
		})
	}
	if why := c.lastRefusal.Load(); why != nil {
		out.LastRefusal = *why
	}
	return out
}
