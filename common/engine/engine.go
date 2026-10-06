// Package engine owns every jitllm object the app touches, on one goroutine.
//
// The UI goroutine never calls jitllm: gogpu.App.Run holds the UI thread, so a
// Prefill on it would freeze the window, and a terminal UI's update loop would
// stall the same way. Every entry point here queues and returns, as the
// window's screen.Engine requires.
//
// The engine knows no front end. It reports through a [Front] and the [State]
// values the front end hands it; the window and the terminal each implement
// both.
//
// It is one worker rather than a goroutine per job because the objects are not
// safe to share: model.State holds a sequence and a conversion beside a decode
// would contend for the same pool and pager. A single command channel makes "a
// long job cannot overlap a decode" true by construction.
package engine

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/samyfodil/jitllm/common/session"
	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// Engine is the app's single owner of a loaded model.
type Engine struct {
	// loadingPath is the model the worker is opening or switching to, for
	// stage. Worker goroutine only.
	loadingPath string

	sh Front
	st State

	cmds chan func()
	done chan struct{}
	once sync.Once

	// cancel is read by the decode loop and set by Stop. Atomic because Stop
	// runs on the UI goroutine and the loop runs on the worker -- the one
	// piece of state that genuinely crosses.
	cancel atomic.Bool

	// busyJobs is how many jobs the app has to wait for are queued or
	// running; Store.Busy is published from it. See busyEnd.
	busyJobs atomic.Int32

	// Everything below is touched only on the worker goroutine.
	//
	// models is every open model and active the one running. The scalars below
	// (m, sess, dev, gpu, closeDev, path, sessMax) belong to the active entry.
	// See entry's doc for why only one runs.
	models   []*entry
	active   *entry
	priority bool

	m    *model.Model
	sess *model.State
	dev  nn.Device
	// gpu is dev when it is a tier, kept because nn.Device cannot answer
	// "how much of each card is this model using" -- only *tier.GPU can.
	gpu      *tier.GPU
	closeDev func()
	path     string
	// sessMax is the window this session was built for, so the decode loop can
	// tell a context that filled from a cap that was reached.
	sessMax int
	// lastPromptTok is the last prompt's length, read by the reuse probe.
	lastPromptTok int
	// pictures is each picture file's preprocessed pixels and name, by the
	// file's identity, so a picture in every turn's history is decoded once.
	// The rows are the model's image cache's; dropped with the model.
	pictures map[fileID]*model.Picture
}

// fileID is a picture file's identity: its path, size and modification time.
type fileID struct {
	path  string
	size  int64
	mtime int64
}

// queueDepth is small on purpose: the commands are user actions, and a queue
// deep enough to hide a stall is a queue that hides a stall.
const queueDepth = 16

// New starts the worker. Close stops it.
func New(sh Front, st State) *Engine {
	e := &Engine{sh: sh, st: st, cmds: make(chan func(), queueDepth), done: make(chan struct{})}
	go e.run()
	return e
}

func (e *Engine) run() {
	defer close(e.done)
	for fn := range e.cmds {
		fn()
	}
}

// post queues work for the worker. It reports whether the queue took it: a
// full queue means the worker is busy with something the user can see, and
// dropping silently is how a button becomes a button that sometimes does
// nothing.
func (e *Engine) post(what string, fn func()) bool {
	select {
	case e.cmds <- fn:
		return true
	default:
		e.sh.Post(func() { e.sh.SetStatus(what + ": busy, try again in a moment") })
		return false
	}
}

// busyBegin counts a job the app must wait for, at the moment it is queued.
func (e *Engine) busyBegin() {
	e.busyJobs.Add(1)
	e.st.Busy.Set(true)
}

// busyEnd uncounts one and publishes what is left.
//
// The count is read when the post runs, not when the job ends: per-job
// Busy=false posts let a queued switch or Send run with Busy overwritten, which
// left Clear enabled under a live decode.
func (e *Engine) busyEnd() {
	e.busyJobs.Add(-1)
	e.sh.Post(func() { e.st.Busy.Set(e.busyJobs.Load() > 0) })
}

// Run is screen.Engine.Run: a long job that must not overlap a decode. It
// reports whether the queue took it.
func (e *Engine) Run(job func()) bool { return e.post("job", job) }

// Close stops the worker and releases the model and the device.
func (e *Engine) Close() {
	e.once.Do(func() {
		e.cancel.Store(true)
		// Queued, so it runs after whatever is in flight rather than freeing a
		// model another command is still holding.
		e.cmds <- func() { e.release() }
		close(e.cmds)
		<-e.done
	})
}

// release tears down the current model and device. Worker goroutine only.
func (e *Engine) release() {
	e.releaseSession()
	for _, en := range e.models {
		if en.m != nil {
			en.m.Close()
			en.m = nil
		}
	}
	e.models, e.active = nil, nil
}

// releaseSession drops what belongs to the active model and leaves the model
// itself open. A session carries the worker pool, the device tier and the KV
// cache; the model carries the weights and the pager. A switch gives up the
// first and keeps the second, so a backgrounded model is cheap to return to and
// two worker pools never exist at once.
func (e *Engine) releaseSession() {
	if e.sess != nil {
		e.sess.Close()
		e.sess = nil
	}
	if e.closeDev != nil {
		e.closeDev()
		e.closeDev = nil
	}
	e.dev, e.gpu, e.m, e.path, e.sessMax = nil, nil, nil, "", 0
}

// touch says a turn's content changed without the turn count moving.
func (e *Engine) touch() { e.st.Revision.Set(e.st.Revision.Get() + 1) }

// status writes the status bar from a worker.
func (e *Engine) status(f string, a ...any) {
	msg := fmt.Sprintf(f, a...)
	e.sh.Post(func() { e.sh.SetStatus(msg) })
}

// report shows an engine error on the session screen, explained, with the
// fix as a button where there is one.
func (e *Engine) report(stage, path string, err error) {
	title, detail, f := explain(stage, path, err)
	p := session.Problem{Title: title, Detail: detail}
	switch f {
	case fixConvert:
		p.Action, p.Do = "Convert\u2026", func() { e.sh.OfferConvert(path) }
	case fixReconvert:
		p.Action, p.Do = "Reconvert", func() { e.sh.OfferReconvert(path) }
	case fixDevices:
		p.Action, p.Do = "Open Settings", e.sh.GoSettings
	}
	e.sh.Report(p)
}

// Load is screen.Engine.Load. It queues and returns; a model already waiting
// to load is not queued twice.
func (e *Engine) Load(path string) {
	if path == "" {
		return
	}
	if _, ok := session.LoadingOf(e.st.Loads, path); ok {
		return
	}
	e.busyBegin()
	e.status("opening %s", short(path))
	session.QueueLoad(e.st.Loads, session.Loading{Path: path, Stage: "Waiting"})
	if !e.post("load", func() {
		defer e.busyEnd()
		defer e.loading(path)()
		e.stage("Reading the model")
		e.load(path)
	}) {
		e.busyEnd()
		session.EndLoad(e.st.Loads, path)
	}
}

// loading marks path as the load the worker is on, for stage, and returns
// what ends it -- whether it opened or failed: the error, if any, is already
// on its way as a Problem.
func (e *Engine) loading(path string) func() {
	e.loadingPath = path
	return func() {
		e.loadingPath = ""
		e.sh.Post(func() { session.EndLoad(e.st.Loads, path) })
	}
}

// stage says what the load in progress is doing now. It does nothing when no
// load is running: activate also runs on a switch.
func (e *Engine) stage(s string) {
	path := e.loadingPath
	if path == "" {
		return
	}
	e.sh.Post(func() { session.SetLoadStage(e.st.Loads, path, s) })
}

// load runs on the worker. It adds a model to the registry and makes it
// active. Re-loading a model already open is a switch, not a second copy: two
// entries over one file would be two pagers over the same bytes.
func (e *Engine) load(path string) {
	if en := e.find(path); en != nil {
		if err := e.activate(en); err != nil {
			e.report(stageDevice, path, err)
		}
		return
	}

	// The share is computed before the open, because model.Open otherwise reads
	// sched.MemBudget() itself, ignoring what the open models already hold.
	e.models = append(e.models, &entry{path: path})
	e.rebudget()
	en := e.find(path)

	m, err := model.Open(path, model.WithPageBudget(en.grant))
	if err != nil {
		// A GGUF or a stale container is the commonest wrong file; report turns
		// both into the button that fixes them.
		e.drop(en)
		e.report(stageOpen, path, err)
		return
	}
	en.m = m
	// With the model open its real share can be applied, and the models that
	// were already open give up the bytes this one now holds.
	e.rebudget()

	if err := e.activate(en); err != nil {
		e.report(stageDevice, path, err)
	}
}

// drop removes an entry that never opened.
func (e *Engine) drop(en *entry) {
	for i, x := range e.models {
		if x == en {
			e.models = append(e.models[:i], e.models[i+1:]...)
			break
		}
	}
	e.rebudget()
}

// activate gives one open model the session, the device and the worker pool,
// and takes them from whichever model had them.
//
// The previous session is closed first, and the order matters: two live States
// are two worker pools on the same cores, two live tiers over one card do not
// arbitrate its memory, and the JIT's codegen knobs are process-wide.
//
// The history is not carried across. In chat mode every turn renders the whole
// conversation through the model's own template, so the next message
// re-prefills it correctly; the KV namespace carries the model's size and
// mtime, so the new model cannot read the old one's pages.
func (e *Engine) activate(en *entry) error {
	if en == nil || en.m == nil {
		return errNoModel
	}
	e.releaseSession()
	e.active = en

	spec := e.st.DeviceSpec.Get()
	if spec == "" {
		spec = "auto"
	}
	m, path := en.m, en.path

	e.stage("Opening the device")
	dev, closeDev, devName, err := openDevices(spec, en.grant)
	if err != nil {
		// The previous session is already closed, so the screen must stop showing
		// it. The model stays open and listed, ready once the device is fixed.
		e.active = nil
		e.publishModels()
		e.sh.Post(func() {
			e.st.Loaded.Set(false)
			e.st.ModelPath.Set("")
			e.st.ModelSummary.Set("")
			e.st.Vision.Set(false)
			e.st.BlockMap.Set(nil)
			e.st.Alloc.Set(session.Allocation{})
		})
		return err
	}

	window := maxSeq(m, e.st.MaxSeq.Get())
	st := m.NewState(window)
	if dev != nil {
		e.stage("Placing layers on the GPU")
		if err := st.SetDevice(dev); err != nil {
			// The model still runs, on the host; say why the device holds nothing.
			e.report(stageDevice, path, err)
		}
	}
	// A context that outgrows the card hands blocks to the host one at a time
	// -- only where the device choice lets the host run them, as the CLI's
	// -relocate does. A choice of devices alone keeps the model off the CPU, so
	// there the token takes the device's refusal instead.
	relocate, _ := tier.AdmitsHost(spec)
	st.SetRelocate(relocate)

	cacheOff := "turned off in Settings"
	if e.st.KVCache.Get() {
		cacheOff = attachKVCache(st, path)
	}

	e.m, e.sess, e.dev, e.closeDev, e.path, e.sessMax = m, st, dev, closeDev, path, window
	e.gpu, _ = dev.(*tier.GPU)
	// The active model's share changes when it becomes active, so this runs
	// after e.active is set and before anything is reported.
	e.rebudget()
	e.publishModels()

	c := m.Cfg
	// A base model cannot chat, so the header says "completion only" and Chat
	// is switched off at load rather than failing every turn.
	summary := e.headerNow().String()
	chatOK := m.ChatCapable()
	vision := m.Tower() != nil
	// A different model is a different preprocessor: every picture is stale.
	e.pictures = nil
	blocks := placementOf(m, st, e.gpu)
	pager := pagerStat(m)
	alloc := allocStat(m, st, e.gpu, c.NLayer)

	e.sh.Post(func() {
		if !chatOK {
			e.st.Chat.Set(false)
		}
		e.st.ChatCapable.Set(chatOK)
		e.st.Vision.Set(vision)
		e.st.ModelPath.Set(path)
		e.st.ModelSummary.Set(summary)
		e.st.Loaded.Set(true)
		e.st.DeviceReport.Set(devName)
		e.st.BlockMap.Set(blocks)
		e.st.Pager.Set(pager)
		e.st.Alloc.Set(alloc)
		// A model that loaded answers whatever problem the banner was showing.
		e.st.Problem.Set(session.Problem{})
		loaded := "loaded " + short(path)
		if cacheOff != "" {
			loaded += " (prompt cache off: " + cacheOff + ")"
		}
		e.sh.SetStatus(loaded)
	})
	return nil
}

// maxSeq is the context window a session allocates: what the caller asked for,
// bounded by what the model was trained with. The model's context is a ceiling
// (State.ContextClamped reports a clamp); asking for less is legitimate, since
// the KV cache is allocated up front.
func maxSeq(m *model.Model, want int) int { return maxSeqFor(m.Cfg.NCtx, want) }

// maxSeqFor is the arithmetic, separated so it can be gated without a model.
func maxSeqFor(modelCtx, want int) int {
	n := modelCtx
	if n <= 0 {
		n = defaultMaxSeq
	}
	if want > 0 && want < n {
		n = want
	}
	if n > defaultMaxSeq && want <= 0 {
		n = defaultMaxSeq
	}
	return n
}

// defaultMaxSeq is the window when nothing asks for one. It is a cap on the
// engine's own default rather than on the user's request: a model trained to
// 32k still opens at 4k unless asked, because the cache is allocated up front.
const defaultMaxSeq = 4096

// samplerFor turns the app's settings into the engine's sampler.
//
// Temp == 0 is greedy and is passed straight through: model.Sampler
// short-circuits to argmax, and every correctness comparison in the engine
// (llama.cpp goldens, CPU vs device, verify) is a comparison of argmax.
func samplerFor(s session.Sampling) *model.Sampler {
	return &model.Sampler{
		Temp:        float64(s.Temp),
		TopK:        s.TopK,
		TopP:        float64(s.TopP),
		MinP:        float64(s.MinP),
		RepeatPen:   float64(s.RepeatPen),
		RepeatLastN: s.RepeatLastN,
		Seed:        s.Seed,
	}
}

func short(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}

// openDevices mirrors cmd/jitllm's, narrowed to what the app needs.
//
// "cpu" opens nothing, and a spec that names a missing device is an error
// rather than a silent fall back to the CPU. Only "auto" may fall back.
func openDevices(spec string, hostBudget uint64) (nn.Device, func(), string, error) {
	if spec == "" || spec == "cpu" {
		return nil, func() {}, "cpu", nil
	}
	g, err := tier.OpenWith(tier.WithDevices(spec), tier.WithHostBudget(hostBudget))
	if err != nil || g == nil {
		if spec == "auto" {
			return nil, func() {}, "cpu", nil
		}
		if err == nil {
			err = fmt.Errorf("no device matched %q", spec)
		}
		return nil, nil, "", err
	}
	name := spec
	if ds := g.DevStats(); len(ds) > 0 {
		name = fmt.Sprintf("%s (%d device(s))", spec, len(ds))
	}
	return g, func() { g.Close() }, name, nil
}

// blockMap is the memory map's input: one byte a block, saying which device
// runs it and whether its page is in host memory (see session.Placement).
//
// It is built from the runs, not a count, because placement is a set and not a
// prefix: a declined block leaves a hole, and the map exists to show it. devOf
// is the device ordinal of a block the runs place; resident is the host pager's.
func blockMap(nLayer int, runs [][2]int, devOf func(li int) int, resident func(li int) bool) []byte {
	if nLayer <= 0 {
		return nil
	}
	b := make([]byte, nLayer)
	for _, r := range runs {
		for i := max(r[0], 0); i < r[1] && i < nLayer; i++ {
			b[i] = byte(session.OnDevice(devOf(i)))
		}
	}
	for i := range b {
		if resident(i) {
			b[i] |= byte(session.PlaceResident)
		}
	}
	return b
}

// placementOf is blockMap for a live session. A block the tier cannot name a
// device for is drawn on device 0 rather than dropped from the device count.
func placementOf(m *model.Model, st *model.State, g *tier.GPU) []byte {
	ord := map[string]int{}
	if g != nil {
		for i, b := range g.Budgets() {
			ord[b.Name] = i
		}
	}
	devOf := func(li int) int {
		if g != nil {
			if name, ok := g.DeviceOf(li); ok {
				return ord[name]
			}
		}
		return 0
	}
	// The session's runs name the text blocks it placed. The blocks after them
	// -- prediction blocks and the vision tower's -- are blocks of the same
	// space, placed by whichever State runs them, so the tier says where each is.
	runs := st.DeviceBlocks()
	if g != nil {
		for li := m.Cfg.NLayer; li < m.Blocks(); li++ {
			if _, ok := g.DeviceOf(li); ok {
				runs = append(runs, [2]int{li, li + 1})
			}
		}
	}
	return blockMap(m.Blocks(), runs, devOf, m.BlockResident)
}

// allocStat is where the model is: bytes per device and bytes on the host.
//
// Nothing here sums across devices. A pool is per heap: one card reached
// through CUDA and Vulkan is one pool, and unified-memory devices share the
// host pool, so two rows can report the same bytes. Each row carries its pool
// name and a Shared flag instead.
//
// Budgets(), DevStats() and Placed() are parallel arrays indexed by device
// ordinal, fastest first.
func allocStat(m *model.Model, st *model.State, g *tier.GPU, nBlocks int) session.Allocation {
	a := session.Allocation{
		HostUsed:   m.HostBytes(),
		HostBudget: m.PageBudget(),
		Dense:      m.DenseBytes(),
		NBlocks:    nBlocks,
		PageBytes:  m.PageSize(),
	}
	if st != nil {
		a.DeviceBlocks = st.GPULayers()
		a.HostBlocks = a.NBlocks - a.DeviceBlocks
		a.HostKV = st.KVBytes()
		a.Pos, a.MaxSeq = st.Pos(), st.MaxSeq()
	}
	if g == nil {
		a.HostBlocks = a.NBlocks
		return a
	}

	// A pool is found by name: a Budget names its pool and there is no index
	// between the two lists.
	used := map[string][2]uint64{}
	for _, p := range g.Pools() {
		used[p.Name()] = [2]uint64{p.Used(), p.Limit()}
	}

	budgets, stats, placed := g.Budgets(), g.DevStats(), g.Placed()
	for i, b := range budgets {
		d := session.DeviceUse{Name: b.Device, Short: b.Name, Pool: b.Pool, Shared: b.Host, Limit: b.Limit, Weights: b.Used}
		if u, ok := used[b.Pool]; ok {
			d.Used = u[0]
			if u[1] > 0 {
				d.Limit = u[1]
			}
		}
		if i < len(placed) {
			d.Blocks = placed[i]
		}
		if i < len(stats) {
			d.KV = stats[i].KVBytes
			d.KVPool = stats[i].KVPoolBytes
		}
		a.Devices = append(a.Devices, d)
	}
	return a
}

func pagerStat(m *model.Model) session.PagerStat {
	fr, in, out := m.PageStats()
	// Resident is the resident bytes: Frames is what jlm holds, Resident is
	// what that costs.
	return session.PagerStat{
		Frames: fr, PageIns: in, PageOuts: out,
		Resident:  m.HostBytes(),
		BytesRead: m.BytesRead(), Reads: m.PagerReads(),
		ChunkBytes: m.ChunkBytes(),
	}
}

// Relocate moves the seam: it asks the device to hold n blocks and publishes
// what it actually took.
//
// It is queued, not called: model.State has no mutex and nothing may overlap a
// decode, so the command channel puts this between turns. It must not copy
// Stop's shortcut, which bypasses the queue on purpose.
//
// The published count is what SetGPULayers returned, not n: growing is best
// effort and may take fewer blocks, and a shrink whose history migration fails
// leaves the seam unchanged.
func (e *Engine) Relocate(n int) {
	e.post("relocate", func() {
		if e.m == nil || e.sess == nil {
			e.status("no model is loaded")
			return
		}
		if e.gpu == nil {
			e.status("this model is on the host: load it with a device to move blocks")
			return
		}
		was := e.sess.GPULayers()
		got := e.sess.SetGPULayers(n)
		nb := e.m.Cfg.NLayer
		blocks := placementOf(e.m, e.sess, e.gpu)
		alloc := allocStat(e.m, e.sess, e.gpu, nb)
		summary := e.headerNow().String()

		msg := fmt.Sprintf("%d of %d block(s) on a device", got, nb)
		switch {
		case got == was && n != was:
			msg = fmt.Sprintf("the seam did not move: still %d of %d", got, nb)
		case got < n:
			msg = fmt.Sprintf("%d of %d block(s) placed, %d asked for", got, nb, n)
		}
		e.sh.Post(func() {
			e.st.BlockMap.Set(blocks)
			e.st.Alloc.Set(alloc)
			e.st.ModelSummary.Set(summary)
			e.st.Status.Set(msg)
		})
	})
}

// Stop cancels the generation in flight and keeps the model loaded.
func (e *Engine) Stop() { e.cancel.Store(true) }

// Send is screen.Engine.Send. It queues and returns.
func (e *Engine) Send(r session.ChatRequest) {
	if !e.st.Loaded.Get() {
		e.sh.Alert("No model loaded", "Open a model on the Models tab first.")
		return
	}
	e.cancel.Store(false)
	e.busyBegin()
	e.st.Streaming.Set(true)
	if !e.post("send", func() { defer e.busyEnd(); e.generate(r) }) {
		e.busyEnd()
		e.st.Streaming.Set(false)
	}
}

func (e *Engine) generate(r session.ChatRequest) {
	defer e.sh.Post(func() { e.st.Streaming.Set(false) })
	if e.m == nil || e.sess == nil {
		return
	}

	// Whether the template already opened a thinking block, asked of the
	// rendered prompt. It is rendered only to be inspected; the ids still come
	// from ChatIDs.
	preOpened := e.templateOpensThinking(r)

	// The thinking disclosure opens itself once and folds itself once; between
	// and after those edges it is the user's. The flags are one-shot so a latch
	// does not overwrite a click on every token.
	var openedOnce, foldedOnce bool

	// A turn with a picture in its history is spans, not ids: a picture is rows
	// the prefill encodes (model.Picture), named by what the tower reads, so the
	// prefix cache restores past it as it does past text and a restored picture
	// runs no tower block.
	var ids []int32
	var spans []model.Span
	var err error
	if hasImages(r) {
		spans, err = e.imageSpans(r)
	} else {
		ids, err = e.promptIDs(r)
	}
	if err != nil {
		e.report(stagePrompt, e.path, err)
		return
	}
	if len(ids) == 0 && len(spans) == 0 {
		e.status("nothing to send")
		return
	}

	ph := e.m.NewPhaseMeter()
	ph.Mark("load")
	moved0 := e.sess.Relocations()
	_, _, outs0 := e.m.PageStats()
	// The sequence is reset before every turn: a chat prompt is the whole history
	// re-rendered, so prefilling onto a session still holding the last turn would
	// show the model the conversation twice. Reset is cheap, and the prefix cache
	// makes re-prefilling the history affordable.
	e.sess.Reset()

	t0 := time.Now()
	// PrefillCached, because a chat turn is its whole history and a plain Prefill
	// would re-read it from position zero every turn. It falls back to an ordinary
	// prefill when no store is attached.
	var logits []float32
	prompted := len(ids)
	if spans != nil {
		prompted = model.SpanPositions(spans, e.m.Cfg.NEmbd)
		logits, err = e.sess.PrefillCachedMixed(spans...)
	} else {
		logits, err = e.sess.PrefillCached(ids)
	}
	if err != nil {
		e.report(stageReply, e.path, err)
		return
	}
	prefill := time.Since(t0)
	reused := e.sess.KVRestored()
	ph.Mark("prefill")
	promptRate := float64(prompted) / prefill.Seconds()
	// Printed even when zero: a working cache and a silently dead one look the
	// same on the first turn otherwise.
	fails := e.sess.KVStoreFailures()
	e.lastPromptTok = prompted
	kvNote := ""
	if prompted > 0 {
		kvNote = fmt.Sprintf("  |  kv %d/%d reused (%.0f%%)",
			reused, prompted, 100*float64(reused)/float64(prompted))
	}
	if fails > 0 {
		kvNote += fmt.Sprintf("  %d STORE WRITE FAILURE(S)", fails)
	}
	e.sh.Post(func() { e.st.PromptTokS.Set(promptRate) })

	max := r.MaxTokens
	if max <= 0 {
		max = session.DefaultMaxTokens
	}
	sm := samplerFor(r.Sampling)
	out := make([]int32, 0, max)
	t1 := time.Now()
	n := 0

	// Why a reply ended is part of the reply: a cap, a full context and Stop
	// otherwise all look like a finished answer.
	stop := "end of text"
	for i := 0; i < max; i++ {
		if e.cancel.Load() {
			stop = "stopped"
			break
		}
		if i == max-1 {
			stop = fmt.Sprintf("hit the %d-token cap", max)
		}
		if e.sess.Pos() >= e.sessMax-1 {
			stop = fmt.Sprintf("context full at %d positions", e.sessMax)
			break
		}
		next := sm.Sample(logits)
		sm.Observe(next)
		if e.m.Vocab.IsEOG(next) {
			break
		}
		out = append(out, next)
		n++
		// Every token is published. RequestRedraw is coalesced and the post queue
		// drops on overflow, so no throttle is needed here. The real cost is one
		// Decode of the whole id slice per token (quadratic, like cmd/jitllm); the fix
		// for that would be an incremental detokenizer, not a timer.
		raw := display(e.m.Vocab.DecodeChat(out))
		partialThink, partial := splitThinking(holdPartialTag(raw), preOpened)
		// The live rate is published with each token so the readout is not "--"
		// during the generation.
		live := float64(n) / time.Since(t1).Seconds()
		// The two edges are latched on the worker so the posted closure carries a
		// decision rather than re-deriving one from racing state.
		openNow := !openedOnce && partialThink != "" && partial == ""
		foldNow := !foldedOnce && partial != ""
		if openNow {
			openedOnce = true
		}
		if foldNow {
			foldedOnce = true
		}
		e.sh.Post(func() {
			e.st.Stream.Set(partial)
			e.st.StreamThink.Set(partialThink)
			e.st.DecodeTokS.Set(live)
			// Open the reasoning while it is the only output, so a long thinking
			// phase does not look like a frozen window.
			if openNow {
				e.sh.SetThinkOpen(r.Reply, true)
			}
			// Fold it once the answer starts. Once only: a click after this sticks.
			if foldNow && !e.st.ShowThinking.Get() {
				e.sh.SetThinkOpen(r.Reply, false)
			}
			// Touch every token: core/listview caches a row's measured height, so a
			// bubble laid out while empty would keep height zero and the reply would
			// paint over its neighbour. InvalidateData re-measures.
			e.touch()
		})
		if logits, err = e.sess.Forward(next); err != nil {
			e.report(stageReply, e.path, err)
			return
		}
	}

	decode := time.Since(t1)
	dec := ph.Mark("decode")
	rawText := display(e.m.Vocab.DecodeChat(out))
	think, text := splitThinking(rawText, preOpened)
	rate := 0.0
	if decode > 0 {
		rate = float64(n) / decode.Seconds()
	}
	perTok := float64(dec.PerToken(n))
	gbs := dec.Rate() / (1 << 30)
	pager := pagerStat(e.m)
	pager.TurnOuts = pager.PageOuts - outs0
	alloc := allocStat(e.m, e.sess, e.gpu, e.m.Cfg.NLayer)
	blocks := placementOf(e.m, e.sess, e.gpu)
	summary := e.headerNow().String()
	modelName, modelColour := e.activeName()
	reply := r.Reply
	// A block moved to the host runs at host speed from here on, so say so.
	if moved := e.sess.Relocations() - moved0; moved > 0 {
		kvNote += fmt.Sprintf("  |  %d layer(s) moved to the CPU to fit the context", moved)
	}

	e.sh.Post(func() {
		// The final text goes into Stream as well as the turn, and both halves are
		// republished from the complete text: the last partial split may differ from
		// the finished one (holdPartialTag trims up to seven characters). SetTurn
		// does not publish a new count, so Touch is what makes the reply visible.
		e.st.Stream.Set(text)
		e.st.StreamThink.Set(think)
		e.st.DecodeTokS.Set(rate)
		e.st.BytesPerTok.Set(perTok)
		e.st.GBs.Set(gbs)
		e.st.Pager.Set(pager)
		e.st.Alloc.Set(alloc)
		e.st.BlockMap.Set(blocks)
		e.st.ModelSummary.Set(summary)
		e.st.Problem.Set(session.Problem{}) // a turn that finished answers the last one's error
		e.sh.SetTurn(reply, session.Turn{
			Role: session.RoleAssistant, Text: text, Think: think, Tokens: n, TokPerSec: rate,
			Model: modelName, Colour: modelColour,
		})
		e.touch()
		e.sh.SetStatus(fmt.Sprintf("%d token(s) at %.2f tok/s — %s%s",
			n, rate, stop, kvNote))
	})
}

// promptIDs builds the token ids for one request.
//
// The chat path goes through ChatIDs, not ChatPrompt+Encode: a rendered
// template is a complete prompt with its own specials and must be encoded with
// addSpecial=false.
func (e *Engine) promptIDs(r session.ChatRequest) ([]int32, error) {
	if e.m.Vocab == nil {
		return nil, fmt.Errorf("this container has no tokenizer: %v", e.m.TokErr)
	}
	if !r.Chat {
		return e.m.Vocab.Encode(r.Prompt, true), nil
	}
	return e.m.ChatIDs(chatMessages(r), true)
}

// chatMessages is the request's conversation in the model's terms.
func chatMessages(r session.ChatRequest) []model.ChatMessage {
	msgs := make([]model.ChatMessage, 0, len(r.Messages))
	for _, m := range r.Messages {
		msgs = append(msgs, model.ChatMessage{Role: m.Role, Content: m.Content, Images: len(m.Images)})
	}
	return msgs
}

// hasImages reports whether any turn of the request carries a picture.
func hasImages(r session.ChatRequest) bool {
	for _, m := range r.Messages {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// imageSpans lays every picture of the request into the conversation as a
// picture span, which the prefill encodes in the session's own vision segment
// (model.Picture): its JIT and pool, its device, the model's image cache. A
// picture is preprocessed once per file -- pictures maps a file's identity to
// its pixels and name, never to rows.
func (e *Engine) imageSpans(r session.ChatRequest) ([]model.Span, error) {
	if e.m.Tower() == nil {
		return nil, fmt.Errorf("the loaded model has no vision tower, so it cannot take a picture")
	}
	if e.pictures == nil {
		e.pictures = map[fileID]*model.Picture{}
	}
	var imgs []model.Image
	for _, m := range r.Messages {
		for _, p := range m.Images {
			fi, err := os.Stat(p)
			if err != nil {
				return nil, err
			}
			k := fileID{p, fi.Size(), fi.ModTime().UnixNano()}
			pc, ok := e.pictures[k]
			if !ok {
				if pc, err = e.picture(p); err != nil {
					return nil, fmt.Errorf("%s: %w", filepath.Base(p), err)
				}
				e.pictures[k] = pc
			}
			imgs = append(imgs, model.Image{Picture: pc})
		}
	}
	return e.m.ChatSpansImages(chatMessages(r), imgs, true)
}

// picture decodes and preprocesses one picture on the session's vision
// segment.
func (e *Engine) picture(path string) (*model.Picture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return e.sess.Picture(img)
}

// display is the decoded text made safe to put on screen.
//
// It decodes the whole id slice rather than concatenating per-token pieces,
// because Vocab.Text returns raw pieces (SentencePiece "▁" markers). A
// partial must not split a rune, so an incomplete multi-byte tail is held
// back. It does nothing else: reasoning is split by the caller, and
// TestThePipelineKeepsTheReasoning guards that pairing.
func display(s string) string {
	for s != "" && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

// SplitThinking separates a model's reasoning block from its answer.
//
// The answer is the reply and the reasoning is shown in a collapsed section,
// so the two are separated. An unclosed block is all thinking and no answer,
// which is the true state while a reasoning model is still working. The tags
// are matched literally as text.
func SplitThinking(s string) (think, answer string) { return splitThinking(s, false) }

// SplitThinkingOpen is SplitThinking for a template that already opened the
// block, so the output begins inside it.
//
// The engine knows this from the rendered prompt (templateOpensThinking); the
// parser cannot, since reasoning without a tag is indistinguishable from an
// answer.
func SplitThinkingOpen(s string) (think, answer string) { return splitThinking(s, true) }

func splitThinking(s string, inBlock bool) (think, answer string) {
	var tb, ab strings.Builder
	if inBlock {
		i := strings.Index(s, thinkClose)
		if i < 0 {
			// Still inside: all of it is reasoning and there is no answer yet.
			return strings.TrimSpace(s), ""
		}
		tb.WriteString(s[:i])
		s = s[i+len(thinkClose):]
	}

	// A closing tag before any opening one means the template opened the block.
	// This is the fallback for a caller that does not know; the engine uses
	// SplitThinkingOpen.
	if i := strings.Index(s, thinkClose); i >= 0 && !inBlock {
		if j := strings.Index(s, thinkOpen); j < 0 || j > i {
			tb.WriteString(s[:i])
			s = s[i+len(thinkClose):]
		}
	}

	for {
		i := strings.Index(s, thinkOpen)
		if i < 0 {
			ab.WriteString(s)
			break
		}
		ab.WriteString(s[:i])
		rest := s[i+len(thinkOpen):]
		j := strings.Index(rest, thinkClose)
		if j < 0 {
			// Still inside the block: everything after the tag is reasoning.
			if tb.Len() > 0 {
				tb.WriteString("\n")
			}
			tb.WriteString(rest)
			s = ""
			break
		}
		// A separator, because two blocks are two thoughts: concatenated they
		// read as one sentence running into the next.
		if tb.Len() > 0 {
			tb.WriteString("\n")
		}
		tb.WriteString(rest[:j])
		s = rest[j+len(thinkClose):]
	}
	return strings.TrimSpace(tb.String()), strings.TrimSpace(ab.String())
}

// holdPartialTag trims a trailing fragment of a tag that has not fully
// arrived.
//
// Otherwise every block flickers "<", "<t", "<th" on its way in.
func holdPartialTag(s string) string {
	for _, tag := range []string{thinkClose, thinkOpen} {
		for n := len(tag) - 1; n > 0; n-- {
			if strings.HasSuffix(s, tag[:n]) {
				return s[:len(s)-n]
			}
		}
	}
	return s
}

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// attachKVCache gives the session a prefix cache in session.KVCacheDir,
// and returns why not when it cannot. That is not an error: the session still
// runs and re-prefills, but the load says so.
//
// The namespace carries size and mtime, not just the basename: many containers
// are named model.jlm, and the geometry hash cannot separate two fine-tunes of
// one architecture.
func attachKVCache(st *model.State, path string) string {
	dir := session.KVCacheDir()
	if dir == "" {
		return "this system has no cache folder"
	}
	// Bounded, because nothing in a KV cache expires.
	store, err := model.NewFileStoreLimit(dir, kvCacheMax)
	if err != nil {
		return err.Error()
	}
	st.SetKVStore(store)
	ns := "jlm/" + filepath.Base(path)
	if fi, err := os.Stat(path); err == nil {
		ns = fmt.Sprintf("jlm/%s/%d-%d", filepath.Base(path), fi.Size(), fi.ModTime().UnixNano())
	}
	// A refused key leaves the process-local default: the cache still serves
	// this run, just not the next one.
	st.SetCacheKey(ns)
	return ""
}

// kvCacheMax bounds the prefix cache on disk. 8 GiB is cmd/jitllm's default
// and is ~64 prompts of a thousand positions on a 1B model.
const kvCacheMax = 8 << 30

// kvStatus reports the reuse for one turn.
func kvStatus(reused, prompted int, fails int64) string {
	if prompted <= 0 {
		return ""
	}
	msg := fmt.Sprintf("kv cache %d of %d prompt position(s) reused (%.0f%%)",
		reused, prompted, 100*float64(reused)/float64(prompted))
	if fails > 0 {
		msg += fmt.Sprintf("  %d STORE WRITE FAILURE(S)", fails)
	}
	return msg
}

// lastReused and lastFailures expose the prefix cache's own counters for the
// turn just finished. They read the session directly and so must be called
// when no generation is in flight, which a test after pump() is.
func (e *Engine) lastReused() int {
	if e.sess == nil {
		return 0
	}
	return e.sess.KVRestored()
}

func (e *Engine) lastFailures() int64 {
	if e.sess == nil {
		return 0
	}
	return e.sess.KVStoreFailures()
}

// lastPrompt is how many tokens the last prompt was, for the reuse probe.
func (e *Engine) lastPrompt() int { return e.lastPromptTok }

// templateOpensThinking reports whether the rendered chat prompt ends inside a
// reasoning block, which is how qwen3's template is written.
func (e *Engine) templateOpensThinking(r session.ChatRequest) bool {
	if !r.Chat || e.m == nil {
		return false
	}
	rendered, err := e.m.ChatPrompt(chatMessages(r), true)
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.TrimRight(rendered, " \t\r\n"), thinkOpen)
}
