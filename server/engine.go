// Package server is jitllm's serving layer: a ConnectRPC control plane over
// the engine's own orchestration primitives, plus OpenAI- and
// Anthropic-compatible HTTP shims that are thin adapters over the SAME
// internal generate path.
//
// jitllm is an inference OS: it places and moves blocks across every core and
// accelerator, and the caller can force each of those decisions. So
// PlacementService is a first-class surface here, and the compatibility shims
// are adapters beside it.
package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/engine/nn"
	"github.com/samyfodil/jitllm/engine/sched"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
)

// Config is what a server is built with. Every field has a working default.
type Config struct {
	// ModelDir is where ListModels scans and where a bare name resolves
	// (default "models", relative to the working directory).
	ModelDir string

	// HostConcurrency is how many sessions may run on the host at once. The
	// default is 1: every nn.JIT runs sched.DecodeCores() workers, so two host
	// sessions are two pools spinning on the same cores. Raise it only on a
	// host with cores to spare, and measure.
	HostConcurrency int

	// DeviceConcurrency is how many sessions may run on one device at once.
	// It should stay 1 until per-session scratch exists: the tier holds one
	// scratch set per device. It governs the one-at-a-time path; generates on
	// sessions wholly on a device share decode steps instead (MaxBatchRows).
	DeviceConcurrency int

	// MaxBatchRows bounds how many generates of one device model decode as
	// rows of one step (batch.go). Zero takes the engine's bound, the widest
	// step a device runs across sessions; 1 turns batching off, and every
	// generate then runs alone on its gates.
	MaxBatchRows int

	// PromptChunk is how many prompt tokens the step loop feeds into one step
	// beside the rows already decoding, which bounds how much admitting a
	// prompt holds each of their tokens up. Zero takes a device prefill chunk
	// (nn.MaxDevicePrefillChunk), the pieces the device would cut it into
	// anyway; a step never carries more than model.MaxStepRows rows in all.
	PromptChunk int

	// JointSteps is how a decode step whose rows could run as one joint step
	// does run: measured per row count (the default), always joint, or never.
	JointSteps JointSteps

	// DefaultMaxSeq is the KV capacity a session gets when it asks for none.
	// Zero takes the model's own context length.
	DefaultMaxSeq int

	// Version is reported by GetServerInfo.
	Version string

	// Probe enumerates the hardware. nil takes the real one, which is
	// expensive and exclusive: backend.Open() opens and closes every backend.
	// Tests and deployments that know their topology supply it here.
	Probe func() ([]DeviceInfo, error)
}

func (c *Config) withDefaults() {
	if c.ModelDir == "" {
		c.ModelDir = "models"
	}
	if c.HostConcurrency < 1 {
		c.HostConcurrency = 1
	}
	if c.DeviceConcurrency < 1 {
		c.DeviceConcurrency = 1
	}
	if c.PromptChunk <= 0 {
		c.PromptChunk = nn.MaxDevicePrefillChunk
	}
	if c.Version == "" {
		c.Version = "dev"
	}
}

// Engine owns every loaded model, every open session, and the gates that
// serialise them onto hardware. It is the whole of the server's state; the six
// Connect services and the two HTTP shims are projections of it.
type Engine struct {
	cfg     Config
	started time.Time

	mu       sync.RWMutex
	models   map[string]*LoadedModel
	sessions map[string]*Session

	// gateMu is separate from mu on purpose. Generate holds a session's own
	// lock while it asks for a gate, and every read path takes mu; one mutex
	// over both would make the lock order depend on which handler ran.
	gateMu sync.Mutex
	gates  map[string]*gate
	// aliases maps a device id a caller may write ("cuda:1", "gpu:0",
	// "vulkan:0") to the gate key of the physical device a loaded tier
	// resolved it to (deviceGateKeys). Guarded by gateMu.
	aliases map[string]string

	seq atomic.Uint64

	// devices is the hardware probe, taken once. See devices.go for why it is
	// not re-taken on every list.
	devMu   sync.Mutex
	devices []DeviceInfo
	devErr  error
	probed  bool
}

// New builds an Engine. It does not touch the hardware; the first ListDevices
// or LoadModel does.
func New(cfg Config) *Engine {
	cfg.withDefaults()
	return &Engine{
		cfg:      cfg,
		started:  time.Now(),
		models:   map[string]*LoadedModel{},
		sessions: map[string]*Session{},
		gates:    map[string]*gate{},
		aliases:  map[string]string{},
	}
}

// Config returns the configuration the Engine was built with, its defaults
// filled in.
func (e *Engine) Config() Config { return e.cfg }

// gate returns the gate for a device id, creating it on first use. The host's
// width comes from HostConcurrency and a device's from DeviceConcurrency.
func (e *Engine) gate(id string) *gate {
	e.gateMu.Lock()
	defer e.gateMu.Unlock()
	if g, ok := e.gates[id]; ok {
		return g
	}
	var g *gate
	if id == HostGateID {
		g = newGate(id, e.cfg.HostConcurrency,
			"every nn.JIT pins its own pool of decode cores, so concurrent host sessions "+
				"oversubscribe the same physical cores; width is Config.HostConcurrency")
	} else {
		g = newGate(id, e.cfg.DeviceConcurrency,
			"generates on sessions wholly on this device decode as rows of their model's "+
				"step loop, every row in one step; any other session holds the device's one "+
				"scratch set (g.bs/g.vbs/g.bbs) alone, so it is queued, never refused")
	}
	e.gates[id] = g
	return g
}

func (e *Engine) nextID(prefix string) string {
	return fmt.Sprintf("%s-%d-%d", prefix, time.Now().UnixNano()/1e6, e.seq.Add(1))
}

// ---------------------------------------------------------------- models

// LoadedModel is one model the Engine holds open: its model.Model, the
// device tier its sessions share, and the sessions created on it.
type LoadedModel struct {
	id       string
	path     string
	name     string
	m        *model.Model
	loadedAt time.Time

	// dev is the tier, shared by every session of this model: the tier owns
	// the model, a session owns the sequence. nil for a host-only model.
	dev       nn.Device
	gpu       *tier.GPU
	closeDev  func()
	deviceIDs []string
	// gateKeys is the physical devices the tier opened, one gate key each
	// (deviceGateKeys): what a session of this model serialises on. deviceIDs
	// is the spec as written and keys nothing, since `cuda` and `cuda:1` are
	// different text for one shared card.
	gateKeys []string
	// budget is the total the pager is divided from -- the load's, or the
	// last SetPageBudget's -- before the subtractions in hostWeightShare.
	// Guarded by mu.
	budget    uint64
	maxBlocks int

	// loop batches this model's device generates (batch.go). nil for a
	// host-only model, or with batching off. Set before the model is
	// published and never changed.
	loop *stepLoop

	// mu guards the model-level mutations: page budget, and the placement
	// default. Session work does not take it.
	mu sync.Mutex

	sessions map[string]*Session

	// embMu guards emb, the idle embedders Embed reuses (embed.go).
	embMu sync.Mutex
	emb   embedders

	tokensGenerated atomic.Int64
	tokensPrefilled atomic.Int64
}

// gateIDs is what a session of this model must hold to run: one gate per
// physical device its tier opened, keyed on the device and not on the spec
// text, so models loaded with `cuda` and `cuda:1` share card 1's gate and no
// other. A model with no device runs on the host and queues there.
func (lm *LoadedModel) gateIDs() []string {
	if len(lm.gateKeys) == 0 {
		return []string{HostGateID}
	}
	// A partial placement runs its remaining blocks on the host, so the host
	// gate is held too. That is not over-locking: those blocks really do run
	// on the same cores another host session would want.
	return append(append([]string{}, lm.gateKeys...), HostGateID)
}

// deviceGateKeys is the gate key of every device a tier opened -- the
// physical identity it already dedupes on, or the backend and ordinal where
// the backend reports none (tier.Hardware) -- and records the ids a caller may
// name each device by, so GetDeviceQueue and ListSessions find its gate.
func (e *Engine) deviceGateKeys(g *tier.GPU) []string {
	hw := g.Hardware()
	keys := make([]string, 0, len(hw))
	e.gateMu.Lock()
	defer e.gateMu.Unlock()
	for _, h := range hw {
		keys = append(keys, h.Key)
		e.aliases[h.Ref] = h.Key
		if h.Name != "" {
			e.aliases[h.Name] = h.Key
		}
	}
	return keys
}

// gateKey is the gate a device id a caller wrote names: the physical device a
// loaded tier resolved it to, or the probe's identity for it, or the id itself
// when neither knows it (a device no model has opened has no queue).
func (e *Engine) gateKey(id string) string {
	if id == HostGateID {
		return id
	}
	e.gateMu.Lock()
	k, ok := e.aliases[id]
	e.gateMu.Unlock()
	if ok {
		return k
	}
	e.devMu.Lock()
	defer e.devMu.Unlock()
	for _, d := range e.devices {
		if d.ID == id && d.PhysicalID != "" {
			return d.PhysicalID
		}
	}
	return id
}

// infoGateKey is the gate a listed device's queue is on: its physical
// identity when the probe read one, so a card listed under two backends
// reports one queue.
func (e *Engine) infoGateKey(d DeviceInfo) string {
	if d.PhysicalID != "" {
		return d.PhysicalID
	}
	return e.gateKey(d.ID)
}

// LoadOptions is LoadModel's input in Go terms.
type LoadOptions struct {
	Path            string
	ModelID         string
	PageBudgetBytes uint64
	DeviceIDs       []string
	MaxDeviceBlocks int // -1 means "as many as fit"
	KVF16           *bool
	// Sessions is how many concurrent sessions every placed block reserves a
	// history for (tier.Config.Sessions). 0 and 1 are one: a second session
	// then gets what the first left over, and on a full card that is the host.
	Sessions int

	// tierConfig adjusts the tier after Sessions is set. Tests use it to put
	// the device into a configuration a gate has to run (a joint step the
	// device refuses); nothing on the wire reaches it.
	tierConfig func(*tier.Config)
}

// ErrNotFound is returned for an unknown model or session id.
var ErrNotFound = errors.New("server: not found")

// ErrInUse is returned when an unload would orphan live sessions.
var ErrInUse = errors.New("server: still in use")

// ErrExists is returned for a model or session id that is already taken.
var ErrExists = errors.New("server: already exists")

// ErrInvalid is returned for a request the caller has to change before it can
// succeed. It is never a server fault: mapped to Internal, it would tell a
// client to retry the identical request.
var ErrInvalid = errors.New("server: invalid request")

// ResolvePath turns a bare name into a path under ModelDir. An absolute path
// is taken as given.
func (e *Engine) ResolvePath(p string) string {
	if p == "" {
		return p
	}
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(e.cfg.ModelDir, p)
}

// LoadModel opens a container and, when devices were named, the tier that will
// hold its blocks. A GGUF is refused by model.Open with a NotConvertedError,
// which errors.go turns into a FailedPrecondition carrying the convert command.
func (e *Engine) LoadModel(o LoadOptions) (*LoadedModel, error) {
	if o.MaxDeviceBlocks == 0 {
		o.MaxDeviceBlocks = -1
	}
	path := e.ResolvePath(o.Path)
	id := o.ModelID
	if id == "" {
		id = e.nextID("m")
	}

	e.mu.Lock()
	if _, ok := e.models[id]; ok {
		e.mu.Unlock()
		return nil, fmt.Errorf("%w: model id %q is already loaded", ErrExists, id)
	}
	e.mu.Unlock()

	hostBudget := o.PageBudgetBytes
	if hostBudget == 0 {
		// Resolved once, before anything is allocated: sched.MemBudget reads
		// MemAvailable, so asking again after the weights are resident would
		// count the engine's own footprint.
		hostBudget = sched.MemBudget()
	}

	var (
		dev      nn.Device
		gpu      *tier.GPU
		closeDev = func() {}
		devIDs   []string
		gateKeys []string
	)
	if spec := strings.Join(o.DeviceIDs, ","); spec != "" && spec != HostGateID {
		es, err := tier.ParseDevices(spec)
		if err != nil {
			return nil, err
		}
		wantsDevice := false
		for _, x := range es {
			wantsDevice = wantsDevice || x.Device()
		}
		if wantsDevice {
			g, err := tier.OpenWith(
				tier.WithDevices(spec),
				tier.WithHostBudget(hostBudget),
				tier.WithConfig(func(c *tier.Config) {
					c.Sessions = o.Sessions
					// The step loop (newStepLoop below) runs steps of up to
					// MaxStepRows rows: the device reserves their scratch at
					// placement, or a card the blocks filled refuses the step.
					if e.cfg.MaxBatchRows != 1 {
						c.StepRows = model.MaxStepRows
					}
					if o.tierConfig != nil {
						o.tierConfig(c)
					}
				}),
			)
			if err != nil {
				// Only "auto" degrades to the host. A spec that names a
				// device asked for that device; silently answering on the
				// CPU would be a wrong answer with a right shape.
				if len(es) != 1 || es[0].API != "auto" {
					return nil, err
				}
			} else {
				dev, gpu, closeDev = g, g, g.Close
				devIDs = normaliseDeviceIDs(o.DeviceIDs)
				gateKeys = e.deviceGateKeys(g)
			}
		}
	}

	opts := []model.Option{model.WithPageBudget(hostBudget)}
	if o.KVF16 != nil {
		opts = append(opts, model.WithKVF16(*o.KVF16))
	}
	m, err := model.Open(path, opts...)
	if err != nil {
		closeDev()
		return nil, err
	}

	lm := &LoadedModel{
		id:        id,
		path:      path,
		name:      strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		m:         m,
		loadedAt:  time.Now(),
		dev:       dev,
		gpu:       gpu,
		closeDev:  closeDev,
		deviceIDs: devIDs,
		gateKeys:  gateKeys,
		budget:    hostBudget,
		maxBlocks: o.MaxDeviceBlocks,
		sessions:  map[string]*Session{},
	}
	if gpu != nil && e.cfg.MaxBatchRows != 1 {
		lm.loop = newStepLoop(e, lm)
	}

	e.mu.Lock()
	e.models[id] = lm
	e.mu.Unlock()
	return lm, nil
}

// normaliseDeviceIDs drops the specs that name no device, so gateIDs does not
// queue on a gate called "auto".
func normaliseDeviceIDs(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" || id == HostGateID {
			continue
		}
		// Strip a `=BYTES` budget suffix: it names a limit, not a device.
		if j := strings.IndexByte(id, '='); j >= 0 {
			id = id[:j]
		}
		out = append(out, id)
	}
	return out
}

// Model returns the loaded model with this id, or ErrNotFound.
func (e *Engine) Model(id string) (*LoadedModel, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	lm, ok := e.models[id]
	if !ok {
		return nil, fmt.Errorf("%w: model %q", ErrNotFound, id)
	}
	return lm, nil
}

// Models returns every loaded model, sorted by id.
func (e *Engine) Models() []*LoadedModel {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*LoadedModel, 0, len(e.models))
	for _, lm := range e.models {
		out = append(out, lm)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// UnloadModel closes a model. Every State must be closed before its Model, so
// force closes the sessions first rather than letting the ordering contract be
// violated by a caller who did not know about it.
func (e *Engine) UnloadModel(id string, force bool) (closed int, err error) {
	e.mu.Lock()
	lm, ok := e.models[id]
	if !ok {
		e.mu.Unlock()
		return 0, fmt.Errorf("%w: model %q", ErrNotFound, id)
	}
	var victims []*Session
	for _, s := range lm.sessions {
		victims = append(victims, s)
	}
	if len(victims) > 0 && !force {
		e.mu.Unlock()
		return 0, fmt.Errorf("%w: model %q has %d open session(s); pass force to close them",
			ErrInUse, id, len(victims))
	}
	delete(e.models, id)
	for _, s := range victims {
		delete(e.sessions, s.id)
	}
	e.mu.Unlock()

	for _, s := range victims {
		s.close()
	}
	closed = len(victims)
	// The loop goes after the sessions -- their generates hold its rows until
	// they end -- and before the model it steps.
	lm.closeLoop()
	lm.closeEmbedders()
	if cerr := lm.m.Close(); cerr != nil {
		err = cerr
	}
	lm.closeDev()
	return closed, err
}

// Close tears the whole engine down in the order the engine requires.
func (e *Engine) Close() {
	e.mu.Lock()
	models := make([]*LoadedModel, 0, len(e.models))
	for _, lm := range e.models {
		models = append(models, lm)
	}
	sessions := make([]*Session, 0, len(e.sessions))
	for _, s := range e.sessions {
		sessions = append(sessions, s)
	}
	e.models = map[string]*LoadedModel{}
	e.sessions = map[string]*Session{}
	e.mu.Unlock()

	for _, s := range sessions {
		s.close()
	}
	for _, lm := range models {
		lm.closeLoop()
		lm.closeEmbedders()
		lm.m.Close()
		lm.closeDev()
	}
}

func (lm *LoadedModel) closeLoop() {
	if lm.loop != nil {
		lm.loop.close()
	}
}

// ---------------------------------------------------------------- sessions

// Session is one sequence on a loaded model: a model.State, its sampler and
// the counters the stats calls report. Work on it is serialised by its own
// lock; a generate with only a model_id runs on an ephemeral one.
type Session struct {
	id      string
	modelID string
	lm      *LoadedModel
	st      *model.State
	maxSeq  int

	// mu serialises this session's own work. A session is one sequence, so two
	// concurrent generates on it are a caller error rather than a queue.
	mu sync.Mutex

	sampling model.Sampler
	created  time.Time

	lastUsed  atomic.Int64
	generated atomic.Int64
	prefilled atomic.Int64
	running   atomic.Bool
	queuePos  atomic.Int32

	// Every reported figure is a snapshot taken under s.mu. model.State's
	// fields are written by Forward with no lock, so reading them from a stats
	// handler mid-generate is a data race, and taking s.mu there would block
	// for the whole generation. The session publishes a snapshot at create,
	// after every placement change and at the end of every generate.
	snapPos       atomic.Int32
	snapKV        atomic.Uint64
	snapDevBlocks atomic.Int32
	snapAt        atomic.Int64
	snapPlacement atomic.Pointer[placementSnapshot]
	// snapBatched is whether a generate here would decode as a row of the
	// model's step loop (Engine.joins), as of the snapshot.
	snapBatched atomic.Bool

	cancelMu sync.Mutex
	cancel   context.CancelFunc

	closed atomic.Bool
}

// SessionOptions is CreateSession's input.
type SessionOptions struct {
	ModelID         string
	SessionID       string
	MaxSeq          int
	DeviceIDs       []string
	MaxDeviceBlocks int
	Relocate        bool
	Sampling        model.Sampler
}

// CreateSession builds a model.State and offers its blocks to the model's
// tier. The tier's Attach gives this State its own device session, which is
// what makes two sessions on one card correct -- see tier.GPU.Attach.
func (e *Engine) CreateSession(o SessionOptions) (*Session, error) {
	lm, err := e.Model(o.ModelID)
	if err != nil {
		return nil, err
	}
	if lm.m.IsEncoder() {
		return nil, fmt.Errorf("%w: model %q is an encoder (%s): it has no decoder session to generate "+
			"from; embed with it instead", ErrInvalid, lm.id, lm.m.Cfg.Arch)
	}
	maxSeq := o.MaxSeq
	if maxSeq <= 0 {
		maxSeq = e.cfg.DefaultMaxSeq
	}
	if maxSeq <= 0 {
		maxSeq = lm.m.Cfg.NCtx
	}
	id := o.SessionID
	if id == "" {
		id = e.nextID("s")
	}

	e.mu.Lock()
	if _, ok := e.sessions[id]; ok {
		e.mu.Unlock()
		return nil, fmt.Errorf("%w: session id %q", ErrExists, id)
	}
	e.mu.Unlock()

	st := lm.m.NewState(maxSeq)
	if lm.dev != nil {
		max := o.MaxDeviceBlocks
		if max == 0 {
			max = lm.maxBlocks
		}
		if max == 0 {
			max = -1
		}
		if err := st.SetDeviceLayers(lm.dev, max); err != nil {
			st.Close()
			return nil, err
		}
	}
	if o.Relocate {
		st.SetRelocate(true)
	}
	e.applyPageBudget(lm, st)

	s := &Session{
		id:       id,
		modelID:  lm.id,
		lm:       lm,
		st:       st,
		maxSeq:   maxSeq,
		sampling: o.Sampling,
		created:  time.Now(),
	}
	s.lastUsed.Store(time.Now().UnixMilli())
	s.mu.Lock()
	s.refresh()
	s.mu.Unlock()

	e.mu.Lock()
	// The model may have been unloaded while this State was being placed. An
	// unload closes the sessions it finds, so one registered after it would be
	// a State outliving its Model.
	if e.models[lm.id] != lm {
		e.mu.Unlock()
		st.Close()
		return nil, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.id)
	}
	if _, ok := e.sessions[id]; ok {
		e.mu.Unlock()
		st.Close()
		return nil, fmt.Errorf("%w: session id %q", ErrExists, id)
	}
	e.sessions[id] = s
	lm.sessions[id] = s
	e.mu.Unlock()
	return s, nil
}

// placementSnapshot is the session's placement as of the last time it was safe
// to read. See the snap* fields on session.
type placementSnapshot struct {
	runs        [][2]int
	declines    []model.DeviceDecline
	headOnDev   bool
	demotions   int
	relocations int
	reclaims    int
	relocating  bool
	takenAt     time.Time
}

// refresh republishes the snapshot. The caller MUST hold s.mu.
func (s *Session) refresh() {
	s.snapPos.Store(int32(s.st.Pos()))
	s.snapKV.Store(s.st.KVBytes())
	s.snapDevBlocks.Store(int32(s.st.GPULayers()))
	s.snapAt.Store(time.Now().UnixMilli())
	s.snapBatched.Store(s.lm.loop != nil && s.st.Steppable())
	s.snapPlacement.Store(&placementSnapshot{
		runs:        s.st.DeviceBlocks(),
		declines:    s.st.DeviceDeclines(),
		headOnDev:   s.st.HeadOnDevice(),
		demotions:   s.st.DeviceDemotions(),
		relocations: s.st.Relocations(),
		reclaims:    s.st.Reclaims(),
		relocating:  s.st.Relocating(),
		takenAt:     time.Now(),
	})
}

// placement returns the last published snapshot, never nil.
func (s *Session) placement() *placementSnapshot {
	if p := s.snapPlacement.Load(); p != nil {
		return p
	}
	return &placementSnapshot{}
}

// Session returns the open session with this id, or ErrNotFound.
func (e *Engine) Session(id string) (*Session, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	s, ok := e.sessions[id]
	if !ok {
		return nil, fmt.Errorf("%w: session %q", ErrNotFound, id)
	}
	return s, nil
}

// Sessions returns every open session, sorted by id.
func (e *Engine) Sessions() []*Session {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*Session, 0, len(e.sessions))
	for _, s := range e.sessions {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// CloseSession removes the session from the Engine and its model, cancels a
// generate in flight on it and closes its State once that generate has let
// go. An unknown id is ErrNotFound.
func (e *Engine) CloseSession(id string) error {
	e.mu.Lock()
	s, ok := e.sessions[id]
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf("%w: session %q", ErrNotFound, id)
	}
	delete(e.sessions, id)
	delete(s.lm.sessions, id)
	e.mu.Unlock()
	s.close()
	return nil
}

func (s *Session) close() {
	if s.closed.Swap(true) {
		return
	}
	s.cancelGeneration()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Close()
}

func (s *Session) cancelGeneration() bool {
	s.cancelMu.Lock()
	defer s.cancelMu.Unlock()
	if s.cancel == nil {
		return false
	}
	s.cancel()
	return true
}

// ---------------------------------------------------------------- generate

// PromptKind says which member of Prompt is meant. An explicit tag beats
// guessing from which field is non-empty: a caller may legitimately ask to
// generate from an EMPTY continuation of an existing session.
type PromptKind int

// The prompt kinds: none (a continuation of the session's own sequence),
// text to encode, token ids as given, and chat messages to render through the
// model's template.
const (
	PromptNone PromptKind = iota
	PromptText
	PromptIDs
	PromptChat
)

// ChatInput is a chat prompt: the messages, an optional system prompt given
// apart from them (HasSystem says it was), and what the template is asked to
// render.
type ChatInput struct {
	Messages            []model.ChatMessage
	System              string
	HasSystem           bool
	AddGenerationPrompt bool
	TemplateName        string
	// Tools is the tool list the template renders, a JSON array in the
	// OpenAI shape ([{"type": "function", "function": {...}}]); nil for none.
	Tools []byte
}

// Prompt is what a generate runs: the member Kind names is the one read.
type Prompt struct {
	Kind PromptKind
	Text string
	IDs  []int32
	Chat *ChatInput
}

// GenerateOptions is the one request shape: the Connect InferenceService and
// both HTTP shims build one, so there is no second generate path.
type GenerateOptions struct {
	SessionID string
	ModelID   string

	Prompt       Prompt
	MaxTokens    int
	Sampling     *model.Sampler
	Stop         []string
	Continue     bool
	Echo         bool
	QueueTimeout time.Duration
	// IgnoreEOS keeps generating past an end-of-generation token, up to
	// MaxTokens: the token is emitted and fed like any other (vLLM's
	// ignore_eos). Stop strings still end the generate.
	IgnoreEOS bool
}

// EventKind discriminates Event.
type EventKind int

// The event kinds, in the order a generate emits them: one Started, a Token
// per sampled token, one Finished.
const (
	EventStarted EventKind = iota
	EventToken
	EventFinished
)

// Started is a generate's first event, sent once the prompt is prefilled:
// how long the request queued and behind how many, where the model's blocks
// ran, and how long the prefill took.
type Started struct {
	SessionID    string
	ModelID      string
	Ephemeral    bool
	PromptTokens int
	QueuedFor    time.Duration
	QueueDepth   int32
	DeviceBlocks int
	HostBlocks   int
	DeviceIDs    []string
	Prefill      time.Duration
	Execution    ExecutionMode
	// Batched is set when the generate decodes as a row of its model's step
	// loop (batch.go) rather than alone on its gates.
	Batched bool
}

// Token is one step of the output. Every sampled token is sent, its Text
// empty while held back for a stop string or a partial rune; ID -1 carries
// text alone (the echoed prompt at Index -1, or the held-back tail).
type Token struct {
	ID    int32
	Text  string
	Index int
}

// FinishReason mirrors the proto enum.
type FinishReason int

// The reasons a generate ends: a stop string matched, MaxTokens reached, an
// end-of-generation token sampled, the context cancelled, or an error.
const (
	FinishUnspecified FinishReason = iota
	FinishStop
	FinishMaxTokens
	FinishEOS
	FinishCancelled
	FinishError
)

// Finished is a generate's last event: why it ended, the token counts, the
// prefill and decode times, the decode rate with the bytes a token reads (so
// the rate can be checked against the read wall), and the session's position.
type Finished struct {
	Reason           FinishReason
	StopMatched      string
	PromptTokens     int
	CompletionTokens int
	Prefill          time.Duration
	Decode           time.Duration
	TokensPerSecond  float64
	BytesPerToken    uint64
	Position         int
}

// Event is one message of a generate's stream: Kind says which of Started,
// Token and Finished is set.
type Event struct {
	Kind     EventKind
	Started  *Started
	Token    *Token
	Finished *Finished
}

// tokenLimit is a request's completion length: what it asked for, never more
// than room, the positions left in the session's context. A request that asks
// for none runs until the model ends its reply or fills the context, as
// OpenAI's API reads an absent max_tokens. Either way a reply that reaches the
// end of the context finishes as FinishMaxTokens, not on a full KV cache.
func tokenLimit(asked, room int) int {
	room = max(room, 0)
	if asked <= 0 || asked > room {
		return room
	}
	return asked
}

// Generate is the one generate path. The first event reports how long the
// request queued -- for a gate, or for a row of its model's step loop -- and
// how many were ahead, so a caller does not mistake a queue for a slow model.
func (e *Engine) Generate(ctx context.Context, o GenerateOptions, emit func(Event) error) (err error) {
	s, ephemeral, err := e.resolveSession(o)
	if err != nil {
		return err
	}
	if ephemeral {
		defer e.CloseSession(s.id)
	}

	// One sequence, one generate. A second concurrent call on the same session
	// is a caller error and blocking here is the kind answer to it.
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() {
		return fmt.Errorf("%w: session %q is closed", ErrNotFound, s.id)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	s.cancelMu.Lock()
	s.cancel = cancel
	s.cancelMu.Unlock()
	defer func() {
		s.cancelMu.Lock()
		s.cancel = nil
		s.cancelMu.Unlock()
	}()

	lm := s.lm
	if lm.m.Vocab == nil {
		return fmt.Errorf("server: model %q has no tokenizer: %v", lm.id, lm.m.TokErr)
	}

	ids, err := e.encode(lm, o.Prompt)
	if err != nil {
		return err
	}
	if len(ids) == 0 && !o.Continue {
		return fmt.Errorf("%w: the prompt is empty and continue_session was not set", ErrInvalid)
	}
	if len(ids) == 0 {
		return fmt.Errorf("%w: continue_session with no prompt has no token to run", ErrInvalid)
	}

	if lp := e.joins(s); lp != nil {
		return e.generateBatched(ctx, lp, s, o, ids, ephemeral, emit)
	}

	// ---- the queue, timed.
	gs := e.gatesFor(lm.gateIDs())
	s.queuePos.Store(1)
	waited, depth, err := gs.acquire(ctx, s.id, o.QueueTimeout)
	s.queuePos.Store(0)
	if err != nil {
		return err
	}
	defer gs.release()

	s.running.Store(true)
	defer s.running.Store(false)
	s.lastUsed.Store(time.Now().UnixMilli())

	// ---- prefill.
	if !o.Continue {
		s.st.Reset()
	}
	prefillStart := time.Now()
	logits, err := s.st.Prefill(ids)
	if err != nil {
		return err
	}
	s.prefilled.Add(int64(len(ids)))
	lm.tokensPrefilled.Add(int64(len(ids)))
	prefill := time.Since(prefillStart)

	devBlocks := s.st.GPULayers()
	if err := emit(Event{Kind: EventStarted, Started: &Started{
		SessionID:    s.id,
		ModelID:      lm.id,
		Ephemeral:    ephemeral,
		PromptTokens: len(ids),
		QueuedFor:    waited,
		QueueDepth:   depth,
		DeviceBlocks: devBlocks,
		HostBlocks:   lm.m.Cfg.NLayer - devBlocks,
		DeviceIDs:    lm.deviceIDs,
		Prefill:      prefill,
		Execution:    gs.mode(),
	}}); err != nil {
		return err
	}

	if o.Echo {
		if err := emit(Event{Kind: EventToken, Token: &Token{
			ID: -1, Text: lm.m.Vocab.Decode(ids), Index: -1,
		}}); err != nil {
			return err
		}
	}

	// ---- decode.
	sampler := s.sampling
	if o.Sampling != nil {
		sampler = *o.Sampling
	}
	maxTokens := tokenLimit(o.MaxTokens, s.st.MaxSeq()-s.st.Pos())

	st := newStreamText(lm.m.Vocab, o.Stop)
	reason := FinishMaxTokens
	stopMatched := ""
	out := make([]int32, 0, min(maxTokens, 4096))
	decodeStart := time.Now()
	n := 0

	for ; n < maxTokens; n++ {
		if ctx.Err() != nil {
			reason = FinishCancelled
			break
		}
		next := sampler.Sample(logits)
		sampler.Observe(next)
		if !o.IgnoreEOS && lm.m.Vocab.IsEOG(next) {
			reason = FinishEOS
			break
		}
		out = append(out, next)

		// Every token is sent, its text empty while held back (a partial rune,
		// or a possible stop string's start): a client counting ids must see
		// all of them, as GenerateToken says.
		chunk, hit, match := st.push(out)
		if err := emit(Event{Kind: EventToken, Token: &Token{ID: next, Text: chunk, Index: n}}); err != nil {
			return err
		}
		if hit {
			reason, stopMatched = FinishStop, match
			n++
			break
		}
		if logits, err = s.st.Forward(next); err != nil {
			return err
		}
	}
	if reason == FinishMaxTokens || reason == FinishEOS || reason == FinishCancelled {
		// Flush whatever was being held back for a stop string that never came.
		if tail := st.flush(); tail != "" {
			if err := emit(Event{Kind: EventToken, Token: &Token{ID: -1, Text: tail, Index: n}}); err != nil {
				return err
			}
		}
	}
	decode := time.Since(decodeStart)
	s.generated.Add(int64(n))
	lm.tokensGenerated.Add(int64(n))
	s.lastUsed.Store(time.Now().UnixMilli())
	s.refresh()

	rate := 0.0
	if decode > 0 && n > 0 {
		rate = float64(n) / decode.Seconds()
	}
	return emit(Event{Kind: EventFinished, Finished: &Finished{
		Reason:           reason,
		StopMatched:      stopMatched,
		PromptTokens:     len(ids),
		CompletionTokens: n,
		Prefill:          prefill,
		Decode:           decode,
		TokensPerSecond:  rate,
		// Bytes per token makes the rate checkable against the read wall.
		BytesPerToken: lm.m.BytesPerToken(),
		Position:      s.st.Pos(),
	}})
}

// resolveSession returns the named session, or creates an ephemeral one for a
// model_id request.
func (e *Engine) resolveSession(o GenerateOptions) (*Session, bool, error) {
	if o.SessionID != "" && o.ModelID != "" {
		return nil, false, fmt.Errorf("%w: give session_id or model_id, not both", ErrInvalid)
	}
	if o.SessionID != "" {
		s, err := e.Session(o.SessionID)
		return s, false, err
	}
	if o.ModelID == "" {
		return nil, false, fmt.Errorf("%w: one of session_id or model_id is required", ErrInvalid)
	}
	s, err := e.CreateSession(SessionOptions{ModelID: o.ModelID})
	if err != nil {
		return nil, false, err
	}
	return s, true, nil
}

// encode turns a Prompt into token ids. A rendered chat template is a
// complete prompt and must be encoded without added specials, or the first
// turn gets a doubled BOS; model.ChatIDs owns that pairing, so chat goes
// through it rather than ChatPrompt-then-Encode.
func (e *Engine) encode(lm *LoadedModel, p Prompt) ([]int32, error) {
	switch p.Kind {
	case PromptIDs:
		return p.IDs, nil
	case PromptText:
		return lm.m.Vocab.Encode(p.Text, true), nil
	case PromptChat:
		if p.Chat == nil {
			return nil, fmt.Errorf("%w: chat prompt with no messages", ErrInvalid)
		}
		if !lm.m.ChatCapable() {
			return nil, fmt.Errorf("%w: model %q carries no chat template, so a chat request "+
				"would run as a RAW COMPLETION -- which produces fluent output and no signal at all; "+
				"send a text prompt instead", ErrInvalid, lm.id)
		}
		msgs := p.Chat.Messages
		if p.Chat.HasSystem {
			// Anthropic puts the system prompt in a top-level field; the
			// template takes a message list, so it is prepended.
			msgs = append([]model.ChatMessage{{Role: "system", Content: p.Chat.System}}, msgs...)
		}
		return lm.m.ChatIDsTools(msgs, p.Chat.Tools, p.Chat.AddGenerationPrompt)
	case PromptNone:
		return nil, nil
	}
	return nil, fmt.Errorf("%w: unknown prompt kind %d", ErrInvalid, p.Kind)
}

// ---------------------------------------------------------------- streaming text

// streamText turns a growing token id list into incremental text, and applies
// stop strings.
//
// Neither tokenizer decodes per token (a BPE token can split a rune, SPM's
// leading space depends on position), so each step re-decodes the whole id
// list and diffs the prefix. DecodeChat, not Decode, so a harmony vocabulary
// does not leave role and channel headers glued to the text.
//
// The tail is held back so the start of a stop string never reaches the
// client before the match completes; the window is one byte short of the
// longest stop string.
type streamText struct {
	vocab   vocabDecoder
	stops   []string
	hold    int
	decoded string
	emitted int
}

type vocabDecoder interface {
	DecodeChat(ids []int32) string
	Decode(ids []int32) string
}

func newStreamText(v vocabDecoder, stops []string) *streamText {
	hold := 0
	for _, s := range stops {
		if len(s) > hold {
			hold = len(s)
		}
	}
	if hold > 0 {
		hold--
	}
	return &streamText{vocab: v, stops: stops, hold: hold}
}

// push takes the full id list so far and returns the text safe to emit now,
// whether a stop string matched, and which one.
func (t *streamText) push(ids []int32) (chunk string, stopped bool, match string) {
	t.decoded = t.vocab.DecodeChat(ids)

	// A stop string is matched against the text generated SINCE the emit
	// pointer started, i.e. the whole completion, because a stop may straddle
	// any number of tokens.
	for _, s := range t.stops {
		if s == "" {
			continue
		}
		if i := strings.Index(t.decoded, s); i >= 0 {
			// Emit up to the stop and no further. The stop string itself is
			// never part of the completion.
			if i > t.emitted {
				chunk = t.decoded[t.emitted:i]
			}
			t.emitted = len(t.decoded)
			return truncPartialRune(chunk), true, s
		}
	}

	safe := len(t.decoded) - t.hold
	if safe <= t.emitted {
		return "", false, ""
	}
	chunk = truncPartialRune(t.decoded[t.emitted:safe])
	t.emitted += len(chunk)
	return chunk, false, ""
}

// flush releases the held-back tail once no stop string can still match it.
func (t *streamText) flush() string {
	if len(t.decoded) <= t.emitted {
		return ""
	}
	chunk := t.decoded[t.emitted:]
	t.emitted = len(t.decoded)
	return chunk
}

// truncPartialRune drops a trailing incomplete UTF-8 sequence so a multi-byte
// rune is never split across two writes.
func truncPartialRune(s string) string {
	for s != "" && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

// ---------------------------------------------------------------- scanning

// ModelFileInfo is one entry ListModels reports from disk.
type ModelFileInfo struct {
	Path           string
	Name           string
	Size           uint64
	IsContainer    bool
	ConvertCommand string
}

// ScanModels lists containers and GGUFs in a directory. A GGUF is listed WITH
// the command that would convert it, because the engine reads one weight
// format and a caller who cannot see the other file has no way to learn why
// their model is missing.
func (e *Engine) ScanModels(dir string) ([]ModelFileInfo, string, error) {
	if dir == "" {
		dir = e.cfg.ModelDir
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, dir, err
	}
	var out []ModelFileInfo
	for _, ent := range ents {
		if ent.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(ent.Name()))
		if ext != model.Ext && ext != ".gguf" {
			continue
		}
		fi, err := ent.Info()
		if err != nil {
			continue
		}
		p := filepath.Join(dir, ent.Name())
		m := ModelFileInfo{
			Path:        p,
			Name:        strings.TrimSuffix(ent.Name(), filepath.Ext(ent.Name())),
			Size:        uint64(fi.Size()),
			IsContainer: ext == model.Ext,
		}
		if !m.IsContainer {
			m.ConvertCommand = fmt.Sprintf("jitllm convert %s %s",
				p, strings.TrimSuffix(p, filepath.Ext(p))+model.Ext)
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, dir, nil
}

// ---------------------------------------------------------------- accessors
//
// The engine's own fields stay unexported so a caller cannot reach past the
// gates into a model.State. These are the read-only handles the command and
// the handlers need.

// ID is the stable id this model was loaded under.
func (lm *LoadedModel) ID() string { return lm.id }

// Name is the container's file name without its extension. The container
// carries no model name, so this is not an identity.
func (lm *LoadedModel) Name() string { return lm.name }

// Path is the container this model was opened from.
func (lm *LoadedModel) Path() string { return lm.path }

// DeviceIDs is the tier this model's blocks are offered to. Empty means the
// model is host-only and has no tier at all.
func (lm *LoadedModel) DeviceIDs() []string {
	return append([]string(nil), lm.deviceIDs...)
}

// ID is the stable id this session was created under.
func (s *Session) ID() string { return s.id }

// ModelID is the model this session holds a sequence of.
func (s *Session) ModelID() string { return s.modelID }

// hostWeightShare is what the HOST may spend on paged weights, given a total
// budget: the dense region never pages, and skipping that subtraction
// double-spends -- a budget one page short costs a read every token. What a
// tier on the host's own memory holds (tier.GPU.HostReserved) is not taken
// here: Model.SetPageBudget takes it off itself, and again whenever a block
// moves, so taking it here too would count it twice.
func (lm *LoadedModel) hostWeightShare(total uint64) uint64 {
	avail := total
	if d := lm.m.DenseBytes(); d < avail {
		avail -= d
	} else {
		return 0
	}
	return avail
}

// pagerBudget is a share as the pager must be handed it. The pager reads ZERO
// as "unlimited", so a share that came to nothing -- a total smaller than what
// the dense region and the KV already claim -- becomes
// one byte: one page at a time, faulted in and out, which is what so small a
// budget means. Handed over as zero it would make the smallest budget a
// caller can ask for into the largest.
func pagerBudget(share uint64) uint64 { return max(share, 1) }

// applyPageBudget re-sizes the pager now that this model's placement and its
// sessions' KV caches are known: the host needs pages only for the blocks the
// devices did not take, and KV is committed memory the weight budget must not
// also claim. KV is summed over every session of the model.
//
// It is an Engine method because lm.sessions is written under e.mu, not
// lm.mu.
func (e *Engine) applyPageBudget(lm *LoadedModel, newest *model.State) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	if lm.m.PageSize() == 0 || lm.budget == 0 {
		return // not a container, or no budget to divide
	}
	avail := lm.hostWeightShare(lm.budget)

	// The newest session is not in the map yet, so it is counted separately.
	kv := uint64(newest.MaxSeq()) * uint64(lm.m.Cfg.KVDim()) * 4 * 2 * uint64(lm.m.Cfg.NLayer)
	minDev := newest.GPULayers()
	e.mu.RLock()
	for _, s := range lm.sessions {
		kv += s.snapKV.Load()
		if d := int(s.snapDevBlocks.Load()); d < minDev {
			minDev = d
		}
	}
	e.mu.RUnlock()

	if kv < avail {
		avail -= kv
	} else {
		avail = 0
	}
	// The host block count follows the session with the fewest device
	// blocks: any block some session runs on the host needs a page.
	if lm.m.Cfg.NLayer-minDev <= 0 {
		return // every block is on a device for every session; nothing faults here
	}
	lm.m.SetPageBudget(pagerBudget(avail))
}
