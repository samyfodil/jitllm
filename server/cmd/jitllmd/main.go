// Command jitllmd is the engine's daemon, and its client.
//
// It serves one Engine over three surfaces at once: a ConnectRPC control
// plane, an OpenAI-compatible HTTP API and an Anthropic-compatible one. All
// three are projections of the same server.Engine; there is no second
// inference path behind any of them.
//
//	jitllmd serve -addr :8080 -models /path/to/models
//	jitllmd serve -load Llama-3.2-1B-Instruct-Q4_K_M.jlm -devices cuda:0
//
// The other verbs are a client of that same control plane, through the same
// generated stubs:
//
//	jitllmd run -addr box:8080 -n 64 "the capital of France is"
//	jitllmd devices           jitllmd models          jitllmd stats -watch
//	jitllmd sessions -new -model m-1 -devices cuda:0
//	jitllmd place -session s-1 -gpu-layers 22
//
// The client lives here, not in cmd/jitllm, so the root module keeps a single
// dependency. Verbs rather than bare flags, matching cmd/jitllm.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/samyfodil/jitllm/engine/model"
	"github.com/samyfodil/jitllm/internal/cmd/goheap"
	"github.com/samyfodil/jitllm/jit/gpu/tier"
	"github.com/samyfodil/jitllm/server"
)

// verb is one client subcommand. `serve` is not one: it owns the process for
// its whole life and exits through the signal handler rather than by returning.
type verb struct {
	name string
	run  func(context.Context, cli, []string) error
	one  string
}

func verbs() []verb {
	return []verb{
		{"run", cmdRun, "stream a generation from a running daemon"},
		{"models", cmdModels, "list, describe, load and unload models"},
		{"devices", cmdDevices, "enumerate the hardware and the memory that is actually spendable"},
		{"sessions", cmdSessions, "create, list, describe, reset and close sessions"},
		{"place", cmdPlace, "read and move the CPU/device block seam, and the pager budget"},
		{"stats", cmdStats, "engine counters, once or streamed"},
	}
}

// version is the release version, set at link time
// (-ldflags "-X main.version=v0.1.0"; see .goreleaser.yaml).
var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		serve(os.Args[2:])
		return
	case "-h", "--help", "help":
		usage()
		return
	}
	for _, v := range verbs() {
		if v.name != os.Args[1] {
			continue
		}
		// ^C cancelling the context is the only bound on a generate (see
		// dial); the server reads the closed stream as a cancellation.
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		exit(v.run(ctx, stdio(), os.Args[2:]))
		return
	}
	fmt.Fprintf(os.Stderr, "jitllmd: unknown command %q\n\n", os.Args[1])
	usage()
	os.Exit(2)
}

// exit turns a verb's error into a status: 2 for a usage error, 1 for a
// failed operation. A flag error has already been printed by the FlagSet.
func exit(err error) {
	if err == nil {
		return
	}
	if errors.Is(err, errHandled) {
		os.Exit(2)
	}
	fmt.Fprintf(os.Stderr, "jitllmd: %v\n", err)
	var ue usageErr
	if errors.As(err, &ue) {
		os.Exit(2)
	}
	os.Exit(1)
}

func usage() {
	fmt.Fprint(os.Stderr, `jitllmd serves the jitllm engine over the network, and talks to one.

serving:
  jitllmd serve [flags]     run the daemon

client (add -addr host:port; the default is `+defaultAddr+`):
`)
	for _, v := range verbs() {
		fmt.Fprintf(os.Stderr, "  jitllmd %-9s %s\n", v.name, v.one)
	}
	fmt.Fprint(os.Stderr, `
  jitllmd <verb> -h         the flags for a verb

Three surfaces are served on one address: the ConnectRPC control plane
(models, devices, placement, sessions, inference, telemetry), an
OpenAI-compatible API at /v1/chat/completions, and an Anthropic-compatible
one at /v1/messages. The client verbs above use the first of the three.
`)
}

// defaultModels is -models' default: JITLLM_MODELS, the variable `jitllm
// fetch` and `jitllm convert` read, or ./models.
func defaultModels() string {
	if d := os.Getenv("JITLLM_MODELS"); d != "" {
		return d
	}
	return "models"
}

func serve(args []string) {
	fs := flag.NewFlagSet("jitllmd serve", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "listen address")
	models := fs.String("models", defaultModels(),
		"directory scanned by ListModels and used to resolve a bare model name (JITLLM_MODELS sets the default)")
	load := fs.String("load", "",
		"a .jlm container to load at startup, absolute or relative to -models")
	modelID := fs.String("id", "", "model id for -load (default: derived)")
	devices := fs.String("devices", "",
		"devices for -load: auto|all|cpu|gpu[:N]|(cuda|vulkan|metal)[:DEVICE][=BYTES], comma-separated; "+
			"a bare gpu, cuda or vulkan is every such device, a :DEVICE selector pins one")
	maxmem := fs.String("maxmem", "",
		"page budget for -load, e.g. 8G (default: the engine's own budget)")
	gpuLayers := fs.Int("gpu-layers", -1, "blocks to place on a device for -load; -1 is as many as fit")
	sessions := fs.Int("sessions", 1, "concurrent sessions each linear device block reserves a recurrent state for, for -load (attention history is paged)")
	maxSeq := fs.Int("max-seq", 0, "default KV capacity per session, in positions (default: the model's context length)")
	maxBatch := fs.Int("max-batch", 0,
		"generates of one device model that decode as rows of one step. 0 is the engine's "+
			"bound (the widest step a device runs across sessions); 1 turns batching off")
	promptChunk := fs.Int("prompt-chunk", 0,
		"prompt tokens fed into one shared step when a request joins a batch. 0 is one "+
			"device prefill chunk")
	jointSteps := fs.String("joint-steps", "auto",
		"how a batch's decode step runs: auto (time joint against one session after another, "+
			"per row count, and run the faster), always (one joint step) or never (each session alone)")
	warm := fs.Bool("warm", true,
		"after every load, run a short prefill and a decode step so the first request does not pay for them")
	noStore := fs.Bool("no-mem-cache", false,
		"turn off each model's memory cache: every request prefills its whole prompt")
	storeMax := fs.String("mem-cache-max", "",
		"bound each model's memory cache, e.g. 2G (default: an eighth of the model's host share, "+
			"and no more than an eighth of what the host has free)")
	hostMem := fs.String("host-mem", "",
		"host memory the loaded models divide, e.g. 64G (default: re-read at each load, eight tenths "+
			"of the smallest of the cgroup limit, the bound NUMA nodes' memory and what is available)")
	pool := fs.Int("session-pool", 0,
		"reset sessions each model keeps for model_id requests; 0 is -sessions, -1 none")
	maxQueue := fs.Int("max-queue", 0,
		"requests one model holds, running or waiting, before it answers 429; 0 is 64, -1 unbounded")
	retry := fs.Duration("retry-after", time.Second, "the Retry-After a 429 for a full queue names")
	kvF16 := fs.Bool("kv-f16", false,
		"the KV cache width of every load that names none: true binary16, false f32 (default: the engine's per-host choice)")
	kvCache := fs.String("kv-cache", "",
		"a directory sessions created with prompt_cache keep their prompt prefixes in; empty refuses such a session")
	kvCacheMax := fs.String("kv-cache-max", "8G", "what -kv-cache may occupy before its least recently used pages go; 0 is unbounded")
	version := fs.String("version", version, "version string reported by GetServerInfo")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if err := misplacedFlag(fs); err != nil {
		fatal("%v", err)
	}

	// Before anything is loaded: the daemon holds page frames for as long as
	// it lives, and a collector that cannot see the cgroup plans a heap goal
	// past it (AGENTS.md RULE 2f).
	if lim := goheap.Cap(); lim > 0 {
		fmt.Fprintf(os.Stderr, "go memory limit %.2f GiB\n", float64(lim)/(1<<30))
	}

	var budget uint64
	if *maxmem != "" {
		b, err := tier.ParseBytes(*maxmem)
		if err != nil {
			fatal("-maxmem: %v", err)
		}
		budget = b
	}

	var storeBytes uint64
	if *storeMax != "" {
		b, err := tier.ParseBytes(*storeMax)
		if err != nil {
			fatal("-mem-cache-max: %v", err)
		}
		storeBytes = b
	}

	var hostBytes uint64
	if *hostMem != "" {
		b, err := tier.ParseBytes(*hostMem)
		if err != nil {
			fatal("-host-mem: %v", err)
		}
		hostBytes = b
	}

	joint, ok := map[string]server.JointSteps{
		"auto": server.JointMeasured, "always": server.JointAlways, "never": server.JointNever,
	}[*jointSteps]
	if !ok {
		fatal("-joint-steps %q: want auto, always or never", *jointSteps)
	}
	var kvWidth *bool
	if wasSet(fs, "kv-f16") {
		kvWidth = kvF16
	}
	var store model.KVStore
	if *kvCache != "" {
		max, err := tier.ParseBytes(*kvCacheMax)
		if err != nil {
			fatal("-kv-cache-max: %v", err)
		}
		fsStore, err := model.NewFileStoreLimit(*kvCache, max)
		if err != nil {
			fatal("-kv-cache: %v", err)
		}
		store = fsStore
	}
	e := server.New(server.Config{
		KVF16:         kvWidth,
		OffHeap:       goheap.OffHeap,
		PromptStore:   store,
		ModelDir:      *models,
		MaxBatchRows:  *maxBatch,
		PromptChunk:   *promptChunk,
		JointSteps:    joint,
		DefaultMaxSeq: *maxSeq,
		Version:       *version,

		NoMemCache:    *noStore,
		MemCacheBytes: storeBytes,
		HostBudget:    hostBytes,
		SessionPool:   *pool,
		MaxQueue:      *maxQueue,
		RetryAfter:    *retry,
		WarmLoads:     *warm,
	})
	defer e.Close()

	if *load != "" {
		o := server.LoadOptions{
			Path:            *load,
			ModelID:         *modelID,
			PageBudgetBytes: budget,
			MaxDeviceBlocks: *gpuLayers,
			Sessions:        *sessions,
		}
		if *devices != "" {
			o.DeviceIDs = strings.Split(*devices, ",")
		}
		start := time.Now()
		lm, err := e.LoadModel(o)
		if err != nil {
			// Print the engine's own error: for a GGUF it is the convert
			// command.
			fatal("%v", err)
		}
		fmt.Fprintf(os.Stderr, "loaded %s in %s\n", lm.ID(), time.Since(start).Round(time.Millisecond))
	}

	srv := &http.Server{
		Addr:    *addr,
		Handler: e.Handler(),
		// No WriteTimeout: a streaming generate on an over-committed model
		// can run for minutes. The write side is bounded by the client's
		// context.
		ReadHeaderTimeout: 20 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errc := make(chan error, 1)
	go func() {
		fmt.Fprintf(os.Stderr, "jitllmd on %s\n", *addr)
		fmt.Fprintf(os.Stderr, "  connect   http://%s/jitllm.v1.ModelService/ListModels\n", hostOf(*addr))
		fmt.Fprintf(os.Stderr, "  openai    http://%s/v1/chat/completions\n", hostOf(*addr))
		fmt.Fprintf(os.Stderr, "  anthropic http://%s/v1/messages\n", hostOf(*addr))
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fatal("%v", err)
		}
	case <-ctx.Done():
		fmt.Fprintln(os.Stderr, "\nshutting down")
		sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(sctx)
	}
}

// misplacedFlag refuses a flag that arrived after the first positional, which
// Go's flag package would otherwise ignore silently (the same guard as
// cmd/jitllm). It matches the flags the set defines rather than a leading
// dash, so an argument that merely looks like a flag is still an argument.
func misplacedFlag(fs *flag.FlagSet) error {
	known := map[string]bool{}
	fs.VisitAll(func(f *flag.Flag) { known[f.Name] = true })
	for _, a := range fs.Args() {
		if !strings.HasPrefix(a, "-") {
			continue
		}
		name := strings.TrimLeft(a, "-")
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		if known[name] {
			return fmt.Errorf("flag -%s came after a positional argument, where Go's flag "+
				"package stops parsing; put it before", name)
		}
	}
	return nil
}

func hostOf(addr string) string {
	if strings.HasPrefix(addr, ":") {
		return "localhost" + addr
	}
	return addr
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "jitllmd: "+format+"\n", a...)
	os.Exit(1)
}
