package main

import (
	"context"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/samyfodil/jitllm/server"
	v1 "github.com/samyfodil/jitllm/server/gen/jitllm/v1"
)

// humanGiB reads the byte figure pattern captures out of text, in GiB.
func humanGiB(t *testing.T, text, pattern string) float64 {
	t.Helper()
	m := regexp.MustCompile(pattern).FindStringSubmatch(text)
	if m == nil {
		t.Fatalf("no %s in:\n%s", pattern, text)
	}
	num, unit, _ := strings.Cut(m[1], " ")
	v, err := strconv.ParseFloat(num, 64)
	if err != nil {
		t.Fatalf("%q is not a byte figure", m[1])
	}
	scale := map[string]float64{"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}[unit]
	if scale == 0 {
		t.Fatalf("%q has no unit this reads", m[1])
	}
	return v * scale / (1 << 30)
}

// ---------------------------------------------------------------- the wire

// TestTheVerbsDriveARealServerOverRealHTTP runs every verb over a socket
// against the real server.Engine behind server.Handler(). None of these need a
// model on disk.
func TestTheVerbsDriveARealServerOverRealHTTP(t *testing.T) {
	addr, _ := engineServer(t, server.Config{Version: "test-1"})

	t.Run("stats", func(t *testing.T) {
		got := out(t, cmdStats, "-addr", addr)
		if !strings.Contains(got, "jitllmd test-1") {
			t.Fatalf("the server's own version did not reach the client:\n%s", got)
		}
		if !strings.Contains(got, "0 model(s), 0 session(s)") {
			t.Fatalf("stats did not report the empty engine:\n%s", got)
		}
	})

	t.Run("models", func(t *testing.T) {
		got := out(t, cmdModels, "-addr", addr)
		if !strings.Contains(got, "loaded (0)") {
			t.Fatalf("models did not report an empty load list:\n%s", got)
		}
		if !strings.Contains(got, "directory ") {
			t.Fatalf("models did not report which directory it scanned:\n%s", got)
		}
	})

	t.Run("sessions", func(t *testing.T) {
		got := out(t, cmdSessions, "-addr", addr)
		if !strings.Contains(got, "0 session(s)") {
			t.Fatalf("sessions did not report an empty list:\n%s", got)
		}
	})

	t.Run("device queue reports WHY it is a queue", func(t *testing.T) {
		got := out(t, cmdSessions, "-addr", addr, "-queue", "cuda:0")
		if !strings.Contains(got, "serialised") {
			t.Fatalf("the queue did not report its execution mode:\n%s", got)
		}
		// The note explains why the device serialises sessions.
		if !strings.Contains(got, "scratch") {
			t.Fatalf("the queue did not carry the reason it serialises:\n%s", got)
		}
	})
}

// ---------------------------------------------------------------- errors

// TestAServerErrorArrivesAsTheServersOwnSentence: a .gguf load must print
// the server's sentence, which carries the convert command. It needs no file
// (Open checks the extension first). With clientError reduced to "request
// failed" it fails with "the server's sentence did not reach the user".
func TestAServerErrorArrivesAsTheServersOwnSentence(t *testing.T) {
	addr, _ := engineServer(t, server.Config{})

	var o, e syncBuffer
	err := runVerb(t, cmdModels, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-load", "/mnt/models/tinyllama.gguf"},
		20*time.Second)
	if err == nil {
		t.Fatal("loading a .gguf succeeded; the engine reads one weight format")
	}
	msg := err.Error()
	if !strings.Contains(msg, "jitllm convert") {
		t.Fatalf("the server's sentence did not reach the user; a GGUF's error carries the "+
			"convert command and this one says only: %s", msg)
	}
	// The code travels with the sentence, for scripts to branch on.
	if !strings.Contains(msg, "failed_precondition") {
		t.Fatalf("the connect code did not reach the user, so a caller cannot tell a mistake "+
			"from a fault: %s", msg)
	}
}

// TestAnUnknownIdIsNotFoundAndSaysWhichOne: not_found, not internal.
func TestAnUnknownIdIsNotFoundAndSaysWhichOne(t *testing.T) {
	addr, _ := engineServer(t, server.Config{})

	var o, e syncBuffer
	err := runVerb(t, cmdPlace, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-session", "nope"}, 20*time.Second)
	if err == nil {
		t.Fatal("placement on an unknown session succeeded")
	}
	if !strings.Contains(err.Error(), "not_found") || !strings.Contains(err.Error(), `"nope"`) {
		t.Fatalf("the refusal does not name the missing session as not_found: %v", err)
	}
}

// TestNoDaemonSaysWhichAddressItTried: when nothing answers, the client says
// which address it dialled and how to start a daemon.
func TestNoDaemonSaysWhichAddressItTried(t *testing.T) {
	// Port 1 needs no listener to be refused, and is not something a developer
	// box has running.
	var o, e syncBuffer
	err := runVerb(t, cmdDevices, cli{out: &o, err: &e},
		[]string{"-addr", "127.0.0.1:1"}, 20*time.Second)
	if err == nil {
		t.Fatal("dialling a dead port succeeded")
	}
	if !strings.Contains(err.Error(), "127.0.0.1:1") {
		t.Fatalf("the failure does not name the address that was tried: %v", err)
	}
	if !strings.Contains(err.Error(), "jitllmd serve") {
		t.Fatalf("the failure does not say how to get a daemon listening: %v", err)
	}
}

// ---------------------------------------------------------------- memory

// TestDevicesPrintsTheSpendableTotalAndNamesAUnifiedHeap: the client must print
// the server's spendable_total, not its own sum that would count an integrated
// GPU's heap. The probe is injected through Config.Probe. With cmdDevices
// summing total_memory itself it fails with "which counts the integrated
// heap".
func TestDevicesPrintsTheSpendableTotalAndNamesAUnifiedHeap(t *testing.T) {
	probe := func() ([]server.DeviceInfo, error) {
		return []server.DeviceInfo{
			// vulkan:0 and cuda:0 are one card (same UUID). The client must
			// print the entry and say it is not counted.
			{ID: "vulkan:0", Backend: "vulkan", Name: "RTX 3050 Ti Laptop", Kind: server.KindDiscrete,
				TotalMemory: 4 << 30, FreeMemory: 2 << 30, Available: true,
				PhysicalID: "uuid:5f3c1a2e9b7d4e6f8a0b1c2d3e4f5a6b"},
			{ID: "cuda:0", Backend: "cuda", Name: "RTX 3050 Ti Laptop", Kind: server.KindDiscrete,
				TotalMemory: 4 << 30, FreeMemory: 2 << 30, Available: true,
				PhysicalID: "uuid:5f3c1a2e9b7d4e6f8a0b1c2d3e4f5a6b"},
			{ID: "vulkan:1", Backend: "vulkan", Name: "Intel Iris Xe", Kind: server.KindIntegrated,
				TotalMemory: 21 << 30, FreeMemory: 10 << 30, CountsTowardHostBudget: true, Available: true},
			{ID: "vulkan:2", Backend: "vulkan", Name: "a card with no compute queue",
				Kind: server.KindDiscrete, TotalMemory: 8 << 30, Available: false,
				Why: "no compute queue"},
		}, nil
	}
	addr, _ := engineServer(t, server.Config{Probe: probe})
	got := out(t, cmdDevices, "-addr", addr)

	// Pinned to the server's own spendable_total rather than a literal: the
	// host budget depends on the host's cgroup, and a literal would let a
	// client that summed every heap pass.
	cn, err := dial(addr)
	if err != nil {
		t.Fatal(err)
	}
	topo, err := cn.Device.GetMemoryTopology(context.Background(),
		connect.NewRequest(&v1.GetMemoryTopologyRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	spendable := topo.Msg.GetTopology().GetSpendableTotal()
	budget := topo.Msg.GetTopology().GetHost().GetWeightBudget().GetBytes()

	// The contract itself, in bytes: the host's budget plus the ONE discrete
	// card that is usable. The 21 GiB integrated heap and the 8 GiB card with
	// no compute queue are both excluded.
	if want := budget + 4<<30; spendable.GetBytes() != want {
		t.Fatalf("the server's spendable_total is %d, want %d (host budget %d + 4 GiB of real "+
			"VRAM); the 21 GiB integrated heap is host memory under another name and the 8 GiB "+
			"card cannot run a kernel", spendable.GetBytes(), want, budget)
	}
	// The same contract read off what the client PRINTED, from its own one
	// response: the host budget is MemAvailable-derived and moves between two
	// calls, so comparing the printed figure against a second request
	// compares two readings of a moving value.
	printedBudget := humanGiB(t, got, `host     weight budget ([0-9.]+ [KMGT]?i?B) of`)
	printedSpend := humanGiB(t, got, `\nspendable ([0-9.]+ [KMGT]?i?B) `)
	// Each figure is rounded to two decimals of its unit.
	if d := printedSpend - (printedBudget + 4); d > 0.011 || d < -0.011 {
		t.Fatalf("the client printed a spendable total of %.2f GiB beside a host budget of %.2f GiB; "+
			"the server's contract is the budget plus the one usable 4 GiB card, and re-deriving it "+
			"client-side is how an integrated heap gets added back in:\n%s", printedSpend, printedBudget, got)
	}
	// The unified device is called out by name, not merely excluded.
	if !strings.Contains(got, "SUBTRACT") {
		t.Fatalf("the integrated device is not flagged as host memory under another name:\n%s", got)
	}
	// An unavailable device is listed with its reason.
	if !strings.Contains(got, "no compute queue") {
		t.Fatalf("a device that enumerated but cannot run a kernel was dropped from the list "+
			"instead of being reported with its reason:\n%s", got)
	}
	// The second view of one card is listed and marked.
	if !strings.Contains(got, "vulkan:0") {
		t.Fatalf("the Vulkan view of the shared card vanished from the list:\n%s", got)
	}
	if !strings.Contains(got, "SAME physical device as cuda:0") {
		t.Fatalf("vulkan:0 and cuda:0 are one card and the list does not say so; a reader "+
			"adding the two rows gets 8 GiB of a 4 GiB card:\n%s", got)
	}
}

// ---------------------------------------------------------------- requests

// TestTheFlagsReachTheEngineAsTheyWereTyped catches a flag parsed but not
// sent, which output cannot show. The scripted backend records the
// GenerateOptions the engine received.
func TestTheFlagsReachTheEngineAsTheyWereTyped(t *testing.T) {
	s := &scripted{tokens: []string{"ok"}, reason: server.FinishEOS}
	addr, _ := scriptServer(t, s)

	out(t, cmdRun, "-addr", addr, "-model", "m-1", "-n", "64",
		"-stop", "END,###", "-temp", "0.7", "-top-k", "40", "-seed", "99",
		"-echo", "-queue-timeout", "2500", "the capital of France is")

	o := s.opts()
	if o.ModelID != "m-1" {
		t.Fatalf("model reached the engine as %q", o.ModelID)
	}
	if o.MaxTokens != 64 {
		t.Fatalf("-n reached the engine as %d, want 64", o.MaxTokens)
	}
	if got := strings.Join(o.Stop, ","); got != "END,###" {
		t.Fatalf("-stop reached the engine as %q, want %q", got, "END,###")
	}
	if !o.Echo {
		t.Fatal("-echo did not reach the engine")
	}
	if o.QueueTimeout != 2500*time.Millisecond {
		t.Fatalf("-queue-timeout reached the engine as %v, want 2.5s", o.QueueTimeout)
	}
	if o.Prompt.Kind != server.PromptText || o.Prompt.Text != "the capital of France is" {
		t.Fatalf("the prompt reached the engine as %+v", o.Prompt)
	}
	if o.Sampling == nil {
		t.Fatal("the sampler did not reach the engine at all")
	}
	// SamplingParams.temperature is a proto float, so 0.7 arrives as the
	// nearest binary32; exact equality is not promised.
	if d := o.Sampling.Temp - 0.7; d > 1e-6 || d < -1e-6 {
		t.Fatalf("-temp reached the engine as %v, want 0.7", o.Sampling.Temp)
	}
	if o.Sampling.TopK != 40 || o.Sampling.Seed != 99 {
		t.Fatalf("the sampler reached the engine as top-k=%d seed=%d", o.Sampling.TopK, o.Sampling.Seed)
	}
}

// TestAnUntypedSamplerIsNotSentAtAll: unset sampling fields mean "the
// session's own sampler", so client defaults must not be sent.
//
// VIOLATION SIGNATURE. Make samplingFrom always return a fully populated
// SamplingParams and this fails with
//
//	the client sent a sampler nobody asked for: &{Temp:0 TopP:0 ...}
func TestAnUntypedSamplerIsNotSentAtAll(t *testing.T) {
	s := &scripted{tokens: []string{"ok"}, reason: server.FinishEOS}
	addr, _ := scriptServer(t, s)
	out(t, cmdRun, "-addr", addr, "-model", "m-1", "hi")

	if o := s.opts(); o.Sampling != nil {
		t.Fatalf("the client sent a sampler nobody asked for: %+v -- the session's own sampler "+
			"is silently replaced by this client's flag defaults", o.Sampling)
	}
}

// TestChatSendsAChatPromptWithItsSystemTurnSeparate: the server renders the
// template, and the system turn keeps its own field.
//
// VIOLATION SIGNATURE. Have promptFrom fold the system message into the message
// list and this fails with
//
//	the system turn was folded into the message list: 2 message(s), want 1
func TestChatSendsAChatPromptWithItsSystemTurnSeparate(t *testing.T) {
	s := &scripted{tokens: []string{"ok"}, reason: server.FinishEOS}
	addr, _ := scriptServer(t, s)
	out(t, cmdRun, "-addr", addr, "-model", "m-1", "-chat", "-system", "be terse", "what is 2+2?")

	o := s.opts()
	if o.Prompt.Kind != server.PromptChat || o.Prompt.Chat == nil {
		t.Fatalf("-chat did not reach the engine as a chat prompt: %+v", o.Prompt)
	}
	if !o.Prompt.Chat.HasSystem || o.Prompt.Chat.System != "be terse" {
		t.Fatalf("the system turn reached the engine as (%v, %q)",
			o.Prompt.Chat.HasSystem, o.Prompt.Chat.System)
	}
	if n := len(o.Prompt.Chat.Messages); n != 1 {
		t.Fatalf("the system turn was folded into the message list: %d message(s), want 1", n)
	}
	if !o.Prompt.Chat.AddGenerationPrompt {
		t.Fatal("add_generation_prompt was not set, so the model is asked to continue the user's " +
			"turn rather than to answer it")
	}

	// An absent system must not read as an empty one: a template may branch
	// on whether a system turn exists.
	s2 := &scripted{tokens: []string{"ok"}, reason: server.FinishEOS}
	addr2, _ := scriptServer(t, s2)
	out(t, cmdRun, "-addr", addr2, "-model", "m-1", "-chat", "hi")
	if s2.opts().Prompt.Chat.HasSystem {
		t.Fatal("an unset -system arrived as HasSystem=true; an empty system turn is not the " +
			"same prompt as no system turn")
	}
}

// TestSystemWithoutChatIsRefused: without a template there is no system turn
// to render.
func TestSystemWithoutChatIsRefused(t *testing.T) {
	var o, e syncBuffer
	err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", "127.0.0.1:1", "-model", "m", "-system", "be terse", "hi"},
		10*time.Second)
	if err == nil || !strings.Contains(err.Error(), "-system needs -chat") {
		t.Fatalf("a -system without -chat was accepted: %v", err)
	}
}

// TestSessionAndModelTogetherAreRefused before the round trip, as the engine
// would refuse them.
func TestSessionAndModelTogetherAreRefused(t *testing.T) {
	var o, e syncBuffer
	err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", "127.0.0.1:1", "-model", "m", "-session", "s", "hi"}, 10*time.Second)
	if err == nil || !strings.Contains(err.Error(), "not both") {
		t.Fatalf("-model with -session was accepted: %v", err)
	}
}

// TestRunRefusesToGuessBetweenSeveralModels: with exactly one model loaded a
// bare run picks it and says so; with none or several it refuses. This covers
// the "none" branch, whose refusal says how to load one.
func TestRunRefusesToGuessBetweenSeveralModels(t *testing.T) {
	// The real engine with nothing loaded: the "no model" branch.
	addr, _ := engineServer(t, server.Config{})
	var o, e syncBuffer
	err := runVerb(t, cmdRun, cli{out: &o, err: &e}, []string{"-addr", addr, "hi"}, 20*time.Second)
	if err == nil {
		t.Fatal("run with no -model and nothing loaded succeeded")
	}
	if !strings.Contains(err.Error(), "no model is loaded") {
		t.Fatalf("the refusal does not say that nothing is loaded: %v", err)
	}
	if !strings.Contains(err.Error(), "models -load") {
		t.Fatalf("the refusal does not say how to load one: %v", err)
	}
}

// ---------------------------------------------------------------- addresses

// TestBaseURLMeansTheSameThingOnBothSidesOfTheBinary: `-addr :8080` must mean
// what it means to `jitllmd serve`.
func TestBaseURLMeansTheSameThingOnBothSidesOfTheBinary(t *testing.T) {
	ok := []struct{ in, want string }{
		{"", "http://localhost:8080"},
		{":8080", "http://localhost:8080"},
		{":9090", "http://localhost:9090"},
		{"localhost:8080", "http://localhost:8080"},
		{"box.local:7000", "http://box.local:7000"},
		{"http://box:7000", "http://box:7000"},
		{"https://box", "https://box"},
		{"http://box:7000/", "http://box:7000"},
	}
	for _, c := range ok {
		got, err := baseURL(c.in)
		if err != nil {
			t.Errorf("baseURL(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("baseURL(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// A bare host has no port, and silently appending one would dial somewhere
	// the caller did not name.
	for _, bad := range []string{"box", "localhost:http", "ftp://box:21"} {
		if got, err := baseURL(bad); err == nil {
			t.Errorf("baseURL(%q) = %q, want a refusal", bad, got)
		}
	}
}
