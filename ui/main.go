// Command jitllm-ui is the desktop front end for jitllm: an alternative to the
// CLI, not a wrapper around it.
//
// The engine is driven in process: shelling out to the jitllm binary cannot
// hold a model open across turns, and it would turn counters back into text.
package main

import (
	"flag"
	"log"
	"os"

	_ "github.com/gogpu/gg/gpu" // enable GPU SDF acceleration; required

	"github.com/gogpu/gogpu"
	"github.com/gogpu/ui/desktop"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/engine"
	"github.com/jitllm/jitllm/ui/mock"
	"github.com/jitllm/jitllm/ui/screen"
)

// register installs every screen in tab order and attaches what the screens
// reach outside the widget tree: the engine, the hardware and the model files.
// Without an engine the Models tab opens the "No engine wired" dialog.
func register(sh *app.Shell, d screen.Deps) {
	screen.Install(sh, d)

	// Drop a .jlm or a .gguf anywhere on the window and it lands in the
	// catalog: a container is revealed in the table, a GGUF fills the convert
	// form. Drop a picture and it is attached to the next message.
	// Shell.OnFilesDropped holds one handler, so DropFiles dispatches the
	// window's whole drag-and-drop behaviour.
	sh.OnFilesDropped(func(paths []string) { screen.DropFiles(sh, paths) })
}

func main() {
	mockMode := flag.Bool("mock", false, "run against package mock: no model, no GPU, fixtures only")
	flag.Parse()
	logToFileWithoutAConsole()
	app.UseEnv(app.Env{Models: os.Getenv("JITLLM_MODELS"), DataHome: os.Getenv("XDG_DATA_HOME")})
	cfg := app.LoadConfig()

	gpuApp := gogpu.NewApp(gogpu.DefaultConfig().
		WithTitle("jitllm").
		WithSize(cfg.WindowW, cfg.WindowH).
		WithIcon(app.Icon()).
		WithMinSize(960, 640))

	// Before the first frame: a font registered after a widget has measured
	// itself is a font the layout has already decided without.
	app.LoadFonts()

	sh := app.NewShell(gpuApp, cfg)
	if *mockMode {
		d, eng := mock.Deps(sh)
		eng.Live = true
		register(sh, d)
	} else {
		eng := engine.New(sh)
		sh.ServeAPI(eng.Server())
		// Before the window closes, so a decode in flight is cancelled and the
		// container and device are released rather than leaked past the window.
		sh.OnShutdown(eng.Close)
		register(sh, screen.Deps{Engine: eng})
		// The macOS app's first launch offers to put its jitllm and jitllmd
		// on PATH; anywhere else it does nothing.
		screen.OfferCLI(sh)
	}

	sh.UI.SetRoot(sh.Build())

	// The async -> UI bridge. OnUpdate runs inside the frame on the main
	// thread, and it only fires when the loop is awake -- which is why
	// Shell.Post pairs every send with a RequestRedraw.
	gpuApp.OnUpdate(sh.DrainQueue)

	// Shutdown runs after desktop.Run returns, not from OnClose: gogpu's OnClose
	// and OnDragDrop are single-slot setters that desktop.Run overwrites, so a
	// handler set here never ran and settings were never saved. Returning from Run
	// is the shutdown; Shell.Close reads the window size from GPU.Size().

	err := desktop.Run(gpuApp, sh.UI)
	sh.Close()
	if err != nil {
		log.Fatal(err)
	}
}
