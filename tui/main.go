// Command tui is jitllm's terminal interface, driven through the same engine
// and the same settings and chats as the window (common/...).
//
//	tui [-chat=true|false] [-device auto] [-ctx 8192] [-models dir] [-mouse] [model.jlm]
//
// A flag overrides its setting for this run. With no model it opens on the
// models screen.
package main

import (
	"flag"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/muesli/termenv"
	"github.com/samyfodil/jitllm/common/config"
	"github.com/samyfodil/jitllm/common/engine"
)

func main() {
	config.UseEnv(config.Env{Models: os.Getenv("JITLLM_MODELS"), DataHome: os.Getenv("XDG_DATA_HOME")})
	cfg := config.LoadConfig()
	chat := flag.Bool("chat", cfg.Chat, "apply the model's chat template")
	device := flag.String("device", cfg.DeviceSpec, "device spec, as the window's setting")
	ctx := flag.Int("ctx", cfg.MaxSeq, "context length")
	dir := flag.String("models", "", "list this folder instead of the configured ones")
	mouse := flag.Bool("mouse", false, "scroll with the mouse wheel (the terminal's own text selection then needs shift)")
	flag.Parse()
	if flag.NArg() > 1 {
		fmt.Fprintln(os.Stderr, "usage: tui [-chat=true|false] [-device auto] [-ctx 8192] [-models dir] [-mouse] [model.jlm]")
		os.Exit(2)
	}
	dirs := cfg.ModelDirs
	if *dir != "" {
		dirs = []string{*dir}
	}
	cfg.Chat, cfg.DeviceSpec, cfg.MaxSeq = *chat, *device, *ctx

	f := newFront()
	f.st = newState(f, cfg.DeviceSpec, cfg.MaxSeq, cfg.Chat, !cfg.NoKVCache)
	e := engine.New(f, f.st)
	defer e.Close()

	r := newRoot(f, e, cfg, dirs)
	probeMachine(r.env)
	if flag.NArg() == 1 {
		r.cur = scrChat
		e.Load(flag.Arg(0))
	}

	opts := []tea.ProgramOption{tea.WithAltScreen()}
	if *mouse {
		opts = append(opts, tea.WithMouseCellMotion())
	}
	// The app paints on its own ground, whatever the terminal's theme: the
	// terminal's background is set for the run and reset to its default on
	// exit (OSC 111).
	termenv.NewOutput(os.Stdout).SetBackgroundColor(termenv.RGBColor(string(cBg)))
	defer fmt.Print("\x1b]111\x07")

	p := tea.NewProgram(program{r}, opts...)
	go func() {
		for msg := range f.queue {
			p.Send(msg)
		}
	}()
	if _, err := p.Run(); err != nil {
		fmt.Print("\x1b]111\x07")
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
