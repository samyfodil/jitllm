package main

import (
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"runtime/debug"

	"github.com/samyfodil/jitllm/common/config"
)

// logToFileWithoutAConsole sends the app's log output to jitllm-ui.log beside
// the settings file when nothing would read standard error: a Windows GUI
// program started from Explorer has no console (its handle does not stat), and
// a macOS app opened from Finder or `open` has its standard error on
// /dev/null. Started from a terminal, the output stays there. A crash's
// traceback goes to the same file, since the runtime writes it to a standard
// error nobody sees.
func logToFileWithoutAConsole() {
	if stderrIsRead() {
		return
	}
	p := config.ConfigPath()
	if p == "" {
		return
	}
	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	// One run's log, truncated at start, so it never grows without bound.
	f, err := os.OpenFile(filepath.Join(dir, "jitllm-ui.log"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return
	}
	os.Stdout, os.Stderr = f, f
	log.SetOutput(f)
	slog.SetDefault(slog.New(slog.NewTextHandler(f, nil)))
	// Failing here leaves a crash's traceback where it was before; the log
	// above still reaches the file.
	if err := debug.SetCrashOutput(f, debug.CrashOptions{}); err != nil {
		log.Printf("crash output stays on stderr: %v", err)
	}
}

// stderrIsRead reports whether standard error goes anywhere a person sees.
func stderrIsRead() bool {
	fi, err := os.Stderr.Stat()
	if err != nil {
		// A GUI program run from a terminal has no console of its own; it
		// writes to the terminal's when it can attach to it.
		return attachParentConsole()
	}
	null, err := os.Stat(os.DevNull)
	if err != nil {
		return true
	}
	return !os.SameFile(fi, null)
}
