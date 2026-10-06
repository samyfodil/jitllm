package config

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// FolderOf is the folder a path names, or holds when it is a file.
func FolderOf(path string) string {
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return path
	}
	for d := path; ; {
		parent := filepath.Dir(d)
		if parent == d {
			return d
		}
		if fi, err := os.Stat(parent); err == nil && fi.IsDir() {
			return parent
		}
		d = parent
	}
}

// OpenFolder shows a folder in the system's file browser. An empty dir does
// nothing.
func OpenFolder(dir string) error {
	if dir == "" {
		return nil
	}
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", dir)
	case "windows":
		cmd = exec.Command("explorer", dir)
	default:
		cmd = exec.Command("xdg-open", dir)
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	// Reaped in the background: the file browser outlives this call.
	go cmd.Wait()
	return nil
}
