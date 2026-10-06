package catalog

import (
	"os"
	"path/filepath"
	"runtime"
)

// DefaultDir is where a first run looks for models and puts downloads: models
// ($JITLLM_MODELS, the variable the engine's own tests read) when set, and
// otherwise a models folder in this system's place for application data.
// dataHome is $XDG_DATA_HOME; the library reads no environment, so the
// command hands both over.
func DefaultDir(models, dataHome string) string {
	if models != "" {
		return models
	}
	return filepath.Join(dataDir(dataHome), "jitllm", "models")
}

// dataDir is where this system keeps an application's data: XDG's data home
// on Linux, Application Support on macOS, the local AppData on Windows.
func dataDir(dataHome string) string {
	switch runtime.GOOS {
	case "darwin":
		if d, err := os.UserConfigDir(); err == nil {
			return d
		}
	case "windows":
		if d, err := os.UserCacheDir(); err == nil {
			return d
		}
	default:
		if dataHome != "" {
			return dataHome
		}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return os.TempDir()
	}
	return filepath.Join(home, ".local", "share")
}
