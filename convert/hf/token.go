package hf

import (
	"os"
	"path/filepath"
	"strings"
)

// FindToken finds a Hub token the way the Hub's own tooling does: the
// environment first, then the file `huggingface-cli login` writes.
//
// It takes its lookups as arguments because a library reads no environment;
// callers pass os.Getenv and os.UserHomeDir. The file matters most: login
// writes ~/.cache/huggingface/token and exports nothing.
func FindToken(getenv func(string) string, home func() (string, error)) string {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
		if v := strings.TrimSpace(getenv(k)); v != "" {
			return v
		}
	}
	dir := strings.TrimSpace(getenv("HF_HOME"))
	if dir == "" {
		h, err := home()
		if err != nil {
			return ""
		}
		dir = filepath.Join(h, ".cache", "huggingface")
	}
	b, err := os.ReadFile(filepath.Join(dir, "token"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}
