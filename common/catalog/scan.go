package catalog

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/samyfodil/jitllm/convert"
)

// Scan walks dirs one level deep and classifies what it finds. It stats and
// nothing more: a container's contents are read by [Probe], which the caller
// runs on the engine goroutine so a directory of large files does not stall a
// frame.
//
// A directory that does not exist is skipped rather than reported: model
// directories may live on removable media, and an unplugged disk is not an
// error worth a dialog.
func Scan(dirs []string) []Entry {
	seen := make(map[string]bool)
	var out []Entry

	for _, dir := range dirs {
		ents, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, de := range ents {
			if de.IsDir() {
				continue
			}
			k := KindOf(de.Name())
			if k == KindUnknown {
				continue
			}
			path := filepath.Join(dir, de.Name())
			if seen[path] {
				continue
			}
			seen[path] = true

			e := Entry{Path: path, Name: de.Name(), Kind: k, Quant: QuantFromName(de.Name())}
			// os.Stat, not de.Info(), so a symlinked model is sized as its target.
			if fi, err := os.Stat(path); err == nil {
				e.Size, e.MTime = fi.Size(), fi.ModTime()
			}
			if k == KindGGUF || k == KindSafetensors {
				// Where converting it writes: the first model directory, whatever disk
				// the source is on (convert.DestFor). A container that already exists
				// anywhere wins, so a model converted elsewhere is not offered again.
				e.Target = convert.DestFor(path, dirs[0])
				e.TargetVersion, _ = ReadVersion(e.Target) // zero when absent or unreadable
				if e.TargetVersion == 0 {
					if beside := convert.DestFor(path, ""); beside != e.Target {
						if v, err := ReadVersion(beside); err == nil && v != 0 {
							e.Target, e.TargetVersion = beside, v
						}
					}
				}
				e.Tower = strings.Contains(strings.ToLower(de.Name()), "mmproj")
			}
			out = append(out, e)
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return strings.ToLower(out[i].Name) < strings.ToLower(out[j].Name)
	})
	return out
}

// KindOf classifies a file by name.
func KindOf(name string) Kind {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".jlm":
		return KindContainer
	case ".gguf":
		return KindGGUF
	case ".safetensors":
		return KindSafetensors
	}
	return KindUnknown
}

// CacheKey is the namespace a KV cache entry is stored under for this file.
// It matches the CLI's derived namespace so two runs of the same file share a
// cache and two different models never do.
func CacheKey(e Entry) string {
	return filepath.Base(e.Path) + "/" + itoa(e.Size) + "-" + itoa(e.MTime.Unix())
}

func itoa(v int64) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// Disk is the filesystem, as screen.Files reads it.
type Disk struct{}

// Scan is [Scan].
func (Disk) Scan(dirs []string) []Entry { return Scan(dirs) }

// Probe is [Probe].
func (Disk) Probe(e *Entry) { Probe(e) }
