// Package hip binds AMD's ROCm runtime (libamdhip64) and code object manager
// (libamd_comgr) through jit/gpu/ffi, with no cgo.
//
// Both libraries are found and loaded at run time, as jit/gpu/cuda finds
// libcuda: a binary built with CGO_ENABLED=0 runs on a host with ROCm and on
// one without, and on the second it simply has no HIP devices. Nothing here
// runs an external program -- no hipcc, clang, ld.lld or rocminfo. Kernels are
// jitllm's own LLVM IR (jit/gpu/amdgpu), turned into a code object by comgr's
// library actions (Compile), and devices are described by HIP calls alone.
//
// The package reads no environment. A caller that wants a ROCm outside the
// places searched by default names it in Config.Path, which cmd/jitllm and
// jitllmd fill from JITLLM_ROCM.
package hip

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/samyfodil/jitllm/jit/gpu/ffi"
)

// Config is where ROCm is looked for. The zero value searches the default
// places (DefaultDirs, then the loader's own path by soname).
type Config struct {
	// Path is a ROCm library directory (the one holding libamdhip64.so), or
	// empty for the default search. When set, ONLY that directory is tried: a
	// caller who named a ROCm gets that one or none, never a different one
	// found elsewhere.
	Path string
}

// HIPNames and ComgrNames are the sonames tried, newest ABI first. Package
// variables so a gate can read them.
var (
	HIPNames   = []string{"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"}
	ComgrNames = []string{"libamd_comgr.so.3", "libamd_comgr.so.2", "libamd_comgr.so"}
)

// DefaultDirs is where a ROCm install puts its libraries, in the order tried:
// /opt/rocm (the distribution's symlink to the active version), then every
// /opt/rocm-X.Y.Z newest first, then the distribution's own library directory.
// The loader's search path is tried before any of them.
func DefaultDirs() []string {
	dirs := []string{"/opt/rocm/lib"}
	vers, _ := filepath.Glob("/opt/rocm-*/lib")
	sort.Slice(vers, func(i, j int) bool { return versionLess(vers[j], vers[i]) })
	dirs = append(dirs, vers...)
	return append(dirs, "/usr/lib/x86_64-linux-gnu")
}

// versionLess orders two /opt/rocm-X.Y.Z/lib paths by their numeric version,
// so 6.10 sorts after 6.9.
func versionLess(a, b string) bool {
	pa, pb := versionOf(a), versionOf(b)
	for i := 0; i < len(pa) && i < len(pb); i++ {
		if pa[i] != pb[i] {
			return pa[i] < pb[i]
		}
	}
	return len(pa) < len(pb)
}

func versionOf(p string) []int {
	base := filepath.Base(filepath.Dir(p))
	var out []int
	for _, f := range strings.Split(strings.TrimPrefix(base, "rocm-"), ".") {
		n := 0
		for _, c := range f {
			if c < '0' || c > '9' {
				break
			}
			n = n*10 + int(c-'0')
		}
		out = append(out, n)
	}
	return out
}

// candidates is every path tried for one library under a config: the named
// directory alone, or the sonames bare (the loader's search path) and then
// under each default directory.
func candidates(c Config, names []string) []string {
	var out []string
	if c.Path != "" {
		for _, n := range names {
			out = append(out, filepath.Join(c.Path, n))
		}
		return out
	}
	out = append(out, names...)
	for _, d := range DefaultDirs() {
		for _, n := range names {
			p := filepath.Join(d, n)
			if _, err := os.Stat(p); err == nil {
				out = append(out, p)
			}
		}
	}
	return out
}

// lib is one load of ROCm: both libraries' entry points, bound once per
// configured path. Two configs naming different directories are two loads.
type lib struct {
	hip   *hipAPI
	comgr *comgrAPI
	// hipErr and comgrErr say why a half is missing. comgr alone is enough to
	// compile (the offline gates), and HIP alone is not enough to run.
	hipErr, comgrErr error
}

var (
	libsMu sync.Mutex
	libs   = map[string]*lib{}
)

// load binds the libraries a config names, once per path. A library that is
// absent, or present without an entry point this package needs, is an error in
// the result and never a panic: ffi panics on a missing symbol, which here is
// recovered into the reason.
func load(c Config) *lib {
	libsMu.Lock()
	defer libsMu.Unlock()
	if l, ok := libs[c.Path]; ok {
		return l
	}
	l := &lib{}
	if runtime.GOOS != "linux" {
		err := fmt.Errorf("hip: ROCm is a Linux runtime; this is %s", runtime.GOOS)
		l.hipErr, l.comgrErr = err, err
		libs[c.Path] = l
		return l
	}
	l.comgr, l.comgrErr = bindComgr(candidates(c, ComgrNames))
	l.hip, l.hipErr = bindHIP(candidates(c, HIPNames))
	libs[c.Path] = l
	return l
}

// bindSafely runs a binder that panics on a missing symbol and returns the
// panic as an error naming it.
func bindSafely(what string, f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("hip: %s: %v", what, r)
		}
	}()
	f()
	return nil
}

// openFirst is ffi.Open with the candidate list in its error.
func openFirst(what string, names []string) (*ffi.Lib, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("hip: no %s candidate to load", what)
	}
	l, err := ffi.Open(names...)
	if err != nil {
		return nil, fmt.Errorf("hip: %s is not loadable: %w", what, err)
	}
	return l, nil
}

// Searched is where a config looks for ROCm, for a report: the named
// directory, or the loader's search path and every default directory.
func Searched(c Config) string {
	if c.Path != "" {
		return c.Path
	}
	return "the loader's path, " + strings.Join(DefaultDirs(), ", ")
}
