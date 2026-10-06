//go:build unix

package convertjob

import (
	"os"
	"syscall"
)

// written is how much of a file has actually been written: allocated blocks,
// not the size, because jlm.Write presizes its .part file with a sparse
// Truncate.
func written(fi os.FileInfo) int64 {
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		return st.Blocks * 512
	}
	return fi.Size()
}
