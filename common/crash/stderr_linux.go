package crash

import (
	"os"
	"syscall"
)

// redirectStderr makes f descriptor 2, which the runtime writes a fatal
// error's own lines to.
func redirectStderr(f *os.File) error { return syscall.Dup3(int(f.Fd()), 2, 0) }
