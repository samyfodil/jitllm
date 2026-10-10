package crash

import (
	"os"
	"syscall"
)

// redirectStderr makes f the process's standard error handle. The runtime
// asks GetStdHandle on every write (runtime.write1), so a fatal error's own
// lines -- "fatal error: ...", "Exception 0xc0000005 ..." -- reach f, where a
// GUI program would otherwise lose them.
func redirectStderr(f *os.File) error {
	const stdErrorHandle = ^uintptr(11) // STD_ERROR_HANDLE, (DWORD)-12
	set := syscall.NewLazyDLL("kernel32.dll").NewProc("SetStdHandle")
	if r, _, err := set.Call(stdErrorHandle, f.Fd()); r == 0 {
		return err
	}
	return nil
}
