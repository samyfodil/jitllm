package crash

import "syscall"

// pathConv is GetShortPathName's and GetLongPathName's shape.
type pathConv func(path *uint16, buf *uint16, buflen uint32) (uint32, error)

// pathForms is the spellings of a path a report may carry: as given, its 8.3
// short form (C:\Users\RUNNER~1, which %TEMP% often uses) and its long form.
// A form the file system does not have is left out.
func pathForms(p string) []string {
	forms := []string{p}
	for _, conv := range []pathConv{syscall.GetShortPathName, syscall.GetLongPathName} {
		if f := convertPath(p, conv); f != "" && f != p {
			forms = append(forms, f)
		}
	}
	return forms
}

func convertPath(p string, conv pathConv) string {
	in, err := syscall.UTF16PtrFromString(p)
	if err != nil {
		return ""
	}
	buf := make([]uint16, 1024)
	n, err := conv(in, &buf[0], uint32(len(buf)))
	if err != nil || n == 0 || int(n) > len(buf) {
		return ""
	}
	return syscall.UTF16ToString(buf[:n])
}
