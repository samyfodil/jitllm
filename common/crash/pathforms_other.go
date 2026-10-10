//go:build !windows

package crash

// pathForms is the spellings of a path a report may carry. Outside Windows a
// path has one.
func pathForms(p string) []string { return []string{p} }
