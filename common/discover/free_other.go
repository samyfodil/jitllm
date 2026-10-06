//go:build !unix

package discover

// FreeBytes cannot answer here, and the download goes ahead unchecked.
func FreeBytes(string) (int64, bool) { return 0, false }
