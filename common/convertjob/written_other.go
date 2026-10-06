//go:build !unix

package convertjob

import "os"

// written is the file's size where no allocation count is available. The bar
// jumps early here, since jlm.Write presizes its file.
func written(fi os.FileInfo) int64 { return fi.Size() }
