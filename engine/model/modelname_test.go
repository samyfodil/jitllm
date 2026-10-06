package model

// nameOf is the subtest name for a model path. It lives in an untagged file
// because untagged gates use it.
func nameOf(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[i+1:]
		}
	}
	return p
}
