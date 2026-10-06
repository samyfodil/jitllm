//go:build jitllmbench && !linux

package nn

// hugeCopy has no transparent huge pages to ask for here; the weights stay
// where they are.
func hugeCopy(b []byte) []byte { return b }
