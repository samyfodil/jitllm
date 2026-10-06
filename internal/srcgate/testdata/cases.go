// Package cases is the fresh-copy gate's violation and control: each function
// is named for what a copy of its result is. The gate parses this file and
// nothing compiles it, so the copies in use stand as bare calls.
package cases

type state struct {
	on   []bool
	keep [][2]int
}

// runs builds its result, so a copy of it is waste.
func (s *state) runs() [][2]int {
	var out [][2]int
	for i := range s.on {
		out = append(out, [2]int{i, i + 1})
	}
	return out
}

func built() []int { return make([]int, 4) }

func viaCall() [][2]int { return (&state{}).runs() }

func cloned() []string { return []string{"a"} }

// Get hands out memory the receiver keeps: a copy before appending is needed.
func (s *state) Get() [][2]int { return s.keep }

func param(xs []int) []int { return xs }

// mixed returns its own memory on one path.
func (s *state) mixed(b bool) [][2]int {
	if b {
		return s.keep
	}
	return nil
}

func sliced(xs []int) []int {
	ys := xs[1:]
	return ys
}

func use(s *state, xs []int) {
	append([][2]int(nil), s.runs()...)
	append([]int(nil), built()...)
	append([][2]int{}, viaCall()...)
	slices.Clone(cloned())
	append([][2]int(nil), s.Get()...)
	append([]int(nil), param(xs)...)
	append([][2]int(nil), s.mixed(true)...)
	append([]int(nil), sliced(xs)...)
}
