//go:build race

package bench

// The physics guard is disabled under -race: the detector instruments MemWall
// (Go) but not the generated kernels, so the ceiling reads about a third of
// normal and correct results look impossible. Timings under -race are
// meaningless anyway.
const raceEnabled = true
