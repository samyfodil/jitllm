package jinja

import (
	"math"
	"strconv"
	"strings"
)

// pyFloat is Python's repr of a float, which str, print and json.dumps all
// write: the shortest digits that read back exactly, positional from 1e-4 up to
// 1e16 with ".0" on an integral value (18.0, not 18), exponent form outside it
// (1e+16, 1.5e-07).
func pyFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	exp, err := strconv.Atoi(e[strings.LastIndexByte(e, 'e')+1:])
	if err != nil || exp < -4 || exp >= 16 {
		return e
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}
