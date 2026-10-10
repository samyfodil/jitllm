package app

import (
	"time"

	"github.com/jitllm/jitllm/common/session"
)

// Formatting helpers shared by every screen, re-exported from
// [session] so both front ends render a figure the same way.

// Bytes renders a byte count in binary units; see [session.Bytes].
func Bytes(n uint64) string { return session.Bytes(n) }

// Rate renders a decode or prefill rate; see [session.Rate].
func Rate(tokPerSec float64) string { return session.Rate(tokPerSec) }

// GBs renders an effective bandwidth; see [session.GBs].
func GBs(bytesPerSec float64) string { return session.GBs(bytesPerSec) }

// Percent renders a fraction as a percentage; see [session.Percent].
func Percent(frac float64) string { return session.Percent(frac) }

// Dur renders a wall-clock duration; see [session.Dur].
func Dur(d time.Duration) string { return session.Dur(d) }

// BytesPerTokenRate is the bandwidth a bytes-per-token figure and a rate
// imply; see [session.BytesPerTokenRate].
func BytesPerTokenRate(bytesPerTok uint64, tokPerSec float64) float64 {
	return session.BytesPerTokenRate(bytesPerTok, tokPerSec)
}
