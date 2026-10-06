package session

import (
	"fmt"
	"math"
	"strconv"
	"time"
)

const (
	kib = 1 << 10
	mib = 1 << 20
	gib = 1 << 30
	tib = 1 << 40
)

// Bytes renders a byte count in binary units (KiB/MiB/GiB), the same units
// jitllm's own budget arithmetic uses. Sub-kilobyte values stay exact.
func Bytes(n uint64) string {
	switch {
	case n >= tib:
		return fmt.Sprintf("%.2f TiB", float64(n)/tib)
	case n >= gib:
		return fmt.Sprintf("%.2f GiB", float64(n)/gib)
	case n >= mib:
		return fmt.Sprintf("%.1f MiB", float64(n)/mib)
	case n >= kib:
		return fmt.Sprintf("%.1f KiB", float64(n)/kib)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// Rate renders a decode or prefill rate. Zero reads as a dash rather than
// "0.00 tok/s", because "not measured yet" and "stopped" are different facts.
func Rate(tokPerSec float64) string {
	if tokPerSec <= 0 || math.IsNaN(tokPerSec) {
		return "--"
	}
	return fmt.Sprintf("%.2f tok/s", tokPerSec)
}

// GBs renders an effective bandwidth in decimal GB/s, which is the unit every
// roofline figure in this project is quoted in.
func GBs(bytesPerSec float64) string {
	if bytesPerSec <= 0 || math.IsNaN(bytesPerSec) {
		return "--"
	}
	return fmt.Sprintf("%.1f GB/s", bytesPerSec/1e9)
}

// Percent renders a fraction as a percentage. A fraction of exactly 0 reads as
// a dash: "0% of wall" is almost always an unmeasured wall, not a stalled bus.
func Percent(frac float64) string {
	if frac <= 0 || math.IsNaN(frac) {
		return "--"
	}
	return fmt.Sprintf("%.0f%%", frac*100)
}

// Dur renders a wall-clock duration at a precision that suits its size.
func Dur(d time.Duration) string {
	switch {
	case d <= 0:
		return "--"
	case d < time.Millisecond:
		return fmt.Sprintf("%d us", d.Microseconds())
	case d < time.Second:
		return fmt.Sprintf("%.0f ms", float64(d)/float64(time.Millisecond))
	case d < time.Minute:
		return fmt.Sprintf("%.2f s", d.Seconds())
	default:
		return fmt.Sprintf("%dm %02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
}

// BytesPerTokenRate turns a bytes-per-token figure and a rate into the
// effective bandwidth those two imply. It is the one derived quantity every
// telemetry panel needs, and deriving it once stops two panels disagreeing.
func BytesPerTokenRate(bytesPerTok uint64, tokPerSec float64) float64 {
	if bytesPerTok == 0 || tokPerSec <= 0 {
		return 0
	}
	return float64(bytesPerTok) * tokPerSec
}

// Thousands writes n with comma separators.
func Thousands(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
