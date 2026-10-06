package session

import (
	"testing"
	"time"
)

func TestBytes(t *testing.T) {
	cases := []struct {
		in   uint64
		want string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1 << 10, "1.0 KiB"},
		{1 << 20, "1.0 MiB"},
		{858316800, "818.6 MiB"}, // a container size
		{1 << 30, "1.00 GiB"},
		{46230000000, "43.06 GiB"},
	}
	for _, c := range cases {
		if got := Bytes(c.in); got != c.want {
			t.Errorf("Bytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestRateAndPercentUseADashForUnmeasured(t *testing.T) {
	if got := Rate(0); got != "--" {
		t.Errorf("Rate(0) = %q, want a dash: an unmeasured rate is not zero", got)
	}
	if got := Percent(0); got != "--" {
		t.Errorf("Percent(0) = %q, want a dash", got)
	}
	if got := GBs(0); got != "--" {
		t.Errorf("GBs(0) = %q, want a dash", got)
	}
	if got := Rate(31.04); got != "31.04 tok/s" {
		t.Errorf("Rate(31.04) = %q", got)
	}
	if got := Percent(0.69); got != "69%" {
		t.Errorf("Percent(0.69) = %q", got)
	}
}

func TestBytesPerTokenRate(t *testing.T) {
	// 31.04 tok/s against 816,010,912 B/token is 25.3 GB/s.
	got := BytesPerTokenRate(816010912, 31.04)
	if s := GBs(got); s != "25.3 GB/s" {
		t.Errorf("GBs(BytesPerTokenRate(...)) = %q, want 25.3 GB/s", s)
	}
	if BytesPerTokenRate(0, 31.04) != 0 {
		t.Error("an unknown byte count must give 0, not a fabricated bandwidth")
	}
}

func TestDur(t *testing.T) {
	cases := []struct {
		in   time.Duration
		want string
	}{
		{0, "--"},
		{500 * time.Microsecond, "500 us"},
		{81 * time.Millisecond, "81 ms"},
		{7126 * time.Millisecond, "7.13 s"},
		{90 * time.Second, "1m 30s"},
	}
	for _, c := range cases {
		if got := Dur(c.in); got != c.want {
			t.Errorf("Dur(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}
