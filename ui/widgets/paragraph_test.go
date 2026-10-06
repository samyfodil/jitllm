package widgets

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// No wrapped line may exceed the column budget, because the canvas does not
// clip: a line measured too long runs off the bubble and off the window.
func TestWrapTextNeverExceedsTheWidth(t *testing.T) {
	const cols = 20
	for _, in := range []string{
		"the quick brown fox jumps over the lazy dog and keeps going",
		"supercalifragilisticexpialidocious is one word",
		"https://example.com/a/very/long/path/that/cannot/be/broken/on/a/space",
		"short",
		strings.Repeat("a", 97),
		"trailing spaces      and   runs   between   words",
	} {
		for _, line := range WrapText(in, cols) {
			if n := utf8.RuneCountInString(line); n > cols {
				t.Errorf("%q wrapped to a %d-rune line (max %d): %q", in, n, cols, line)
			}
		}
	}
}

// Wrapping must not lose or invent characters: the words that went in are the
// words that come out, in order.
func TestWrapTextKeepsEveryWord(t *testing.T) {
	const in = "the quick brown fox jumps over the lazy dog"
	got := strings.Join(strings.Fields(strings.Join(WrapText(in, 12), " ")), " ")
	if want := strings.Join(strings.Fields(in), " "); got != want {
		t.Errorf("wrapping changed the words:\n got %q\nwant %q", got, want)
	}
}

// A newline the model emitted is a line break, not a space.
func TestWrapTextHonoursNewlines(t *testing.T) {
	got := WrapText("one\ntwo\n\nthree", 40)
	want := []string{"one", "two", "", "three"}
	if len(got) != len(want) {
		t.Fatalf("got %d line(s) %q, want %d %q", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// A word longer than the line is split rather than looping for ever or
// overflowing -- a 200-character token and a URL both reach this path.
func TestWrapTextSplitsAnUnbreakableWord(t *testing.T) {
	lines := WrapText(strings.Repeat("x", 25), 10)
	if len(lines) != 3 {
		t.Fatalf("got %d line(s), want 3: %q", len(lines), lines)
	}
	if strings.Join(lines, "") != strings.Repeat("x", 25) {
		t.Errorf("the split lost or duplicated characters: %q", lines)
	}
}

// Degenerate inputs must terminate and never index out of range.
func TestWrapTextDegenerateInputs(t *testing.T) {
	if got := WrapText("", 10); got != nil {
		t.Errorf("empty text wrapped to %q, want nil", got)
	}
	for _, cols := range []int{0, -1, 1} {
		lines := WrapText("ab cd", cols)
		for _, l := range lines {
			if utf8.RuneCountInString(l) > max(cols, 1) {
				t.Errorf("cols=%d produced %q", cols, l)
			}
		}
	}
}

// A wrapped line never begins with the space that broke it, which is what
// makes a paragraph's left edge straight.
func TestWrapTextDoesNotIndentAWrappedLine(t *testing.T) {
	for _, l := range WrapText("alpha beta gamma delta epsilon zeta eta theta", 12) {
		if l != "" && strings.HasPrefix(l, " ") {
			t.Errorf("line begins with a space: %q", l)
		}
	}
}
