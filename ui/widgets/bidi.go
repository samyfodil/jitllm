package widgets

import (
	"strings"

	"golang.org/x/text/unicode/bidi"
)

// visualOrder rearranges one already-wrapped line from logical order into the
// order it is read, using the Unicode bidirectional algorithm from x/text, so
// no script is named here.
//
// It is safe only because nothing shapes: gg/text emits base cmap glyphs for
// Arabic (no init/medi/fina), so reversing breaks no joins. If the text stack
// ever shapes, this transform has to go.
//
// The line must already be wrapped in logical order; Draw wraps first and
// calls this per line. The result no longer maps index-for-index onto the
// input, which is acceptable for the read-only transcript but not an editor.
func visualOrder(s string) string {
	if !hasRTL(s) {
		return s
	}
	var p bidi.Paragraph
	if _, err := p.SetString(s); err != nil {
		return s
	}
	ord, err := p.Order()
	if err != nil || ord.NumRuns() == 0 {
		return s
	}

	runs := make([]string, ord.NumRuns())
	for i := range runs {
		r := ord.Run(i)
		t := r.String()
		if r.Direction() == bidi.RightToLeft {
			t = reverseRunes(t)
		}
		runs[i] = t
	}
	// In a right-to-left paragraph the runs themselves read right to left, so
	// an embedded English phrase sits to the left of the Arabic that follows
	// it rather than to the right.
	if p.Direction() == bidi.RightToLeft {
		for i, j := 0, len(runs)-1; i < j; i, j = i+1, j-1 {
			runs[i], runs[j] = runs[j], runs[i]
		}
	}
	return strings.Join(runs, "")
}

// hasRTL is a fast path, not a rule: below U+0590 there is no right-to-left
// character in Unicode, so a line of Latin, Greek, Cyrillic or CJK skips the
// whole analysis. Anything at or above it goes through the real algorithm,
// which is what decides.
func hasRTL(s string) bool {
	for _, r := range s {
		if r >= 0x0590 {
			return true
		}
	}
	return false
}

func reverseRunes(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}
