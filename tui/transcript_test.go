package main

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

// A reply wraps greedily, whatever hyphens and the model's own line breaks it
// carries: no line could have taken the next line's first word, and no line
// keeps glamour's margin. The text is a Qwen3 reply that left "(experts)" on
// a line of its own under glamour v1.
func TestAReplyWrapsWithoutStrandingAWord(t *testing.T) {
	text := "A mixture-of-experts (MoE) model combines multiple specialized sub-models (experts) to\n" +
		"handle different aspects of a task, with each expert focusing on specific data\n" +
		"patterns or regions. A routing mechanism, often called a \"gating network,\"\n" +
		"dynamically selects which experts to activate for each input based on its features."
	for _, w := range []int{40, 60, 85, 88, 120} {
		var md markdown
		var lines []string
		for _, l := range strings.Split(md.render(text, w), "\n") {
			lines = append(lines, strings.TrimRight(ansi.Strip(l), " "))
		}
		for i, l := range lines {
			if strings.HasPrefix(l, " ") {
				t.Errorf("w=%d: line %d keeps a margin: %q", w, i, l)
			}
			if ansi.StringWidth(l) > w {
				t.Errorf("w=%d: line %d is %d cells: %q", w, i, ansi.StringWidth(l), l)
			}
		}
		// The margin is checked above; the wrap is checked on the words.
		limit := 0
		for i, l := range lines {
			lines[i] = strings.TrimLeft(l, " ")
			limit = max(limit, ansi.StringWidth(lines[i]))
		}
		for i := 0; i+1 < len(lines); i++ {
			next, _, _ := strings.Cut(lines[i+1], " ")
			if lines[i] != "" && next != "" && ansi.StringWidth(lines[i])+1+ansi.StringWidth(next) <= limit {
				t.Errorf("w=%d: %q could have taken %q from the next line", w, lines[i], next)
			}
		}
	}
}
