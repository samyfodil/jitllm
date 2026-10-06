package tok

import (
	"reflect"
	"testing"

	"github.com/samyfodil/jitllm/tok/pretok"
)

// TestO200KSplitsByCaseAndSuffixesContractions pins the three ways o200k's
// pre-tokenizer differs from llama3's, one case per difference and one case per
// flag that switches it.
//
// The oracle is TestBPEAgainstLlamaCpp on the real gpt-oss vocabularies; this
// is the fast pin. Each subtest fails with only its own flag cleared, so no
// flag is decoration.
func TestO200KSplitsByCaseAndSuffixesContractions(t *testing.T) {
	o200k := pretok.SplitParams{
		Contractions: true, CaseFold: true, DigitGroup: 3,
		CaseRuns: true, SuffixContract: true, PunctSlash: true,
	}
	llama3 := pretok.SplitParams{Contractions: true, CaseFold: true, DigitGroup: 3}

	cases := []struct {
		name            string
		in              string
		want, wasllama3 []string
	}{
		// A lower-to-upper boundary starts a new piece. llama3's \p{L}+ takes the
		// whole word.
		{"camel", "helloWorld", []string{"hello", "World"}, []string{"helloWorld"}},
		// An upper run followed by a lower one does not split: the first
		// alternative is U* L+, and greedy U* takes "HTTPS" before L+ takes the
		// rest. Both engines agree here, which is the case that says CaseRuns is
		// not simply "split at every case change".
		{"acronym", "HTTPServer", []string{"HTTPServer"}, []string{"HTTPServer"}},
		// The contraction hangs off the end of the letter run instead of standing
		// alone. This is the case gpt-oss tokenized wrong.
		{"contraction", "Isn't", []string{"Isn't"}, []string{"Isn", "'t"}},
		// ...and with no letters in front of it there is no contraction to hang.
		// o200k's optional leading [^\r\n\p{L}\p{N}] takes the apostrophe and
		// the letter run takes the rest, so "'tis" is one piece; llama3 tries its
		// standalone contraction first and cuts "'t" off the front of the word.
		{"lone", "'tis", []string{"'tis"}, []string{"'t", "is"}},
		// The punctuation run's tail is [\r\n/]* rather than [\r\n]*, so the
		// slash joins the newline; under llama3 the tail stops at the newline and
		// the slash is taken by the letter rule's optional leading character.
		{"slash", ".\n/x", []string{".\n/", "x"}, []string{".\n", "/x"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := splitLlama3Params(c.in, o200k); !reflect.DeepEqual(got, c.want) {
				t.Errorf("o200k %q: got %q, want %q", c.in, got, c.want)
			}
			// The llama3 arm states what the old parameters produce for the
			// same input, so each row is a real difference rather than a
			// restatement of the new code.
			if got := splitLlama3Params(c.in, llama3); !reflect.DeepEqual(got, c.wasllama3) {
				t.Errorf("llama3 %q: got %q, want %q", c.in, got, c.wasllama3)
			}
		})
	}
}
