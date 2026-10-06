package tok

import (
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/samyfodil/jitllm/format/jlm"
	"github.com/samyfodil/jitllm/internal/testmodels"
)

// TestEncodeIsSafeConcurrently encodes on one vocabulary from many goroutines
// at once and holds every result to the serial one.
//
// Encode borrows its scratch from a pool shared by every caller, so two
// concurrent calls must never be handed the same buffers; run it under -race.
// Texts of different lengths make the pooled buffers change size between uses.
func TestEncodeIsSafeConcurrently(t *testing.T) {
	doc, err := os.ReadFile("../AGENTS.md")
	if err != nil {
		t.Fatal(err)
	}
	var texts []string
	for n := 64; n < len(doc) && len(texts) < 12; n *= 3 {
		texts = append(texts, strings.ToValidUTF8(string(doc[:n]), "?"))
	}
	texts = append(texts, "bad \xff\xfe utf8 \xe3\x81 end", "x"+strings.Repeat(" ", 40)+"y 12345678")
	checked := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.jlm", "Qwen3-1.7B-Q4_K_M.jlm", "gemma-3-1b-it-Q4_K_M.jlm"} {
		f, err := jlm.Open(testmodels.Path(name))
		if err != nil {
			continue
		}
		vc := f.Vocab()
		f.Close()
		v, err := New(vc)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		checked++
		want := make([][]int32, len(texts))
		for i, s := range texts {
			want[i] = v.Encode(s, true)
		}
		var wg sync.WaitGroup
		errs := make(chan string, 64)
		for g := 0; g < 16; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				for r := 0; r < 4; r++ {
					i := (g + r) % len(texts)
					if got := v.Encode(texts[i], true); !slicesEqual(got, want[i]) {
						errs <- name
						return
					}
				}
			}(g)
		}
		wg.Wait()
		close(errs)
		for e := range errs {
			t.Fatalf("%s: a concurrent Encode differed from the serial one", e)
		}
	}
	if checked == 0 {
		testmodels.Missing(t, "%s", "no container found under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory): this gate proved nothing")
	}
}
