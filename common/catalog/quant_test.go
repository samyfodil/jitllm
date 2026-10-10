package catalog

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

func TestQuantFromNameReadsThePublishersLabel(t *testing.T) {
	for name, want := range map[string]string{
		"Llama-3.2-1B-Instruct-Q4_K_M.jlm":   "Q4_K_M",
		"tinyllama-1.1b-q3_K_M.gguf":         "Q3_K_M",
		"SmolVLM-256M-Instruct-Q8_0-vlm.jlm": "Q8_0",
		"llava-phi-3-mini-mmproj-f16.gguf":   "F16",
		"llava-phi-3-mini-int4.gguf":         "INT4",
		"model-IQ3_XS.gguf":                  "IQ3_XS",
		"gemma-2b.jlm":                       "",
		"Qwen3-Next-80B-A3B-Instruct.jlm":    "",
	} {
		if got := QuantFromName(name); got != want {
			t.Errorf("QuantFromName(%q) = %q, want %q", name, got, want)
		}
	}
}

// A container whose name says nothing must still get a quant from its tensors.
func TestProbeReadsTheQuantFromTheTensorsWhenTheNameIsSilent(t *testing.T) {
	path := testmodels.Path("gemma-2b.jlm")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	e := Entry{Path: path, Name: "gemma-2b.jlm", Kind: KindContainer}
	Probe(&e)
	if e.ProbeErr != "" {
		t.Fatal(e.ProbeErr)
	}
	// gemma-2b's GGUF is Q4_0 with a Q8_0 embedding and head a third of it.
	if e.Quant != "Q4_0 + Q8_0" {
		t.Errorf("gemma-2b reads as %q", e.Quant)
	}
}
