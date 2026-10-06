package jinja

import (
	"os"
	"path/filepath"
	"testing"
)

// One binary, one knob, both arms live and interleaved. scopeSlots=0 is the
// map-only scope; 4 is the inline slots.
func benchScopeArm(b *testing.B, slots int) {
	old := scopeSlots
	scopeSlots = slots
	defer func() { scopeSlots = old }()
	src, err := os.ReadFile(filepath.Join("testdata", "templates", "qwen3.jinja"))
	if err != nil {
		b.Fatal(err)
	}
	tpl, err := Compile(string(src))
	if err != nil {
		b.Fatal(err)
	}
	data := map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": "write a hello world http server in rust"}},
		"add_generation_prompt": true,
		"bos_token":             "<|begin_of_text|>",
		"eos_token":             "<|eot_id|>",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := tpl.Render(data); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkScopeA_Map(b *testing.B)    { benchScopeArm(b, 0) }
func BenchmarkScopeB_Inline(b *testing.B) { benchScopeArm(b, scopeInline) }
