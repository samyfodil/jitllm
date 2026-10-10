package jinja_test

import (
	"testing"

	"github.com/jitllm/jitllm/tok/jinja"
)

// benchTemplate renders a family's template with the data engine/model binds
// for one user turn and no tools -- the workload every chat request starts as.
func benchTemplate(b *testing.B, file string) {
	tpl, err := jinja.Compile(readTemplate(b, file))
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

func BenchmarkChatRender_Llama3(b *testing.B) { benchTemplate(b, "llama3.jinja") }
func BenchmarkChatRender_Qwen3(b *testing.B)  { benchTemplate(b, "qwen3.jinja") }
func BenchmarkChatRender_Qwen2(b *testing.B)  { benchTemplate(b, "qwen2.jinja") }
func BenchmarkChatRender_Zephyr(b *testing.B) { benchTemplate(b, "zephyr.jinja") }

func BenchmarkChatCompile_Llama3(b *testing.B) {
	s := readTemplate(b, "llama3.jinja")
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := jinja.Compile(s); err != nil {
			b.Fatal(err)
		}
	}
}
