package engine

import (
	"strings"
	"testing"
)

// The header must say where the model is and what it can do, in words.
//
// It replaces a geometry dump ("llama L=16 d=2048 heads=32/8 ...").
func TestTheHeaderSaysWhereTheModelIsInWords(t *testing.T) {
	cases := []struct {
		h    header
		want []string
		not  []string
	}{
		{header{Name: "Llama-3.2-1B-Instruct-Q4_K_M", Layers: 16, OnDevice: 16, Device: "NVIDIA GeForce RTX 3050 Ti Laptop GPU", Devices: 1, Context: 8192, Asked: 8192, Chat: true},
			[]string{"Llama-3.2-1B-Instruct-Q4_K_M", "all 16 layers on the NVIDIA GeForce RTX 3050 Ti Laptop GPU", "context 8,192", "chat"},
			[]string{"L=", "blocks", "(the most"}},
		{header{Name: "Qwen3-30B-A3B", Layers: 48, OnDevice: 4, Device: "RTX", Devices: 1, Context: 4096, Asked: 4096, Chat: true},
			[]string{"4 of 48 layers on the RTX, the rest on the CPU"}, nil},
		{header{Name: "stories15M", Layers: 6, Context: 256, Asked: 4096},
			[]string{"on the CPU", "context 256 (the most it supports)", "completion only"}, []string{"layers"}},
		{header{Name: "SmolVLM", Layers: 30, OnDevice: 30, Devices: 2, Context: 8192, Asked: 8192, Chat: true, Vision: true},
			[]string{"all 30 layers on 2 GPUs", "sees images"}, nil},
	}
	for _, c := range cases {
		got := c.h.String()
		// The device last: the header is cut to one line, and a long card name
		// first would push out "chat" and "sees images".
		if i, j := strings.Index(got, "context"), strings.LastIndex(got, " on "); i < 0 || j < i {
			t.Errorf("header %q does not end with where the layers are", got)
		}
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("header %q does not say %q", got, w)
			}
		}
		for _, n := range c.not {
			if strings.Contains(got, n) {
				t.Errorf("header %q still says %q", got, n)
			}
		}
	}
}

func TestADeviceLabelIsTheCard(t *testing.T) {
	for in, want := range map[string]string{
		"0:NVIDIA GeForce RTX 3050 Ti Laptop GPU [cuda]": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
		"1:Intel(R) Iris(R) Xe Graphics [vulkan]":        "Intel(R) Iris(R) Xe Graphics",
		"Apple M4": "Apple M4",
	} {
		if got := deviceLabel(in); got != want {
			t.Errorf("deviceLabel(%q) = %q, want %q", in, got, want)
		}
	}
}
