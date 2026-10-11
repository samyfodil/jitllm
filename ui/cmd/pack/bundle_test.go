package main

import "testing"

// TestSigningIDNamesEveryShippedProgram: each command-line program the release
// signs on macOS has its identifier, and anything else keeps codesign's default.
func TestSigningIDNamesEveryShippedProgram(t *testing.T) {
	for name, want := range map[string]string{
		"jitllm":     "org.jitllm.cli",
		"jitllmd":    "org.jitllm.service",
		"jitllm-tui": "org.jitllm.tui",
		"other":      "",
	} {
		if got := signingID(name); got != want {
			t.Errorf("signingID(%q) = %q, want %q", name, got, want)
		}
	}
}
