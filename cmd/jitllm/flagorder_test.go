package main

import (
	"flag"
	"io"
	"strings"
	"testing"
)

// A flag after the file is swallowed into the prompt by Go's parser, and the
// engine then answers a question nobody asked in a configuration nobody chose:
// "-chat what is the capital of France?" generates a fluent raw completion
// whose only trace is a stray token at the head of the ids.
func TestAFlagAfterTheFileIsRefusedAndAPromptIsNot(t *testing.T) {
	mk := func(argv ...string) *flag.FlagSet {
		fs := flag.NewFlagSet("run", flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		fs.Bool("chat", false, "")
		fs.String("system", "", "")
		fs.Int("n", 32, "")
		if err := fs.Parse(argv); err != nil {
			t.Fatalf("parse %v: %v", argv, err)
		}
		return fs
	}
	refused := []struct {
		what string
		argv []string
	}{
		{"the one that shipped", []string{"m.jlm", "-chat", "what is the capital of France?"}},
		{"double dash", []string{"m.jlm", "--chat", "hi"}},
		{"with a value attached", []string{"m.jlm", "-n=8", "hi"}},
		{"a value-taking flag", []string{"m.jlm", "-system", "be terse", "hi"}},
	}
	for _, c := range refused {
		err := misplacedFlag(mk(c.argv...))
		if err == nil {
			t.Errorf("%s: %v was accepted; it becomes prompt text", c.what, c.argv)
			continue
		}
		t.Logf("%s: %v", c.what, strings.SplitN(err.Error(), "\n", 2)[0])
	}

	// The refusal must not be "starts with a dash", which would eat a
	// legitimate prompt. It matches the flags this set defines.
	allowed := [][]string{
		{"m.jlm", "-42 degrees is"},
		{"m.jlm", "--", "a literal dash dash"},
		{"m.jlm", "-notaflag", "hi"},
		{"m.jlm"},
		{},
	}
	for _, argv := range allowed {
		if err := misplacedFlag(mk(argv...)); err != nil {
			t.Errorf("%v was refused: %v", argv, err)
		}
	}
}
