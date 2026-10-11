package crash

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The 8.3 short form of the home directory (C:\Users\RUNNER~1, which %TEMP%
// spells paths with) is scrubbed as the long one is. Against the violation
// (pathForms returning the home alone) the short name survives.
func TestScrubShortensTheShortFormOfHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	forms := pathForms(home)
	short := ""
	for _, f := range forms[1:] {
		if strings.Contains(f, "~") {
			short = f
		}
	}
	if short == "" {
		t.Skipf("this volume has no 8.3 name for %s (forms %q): the gate proved nothing", home, forms)
	}
	out := Scrub(`at ` + short + `\AppData\Local\Temp\x.go:12 and ` + filepath.ToSlash(short) + "/y")
	if strings.Contains(strings.ToLower(out), strings.ToLower(filepath.Base(short))) {
		t.Errorf("the short form %q survived: %q", short, out)
	}
}
