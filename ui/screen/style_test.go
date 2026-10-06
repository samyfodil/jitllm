package screen

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/samyfodil/jitllm/internal/testmodels"
)

// No screen may invent a font size: a hardcoded size drops the line height
// that comes with the theme's type role. The gate is a grep because the
// property is syntactic.
func TestNoScreenInventsAFontSize(t *testing.T) {
	testmodels.SourceTree(t)
	// FontSize( followed by a digit: a literal rather than a role.
	lit := regexp.MustCompile(`FontSize\(\s*\d`)

	var bad []string
	for _, dir := range []string{".", "../app", "../widgets"} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			name := e.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			for i, l := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(strings.TrimSpace(l), "//") {
					continue
				}
				if lit.MatchString(l) {
					bad = append(bad, filepath.Join(dir, name)+":"+itoa(i+1)+" "+strings.TrimSpace(l))
				}
			}
		}
	}
	if len(bad) > 0 {
		t.Errorf("%d hardcoded font size(s); take the role from sh.P.Type and the "+
			"line height comes with it:\n  %s", len(bad), strings.Join(bad, "\n  "))
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
