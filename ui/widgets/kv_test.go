package widgets

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

// A long value must be cut to its row with an ellipsis, not painted past it.
// Against the violation (a plain primitives.Text) the whole path is drawn in
// one run far wider than the row.
func TestAKVValueIsCutToItsRow(t *testing.T) {
	path := "/mnt/models/library/qwen3-next/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-with-a-long-suffix.jlm"
	row := KVSignal("Path", state.NewSignal(path).AsReadonly(), widget.ColorBlack, widget.ColorBlack, 14)
	ctx := uitest.NewMockContext()
	widget.MountTree(row, ctx)
	row.Layout(ctx, geometry.Constraints{MaxWidth: 520, MaxHeight: 200})
	c := &uitest.MockCanvas{}
	row.Draw(ctx, c)

	var drawn []string
	for _, x := range c.Texts {
		drawn = append(drawn, x.Text)
	}
	for _, x := range c.StyledTexts {
		drawn = append(drawn, x.Text)
	}
	for _, s := range drawn {
		if s == "Path" {
			continue
		}
		if s == path {
			t.Fatalf("the whole path was drawn in one run: %q", s)
		}
		if w := float32(len([]rune(s))) * 14 * charWidthRatio; w > 520-190-8 {
			t.Errorf("value %q is ~%.0f px in a %d px column", s, w, 520-190-8)
		}
	}
	if !strings.Contains(strings.Join(drawn, "|"), "…") {
		t.Errorf("a cut value does not say it was cut: %q", drawn)
	}
}
