package screen

import (
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
)

// The map is built before any model loads, so it is first measured empty. A
// load must grow it -- through the section it sits in -- or it draws its lanes
// over the rows below.
func TestTheMemoryMapGrowsWhenAModelLoads(t *testing.T) {
	sh := app.NewShell(nil, nil)
	ctx := mockCtx()
	w := section(sh, "Memory", placement(sh))
	widget.MountTree(w, ctx)
	cons := geometry.Constraints{MaxWidth: 1200, MaxHeight: geometry.Infinity}
	before := widget.LayoutChild(w, ctx, cons).Height

	sh.Store.BlockMap.Set(splitBlocks(16, 16))
	sh.Store.Alloc.Set(app.Allocation{NBlocks: 16, HostUsed: 1 << 30, PageBytes: 64 << 20})
	after := widget.LayoutChild(w, ctx, cons).Height
	if after < before+150 {
		t.Fatalf("the section measured %v empty and %v loaded: the map's lanes did not reach the layout", before, after)
	}
}
