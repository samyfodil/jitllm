package screen

import (
	"fmt"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/slider"

	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/widget"

	"github.com/samyfodil/jitllm/common/session"
	"github.com/samyfodil/jitllm/ui/app"
)

// HostLine is the host's share; see [session.HostLine].
func HostLine(a app.Allocation) string { return session.HostLine(a) }

// DeviceUseLine is one device's share; see [session.DeviceUseLine].
func DeviceUseLine(d app.DeviceUse) string { return session.DeviceUseLine(d) }

// relocate is the control that moves the seam while the model is loaded. It
// commits on a button, not on the drag: both queues drop on overflow and a
// move re-reads, repacks and uploads every block that crossed, so it is one
// commit per intent. The slider is a fraction because its maximum is fixed
// at build time, before the block count is known.
func relocate(sh *app.Shell) widget.Widget {
	caption := reactive(sh, sh.Store.Alloc.AsReadonly(), func() string {
		a := sh.Store.Alloc.Get()
		if a.NBlocks == 0 {
			return "load a model to move its blocks"
		}
		return fmt.Sprintf("%d of %d block(s) on a device now", a.DeviceBlocks, a.NBlocks)
	})

	target := reactive(sh, sh.Store.SeamTarget.AsReadonly(), func() string {
		a := sh.Store.Alloc.Get()
		if a.NBlocks == 0 {
			return ""
		}
		return fmt.Sprintf("move to %d", seamBlocks(sh))
	})

	apply := button.New(
		button.TextOpt("Apply"),
		button.SizeOpt(button.Small),
		button.PainterOpt(sh.P.Button),
		button.OnClick(func() {
			eng := depsOf(sh).Engine
			if eng == nil {
				sh.Store.Status.Set("no engine is attached: see screen.Attach")
				return
			}
			if sh.Store.Alloc.Get().NBlocks == 0 {
				sh.Store.Status.Set("no model is loaded")
				return
			}
			eng.Relocate(seamBlocks(sh))
		}),
	)

	return primitives.VBox(
		line(sh, caption).Color(sh.P.Text()),
		slider.New(
			slider.ValueSignal(sh.Store.SeamTarget),
			slider.Min(0), slider.Max(1), slider.Step(0.02),
			slider.PainterOpt(sh.P.Slider),
		),
		primitives.HBox(
			primitives.Expanded(line(sh, target)),
			apply,
		).Gap(8),
		// The reading is what the device took, never the request: growing is
		// best-effort, and a move that would lose the attention history or
		// recurrent summary is refused. It is queued behind the reply in flight.
		note(sh,
			"Moves layers between the GPU and the CPU. The count above is what the GPU took,",
			"which can be fewer than asked. A move waits for the current reply to finish.",
		),
	)
}

// seamBlocks is the slider's fraction as a block count.
func seamBlocks(sh *app.Shell) int {
	a := sh.Store.Alloc.Get()
	n := int(float64(sh.Store.SeamTarget.Get())*float64(a.NBlocks) + 0.5)
	if n < 0 {
		n = 0
	}
	if n > a.NBlocks {
		n = a.NBlocks
	}
	return n
}
