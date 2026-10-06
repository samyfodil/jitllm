package widgets

import (
	"github.com/gogpu/ui/core/progressbar"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Gauge is a labelled progress bar with a formatted caption: a label on the
// left, a caption on the right, and the bar underneath. The caption and value
// are separate signals so a fraction is always shown with its figures.
func Gauge(
	label string,
	value state.ReadonlySignal[float64],
	caption state.ReadonlySignal[string],
	colors progressbar.ProgressBarColorScheme,
	labelColor widget.Color,
	// size is the caller's type role, so a gauge does not invent one.
	size float32,
) *primitives.BoxWidget {
	return primitives.VBox(
		primitives.HBox(
			primitives.Expanded(primitives.Text(label).FontSize(size).Color(labelColor)),
			primitives.Text("").ContentSignal(caption).FontSize(size).Color(labelColor),
		).Gap(8),
		progressbar.New(
			progressbar.ValueReadonlySignal(value),
			progressbar.ColorSchemeOpt(colors),
			progressbar.Height(6),
		),
	).Gap(3).PaddingXY(0, 4)
}
