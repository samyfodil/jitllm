package widgets

import (
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// KV is one label/value row, the unit every detail pane in this app is built
// from. The value is static.
//
// The value is a one-line Paragraph, not primitives.Text, which paints past
// its bounds (UPSTREAM.md #10).
func KV(label, value string, labelColor, valueColor widget.Color, size float32) *primitives.BoxWidget {
	return primitives.HBox(
		primitives.Box(primitives.Text(label).FontSize(size).Color(labelColor)).Width(190),
		primitives.Expanded(NewParagraph(value).FontSize(size).Color(valueColor).MaxLines(1)),
	).Gap(8).PaddingXY(0, 4)
}

// KVSignal is [KV] with a live value.
func KVSignal(label string, value state.ReadonlySignal[string], labelColor, valueColor widget.Color, size float32) *primitives.BoxWidget {
	return primitives.HBox(
		primitives.Box(primitives.Text(label).FontSize(size).Color(labelColor)).Width(190),
		primitives.Expanded(NewParagraph("").ContentSignal(value).FontSize(size).Color(valueColor).MaxLines(1)),
	).Gap(8).PaddingXY(0, 4)
}
