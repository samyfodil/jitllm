// Package screen holds the top-level screens.
//
// Each screen is an [app.ScreenFunc]: it takes the shell and returns a root
// widget. It may be called more than once -- a theme swap rebuilds the tree --
// so nothing that matters may live in a closure over the first build. Mutable
// state goes in the shell's Store, or in a registry keyed by the shell where a
// screen's own form state has no Store field (see modelsFor).
//
// Among them:
//
//	Session  the conversation: prompt, transcript, live reply, rate strip
//	Models   the catalog: scan, probe, load
//	Convert  the GGUF and safetensors sources and the convert form
//	Devices  the machine: host, budgets, kernels, devices, placement
//
// Screens with an engine seam ([Attach]) stay inert and
// say so when nothing is wired into it.
package screen
