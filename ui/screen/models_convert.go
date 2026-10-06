package screen

import (
	"github.com/samyfodil/jitllm/common/convertjob"
	"github.com/samyfodil/jitllm/ui/app"
)

// Conversion, the other half of this screen. The work is
// [convertjob]'s, which the terminal drives too; this file puts its progress
// on the shell's bar and status line.

// ConvertRequest is one conversion; see [convertjob.Request].
type ConvertRequest = convertjob.Request

// DestFor is the container a source converts to; see [convertjob.DestFor].
func DestFor(src, dir string) string { return convertjob.DestFor(src, dir) }

// primaryDir is the model directory conversions write to; see
// [convertjob.PrimaryDir].
func primaryDir(dirs []string) string { return convertjob.PrimaryDir(dirs) }

// PlanConvert resolves a request and refuses the ones that cannot work; see
// [convertjob.Plan].
func PlanConvert(req ConvertRequest) (ConvertRequest, error) { return convertjob.Plan(req) }

// RunConvert converts one model and reports through the shell's progress
// signals. It blocks; the caller decides where it runs, and that should be the
// engine worker when there is one, because a conversion beside a decode ruins
// both ([ModelsHooks.Run]). Without an engine it runs on its own goroutine.
func RunConvert(sh *app.Shell, req ConvertRequest) error {
	_, err := convertjob.Run(req, func(u convertjob.Update) {
		switch u.Stage {
		case convertjob.Started, convertjob.Writing:
			sh.Store.Progress.Set(u.Fraction)
			sh.Store.ProgressLabel.Set(u.Label)
		case convertjob.Done:
			sh.Store.ProgressLabel.Set(u.Label)
			sh.SetStatus(u.Label)
			sh.Store.Progress.Set(0)
		case convertjob.Failed:
			sh.Store.ProgressLabel.Set("")
			sh.Store.Progress.Set(0)
		}
	})
	return err
}
