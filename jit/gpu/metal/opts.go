package metal

import "time"

// Opts are the choices a context is opened with. Zero is the shipping default.
// They are per context, so two tiers in one process open theirs differently.
type Opts struct {
	// FastMath compiles libraries with Metal's fast math instead of the IEEE
	// rounding the other backends lower to; see strictMath.
	FastMath bool
	// Spin is how long a Wait polls a command buffer's status before it blocks
	// in waitUntilCompleted. Zero takes the default, 20ms; negative blocks at
	// once, which is waitUntilCompleted's own behaviour.
	Spin time.Duration
}

func (o Opts) spin() time.Duration {
	switch {
	case o.Spin < 0:
		return 0
	case o.Spin == 0:
		return 20 * time.Millisecond
	}
	return o.Spin
}
