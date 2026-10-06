package cpu

// RowMode is which of the row-statistic kernels the layer norm's passes
// build: the layer norm itself, Gemma 3n's gaussian top-k (EmitGaussTopK) or
// AltUp's magnitude match (EmitMagMatch). They share the mean and variance
// passes and differ in what the statistics become.
type RowMode int

const (
	RowLayerNorm RowMode = iota
	RowGauss
	RowMag
)
