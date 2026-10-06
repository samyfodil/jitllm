// Package widgets holds the composite widgets this app needs and gogpu/ui does
// not ship.
//
// It is a leaf on purpose: it imports gogpu/ui and nothing else of this
// application, so both app/ and screen/ can depend on it without a cycle.
// Anything here takes signals and plain values, never the app's Store.
package widgets
