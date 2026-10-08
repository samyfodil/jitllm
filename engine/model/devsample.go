package model

import (
	"sync"
	"time"
)

// Whether a sampled token's candidates are selected on the device
// (nn.Head.SampleK) or the row is read back and the host sampler selects
// them is a measured choice, not a rule: reading a 128k-vocabulary row back
// (513 KB) cost less on an RTX 3050 Ti than the device's penalty and top-k
// passes, measured at about 3% of the token in a paired A/B. Which wins is a
// property of the card, the vocabulary, k and whether the penalty runs, so
// each (device, vocabulary, k, penalty) key is decided by timing both arms on
// real tokens and the winner is kept until the next probe. Both arms return
// the same token (ForwardSample), so a probe costs time and never an answer.

// DeviceSampleMode is WithDeviceSample's setting.
type DeviceSampleMode int

const (
	// DeviceSampleAuto measures both arms per key and keeps the faster.
	DeviceSampleAuto DeviceSampleMode = iota
	// DeviceSampleOn always selects on the device where it can.
	DeviceSampleOn
	// DeviceSampleOff always reads the row back.
	DeviceSampleOff
)

// WithDeviceSample forces the device sampler's choice; the default measures.
func WithDeviceSample(m DeviceSampleMode) Option {
	return func(l *loadOpts) { l.opt.devSample = m }
}

const (
	// sampleProbeRounds is how many ABBA quartets a probe times: each round is
	// device, host, host, device, so a drift in the clock across the round
	// falls on both arms alike (the seam tuner's ordering).
	sampleProbeRounds = 8
	// sampleReprobe is how many tokens a decided key serves before it is
	// measured again: a card shared with another model, or a different
	// context length, can move the answer.
	sampleReprobe = 4096
)

// sampleKey is what the choice is a property of. The device is the State's
// own, so the key is per Model with the device named by its identity.
type sampleKey struct {
	dev   any
	vocab int
	k     int
	pen   bool
}

// sampleChoice is one key's probe and verdict.
type sampleChoice struct {
	// step counts tokens since the probe began; below 4*rounds the probe is
	// running and the arm is ABBA's.
	step int
	// dev and host are the probe's summed token times per arm.
	dev, host time.Duration
	// useDev is the verdict, valid once the probe is over.
	useDev bool
}

// sampleChoices is the Model's verdicts, under mu.
type sampleChoices struct {
	mu sync.Mutex
	m  map[sampleKey]*sampleChoice
}

// arm reports whether this token takes the device arm and whether it is a
// probe token (to be timed and recorded).
func (c *sampleChoice) arm() (dev, probe bool) {
	const n = 4 * sampleProbeRounds
	if c.step < n {
		q := c.step % 4
		return q == 0 || q == 3, true
	}
	return c.useDev, false
}

// record adds one token's time and advances; at the end of a probe it
// decides, and past the re-probe cadence it starts the next probe.
func (c *sampleChoice) record(dev, probe bool, d time.Duration) {
	const n = 4 * sampleProbeRounds
	if probe {
		if dev {
			c.dev += d
		} else {
			c.host += d
		}
	}
	c.step++
	if c.step == n {
		c.useDev = c.dev < c.host
	}
	if c.step >= n+sampleReprobe {
		c.step, c.dev, c.host = 0, 0, 0
	}
}

// sampleChoiceFor is the key's state, made on first use. arm and record run
// under the same lock: sessions on one device share a key.
func (s *State) sampleChoiceFor(k sampleKey) *sampleChoice {
	cs := &s.m.sampleChoices
	cs.mu.Lock()
	defer cs.mu.Unlock()
	c := cs.m[k]
	if c == nil {
		if cs.m == nil {
			cs.m = map[sampleKey]*sampleChoice{}
		}
		c = &sampleChoice{}
		cs.m[k] = c
	}
	return c
}

// DeviceSampleChoice is the auto mode's state for the key a sampler would use
// on this State now: probing, or the device or the host chosen, with the two
// arms' probe times. ok is false where there is no key yet.
func (s *State) DeviceSampleChoice(smp *Sampler) (probing, device bool, dev, host time.Duration, ok bool) {
	cs := &s.m.sampleChoices
	cs.mu.Lock()
	defer cs.mu.Unlock()
	c := cs.m[s.sampleKeyOf(smp)]
	if c == nil {
		return false, false, 0, 0, false
	}
	return c.step < 4*sampleProbeRounds, c.useDev, c.dev, c.host, true
}

func (s *State) sampleKeyOf(smp *Sampler) sampleKey {
	return sampleKey{dev: s.device, vocab: s.c.NVocab, k: smp.TopK, pen: smp.RepeatPen > 1}
}
