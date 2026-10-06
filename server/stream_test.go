package server

import (
	"strings"
	"testing"
)

// streamText is where a streaming shim actually goes wrong, so it is gated on
// its own rather than only through the HTTP surface.

type fakeVocab struct{ pieces map[int32]string }

func (v fakeVocab) Decode(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(v.pieces[id])
	}
	return b.String()
}
func (v fakeVocab) DecodeChat(ids []int32) string { return v.Decode(ids) }

// TestStopStringIsNeverLeakedAcrossTokens is the gate for the held-back tail.
// With the hold-back removed (hold = 0) it fails with `leaked "<" before the
// stop string completed`; a check on the final text alone cannot see this.
func TestStopStringIsNeverLeakedAcrossTokens(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{
		1: "hello", 2: "<", 3: "|end", 4: "|>", 5: "tail",
	}}
	st := newStreamText(v, []string{"<|end|>"})

	var emitted strings.Builder
	var stopped bool
	var match string
	ids := []int32{}
	for _, id := range []int32{1, 2, 3, 4, 5} {
		ids = append(ids, id)
		chunk, hit, m := st.push(ids)
		emitted.WriteString(chunk)
		if !hit && strings.Contains(emitted.String(), "<") {
			t.Fatalf("leaked %q before the stop string completed: emitted %q after ids %v",
				"<", emitted.String(), ids)
		}
		if hit {
			stopped, match = true, m
			break
		}
	}
	if !stopped {
		t.Fatalf("the stop string never matched; emitted %q", emitted.String())
	}
	if match != "<|end|>" {
		t.Fatalf("matched %q, want %q", match, "<|end|>")
	}
	if got := emitted.String(); got != "hello" {
		t.Fatalf("emitted %q, want %q -- the stop string itself is never part of the completion", got, "hello")
	}
}

// TestFlushReleasesTheHeldTailWhenNoStopMatched proves the hold-back is not a
// silent truncation: without flush() the last len(stop)-1 bytes vanish and it
// fails with `after flush got "hell", want "hello"`.
func TestFlushReleasesTheHeldTailWhenNoStopMatched(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{1: "hel", 2: "lo"}}
	st := newStreamText(v, []string{"<|end|>"})
	var b strings.Builder
	ids := []int32{}
	for _, id := range []int32{1, 2} {
		ids = append(ids, id)
		chunk, hit, _ := st.push(ids)
		if hit {
			t.Fatal("no stop string is present, yet one matched")
		}
		b.WriteString(chunk)
	}
	if b.String() == "hello" {
		t.Fatalf("nothing was held back, so this gate cannot see a missing flush; "+
			"hold is %d", st.hold)
	}
	b.WriteString(st.flush())
	if got := b.String(); got != "hello" {
		t.Fatalf("after flush got %q, want %q", got, "hello")
	}
}

// TestAMultiByteRuneIsNeverSplitAcrossTwoWrites: a BPE token can split a
// rune. Without truncPartialRune in push() it fails with `chunk 0 is not
// valid UTF-8`.
func TestAMultiByteRuneIsNeverSplitAcrossTwoWrites(t *testing.T) {
	// "€" is e2 82 ac; the tokenizer splits it across two ids.
	v := fakeVocab{pieces: map[int32]string{1: "\xe2\x82", 2: "\xac", 3: "!"}}
	st := newStreamText(v, nil)
	var b strings.Builder
	ids := []int32{}
	for i, id := range []int32{1, 2, 3} {
		ids = append(ids, id)
		chunk, _, _ := st.push(ids)
		for _, r := range chunk {
			if r == '�' {
				t.Fatalf("chunk %d is not valid UTF-8: %q", i, chunk)
			}
		}
		b.WriteString(chunk)
	}
	b.WriteString(st.flush())
	if got := b.String(); got != "€!" {
		t.Fatalf("reassembled %q, want %q", got, "€!")
	}
}

// TestNoStopStringMeansNoHoldBack: the common case must not be delayed by a
// window it does not need.
func TestNoStopStringMeansNoHoldBack(t *testing.T) {
	v := fakeVocab{pieces: map[int32]string{1: "a", 2: "b"}}
	st := newStreamText(v, nil)
	if st.hold != 0 {
		t.Fatalf("hold is %d with no stop strings; a request with none must stream every byte immediately", st.hold)
	}
	chunk, _, _ := st.push([]int32{1})
	if chunk != "a" {
		t.Fatalf("first chunk %q, want %q", chunk, "a")
	}
}
