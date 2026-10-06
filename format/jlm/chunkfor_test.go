package jlm

import "testing"

// The stored granularity must be the smallest span a reader asks for on its
// own, and it must be what the pager then uses. It is stored in the header
// rather than set as a package global, so there is no ordering to get wrong.
// The expectation is computed from kernels.PackedWords, not from chunkFor's own
// formula.
func TestTheStoredChunkIsTheSmallestRequestedSpan(t *testing.T) {
	for _, tc := range []struct {
		name string
		es   []Entry
		want uint64
	}{{
		// A routed bank lives in expert pages, which are read whole, so its
		// planes must not drag the block pages' granularity down to a d plane.
		name: "bank is read whole",
		es: []Entry{{Role: RoleExpGateBank, Type: TypeQ4S, NDim: 3,
			Dims: [4]uint64{2048, 512, 512, 0}}},
		want: 1 << 20,
	}, {
		// An ordinary 2-D weight is one sheet, and its planes are still asked
		// for separately.
		name: "dense weight",
		es: []Entry{{Role: RoleAttnQ, Type: TypeQ4S, NDim: 2,
			Dims: [4]uint64{2048, 2048, 0, 0}}},
	}, {
		// Nothing paged: every entry is dense-region or expanded, so there is
		// no smallest request and the cap stands.
		name: "nothing paged",
		es: []Entry{
			{Role: RoleAttnQ, Type: TypeQ4S, NDim: 2, Block: DenseBlock,
				Dims: [4]uint64{2048, 2048, 0, 0}},
			{Role: RoleAttnNorm, Type: TypeF32, NDim: 1, Dims: [4]uint64{2048, 0, 0, 0}},
		},
		want: 1 << 20,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			want := tc.want
			if want == 0 {
				smallest := uint64(1) << 20
				for i := range tc.es {
					qs, d, sc, sheets, err := sheetSpan(&tc.es[i])
					if err != nil || sheets == 0 {
						t.Fatalf("fixture entry %d is not packed: %v", i, err)
					}
					for _, n := range [3]uint64{qs, d, sc} {
						if n > 0 && n < smallest {
							smallest = n
						}
					}
				}
				for want = 1 << 20; want > Align && want > smallest; want >>= 1 {
				}
			}
			if got := chunkFor(tc.es); got != want {
				t.Errorf("chunkFor = %d, want %d", got, want)
			}
		})
	}

	// It must be a power of two at least Align, or the page read it sizes is
	// unaligned and the O_DIRECT handle answers EINVAL instead of data.
	for _, es := range [][]Entry{
		{{Role: RoleAttnQ, Type: TypeQ4S, NDim: 2, Dims: [4]uint64{32, 1, 0, 0}}},
		nil,
	} {
		c := chunkFor(es)
		if c < Align || c&(c-1) != 0 {
			t.Errorf("chunkFor(%d entries) = %d, which is not a power of two >= %d",
				len(es), c, Align)
		}
	}
}

// And a header carrying a granularity the pager cannot use is refused rather
// than repaired. The tower-only case (degenerate PageSize, megabyte vision
// pages) is here because the first version of the check refused it.
func TestABadStoredChunkIsRefused(t *testing.T) {
	good := &Header{PageSize: 1 << 20, NBlocks: 1, NTensors: 1, Chunk: 16 << 10,
		CfgOff: HeaderBytes, DataOff: 1 << 20, DenseOff: HeaderBytes,
		TabOff: HeaderBytes, FPOff: HeaderBytes, VocOff: HeaderBytes,
		VisOff: HeaderBytes, VisDataOff: 1 << 20, VisPageSize: Align}
	for _, tc := range []struct {
		name  string
		chunk uint64
		ok    bool
	}{
		{"the converter's value", 16 << 10, true},
		{"zero -- a header that predates the field", 0, false},
		{"not a power of two", 3 << 10, false},
		{"below the alignment", 2048, false},
		{"larger than the page", 2 << 20, false},
	} {
		h := *good
		h.Chunk = tc.chunk
		b := make([]byte, HeaderBytes)
		h.encode(b)
		got, err := decodeHeader(b, 4<<20)
		switch {
		case tc.ok && err != nil:
			t.Errorf("%s: %v", tc.name, err)
		case tc.ok && got.Chunk != tc.chunk:
			t.Errorf("%s: decoded chunk %d, want %d", tc.name, got.Chunk, tc.chunk)
		case !tc.ok && err == nil:
			t.Errorf("%s: chunk %d was accepted", tc.name, tc.chunk)
		}
	}

	// A tower-only container: no text blocks, so PageSize is the degenerate
	// Align and the real pages are the vision array's.
	tower := *good
	tower.PageSize, tower.NBlocks = Align, 0
	tower.VisPageSize, tower.NVisBlocks = 8<<20, 12
	tower.Chunk = 32 << 10
	b := make([]byte, HeaderBytes)
	tower.encode(b)
	if _, err := decodeHeader(b, 128<<20); err != nil {
		t.Errorf("a tower-only container was refused: %v", err)
	}
}

// The stored stream-group count must be the largest split whose per-plane
// transfer still clears minGroupBytes. The gate pins the shape
// of the rule, not a constant.
func TestTheStoredStreamGroupsIsTheLargestSplitThatClearsTheFloor(t *testing.T) {
	// Qwen3-Next-80B's routed banks: k=2048, 512 experts, Q4_K. Its d plane is
	// the smallest of the three and is what limits the split.
	bank := []Entry{
		{Role: RoleExpGateBank, Type: TypeQ4S, NDim: 3, Dims: [4]uint64{2048, 512, 512, 0}},
		{Role: RoleExpUpBank, Type: TypeQ4S, NDim: 3, Dims: [4]uint64{2048, 512, 512, 0}},
		{Role: RoleExpDownBank, Type: TypeQ4S, NDim: 3, Dims: [4]uint64{512, 2048, 512, 0}},
	}
	_, d, _, sheets, err := sheetSpan(&bank[0])
	if err != nil || sheets != 512 {
		t.Fatalf("fixture is not a bank: sheets=%d err=%v", sheets, err)
	}
	if d != 16384 {
		t.Fatalf("the d plane is %d B, not the 16384 the measured row was taken on; "+
			"re-derive the expectation before trusting this gate", d)
	}
	if got := streamGroupsFor(bank, 10); got != 3 {
		t.Errorf("streamGroupsFor(k=10) = %d, want 3: four sheets of a 16384 B "+
			"plane is 64 KiB and clears the floor, three is 48 KiB and does not",
			got)
	}

	// The shape, not the constant: every answer must clear the floor, and one
	// more group must not.
	for _, used := range []int{2, 4, 8, 10, 16, 64, 128} {
		n := streamGroupsFor(bank, used)
		if n < 1 || n > uint32(used) {
			t.Fatalf("k=%d: %d groups is outside [1, k]", used, n)
		}
		per := func(g uint32) uint64 { return (uint64(used) + uint64(g) - 1) / uint64(g) * d }
		if n > 1 && per(n) < minGroupBytes {
			t.Errorf("k=%d: %d groups moves %d B a plane, under the %d B floor",
				used, n, per(n), minGroupBytes)
		}
		if int(n) < used && per(n+1) >= minGroupBytes {
			t.Errorf("k=%d: stopped at %d groups when %d still clears the floor "+
				"(%d B)", used, n, n+1, per(n+1))
		}
	}

	// Nothing streams without a routed bank, and a k of one cannot be split.
	for _, tc := range []struct {
		name string
		es   []Entry
		used int
	}{
		{"dense model", []Entry{{Role: RoleAttnQ, Type: TypeQ4S, NDim: 2,
			Dims: [4]uint64{2048, 2048, 0, 0}}}, 10},
		{"one expert used", bank, 1},
	} {
		if got := streamGroupsFor(tc.es, tc.used); got != 1 {
			t.Errorf("%s: %d groups, want 1", tc.name, got)
		}
	}
}
