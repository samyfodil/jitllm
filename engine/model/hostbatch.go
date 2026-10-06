package model

// hostBatch is a prompt chunk's host scratch: what growBatch and growMoEBatch
// size for a block the host runs. A closing State hands it to the model with
// its promptBuf, and the next State's first prompt takes it back.
//
// It is the largest thing a request allocates: tens of MiB at a 512-row
// chunk, against a decode token's few KiB, so a server opening a State per
// request was handing the collector that much per request -- and under a memory limit near the weight budget, that is
// a cycle per request (AGENTS.md RULE 2f).
//
// Nothing here is read before a chunk writes it, so it is reused as it is,
// as promptBuf's residual rows always were. TestReusedPromptScratchIsNotRead
// poisons a handed-back set with NaN and demands the next State's logits are
// bit-identical to a fresh model's.
type hostBatch struct {
	bh, bq, bxb, bk, bv, bgate, bup, bhf, battf []float32
	qgate, ogate, bqf, bxbf                     []float32
	bmRow                                       []int32
	bmW, bmX, bmY, bmG, bmU                     []float32
}

// scratch is n float32s with room for c, from b when it is big enough.
func scratch(b []float32, n, c int) []float32 {
	c = max(c, n)
	if cap(b) >= c {
		return b[:n]
	}
	return make([]float32, n, c)
}

// takeHostBatch moves this State's host scratch out, leaving it to regrow.
func (s *State) takeHostBatch() hostBatch {
	h := hostBatch{
		bh: s.bh, bq: s.bq, bxb: s.bxb, bk: s.bk, bv: s.bv, bgate: s.bgate, bup: s.bup,
		bhf: s.bhf, battf: s.battf, qgate: s.qgate, ogate: s.ogate, bqf: s.bqf, bxbf: s.bxbf,
		bmRow: s.bmRow, bmW: s.bmW, bmX: s.bmX, bmY: s.bmY, bmG: s.bmG, bmU: s.bmU,
	}
	// One the State took from the model and never grew into is still worth
	// handing on.
	if h.bh == nil {
		h = s.spareHost
	}
	if h.bmRow == nil {
		h.bmRow, h.bmW, h.bmX, h.bmY, h.bmG, h.bmU = s.spareHost.bmRow, s.spareHost.bmW,
			s.spareHost.bmX, s.spareHost.bmY, s.spareHost.bmG, s.spareHost.bmU
	}
	s.bh, s.bq, s.bxb, s.bk, s.bv, s.bgate, s.bup = nil, nil, nil, nil, nil, nil, nil
	s.bhf, s.battf, s.qgate, s.ogate, s.bqf, s.bxbf = nil, nil, nil, nil, nil, nil
	s.bmRow, s.bmW, s.bmX, s.bmY, s.bmG, s.bmU = nil, nil, nil, nil, nil, nil
	s.spareHost = hostBatch{}
	return h
}
