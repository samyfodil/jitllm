package tier

// KVRoom is how many more positions of attention history the devices can take
// without sending a page home: in every layer a device's pool holds, the pages
// it has free and the pages the budget lets it grow by, priced at one page of
// every layer at once, and the fewest of any device, since a sequence grows in
// every layer together. page is the pool's positions a page (0 with no pool);
// a sequence takes whole pages, so a caller adding up sequences rounds each up
// to it. A device with no attention layer does not bound it.
//
// It is what a scheduler admits by: rows whose histories fit it all stay on
// the card and step together, where rows past it send their oldest pages home
// and stream them back every token (kvevict.go) -- correct, and slower by the
// whole history crossing the bus once a step.
func (g *GPU) KVRoom() (positions, page int) {
	positions = -1
	for _, d := range g.devs {
		d.mu.Lock()
		n, p := d.kvRoom()
		d.mu.Unlock()
		if p == 0 {
			continue
		}
		page = p
		if positions < 0 || n < positions {
			positions = n
		}
	}
	return max(positions, 0), page
}

// kvRoom is KVRoom for one device. Callers hold g.mu.
func (g *devTier) kvRoom() (positions, page int) {
	kp := g.kvp
	if kp == nil || len(kp.layers) == 0 {
		return 0, 0
	}
	var per uint64
	free := -1
	for _, l := range kp.layers {
		per += kp.pageBytes(l)
		if n := len(l.free) + len(l.fenced); free < 0 || n < free {
			free = n
		}
	}
	g.settleRounding()
	var grow uint64
	if lim, used := g.limit, g.used+g.reserved(); used < lim {
		grow = lim - used
	}
	if g.pool != nil && g.pool.limit > 0 {
		g.pool.mu.Lock()
		if g.pool.used < g.pool.limit {
			grow = min(grow, g.pool.limit-g.pool.used)
		} else {
			grow = 0
		}
		g.pool.mu.Unlock()
	}
	// The one page eviction keeps free to stream through is not room.
	pages := max(free-1, 0) + int(grow/max(per, 1))
	return pages * kp.p, kp.p
}

// PromiseKV tells the devices that a scheduler has admitted rows whose
// histories will grow by positions more, page-rounded per row, and that the
// room for them is spoken for: a batched scratch of a width nobody reserved
// is not built into it (prepBatch), so a scratch built after admission does
// not take the pages the admitted rows were promised and send their history
// home. 0 withdraws the promise.
func (g *GPU) PromiseKV(positions int) {
	for _, d := range g.devs {
		d.mu.Lock()
		d.kvPromise = max(positions, 0)
		d.mu.Unlock()
	}
}

// promisedBytes is what the promise needs beyond the pool's free pages, at
// one page of every layer a page. Callers hold g.mu.
func (g *devTier) promisedBytes() uint64 {
	kp := g.kvp
	if g.kvPromise <= 0 || kp == nil || len(kp.layers) == 0 || kp.p <= 0 {
		return 0
	}
	var per uint64
	free := -1
	for _, l := range kp.layers {
		per += kp.pageBytes(l)
		if n := len(l.free) + len(l.fenced); free < 0 || n < free {
			free = n
		}
	}
	pages := (g.kvPromise + kp.p - 1) / kp.p
	return uint64(max(pages-max(free-1, 0), 0)) * per
}
