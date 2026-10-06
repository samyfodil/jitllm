package tok

// Hunyuan's pre-tokenizer stages (pretok.OpDigitGroups, OpKanaHanRuns,
// OpHunyuanSplit), each a HuggingFace Split with behavior Isolated: what a
// pattern matches is a piece, and so is every stretch between two matches.

// splitDigitGroupsEach isolates \p{N} in groups of n, left to right: the
// regex \p{N}{1,n} scans greedily without overlap, so a run of five digits at
// n=3 is "123" then "45". Everything else is emitted in the stretches between.
func splitDigitGroupsEach(s string, n int, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		if !ucIsNumber(rs[i]) {
			j := i
			for j < len(rs) && !ucIsNumber(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		j := i
		for j < len(rs) && j-i < n && ucIsNumber(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// isKanaHan is the class [一-龥぀-ゟ゠-ヿ]: the
// ranges as written, not \p{Han}.
func isKanaHan(r rune) bool {
	return r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x309f || r >= 0x30a0 && r <= 0x30ff
}

// isHunyuanMark is the first alternative's class, the ASCII punctuation and
// symbols written out: [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~].
func isHunyuanMark(r rune) bool {
	return r >= '!' && r <= '/' || r >= ':' && r <= '@' || r >= '[' && r <= '`' || r >= '{' && r <= '~'
}

func isASCIILetter(r rune) bool { return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' }

// splitHunyuanEach is pretok.OpHunyuanSplit: at each position the first of
// the six alternatives that matches, as a backtracking regex takes them, and
// the characters no alternative matches kept as pieces of their own. spark
// is pretok.OpSparkSplit: no newline tail on the punctuation run, and a
// newline a piece of its own in place of \s*[\r\n]+.
func splitHunyuanEach(s string, spark bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	n := len(rs)
	fl := func(i int) uint16 { return ucFlags(rs[i]) }
	letter := func(i int) bool { return i < n && fl(i)&ucFlagLetter != 0 }
	lm := func(i int) bool { return i < n && fl(i)&(ucFlagLetter|ucFlagAccentMark) != 0 }
	ps := func(i int) bool { return i < n && fl(i)&(ucFlagPunctuation|ucFlagSymbol) != 0 }
	nl := func(i int) bool { return i < n && (rs[i] == '\r' || rs[i] == '\n') }
	space := func(i int) bool { return i < n && ucIsSpace(rs[i]) }
	run := func(i int, in func(int) bool) int {
		for in(i) {
			i++
		}
		return i
	}
	// The punctuation run's tail: [\r\n]*, or nothing for Spark.
	tail := nl
	if spark {
		tail = func(int) bool { return false }
	}
	at := func(i int) int {
		// [ASCII mark][A-Za-z]+
		if isHunyuanMark(rs[i]) && i+1 < n && isASCIILetter(rs[i+1]) {
			return run(i+1, func(j int) bool { return j < n && isASCIILetter(rs[j]) })
		}
		// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+: the prefix first, then without.
		if !nl(i) && !letter(i) && !ps(i) && lm(i+1) {
			return run(i+1, lm)
		}
		if lm(i) {
			return run(i, lm)
		}
		// ` ?[\p{P}\p{S}]+[\r\n]*`; Spark's has no tail.
		if rs[i] == ' ' && ps(i+1) {
			return run(run(i+1, ps), tail)
		}
		if ps(i) {
			return run(run(i, ps), tail)
		}
		// Spark's [\r\n]: one character, before any whitespace run.
		if spark && nl(i) {
			return i + 1
		}
		if !space(i) {
			return i
		}
		w := run(i, space)
		if spark {
			// \s+(?!\S) then \s+, with no newline alternative in between.
			if w == n || w-i < 2 {
				return w
			}
			return w - 1
		}
		// \s*[\r\n]+ backtracks the greedy \s* to the last newline in the run.
		for k := w - 1; k >= i; k-- {
			if nl(k) {
				return k + 1
			}
		}
		// \s+(?!\S): the whole run at the end of the text, all but its last
		// character before anything else; then \s+.
		if w == n || w-i >= 2 {
			if w == n {
				return w
			}
			return w - 1
		}
		return w
	}
	prev := 0
	for i := 0; i < n; {
		end := at(i)
		if end == i {
			i++
			continue
		}
		if i > prev {
			emit(sc.piece(s, prev, i))
		}
		emit(sc.piece(s, i, end))
		i, prev = end, end
	}
	if prev < n {
		emit(sc.piece(s, prev, n))
	}
}
