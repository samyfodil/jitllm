package jlm

// SetJournalEvery sets how many bytes a resumable write puts down between
// journal points, and returns the call that puts it back.
func SetJournalEvery(n uint64) func() {
	old := journalEvery
	journalEvery = n
	return func() { journalEvery = old }
}
