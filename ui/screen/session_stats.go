package screen

import (
	"github.com/gogpu/ui/state"

	"github.com/samyfodil/jitllm/common/session"
	"github.com/samyfodil/jitllm/ui/app"
)

// What the last generation cost, as one line under the composer, mirroring
// `jitllm run`. A rate alone cannot tell a fully resident run from one a page
// short; the paging clause can. The line itself is [session.SessionFooter],
// which the terminal prints too.

// sessionFooter is the line under the composer: what the last reply cost and
// how much of the context window the conversation holds. Empty until there is
// a reply to describe.
func sessionFooter(sh *app.Shell) state.ReadonlySignal[string] {
	st := sh.Store
	return state.NewComputed(func() string {
		if st.Turns.Get() == 0 || !st.Loaded.Get() {
			return ""
		}
		return session.SessionFooter(st.PromptTokS.Get(), st.DecodeTokS.Get(),
			uint64(st.BytesPerTok.Get()), st.MemWall.Get(), st.Alloc.Get(), st.Pager.Get())
	}, st.Turns.AsReadonly(), st.Loaded.AsReadonly(), st.PromptTokS.AsReadonly(), st.DecodeTokS.AsReadonly(),
		st.BytesPerTok.AsReadonly(), st.MemWall.AsReadonly(), st.Alloc.AsReadonly(), st.Pager.AsReadonly())
}

// transcriptText is the whole conversation as plain text.
func transcriptText(st *app.Store) string {
	turns := make([]app.Turn, st.Turns.Get())
	for i := range turns {
		turns[i] = st.Turn(i)
	}
	return session.TranscriptText(turns)
}
