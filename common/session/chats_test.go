package session

import "testing"

// The list's semantics without a front end: an empty chat is reused, a chat
// keeps its turns while another is current, and removing the last one leaves
// an empty chat.
func TestChatListKeepsItsChats(t *testing.T) {
	l := NewChatList()
	l.Show(0)
	l.Park([]Turn{{Role: RoleUser, Text: "first   question"}}, "draft one")
	if got := l.Infos(); len(got) != 1 || got[0].Title != "first question" {
		t.Fatalf("infos %+v, want one named by its first message", got)
	}
	i := l.New("m.jlm")
	if i != 0 || len(l.Chats) != 2 || l.Chats[0].Model != "m.jlm" {
		t.Fatalf("new chat at %d of %d, model %q", i, len(l.Chats), l.Chats[0].Model)
	}
	l.Show(i)
	if j := l.New(""); j != 0 || len(l.Chats) != 2 {
		t.Fatalf("a second New stacked a blank chat: at %d of %d", j, len(l.Chats))
	}
	if _, ok := l.Select(0); ok {
		t.Fatal("selecting the current chat did something")
	}
	j, ok := l.Select(1)
	if !ok {
		t.Fatal("selecting the other chat did nothing")
	}
	if c := l.Show(j); c.Draft != "draft one" || c.Turns[0].Text != "first   question" {
		t.Fatalf("the first chat came back as %+v", c)
	}
	next, _ := l.Delete(1)
	if len(l.Chats) != 1 || next != 0 {
		t.Fatalf("delete left %d chats, next %d", len(l.Chats), next)
	}
	l.Show(next)
	l.Delete(0)
	if len(l.Chats) != 1 || len(l.Chats[0].Turns) != 0 {
		t.Fatalf("deleting the last chat left %+v, want one empty", l.Chats)
	}
}

func TestChatListSurvivesARestart(t *testing.T) {
	p := t.TempDir() + "/chats.json"
	l := NewChatList()
	l.Show(0)
	l.Park([]Turn{{Role: RoleUser, Text: "kept"}}, "")
	if err := l.Save(p); err != nil {
		t.Fatal(err)
	}
	r := LoadChats(p)
	if len(r.Chats) != 1 || r.Chats[0].Turns[0].Text != "kept" || r.Cur != -1 {
		t.Fatalf("restored %+v", r)
	}
	if e := LoadChats(p + ".missing"); len(e.Chats) != 1 {
		t.Fatalf("a missing file gave %d chats, want one empty", len(e.Chats))
	}
}

func TestChatTitleNamesAPicture(t *testing.T) {
	if got := ChatTitle([]Turn{{Role: RoleUser, Images: []string{"a.png"}}}); got != "A picture" {
		t.Fatalf("title %q", got)
	}
}
