package app

import "testing"

// A chat is named by its first message, keeps its turns while another is on
// screen, and gives them back when selected.
func TestChatsKeepTheirTurns(t *testing.T) {
	s := NewStore()
	s.AppendTurn(Turn{Role: RoleUser, Text: "first question"})
	s.AppendTurn(Turn{Role: RoleAssistant, Text: "first answer"})
	if got := s.Chats.Get(); len(got) != 1 || got[0].Title != "first question" {
		t.Fatalf("chats %+v, want one named by its first message", got)
	}
	s.NewChat()
	if s.Turns.Get() != 0 || len(s.Chats.Get()) != 2 || s.ChatSel.Get() != 0 {
		t.Fatalf("new chat: %d turns, %d chats, at %d", s.Turns.Get(), len(s.Chats.Get()), s.ChatSel.Get())
	}
	s.NewChat() // an empty chat is reused, not stacked
	if len(s.Chats.Get()) != 2 {
		t.Fatalf("%d chats after a second New chat, want 2", len(s.Chats.Get()))
	}
	s.Draft.Set("half written")
	s.AppendTurn(Turn{Role: RoleUser, Text: "second question"})
	s.SelectChat(1)
	if s.Turns.Get() != 2 || s.Turn(1).Text != "first answer" || s.Draft.Get() != "" {
		t.Fatalf("the first chat came back as %d turns, %q, draft %q", s.Turns.Get(), s.Turn(1).Text, s.Draft.Get())
	}
	s.SelectChat(0)
	if s.Turn(0).Text != "second question" || s.Draft.Get() != "half written" {
		t.Fatalf("the second chat came back as %q, draft %q", s.Turn(0).Text, s.Draft.Get())
	}
	s.DeleteChat(0)
	if got := s.Chats.Get(); len(got) != 1 || got[0].Title != "first question" || s.Turn(0).Text != "first question" {
		t.Fatalf("after deleting: %+v showing %q", got, s.Turn(0).Text)
	}
	s.DeleteChat(0)
	if len(s.Chats.Get()) != 1 || s.Turns.Get() != 0 {
		t.Fatalf("deleting the last chat left %d chats and %d turns, want one empty", len(s.Chats.Get()), s.Turns.Get())
	}
}

func TestChatsSurviveARestart(t *testing.T) {
	p := t.TempDir() + "/chats.json"
	s := NewStore()
	s.AppendTurn(Turn{Role: RoleUser, Text: "kept"})
	if err := s.SaveChats(p); err != nil {
		t.Fatal(err)
	}
	r := NewStore()
	r.LoadChats(p)
	if r.Turns.Get() != 1 || r.Turn(0).Text != "kept" {
		t.Fatalf("restored %d turns, %q", r.Turns.Get(), r.Turn(0).Text)
	}
}
