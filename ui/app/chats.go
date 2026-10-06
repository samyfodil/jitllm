package app

import "github.com/samyfodil/jitllm/common/session"

// The chat list is [session.ChatList], which the terminal drives too; the
// Store keeps the current chat's turns in its transcript, where every turn
// method already reads them, and mirrors the list into its signals.

// Chat is one conversation; see [session.Chat].
type Chat = session.Chat

// ChatInfo is what the chat list shows of one chat.
type ChatInfo = session.ChatInfo

// park copies the transcript and draft into the current chat. Lock held.
func (s *Store) park() { s.chats.Park(s.turns, s.Draft.Get()) }

// publishChats mirrors the chat list into its signals. Lock not held.
func (s *Store) publishChats() {
	s.mu.Lock()
	s.park()
	out := s.chats.Infos()
	cur := s.chats.Cur
	s.mu.Unlock()
	s.Chats.Set(out)
	s.ChatSel.Set(cur)
}

// show makes chat i the transcript. Lock not held.
func (s *Store) show(i int) {
	s.mu.Lock()
	c := s.chats.Show(i)
	s.turns = c.Turns
	s.thinkOpen = nil
	s.mu.Unlock()
	s.Draft.Set(c.Draft)
	s.Attach.Set(nil)
	s.Stream.Set("")
	s.StreamThink.Set("")
	s.Turns.Set(len(c.Turns))
	s.Shown.Set(s.Shown.Get() + 1)
	s.publishChats()
}

// NewChat starts an empty conversation, first in the list. An empty chat is
// reused rather than stacked: a list of blank chats is no history. UI
// goroutine only, like every chat method.
func (s *Store) NewChat() {
	s.mu.Lock()
	s.park()
	i := s.chats.New(s.ModelPath.Get())
	s.mu.Unlock()
	s.show(i)
}

// SelectChat makes chat i the conversation on screen.
func (s *Store) SelectChat(i int) {
	s.mu.Lock()
	next, ok := s.chats.Select(i)
	if ok {
		s.park()
	}
	s.mu.Unlock()
	if ok {
		s.show(next)
	}
}

// DeleteChat removes chat i. Removing the last one leaves an empty chat.
func (s *Store) DeleteChat(i int) {
	s.mu.Lock()
	s.park()
	next, ok := s.chats.Delete(i)
	s.mu.Unlock()
	if ok {
		s.show(next)
	}
}

// ChatAt reads chat i as it is now, the current one included.
func (s *Store) ChatAt(i int) (Chat, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.park()
	return s.chats.At(i)
}

// SetChatModel records which model the current chat is talking to.
func (s *Store) SetChatModel(path string) {
	s.mu.Lock()
	s.chats.SetModel(path)
	s.mu.Unlock()
	s.publishChats()
}

// SaveChats writes every chat to path. An empty path saves nothing.
func (s *Store) SaveChats(path string) error {
	if path == "" {
		return nil
	}
	s.mu.Lock()
	s.park()
	b, err := s.chats.Encode()
	s.mu.Unlock()
	if err != nil {
		return err
	}
	return session.WriteChats(path, b)
}

// LoadChats replaces the chats with the ones saved at path, the first shown.
// A missing or unreadable file leaves one empty chat.
func (s *Store) LoadChats(path string) {
	l := session.LoadChats(path)
	s.mu.Lock()
	s.chats = l
	s.mu.Unlock()
	s.show(0)
}
