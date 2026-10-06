package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// Chat is one saved conversation. The file of them (config.Config.ChatsPath)
// is shared by both front ends, so a chat made in one shows in the other.
type Chat struct {
	// Model is the path of the model that last replied in it, so going back to
	// the chat can bring that model back.
	Model string `json:"model,omitempty"`
	Turns []Turn `json:"turns"`
	Draft string `json:"draft,omitempty"`
}

// ChatInfo is what the chat list shows of one chat.
type ChatInfo struct {
	Title string
	Model string
}

// ChatTitle is a chat named by its first message, cut to fit a sidebar.
func ChatTitle(turns []Turn) string {
	for _, t := range turns {
		if t.Role != RoleUser {
			continue
		}
		s := strings.Join(strings.Fields(t.Text), " ")
		if s == "" && len(t.Images) > 0 {
			return "A picture"
		}
		if r := []rune(s); len(r) > 48 {
			return strings.TrimSpace(string(r[:47])) + "…"
		}
		return s
	}
	return ""
}

// ChatList is every conversation and which one is current. The current
// chat's turns live with the front end while it is on screen, which hands
// them back with Park before anything that reads or reorders the list. It is
// not safe for concurrent use; a front end holds its own lock around it.
//
// Every method that changes which chat is current returns the index to show
// and leaves Cur at -1 or the old index; the front end shows that chat and
// sets Cur by calling Show.
type ChatList struct {
	Chats []Chat
	// Cur is the chat on screen, or -1 while none is.
	Cur int
}

// NewChatList is one empty chat, not yet shown.
func NewChatList() ChatList { return ChatList{Chats: []Chat{{}}, Cur: -1} }

// Park copies the transcript and draft into the current chat.
func (l *ChatList) Park(turns []Turn, draft string) {
	if l.Cur < 0 || l.Cur >= len(l.Chats) {
		return
	}
	c := &l.Chats[l.Cur]
	c.Turns = append([]Turn(nil), turns...)
	c.Draft = draft
}

// Show makes chat i current and returns a copy of it for the screen.
func (l *ChatList) Show(i int) Chat {
	l.Cur = i
	c := l.Chats[i]
	c.Turns = append([]Turn(nil), c.Turns...)
	return c
}

// Infos is the list as the sidebar shows it.
func (l *ChatList) Infos() []ChatInfo {
	out := make([]ChatInfo, len(l.Chats))
	for i, c := range l.Chats {
		out[i] = ChatInfo{Title: ChatTitle(c.Turns), Model: c.Model}
	}
	return out
}

// New starts an empty conversation for model, first in the list, and returns
// the chat to show. An empty chat is reused rather than stacked: a list of
// blank chats is no history. Park first.
func (l *ChatList) New(model string) int {
	for i, c := range l.Chats {
		if len(c.Turns) == 0 {
			return i
		}
	}
	l.Chats = append([]Chat{{Model: model}}, l.Chats...)
	l.Cur++
	return 0
}

// Select returns the chat to show for a press on row i, and false when there
// is nothing to do (no such row, or it is already current). Park first.
func (l *ChatList) Select(i int) (int, bool) {
	if i < 0 || i >= len(l.Chats) || i == l.Cur {
		return 0, false
	}
	return i, true
}

// Delete removes chat i and returns the chat to show, false when there is no
// such row. Removing the last one leaves an empty chat. Park first.
func (l *ChatList) Delete(i int) (int, bool) {
	if i < 0 || i >= len(l.Chats) {
		return 0, false
	}
	l.Chats = append(l.Chats[:i], l.Chats[i+1:]...)
	if len(l.Chats) == 0 {
		l.Chats = []Chat{{}}
	}
	next := min(max(l.Cur, 0), len(l.Chats)-1)
	if i < l.Cur {
		next = l.Cur - 1
	}
	l.Cur = -1
	return next, true
}

// At reads chat i. Park first for the current one to be up to date.
func (l *ChatList) At(i int) (Chat, bool) {
	if i < 0 || i >= len(l.Chats) {
		return Chat{}, false
	}
	return l.Chats[i], true
}

// SetModel records which model the current chat is talking to.
func (l *ChatList) SetModel(path string) {
	if l.Cur >= 0 && l.Cur < len(l.Chats) {
		l.Chats[l.Cur].Model = path
	}
}

// Encode is the list as the chats file holds it. Park first.
func (l *ChatList) Encode() ([]byte, error) { return json.MarshalIndent(l.Chats, "", " ") }

// WriteChats writes an encoded list to path, creating its folder. An empty
// path saves nothing.
func WriteChats(path string, b []byte) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}

// Save writes every chat to path. An empty path saves nothing. Park first.
func (l *ChatList) Save(path string) error {
	if path == "" {
		return nil
	}
	b, err := l.Encode()
	if err != nil {
		return err
	}
	return WriteChats(path, b)
}

// LoadChats is the list saved at path, nothing shown yet. A missing or
// unreadable file is one empty chat.
func LoadChats(path string) ChatList {
	var chats []Chat
	if b, err := os.ReadFile(path); err == nil {
		// A corrupt history starts empty rather than refusing to start.
		if json.Unmarshal(b, &chats) != nil {
			chats = nil
		}
	}
	if len(chats) == 0 {
		chats = []Chat{{}}
	}
	return ChatList{Chats: chats, Cur: -1}
}
