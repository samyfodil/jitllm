package session

// Value is one piece of shared state a front end keeps: a signal in the
// window, a field behind a lock in the terminal. Get and Set must be safe
// from any goroutine.
type Value[T any] interface {
	Get() T
	Set(T)
}

// The load list is every model being opened or switched to, in the order
// asked. These are its operations, over whatever Value the front end keeps
// it in.

// LoadingOf is the load of path in the list, if there is one.
func LoadingOf(loads Value[[]Loading], path string) (Loading, bool) {
	for _, l := range loads.Get() {
		if l.Path == path {
			return l, true
		}
	}
	return Loading{}, false
}

// QueueLoad adds a load to the end of the list.
func QueueLoad(loads Value[[]Loading], l Loading) {
	loads.Set(append(append([]Loading(nil), loads.Get()...), l))
}

// SetLoadStage says what the load of path is doing now.
func SetLoadStage(loads Value[[]Loading], path, stage string) {
	ls := append([]Loading(nil), loads.Get()...)
	for i := range ls {
		if ls[i].Path == path {
			ls[i].Stage = stage
			loads.Set(ls)
			return
		}
	}
}

// EndLoad removes the first load of path, done or failed.
func EndLoad(loads Value[[]Loading], path string) {
	ls := loads.Get()
	for i := range ls {
		if ls[i].Path == path {
			loads.Set(append(append([]Loading(nil), ls[:i]...), ls[i+1:]...))
			return
		}
	}
}
