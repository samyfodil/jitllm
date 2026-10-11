package srcgate

import "testing"

// TestRecipesRun is recipes_test.go's on every other host. The recipes are
// bash, and their servers are stopped as a process group; Windows has neither.
func TestRecipesRun(t *testing.T) {
	t.Skip("docs/recipes are bash scripts run as process groups; they run on Linux and macOS")
}
