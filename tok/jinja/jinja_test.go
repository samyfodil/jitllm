package jinja_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok/jinja"
)

// =============================================================================
// Basic engine tests (no model dependency)
// =============================================================================

func TestBasicRender(t *testing.T) {
	tmpl, err := jinja.Compile("Hello {{ name }}!")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"name": "World",
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "Hello World!" {
		t.Errorf("expected 'Hello World!', got %q", result)
	}
}

func TestForLoop(t *testing.T) {
	source := `{% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"items": []any{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "a, b, c" {
		t.Errorf("expected 'a, b, c', got %q", result)
	}
}

func TestIfElse(t *testing.T) {
	source := `{% if x %}yes{% else %}no{% endif %}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{"x": true})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if result != "yes" {
		t.Errorf("expected 'yes', got %q", result)
	}

	result, err = tmpl.Render(map[string]any{"x": false})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if result != "no" {
		t.Errorf("expected 'no', got %q", result)
	}
}

func TestTojsonFilter(t *testing.T) {
	source := `{{ data | tojson }}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"data": map[string]any{"name": "test", "value": 42},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if !strings.Contains(result, `"name"`) || !strings.Contains(result, `"test"`) {
		t.Errorf("tojson output unexpected: %q", result)
	}
}

func TestFromJSONFilter(t *testing.T) {
	source := `{%- set data = data | from_json -%}{{ data.name }}:{{ data.value }}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"data": `{"name":"test","value":42}`,
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "test:42" {
		t.Errorf("got %q, want %q", result, "test:42")
	}
}

func TestUndefinedCallableErrorNamesExpression(t *testing.T) {
	tmpl, err := jinja.Compile(`{{ args.items() }}`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	_, err = tmpl.Render(nil)
	if err == nil {
		t.Fatal("render: expected error")
	}

	const want = "args is undefined, so items cannot be read from it"
	if err.Error() != want {
		t.Errorf("error: got %q, want %q", err, want)
	}
}

func TestNamespace(t *testing.T) {
	source := `{%- set ns = namespace(found=false) -%}
{%- for item in items -%}
{%- if item == "target" -%}
{%- set ns.found = true -%}
{%- endif -%}
{%- endfor -%}
{{ ns.found }}`

	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"items": []any{"a", "target", "b"},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	result = strings.TrimSpace(result)
	if result != "True" {
		t.Errorf("expected 'True', got %q", result)
	}
}

func TestStringMethods(t *testing.T) {
	tests := []struct {
		name   string
		source string
		want   string
	}{
		{"strip", `{{ "  hello  ".strip() }}`, "hello"},
		{"split", `{{ "a,b,c".split(",") | join("-") }}`, "a-b-c"},
		{"startswith", `{{ "hello".startswith("hel") }}`, "True"},
		{"endswith", `{{ "hello".endswith("llo") }}`, "True"},
		{"upper", `{{ "hello".upper() }}`, "HELLO"},
		{"lower", `{{ "HELLO".lower() }}`, "hello"},
		{"replace", `{{ "hello world".replace("world", "jinja") }}`, "hello jinja"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := jinja.Compile(tt.source)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			result, err := tmpl.Render(nil)
			if err != nil {
				t.Fatalf("render: %v", err)
			}
			if result != tt.want {
				t.Errorf("expected %q, got %q", tt.want, result)
			}
		})
	}
}

func TestDictMethods(t *testing.T) {
	source := `{%- for key, value in data.items() -%}{{ key }}={{ value }} {% endfor -%}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"data": map[string]any{"a": 1, "b": 2},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	// Keys are sorted for deterministic output.
	if !strings.Contains(result, "a=1") || !strings.Contains(result, "b=2") {
		t.Errorf("unexpected dict items output: %q", result)
	}
}

func TestFromJSONWithDictItems(t *testing.T) {
	source := `{%- set func = tool['function'] -%}
{%- set args = func['arguments'] -%}
{%- if args is string -%}
  {%- set args = args | from_json -%}
{%- endif -%}
{%- for key, val in args.items() -%}{{ key }}={{ val }}{%- endfor -%}`

	tests := []struct {
		name      string
		arguments any
		want      string
	}{
		{
			name:      "json string",
			arguments: `{"location":"New York City, NY"}`,
			want:      "location=New York City, NY",
		},
		{
			name:      "map",
			arguments: map[string]any{"location": "New York City, NY"},
			want:      "location=New York City, NY",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := jinja.Compile(source)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}

			result, err := tmpl.Render(map[string]any{
				"tool": map[string]any{
					"function": map[string]any{
						"name":      "get_weather",
						"arguments": tt.arguments,
					},
				},
			})
			if err != nil {
				t.Fatalf("render: %v", err)
			}

			if result != tt.want {
				t.Errorf("result: got %q, want %q", result, tt.want)
			}
		})
	}
}

func TestNestedRangePreservesOuterLoop(t *testing.T) {
	source := `{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
{%- set ep = namespace(idx=(loop.index0 - 1), done=false) -%}
{%- for _i in range(loop.index0) -%}
{%- if not ep.done and ep.idx >= 0 -%}
{%- set ep.done = true -%}
{%- endif -%}
{%- endfor -%}
{%- for tool in message['tool_calls'] -%}
{%- set args = tool['function']['arguments'] -%}
{%- if args is string -%}{%- set args = args | from_json -%}{%- endif -%}
{%- for key, val in args.items() -%}{{ key }}={{ val }}{%- endfor -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}`

	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "weather"},
			{
				"role": "assistant",
				"tool_calls": []map[string]any{
					{
						"function": map[string]any{
							"name":      "get_weather",
							"arguments": map[string]any{"location": "New York City, NY"},
						},
					},
				},
			},
			{"role": "tool", "content": `{"temperature":"72°F"}`},
		},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	const want = "location=New York City, NY"
	if result != want {
		t.Errorf("result: got %q, want %q", result, want)
	}
}

func TestInlineIf(t *testing.T) {
	source := `{{ "yes" if x else "no" }}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{"x": true})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if result != "yes" {
		t.Errorf("expected 'yes', got %q", result)
	}
}

func TestIsTests(t *testing.T) {
	tests := []struct {
		name   string
		source string
		want   string
	}{
		{"is defined", `{{ "yes" if x is defined else "no" }}`, "yes"},
		{"is not defined", `{{ "yes" if y is not defined else "no" }}`, "yes"},
		{"is string", `{{ "yes" if x is string else "no" }}`, "yes"},
		{"is none", `{{ "yes" if n is none else "no" }}`, "yes"},
		{"is true", `{{ "yes" if t is true else "no" }}`, "yes"},
		{"is false", `{{ "yes" if f is false else "no" }}`, "yes"},
	}

	data := map[string]any{
		"x": "hello",
		"n": nil,
		"t": true,
		"f": false,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := jinja.Compile(tt.source)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			result, err := tmpl.Render(data)
			if err != nil {
				t.Fatalf("render: %v", err)
			}
			if result != tt.want {
				t.Errorf("expected %q, got %q", tt.want, result)
			}
		})
	}
}

func TestWhitespaceControl(t *testing.T) {
	source := "A\n  {%- if true %}\nB\n  {%- endif %}\nC"
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(nil)
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if !strings.Contains(result, "A") && !strings.Contains(result, "B") && !strings.Contains(result, "C") {
		t.Errorf("whitespace control produced unexpected output: %q", result)
	}
}

func TestSlicing(t *testing.T) {
	source := `{{ items[::-1] | join(",") }}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"items": []any{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "c,b,a" {
		t.Errorf("expected 'c,b,a', got %q", result)
	}
}

func TestStringConcat(t *testing.T) {
	source := `{{ "Hello" ~ " " ~ "World" }}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(nil)
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "Hello World" {
		t.Errorf("expected 'Hello World', got %q", result)
	}
}

func TestLoopVariables(t *testing.T) {
	source := `{%- for item in items -%}{{ loop.index0 }}:{{ item }} {% endfor -%}`
	tmpl, err := jinja.Compile(source)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	result, err := tmpl.Render(map[string]any{
		"items": []any{"x", "y", "z"},
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	if result != "0:x 1:y 2:z " {
		t.Errorf("expected '0:x 1:y 2:z ', got %q", result)
	}
}

// TestDictGet verifies dict.get(), which every tool-calling template reads
// (message.get('tool_calls') and the like).
func TestDictGet(t *testing.T) {
	tests := []struct {
		name   string
		source string
		data   map[string]any
		want   string
	}{
		{
			name:   "get existing key",
			source: `{{ d.get('name') }}`,
			data:   map[string]any{"d": map[string]any{"name": "Alice"}},
			want:   "Alice",
		},
		{
			name:   "get missing key returns none",
			source: `{{ d.get('missing') }}`,
			data:   map[string]any{"d": map[string]any{"name": "Alice"}},
			want:   "",
		},
		{
			name:   "get missing key with default",
			source: `{{ d.get('missing', 'fallback') }}`,
			data:   map[string]any{"d": map[string]any{"name": "Alice"}},
			want:   "fallback",
		},
		{
			name:   "get in or chain",
			source: `{{ d.get('a') or d.get('b') or 'none' }}`,
			data:   map[string]any{"d": map[string]any{"b": "found_b"}},
			want:   "found_b",
		},
		{
			name:   "get in if condition",
			source: `{% if d.get('tool_calls') %}yes{% else %}no{% endif %}`,
			data:   map[string]any{"d": map[string]any{"name": "Alice"}},
			want:   "no",
		},
		{
			name:   "get truthy check",
			source: `{% if d.get('tool_calls') %}yes{% else %}no{% endif %}`,
			data:   map[string]any{"d": map[string]any{"tool_calls": []any{"call1"}}},
			want:   "yes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := jinja.Compile(tt.source)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			result, err := tmpl.Render(tt.data)
			if err != nil {
				t.Fatalf("render: %v", err)
			}
			if result != tt.want {
				t.Errorf("expected %q, got %q", tt.want, result)
			}
		})
	}
}
