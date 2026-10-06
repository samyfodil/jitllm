package jinja

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// FromJSON decodes JSON into a Value, keeping every object's keys in the order
// the document wrote them.
//
// FromGoValue sorts a map's keys; Python's json.loads keeps insertion order,
// and a tool schema rendered in a different order is a different prompt.
func FromJSON(data []byte) (Value, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := decodeJSONValue(d)
	if err != nil {
		return Undefined(), err
	}
	if _, err := d.Token(); err != io.EOF {
		return Undefined(), fmt.Errorf("jinja: trailing data after the JSON value")
	}
	return v, nil
}

func decodeJSONValue(d *json.Decoder) (Value, error) {
	t, err := d.Token()
	if err != nil {
		return Undefined(), err
	}
	switch x := t.(type) {
	case json.Delim:
		switch x {
		case '{':
			out := NewDict()
			dict := out.AsDict()
			for d.More() {
				kt, err := d.Token()
				if err != nil {
					return Undefined(), err
				}
				k, ok := kt.(string)
				if !ok {
					return Undefined(), fmt.Errorf("jinja: object key %v is not a string", kt)
				}
				v, err := decodeJSONValue(d)
				if err != nil {
					return Undefined(), err
				}
				dict.Set(k, v)
			}
			_, err := d.Token() // '}'
			return out, err
		case '[':
			var items []Value
			for d.More() {
				v, err := decodeJSONValue(d)
				if err != nil {
					return Undefined(), err
				}
				items = append(items, v)
			}
			_, err := d.Token() // ']'
			return NewList(items), err
		}
		return Undefined(), fmt.Errorf("jinja: unexpected %v", x)
	default:
		return FromGoValue(x), nil
	}
}
