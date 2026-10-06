// Package ffi binds C functions without cgo, on top of goffi.
//
// goffi is libffi-shaped: prepare a CallInterface, then call with an array of
// argument pointers and a return slot. This wraps it in typed Go functions,
// such as func(*CUdevptr, uint64) CUresult, so the signature is checked by the
// Go compiler rather than a hand-written descriptor list.
//
// goffi replaced purego (about three times cheaper per call, and well under
// 1% of a token, so it is not a throughput change). The two cannot coexist:
// each vendors its own internal/fakecgo, and linking both gives "duplicated
// definition of symbol _cgo_init".
package ffi

import (
	"fmt"
	"reflect"
	"unsafe"

	gffi "github.com/go-webgpu/goffi/ffi"
	"github.com/go-webgpu/goffi/types"
)

// None is the return type of a C function that returns void.
type None struct{}

// Lib is an open shared library.
type Lib struct{ h unsafe.Pointer }

// Open loads the first of names that exists.
func Open(names ...string) (*Lib, error) {
	var last error
	for _, n := range names {
		h, err := gffi.LoadLibrary(n)
		if err == nil {
			return &Lib{h: h}, nil
		}
		last = err
	}
	return nil, fmt.Errorf("ffi: none of %v could be loaded: %w", names, last)
}

func (l *Lib) Close() error { return gffi.FreeLibrary(l.h) }

// Has reports whether the library exports name.
//
// Binding panics on a missing symbol (a wrong signature or a typo is a startup
// bug), so an optional entry point, such as CUDA's graph API on an old driver,
// is checked with Has first.
func (l *Lib) Has(name string) bool {
	_, err := gffi.GetSymbol(l.h, name)
	return err == nil
}

// noescape hides a pointer from escape analysis.
// It keeps the arguments, return slot and pointer array from escaping (three
// allocations per call without it), because the compiler cannot see that
// CallFunction does not retain them. It is sound only because every bound
// entry point is synchronous and never keeps these pointers; the async forms
// that retain a host pointer (cuMemcpyHtoDAsync, cuLaunchHostFunc) are not
// bound.
//
// The array holding these pointers must stay in the frame too: see
// prepared.call and TestCallArgumentsStayOnTheStack before adding a wrapper or
// changing call's signature.
//
// `go vet` flags the uintptr round trip as possible misuse of unsafe.Pointer;
// that pattern is exactly what makes it work, as in the standard library.
//
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

type prepared struct {
	cif types.CallInterface
	fn  unsafe.Pointer
}

// call hands the prepared signature, the return slot and the argument array to
// goffi. av is the array base and n its length, rather than a slice.
//
// A slice parameter escapes, which moved each wrapper's `var ps [N]` array to
// the heap while it held addresses of the caller's frame locals. A heap object
// holding a stack pointer is illegal in Go and fails two ways: the collector
// can scan it after the frame is gone ("found bad pointer in Go heap"), and a
// stack growth before goffi reads it leaves the driver reading the old stack,
// silently. Re-forming the slice through noescape keeps ps in the frame, where
// stack copying adjusts it with the values it points at.
//
// It is sound because goffi copies each argument out of avalue[i] before the
// trampoline and does not retain the slice (internal/arch/amd64/call_unix.go).
//
//go:nosplit
func (p *prepared) call(rv unsafe.Pointer, av *unsafe.Pointer, n int) {
	gffi.CallFunction(&p.cif, p.fn, rv,
		unsafe.Slice((*unsafe.Pointer)(noescape(unsafe.Pointer(av))), n))
}

// desc maps a Go type to goffi's descriptor.
//
// Only what the GPU runtimes use. An unknown type panics at bind time, once at
// startup, rather than making a silently wrong call.
func desc(t reflect.Type) *types.TypeDescriptor {
	if t == nil {
		return types.VoidTypeDescriptor
	}
	switch t.Kind() {
	case reflect.Struct:
		if t.NumField() == 0 {
			return types.VoidTypeDescriptor
		}
		// Structs by value are needed: Metal passes two MTLSize (24 bytes,
		// where arm64 switches to a pointer to a copy). goffi applies the
		// ABI rule from the member layout.
		m := make([]*types.TypeDescriptor, t.NumField())
		for i := range m {
			m[i] = desc(t.Field(i).Type)
		}
		return &types.TypeDescriptor{Kind: types.StructType, Members: m}
	// The 8-bit cases exist for Objective-C's BOOL: arm64 returns it in the low
	// bits of x0 with the rest unspecified, so a wider binding reads garbage.
	// goffi writes back exactly the descriptor's size.
	case reflect.Uint8:
		return types.UInt8TypeDescriptor
	case reflect.Int8:
		return types.SInt8TypeDescriptor
	case reflect.Int32:
		return types.SInt32TypeDescriptor
	case reflect.Uint32:
		return types.UInt32TypeDescriptor
	case reflect.Int64:
		return types.SInt64TypeDescriptor
	case reflect.Uint64, reflect.Uintptr:
		return types.UInt64TypeDescriptor
	case reflect.Float32:
		return types.FloatTypeDescriptor
	case reflect.Float64:
		return types.DoubleTypeDescriptor
	case reflect.Ptr, reflect.UnsafePointer:
		return types.PointerTypeDescriptor
	}
	panic("ffi: no descriptor for " + t.String())
}

func descOf[T any]() *types.TypeDescriptor {
	var v T
	return desc(reflect.TypeOf(v))
}

func prep(l *Lib, name string, ret *types.TypeDescriptor, args ...*types.TypeDescriptor) *prepared {
	sym, err := gffi.GetSymbol(l.h, name)
	if err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	p := &prepared{fn: sym}
	if err := gffi.PrepareCallInterface(&p.cif, types.DefaultConvention(), ret, args); err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	return p
}

// Fn0 binds a C function of 0 argument(s).
func Fn0[R any](l *Lib, name string) func() R {
	p := prep(l, name, descOf[R]())
	return func() R {
		var r R
		var ps [1]unsafe.Pointer
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 0)
		return r
	}
}

// Fn1 binds a C function of 1 argument(s).
func Fn1[R any, A0 any](l *Lib, name string) func(A0) R {
	p := prep(l, name, descOf[R](), descOf[A0]())
	return func(a0 A0) R {
		var r R
		var ps [1]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 1)
		return r
	}
}

// Fn2 binds a C function of 2 argument(s).
func Fn2[R any, A0 any, A1 any](l *Lib, name string) func(A0, A1) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1]())
	return func(a0 A0, a1 A1) R {
		var r R
		var ps [2]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 2)
		return r
	}
}

// Fn3 binds a C function of 3 argument(s).
func Fn3[R any, A0 any, A1 any, A2 any](l *Lib, name string) func(A0, A1, A2) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2]())
	return func(a0 A0, a1 A1, a2 A2) R {
		var r R
		var ps [3]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 3)
		return r
	}
}

// Fn4 binds a C function of 4 argument(s).
func Fn4[R any, A0 any, A1 any, A2 any, A3 any](l *Lib, name string) func(A0, A1, A2, A3) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3) R {
		var r R
		var ps [4]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 4)
		return r
	}
}

// Fn5 binds a C function of 5 argument(s).
func Fn5[R any, A0 any, A1 any, A2 any, A3 any, A4 any](l *Lib, name string) func(A0, A1, A2, A3, A4) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4) R {
		var r R
		var ps [5]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 5)
		return r
	}
}

// Fn6 binds a C function of 6 argument(s).
func Fn6[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5) R {
		var r R
		var ps [6]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 6)
		return r
	}
}

// Fn7 binds a C function of 7 argument(s).
func Fn7[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6) R {
		var r R
		var ps [7]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		ps[6] = noescape(unsafe.Pointer(&a6))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 7)
		return r
	}
}

// Fn8 binds a C function of 8 argument(s).
func Fn8[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7) R {
		var r R
		var ps [8]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		ps[6] = noescape(unsafe.Pointer(&a6))
		ps[7] = noescape(unsafe.Pointer(&a7))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 8)
		return r
	}
}

// Fn9 binds a C function of 9 argument(s).
func Fn9[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8) R {
		var r R
		var ps [9]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		ps[6] = noescape(unsafe.Pointer(&a6))
		ps[7] = noescape(unsafe.Pointer(&a7))
		ps[8] = noescape(unsafe.Pointer(&a8))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 9)
		return r
	}
}

// Fn10 binds a C function of 10 argument(s).
func Fn10[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8](), descOf[A9]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8, a9 A9) R {
		var r R
		var ps [10]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		ps[6] = noescape(unsafe.Pointer(&a6))
		ps[7] = noescape(unsafe.Pointer(&a7))
		ps[8] = noescape(unsafe.Pointer(&a8))
		ps[9] = noescape(unsafe.Pointer(&a9))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 10)
		return r
	}
}

// Fn11 binds a C function of 11 argument(s).
func Fn11[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any, A10 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) R {
	p := prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8](), descOf[A9](), descOf[A10]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8, a9 A9, a10 A10) R {
		var r R
		var ps [11]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		ps[4] = noescape(unsafe.Pointer(&a4))
		ps[5] = noescape(unsafe.Pointer(&a5))
		ps[6] = noescape(unsafe.Pointer(&a6))
		ps[7] = noescape(unsafe.Pointer(&a7))
		ps[8] = noescape(unsafe.Pointer(&a8))
		ps[9] = noescape(unsafe.Pointer(&a9))
		ps[10] = noescape(unsafe.Pointer(&a10))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 11)
		return r
	}
}

// ---------------------------------------------------------------------------
// Entry points that are not exported by name.

// Fn3At is Fn4At for a 3-argument entry point, such as
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
func Fn3At[R any, A0 any, A1 any, A2 any](fn unsafe.Pointer, name string) func(A0, A1, A2) R {
	p := prepAt(fn, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2]())
	return func(a0 A0, a1 A1, a2 A2) R {
		var r R
		var ps [3]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 3)
		return r
	}
}

// Fn4At binds a C function of 4 arguments held as an address rather than a
// symbol name. The Vulkan loader exports core entry points but not extension
// ones, which are reachable only through its dispatch table
// (vkGetDeviceProcAddr). The caller vouches for the pointer; a nil fn panics
// here rather than jumping to zero later.
func Fn4At[R any, A0 any, A1 any, A2 any, A3 any](fn unsafe.Pointer, name string) func(A0, A1, A2, A3) R {
	p := prepAt(fn, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3]())
	return func(a0 A0, a1 A1, a2 A2, a3 A3) R {
		var r R
		var ps [4]unsafe.Pointer
		ps[0] = noescape(unsafe.Pointer(&a0))
		ps[1] = noescape(unsafe.Pointer(&a1))
		ps[2] = noescape(unsafe.Pointer(&a2))
		ps[3] = noescape(unsafe.Pointer(&a3))
		p.call(noescape(unsafe.Pointer(&r)), &ps[0], 4)
		return r
	}
}

// prepAt is prep for an entry point already resolved to an address. name is
// for the panic message only.
func prepAt(fn unsafe.Pointer, name string, ret *types.TypeDescriptor, args ...*types.TypeDescriptor) *prepared {
	if fn == nil {
		panic("ffi: " + name + ": nil entry point")
	}
	p := &prepared{fn: fn}
	if err := gffi.PrepareCallInterface(&p.cif, types.DefaultConvention(), ret, args); err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	return p
}
