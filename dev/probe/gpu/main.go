package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"
	"unsafe"
)

// ---- runtime PTX codegen: constants are baked into the text, not passed as params ----

func genSaxpyPTX(n int, alpha float32) string {
	// alpha baked as a PTX hex-float immediate; n baked as an immediate compare.
	bits := math.Float32bits(alpha)
	return fmt.Sprintf(`.version 7.8
.target sm_86
.address_size 64
.visible .entry k(.param .u64 px, .param .u64 py)
{
 .reg .pred %%p<2>; .reg .f32 %%f<4>; .reg .b32 %%r<5>; .reg .b64 %%rd<8>;
 ld.param.u64 %%rd1,[px];
 ld.param.u64 %%rd2,[py];
 cvta.to.global.u64 %%rd3,%%rd1;
 cvta.to.global.u64 %%rd4,%%rd2;
 mov.u32 %%r1,%%ctaid.x;
 mov.u32 %%r2,%%ntid.x;
 mov.u32 %%r3,%%tid.x;
 mad.lo.s32 %%r4,%%r1,%%r2,%%r3;
 setp.ge.s32 %%p1,%%r4,%d;
 @%%p1 bra D;
 mul.wide.s32 %%rd5,%%r4,4;
 add.s64 %%rd6,%%rd3,%%rd5;
 add.s64 %%rd7,%%rd4,%%rd5;
 ld.global.f32 %%f1,[%%rd6];
 ld.global.f32 %%f2,[%%rd7];
 fma.rn.f32 %%f3,%%f1,0f%08X,%%f2;
 st.global.f32 [%%rd7],%%f3;
D: ret;
}
`, n, bits)
}

const emptyPTX = `.version 7.8
.target sm_86
.address_size 64
.visible .entry k(.param .u64 p) { ret; }
`

func cstr(s string) unsafe.Pointer {
	b := append([]byte(s), 0)
	return unsafe.Pointer(&b[0])
}

func med(v []float64) float64  { sort.Float64s(v); return v[len(v)/2] }
func minf(v []float64) float64 { sort.Float64s(v); return v[0] }

func main() {
	runtime.LockOSThread() // CUDA contexts are per-OS-thread; a Go goroutine that migrates loses it.
	if err := loadCUDA(); err != nil {
		fmt.Println("DLOPEN FAIL:", err)
		os.Exit(1)
	}
	fmt.Println("== purego dlopen libcuda.so.1: OK (no cgo) ==")
	ck(cuInit(0), "cuInit")
	var dv int32
	ck(cuDriverGetVersion(&dv), "cuDriverGetVersion")
	var dev CUdevice
	ck(cuDeviceGet(&dev, 0), "cuDeviceGet")
	var ctx CUctx
	ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate")
	var free, total uint64
	ck(cuMemGetInfo(&free, &total), "cuMemGetInfo")
	fmt.Printf("driver=%d  vram free=%.0f MiB / total=%.0f MiB\n", dv, float64(free)/1048576, float64(total)/1048576)

	// ---------- 1. correctness: generated PTX, load, launch, copy back ----------
	const N = 1 << 20
	ptx := genSaxpyPTX(N, 3.5)
	var mod CUmodule
	t0 := time.Now()
	r := cuModuleLoadData(&mod, cstr(ptx))
	firstLoad := time.Since(t0)
	if r != 0 {
		fmt.Println("cuModuleLoadData FAILED:", errStr(r))
		os.Exit(1)
	}
	var fn CUfunc
	ck(cuModuleGetFunction(&fn, mod, "k"), "cuModuleGetFunction")
	var dx, dy CUdevptr
	ck(cuMemAlloc(&dx, N*4), "cuMemAlloc x")
	ck(cuMemAlloc(&dy, N*4), "cuMemAlloc y")
	hx := make([]float32, N)
	hy := make([]float32, N)
	for i := range hx {
		hx[i] = float32(i % 97)
		hy[i] = 1.0
	}
	ck(cuMemcpyHtoD(dx, unsafe.Pointer(&hx[0]), N*4), "HtoD x")
	ck(cuMemcpyHtoD(dy, unsafe.Pointer(&hy[0]), N*4), "HtoD y")
	args := []unsafe.Pointer{unsafe.Pointer(&dx), unsafe.Pointer(&dy)}
	ck(cuLaunchKernel(fn, N/256, 1, 1, 256, 1, 1, 0, 0, unsafe.Pointer(&args[0]), nil), "launch")
	ck(cuCtxSynchronize(), "sync")
	out := make([]float32, N)
	ck(cuMemcpyDtoH(unsafe.Pointer(&out[0]), dy, N*4), "DtoH")
	bad := 0
	for i := range out {
		want := 3.5*hx[i] + 1.0
		if math.Abs(float64(out[i]-want)) > 1e-4 {
			bad++
		}
	}
	fmt.Printf("PTX-JIT saxpy N=%d alpha=3.5 BAKED: mismatches=%d  out[7]=%.1f (want %.1f)  firstModuleLoad=%v\n",
		N, bad, out[7], 3.5*hx[7]+1.0, firstLoad)

	// ---------- 2. cuModuleLoadData latency: unique PTX (cold JIT) vs identical (cache) ----------
	var cold, warm []float64
	for i := 0; i < 30; i++ {
		p := genSaxpyPTX(N, float32(1.0)+float32(i)*0.0001+rand.Float32()*1e-6) // unique text every time
		var m CUmodule
		s := time.Now()
		ck(cuModuleLoadData(&m, cstr(p)), "loadData cold")
		cold = append(cold, float64(time.Since(s).Microseconds()))
		cuModuleUnload(m)
	}
	same := genSaxpyPTX(N, 2.25)
	for i := 0; i < 30; i++ {
		var m CUmodule
		s := time.Now()
		ck(cuModuleLoadData(&m, cstr(same)), "loadData warm")
		warm = append(warm, float64(time.Since(s).Microseconds()))
		cuModuleUnload(m)
	}
	fmt.Printf("cuModuleLoadData  UNIQUE-ptx median=%.0f us min=%.0f us   |   REPEATED-ptx median=%.0f us min=%.0f us\n",
		med(cold), minf(cold), med(warm), minf(warm))

	// bigger PTX: how does JIT time scale with kernel size?
	var big strings.Builder
	big.WriteString(".version 7.8\n.target sm_86\n.address_size 64\n")
	for i := 0; i < 32; i++ {
		big.WriteString(strings.Replace(genSaxpyPTX(N+i, float32(i)+0.5), ".version 7.8\n.target sm_86\n.address_size 64\n", "", 1))
		s := big.String()
		big.Reset()
		big.WriteString(strings.Replace(s, ".visible .entry k(", fmt.Sprintf(".visible .entry k%d(", i), 1))
	}
	bp := big.String()
	var bm CUmodule
	s := time.Now()
	rb := cuModuleLoadData(&bm, cstr(bp))
	bd := time.Since(s)
	fmt.Printf("cuModuleLoadData  32-kernel PTX (%d bytes): %v  err=%v\n", len(bp), bd, rb == 0)
	if rb == 0 {
		cuModuleUnload(bm)
	}

	// ---------- 3. kernel launch latency ----------
	var em CUmodule
	ck(cuModuleLoadData(&em, cstr(emptyPTX)), "empty ptx")
	var ef CUfunc
	ck(cuModuleGetFunction(&ef, em, "k"), "empty fn")
	dummy := CUdevptr(dx)
	eargs := []unsafe.Pointer{unsafe.Pointer(&dummy)}
	for i := 0; i < 100; i++ {
		cuLaunchKernel(ef, 1, 1, 1, 32, 1, 1, 0, 0, unsafe.Pointer(&eargs[0]), nil)
	}
	ck(cuCtxSynchronize(), "warm")
	// (a) launch+sync round trip
	var rt []float64
	for i := 0; i < 200; i++ {
		s := time.Now()
		ck(cuLaunchKernel(ef, 1, 1, 1, 32, 1, 1, 0, 0, unsafe.Pointer(&eargs[0]), nil), "l")
		ck(cuCtxSynchronize(), "s")
		rt = append(rt, float64(time.Since(s).Nanoseconds()))
	}
	// (b) async issue cost only (pipelined)
	s = time.Now()
	for i := 0; i < 2000; i++ {
		cuLaunchKernel(ef, 1, 1, 1, 32, 1, 1, 0, 0, unsafe.Pointer(&eargs[0]), nil)
	}
	issue := time.Since(s)
	ck(cuCtxSynchronize(), "s")
	fmt.Printf("launch: round-trip(launch+sync) median=%.1f us min=%.1f us | pipelined issue cost=%.2f us/launch\n",
		med(rt)/1000, minf(rt)/1000, float64(issue.Nanoseconds())/2000/1000)

	// ---------- 4. PCIe bandwidth: pageable vs pinned ----------
	sizes := []uint64{64 << 10, 1 << 20, 16 << 20, 128 << 20}
	var dbuf CUdevptr
	ck(cuMemAlloc(&dbuf, 128<<20), "alloc big")
	var pinned unsafe.Pointer
	ck(cuMemHostAlloc(&pinned, 128<<20, 0), "hostAlloc")
	pageable := make([]byte, 128<<20)
	fmt.Println("HtoD bandwidth (GB/s, median of 20):")
	for _, sz := range sizes {
		reps := 20
		var bp, bpi []float64
		for i := 0; i < reps; i++ {
			s := time.Now()
			ck(cuMemcpyHtoD(dbuf, unsafe.Pointer(&pageable[0]), sz), "pgHtoD")
			bp = append(bp, float64(sz)/time.Since(s).Seconds()/1e9)
			s = time.Now()
			ck(cuMemcpyHtoD(dbuf, pinned, sz), "pinHtoD")
			bpi = append(bpi, float64(sz)/time.Since(s).Seconds()/1e9)
		}
		fmt.Printf("  %7.2f MiB   pageable %6.2f   pinned %6.2f   (pinned latency %6.1f us)\n",
			float64(sz)/1048576, med(bp), med(bpi), float64(sz)/med(bpi)/1e9*1e6)
	}
	// DtoH
	hbuf := make([]byte, 128<<20)
	var bd2 []float64
	for i := 0; i < 20; i++ {
		s := time.Now()
		ck(cuMemcpyDtoH(pinned, dbuf, 128<<20), "DtoH pin")
		bd2 = append(bd2, float64(128<<20)/time.Since(s).Seconds()/1e9)
	}
	fmt.Printf("  DtoH 128 MiB pinned %6.2f GB/s\n", med(bd2))
	_ = hbuf
	runGEMM()
	fmt.Println("== DONE ==")
}
