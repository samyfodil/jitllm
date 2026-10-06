package main

import (
	"fmt"
	"strings"
	"time"
	"unsafe"
)

func nowf() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// ---- runtime PTX codegen for an int8 GEMM ----
//
// The question this probe answers: can jitllm generate a useful GPU matmul at load
// time, the way it generates AVX-VNNI and SDOT kernels, and what is that worth
// against the CPU GEMM it already has?
//
// The kernel mirrors the CPU one on purpose, so the comparison means something:
// a register tile of tm weight rows by tn tokens, int8 accumulated in int32 via
// dp4a -- the GPU's VPDPBUSD/SDOT -- and converted to float once at the end.
// Four consecutive k land in one 32-bit load on both sides, which is the same
// property the CPU kernels are built around.
//
// K, the tile and the output stride are baked as immediates, which is what
// makes this code generation rather than one fixed shader.
func genGEMMPTX(k, tm, tn, tokStride int) string {
	var b strings.Builder
	nacc := tm * tn
	// Register bases are derived from the tile, not hand-numbered: fixed bases
	// collide once tm > 10 and give a fast, wrong kernel.
	rW := 20      // tm weight operands
	rA := rW + tm // tn activation operands
	rC := rA + tn // tm*tn accumulators
	rEnd := rC + nacc + 4
	// The register file is declared to fit the tile. Getting it wrong makes
	// ptxas reject the module, which looks exactly like a hardware limit.
	fmt.Fprintf(&b, `.version 7.8
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pW, .param .u64 pA, .param .u64 pO)
{
 .reg .pred %%p<4>;
 .reg .b32 %%r<%d>;
 .reg .f32 %%f<%d>;
 .reg .b64 %%rd<%d>;`, rEnd, 8+nacc, 24+tm+tn)
	b.WriteString(`
 ld.param.u64 %rd1,[pW]; ld.param.u64 %rd2,[pA]; ld.param.u64 %rd3,[pO];
 cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r8,%tid.x;
 mad.lo.s32 %r9,%r1,%r2,%r8;
`)
	// One flat tile index, divided by a baked constant, with a fixed
	// 128-thread block: tying blockDim to tok/tn gave half-warp blocks that
	// capped occupancy.
	ntok := tokStride / tn
	fmt.Fprintf(&b, " div.u32 %%r10,%%r9,%d;\n rem.u32 %%r11,%%r9,%d;\n", ntok, ntok)
	fmt.Fprintf(&b, " mul.lo.s32 %%r3,%%r10,%d;\n", tm) // first row of this tile
	fmt.Fprintf(&b, " mul.lo.s32 %%r4,%%r11,%d;\n", tn) // first token

	// One walking pointer per row and per token, advanced by 4 bytes a step.
	for i := 0; i < tm; i++ {
		fmt.Fprintf(&b, " add.s32 %%r5,%%r3,%d; mul.wide.s32 %%rd%d,%%r5,%d; add.s64 %%rd%d,%%rd1,%%rd%d;\n",
			i, 10+i, k, 10+i, 10+i)
	}
	for j := 0; j < tn; j++ {
		fmt.Fprintf(&b, " add.s32 %%r5,%%r4,%d; mul.wide.s32 %%rd%d,%%r5,%d; add.s64 %%rd%d,%%rd2,%%rd%d;\n",
			j, 10+tm+j, k, 10+tm+j, 10+tm+j)
	}
	for a := 0; a < nacc; a++ {
		fmt.Fprintf(&b, " mov.u32 %%r%d,0;\n", rC+a)
	}
	b.WriteString(" mov.u32 %r6,0;\nL:\n")
	for i := 0; i < tm; i++ {
		fmt.Fprintf(&b, " ld.global.u32 %%r%d,[%%rd%d];\n", rW+i, 10+i)
	}
	for j := 0; j < tn; j++ {
		fmt.Fprintf(&b, " ld.global.u32 %%r%d,[%%rd%d];\n", rA+j, 10+tm+j)
	}
	for i := 0; i < tm; i++ {
		for j := 0; j < tn; j++ {
			fmt.Fprintf(&b, " dp4a.s32.s32 %%r%d,%%r%d,%%r%d,%%r%d;\n",
				rC+i*tn+j, rW+i, rA+j, rC+i*tn+j)
		}
	}
	for i := 0; i < tm; i++ {
		fmt.Fprintf(&b, " add.s64 %%rd%d,%%rd%d,4;\n", 10+i, 10+i)
	}
	for j := 0; j < tn; j++ {
		fmt.Fprintf(&b, " add.s64 %%rd%d,%%rd%d,4;\n", 10+tm+j, 10+tm+j)
	}
	fmt.Fprintf(&b, " add.s32 %%r6,%%r6,4;\n setp.lt.u32 %%p1,%%r6,%d;\n @%%p1 bra L;\n", k)

	// out[row][tok], float32, with the token stride baked.
	for i := 0; i < tm; i++ {
		for j := 0; j < tn; j++ {
			a := i*tn + j
			fmt.Fprintf(&b, " cvt.rn.f32.s32 %%f%d,%%r%d;\n", 1+a, rC+a)
			fmt.Fprintf(&b, " add.s32 %%r7,%%r3,%d; mul.lo.s32 %%r7,%%r7,%d; add.s32 %%r7,%%r7,%%r4; add.s32 %%r7,%%r7,%d;\n",
				i, tokStride, j)
			fmt.Fprintf(&b, " mul.wide.s32 %%rd%d,%%r7,4; add.s64 %%rd%d,%%rd3,%%rd%d; st.global.f32 [%%rd%d],%%f%d;\n", 10+tm+tn, 10+tm+tn, 10+tm+tn, 10+tm+tn, 1+a)
		}
	}
	b.WriteString(" ret;\n}\n")
	return b.String()
}

// runGEMM measures the generated int8 matmul against a CPU reference.
//
// The shape is a real prefill matmul: gemma's 2048-wide projections and
// tinyllama's FFN are this size, and the token count is jitllm's prefill chunk.
func runGEMM() {
	const (
		rows = 4096
		K    = 2048
		tok  = 128
	)
	fmt.Printf("\n== int8 GEMM generated at runtime: rows=%d k=%d tok=%d ==\n", rows, K, tok)

	hw := make([]int8, rows*K)
	ha := make([]int8, tok*K)
	for i := range hw {
		hw[i] = int8((i*37)%251 - 125)
	}
	for i := range ha {
		ha[i] = int8((i*53)%241 - 120)
	}
	out := make([]float32, rows*tok)
	var dw, da, do CUdevptr
	ck(cuMemAlloc(&dw, uint64(len(hw))), "alloc W")
	ck(cuMemAlloc(&da, uint64(len(ha))), "alloc A")
	ck(cuMemAlloc(&do, uint64(len(out)*4)), "alloc O")
	ck(cuMemcpyHtoD(dw, unsafe.Pointer(&hw[0]), uint64(len(hw))), "HtoD W")
	ck(cuMemcpyHtoD(da, unsafe.Pointer(&ha[0]), uint64(len(ha))), "HtoD A")
	args := []unsafe.Pointer{unsafe.Pointer(&dw), unsafe.Pointer(&da), unsafe.Pointer(&do)}
	macs := float64(rows) * float64(K) * float64(tok)

	// The tile sweep asks whether register tiling pays here as it does on the
	// CPU, on a machine whose register allocation jitllm does not own.
	type res struct {
		tm, tn int
		gmac   float64
		jit    float64
		bytes  int
		ok     bool
	}
	var out2 []res
	for _, tl := range [][2]int{{1, 1}, {2, 2}, {4, 2}, {2, 4}, {4, 4}, {8, 4}, {4, 8}, {8, 8}, {8, 16}, {16, 8}} {
		tm, tn := tl[0], tl[1]
		if tok%tn != 0 || rows%tm != 0 {
			continue
		}
		ptx := genGEMMPTX(K, tm, tn, tok)
		t0 := nowf()
		var mod CUmodule
		if r := cuModuleLoadData(&mod, cstr(ptx)); r != 0 {
			fmt.Printf("  %dx%d: ptxas rejected: %s\n", tm, tn, errStr(r))
			continue
		}
		jit := nowf() - t0
		var fn CUfunc
		ck(cuModuleGetFunction(&fn, mod, "k"), "getFunction")

		tiles := (rows / tm) * (tok / tn)
		block := uint32(128)
		grid := uint32((tiles + 127) / 128)
		for i := range out {
			out[i] = 0
		}
		ck(cuMemcpyHtoD(do, unsafe.Pointer(&out[0]), uint64(len(out)*4)), "clear O")
		ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, unsafe.Pointer(&args[0]), nil), "launch")
		ck(cuCtxSynchronize(), "sync")
		ck(cuMemcpyDtoH(unsafe.Pointer(&out[0]), do, uint64(len(out)*4)), "DtoH")

		bad := 0
		for _, r := range []int{0, 1, 17, rows / 2, rows - 1} {
			for _, c := range []int{0, 3, 64, tok - 1} {
				var want int32
				for x := 0; x < K; x++ {
					want += int32(hw[r*K+x]) * int32(ha[c*K+x])
				}
				if float32(want) != out[r*tok+c] {
					bad++
				}
			}
		}
		var times []float64
		for i := 0; i < 15; i++ {
			s := nowf()
			ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, unsafe.Pointer(&args[0]), nil), "launch")
			ck(cuCtxSynchronize(), "sync")
			times = append(times, nowf()-s)
		}
		m := med(times)
		out2 = append(out2, res{tm, tn, macs / m / 1e9, jit * 1e3, len(ptx), bad == 0})
		cuModuleUnload(mod)
	}
	fmt.Println("  tile   Gmac/s    GFLOP/s   ptx    codegen+ptxas   exact")
	best := 0.0
	for _, r := range out2 {
		flag := "  OK"
		if !r.ok {
			flag = "  WRONG"
		}
		if r.gmac > best && r.ok {
			best = r.gmac
		}
		fmt.Printf("  %dx%-3d %8.1f  %9.1f  %5dB  %8.2f ms   %s\n",
			r.tm, r.tn, r.gmac, 2*r.gmac, r.bytes, r.jit, flag)
	}
	fmt.Printf("  best %.1f Gmac/s = %.1f GFLOP/s\n", best, 2*best)
	fmt.Printf("  for scale: jitllm CPU GEMM peaks ~750 Gmac/s across 5 P-cores;\n")
	fmt.Printf("             weights here are %.1f MiB, HtoD ~9 GB/s = %.1f ms\n",
		float64(len(hw))/1048576, float64(len(hw))/9e9*1e3)
	cuMemFree(dw)
	cuMemFree(da)
	cuMemFree(do)
}
