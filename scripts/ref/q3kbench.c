// Head-to-head: llama.cpp's Q3_K NEON dot product, timed exactly as jitllm's
// fmtbench times its own -- same working-set size, same soak, same median.
//
// This is the measurement that settles whether the M4 gap is in the KERNEL.
// Reading both sources said jitllm issues ~80 unpack instructions per super-block
// against llama.cpp's ~76 and pays 16 fewer horizontal reduces, which predicts
// jitllm should be AHEAD. Prediction from a static count is not a measurement.
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

#define QK_K 256
typedef unsigned short ggml_half;
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d; } block_q3_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

typedef struct { uint8x16_t val[2]; } ggml_uint8x16x2_t;
typedef struct { uint8x16_t val[4]; } ggml_uint8x16x4_t;
typedef struct { int8x16_t  val[4]; } ggml_int8x16x4_t;
static inline ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t *p){ ggml_uint8x16x2_t r; r.val[0]=vld1q_u8(p); r.val[1]=vld1q_u8(p+16); return r; }
static inline ggml_int8x16x4_t  ggml_vld1q_s8_x4(const int8_t  *p){ ggml_int8x16x4_t  r; for(int i=0;i<4;i++) r.val[i]=vld1q_s8(p+16*i); return r; }
static inline float ggml_half_to_float(ggml_half h){ __fp16 f; memcpy(&f,&h,2); return (float)f; }
#define GGML_FP16_TO_FP32 ggml_half_to_float
#define ggml_vdotq_s32 vdotq_s32

static void vec_dot_q3_K(int n, float *s, const void *vx, const void *vy) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q3_K *restrict x = vx;
    const block_q8_K *restrict y = vy;
    const int nb = n / QK_K;
#include "lcpp_q3k_body.inc"
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static int cmp(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

// ★ THREADED, BECAUSE A ONE-CORE KERNEL NUMBER CANNOT SPLIT A WHOLE-MODEL GAP.
// jitllm's Mac CPU row is ~21% behind, and the question is how much of that is the
// KERNEL and how much is what the engine does BETWEEN kernels. Only a
// like-for-like comparison answers it: jitllm's fmtbench at JITLLM_CORES=4 against
// llama.cpp's kernel on the same four threads. This harness was single-threaded,
// so that comparison could not be made and the split was being ESTIMATED from a
// one-core ratio -- see RULE 5aa.
//
// Threads take a contiguous row slice each, which is how jitllm's pool splits a
// matvec. macOS has no pthread_barrier_t, so this is a sense-reversing spin
// barrier: workers wait for the generation to change, run their slice, then
// signal completion. Spinning rather than sleeping keeps the measured window
// free of wake-up latency, which at a ~6 ms pass would be several percent.
struct warg { long lo, hi; };
static struct warg *wa;
static block_q3_K *g_w; static block_q8_K *g_a; static float *g_out;
static int g_nb; static long g_nthr;
static _Atomic long gen, done;

static void *worker(void *p){
    struct warg *v = (struct warg*)p;
    long seen = 0;
    for(;;){
        long g;
        while((g = atomic_load_explicit(&gen, memory_order_acquire)) == seen) ;
        if(g < 0) return NULL;
        seen = g;
        for(long r=v->lo;r<v->hi;r++) vec_dot_q3_K(2048,&g_out[r],&g_w[r*g_nb],g_a);
        atomic_fetch_add_explicit(&done, 1, memory_order_release);
    }
}

// The caller is worker 0, exactly as jitllm's pool does it (RULE 11): nthr
// participants means nthr-1 spawned threads, so nthr threads compete for nthr
// cores rather than nthr+1.
static void pass(void){
    atomic_store(&done, 0);
    atomic_fetch_add_explicit(&gen, 1, memory_order_release);
    for(long r=wa[0].lo;r<wa[0].hi;r++) vec_dot_q3_K(2048,&g_out[r],&g_w[r*g_nb],g_a);
    while(atomic_load_explicit(&done, memory_order_acquire) < g_nthr-1) ;
}

int main(int argc, char **argv) {
    const long bytes = 280L<<20;
    const int nb = 2048/QK_K;                    // super-blocks per row, k=2048
    const long rowB = (long)nb*sizeof(block_q3_K);
    const long rows = bytes/rowB;
    // aligned_alloc wants a size that is a multiple of the alignment on macOS.
    const long wsz = ((rows*rowB + 63) / 64) * 64;
    const long asz = ((nb*(long)sizeof(block_q8_K) + 63) / 64) * 64;
    block_q3_K *w = aligned_alloc(64, wsz);
    block_q8_K *a = aligned_alloc(64, asz);
    float *out = malloc(rows*sizeof(float));
    if(!w||!a||!out){ puts("alloc failed"); return 1; }
    srandom(1); for(long i=0;i<rows*rowB;i++) ((uint8_t*)w)[i]=random();
    for(int i=0;i<nb;i++){ a[i].d=0.01f; for(int j=0;j<QK_K;j++) a[i].qs[j]=(int8_t)(random()); for(int j=0;j<QK_K/16;j++) a[i].bsums[j]=0; }

    // ★ THREADED, BECAUSE A ONE-CORE KERNEL NUMBER CANNOT SPLIT A WHOLE-MODEL
    // GAP. jitllm's Mac CPU row is ~21% behind and the question is how much of that
    // is the KERNEL and how much is what the engine does between kernels. Only a
    // like-for-like comparison answers it: jitllm's fmtbench at JITLLM_CORES=4 against
    // llama.cpp's kernel at the same four threads. This harness was
    // single-threaded, so that comparison could not be made and the split was
    // being ESTIMATED from a one-core ratio -- see RULE 5aa.
    //
    // Threads take a contiguous row slice each, which is how jitllm's pool splits a
    // matvec. macOS has no pthread_barrier_t, so this is a sense-reversing spin
    // barrier: workers wait for the generation to change, run their slice, then
    // signal completion. Spinning rather than sleeping keeps the measured window
    // free of wake-up latency, which at a ~6 ms pass would be several percent.
    long nthr = 1;
    if(argc>1) nthr = atol(argv[1]);
    if(nthr<1) nthr = 1;
    g_w=w; g_a=a; g_out=out; g_nb=nb; g_nthr=nthr;
    atomic_store(&gen, 0); atomic_store(&done, 0);
    wa = malloc(nthr*sizeof(struct warg));
    pthread_t *th = malloc(nthr*sizeof(pthread_t));
    for(long t=0;t<nthr;t++){
        wa[t].lo = rows*t/nthr; wa[t].hi = rows*(t+1)/nthr;
        if(t) pthread_create(&th[t], NULL, worker, &wa[t]);
    }

    double deadline = now()+1.5;
    while(now()<deadline) pass();
    double ds[15];
    for(int k=0;k<15;k++){ double t0=now(); pass(); ds[k]=now()-t0; }
    qsort(ds,15,sizeof(double),cmp);
    printf("llama.cpp Q3_K, %ld thread%s, %.0f MB: %.1f GB/s\n",
           nthr, nthr==1?"":"s", (double)(rows*rowB)/(1<<20),
           (double)(rows*rowB)/ds[7]/1e9);
    atomic_store(&gen, -1);
    return 0;
}
