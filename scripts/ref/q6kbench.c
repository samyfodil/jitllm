// Head-to-head: llama.cpp's Q6_K NEON dot product, timed exactly as jitllm's
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

#define QK_K 256
typedef unsigned short ggml_half;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

typedef struct { uint8x16_t val[2]; } ggml_uint8x16x2_t;
typedef struct { uint8x16_t val[4]; } ggml_uint8x16x4_t;
typedef struct { int8x16_t  val[4]; } ggml_int8x16x4_t;
typedef struct { int8x16_t  val[2]; } ggml_int8x16x2_t;
static inline ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t *p){ ggml_int8x16x2_t r; r.val[0]=vld1q_s8(p); r.val[1]=vld1q_s8(p+16); return r; }
typedef struct { int16x8_t val[2]; } ggml_int16x8x2_t;
static inline ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t *p){ ggml_int16x8x2_t r; r.val[0]=vld1q_s16(p); r.val[1]=vld1q_s16(p+8); return r; }
static inline ggml_uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t *p){ ggml_uint8x16x4_t r; for(int i=0;i<4;i++) r.val[i]=vld1q_u8(p+16*i); return r; }
static inline int8x16_t ggml_vqtbl1q_s8(int8x16_t t, uint8x16_t i){ return vqtbl1q_s8(t,i); }
static inline ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t *p){ ggml_uint8x16x2_t r; r.val[0]=vld1q_u8(p); r.val[1]=vld1q_u8(p+16); return r; }
static inline ggml_int8x16x4_t  ggml_vld1q_s8_x4(const int8_t  *p){ ggml_int8x16x4_t  r; for(int i=0;i<4;i++) r.val[i]=vld1q_s8(p+16*i); return r; }
static inline float ggml_half_to_float(ggml_half h){ __fp16 f; memcpy(&f,&h,2); return (float)f; }
#define GGML_FP16_TO_FP32 ggml_half_to_float
#define ggml_vdotq_s32 vdotq_s32

static void vec_dot_q6_K(int n, float *s, const void *vx, const void *vy) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const block_q6_K *restrict x = vx;
    const block_q8_K *restrict y = vy;
    const int nb = n / QK_K;
#include "lcpp_q6k_body.inc"
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static int cmp(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

int main(void) {
    const long bytes = 280L<<20;
    const int nb = 2048/QK_K;                    // super-blocks per row, k=2048
    const long rowB = (long)nb*sizeof(block_q6_K);
    const long rows = bytes/rowB;
    // aligned_alloc wants a size that is a multiple of the alignment on macOS.
    const long wsz = ((rows*rowB + 63) / 64) * 64;
    const long asz = ((nb*(long)sizeof(block_q8_K) + 63) / 64) * 64;
    block_q6_K *w = aligned_alloc(64, wsz);
    block_q8_K *a = aligned_alloc(64, asz);
    float *out = malloc(rows*sizeof(float));
    if(!w||!a||!out){ puts("alloc failed"); return 1; }
    srandom(1); for(long i=0;i<rows*rowB;i++) ((uint8_t*)w)[i]=random();
    for(int i=0;i<nb;i++){ a[i].d=0.01f; for(int j=0;j<QK_K;j++) a[i].qs[j]=(int8_t)(random()); for(int j=0;j<QK_K/16;j++) a[i].bsums[j]=0; }

    // Soak, then median of 15 -- RULE 2c, the same protocol fmtbench uses.
    double deadline = now()+1.5;
    while(now()<deadline) for(long r=0;r<rows;r++) vec_dot_q6_K(2048,&out[r],&w[r*nb],a);
    double ds[15];
    for(int k=0;k<15;k++){ double t0=now(); for(long r=0;r<rows;r++) vec_dot_q6_K(2048,&out[r],&w[r*nb],a); ds[k]=now()-t0; }
    qsort(ds,15,sizeof(double),cmp);
    printf("llama.cpp Q6_K, 1 thread, %.0f MB: %.1f GB/s\n", (double)(rows*rowB)/(1<<20), (double)(rows*rowB)/ds[7]/1e9);
    return 0;
}
