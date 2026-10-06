// Build: gcc -O2 -shared -fPIC -o gshim.so gshim.c -lcudart -ldl
// Use:   LD_PRELOAD=./gshim.so llama-bench ... -n 256   (docs/engineering-history/gpu-kernels.md)
// LD_PRELOAD shim: device-clock time of every cudaGraphLaunch, and the idle
// gap before it, read one launch later (as jitllm's JITLLM_CUDA_TIMING does).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
static cudaEvent_t ev[3][2];
static int ti, init;
static float busy[1 << 16], gap[1 << 16];
static int nb;
static int cmp(const void *a, const void *b) { float x = *(float *)a, y = *(float *)b; return (x > y) - (x < y); }
static float med(float *v, int n) { if (!n) return -1; qsort(v, n, sizeof *v, cmp); return v[n / 2]; }
static void report(void) {
  int n = nb > 100 ? 100 : nb; // the last 100
  static float b[100], g[100];
  for (int i = 0; i < n; i++) { b[i] = busy[nb - n + i]; g[i] = gap[nb - n + i]; }
  fprintf(stderr, "gshim: %d launches, last %d: graph %.1f us, gap before %.1f us (medians)\n", nb, n, med(b, n), med(g, n));
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t s) {
  static cudaError_t (*real)(cudaGraphExec_t, cudaStream_t);
  if (!real) real = dlsym(RTLD_NEXT, "cudaGraphLaunch");
  if (!init) { for (int i = 0; i < 3; i++) { cudaEventCreate(&ev[i][0]); cudaEventCreate(&ev[i][1]); } atexit(report); init = 1; }
  if (ti >= 1 && nb < (1 << 16)) {
    cudaEvent_t *p = ev[(ti - 1) % 3];
    float b = -1, g = -1;
    if (cudaEventQuery(p[1]) == cudaSuccess) {
      cudaEventElapsedTime(&b, p[0], p[1]);
      if (ti >= 2) cudaEventElapsedTime(&g, ev[(ti - 2) % 3][1], p[0]);
      busy[nb] = b * 1e3f; gap[nb] = g * 1e3f; nb++;
    }
  }
  cudaEventRecord(ev[ti % 3][0], s);
  cudaError_t r = real(e, s);
  cudaEventRecord(ev[ti % 3][1], s);
  ti++;
  return r;
}
