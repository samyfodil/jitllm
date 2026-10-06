// Build: gcc -O2 -shared -fPIC -o kshim.so kshim.c -lcudart -ldl
// Use:   LD_PRELOAD=./kshim.so llama-bench ... -n 256
// LD_PRELOAD shim: during a stream capture, an event NODE after every
// cudaLaunchKernel, so each graph replay stamps every kernel's end on the
// device clock with no host in the loop; kernel i's time is its event minus
// kernel i-1's. Read one launch later and summed per kernel -- the same
// measurement jitllm's JITLLM_CUDA_KERNEL_TIMING takes of its own graphs
// (docs/engineering-history/gpu-kernels.md).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#define M 4096
#define K 512
typedef struct { cudaEvent_t ev[M]; int kern[M]; int n; } list;
static list *cap, *built, *launched;
static cudaStream_t capS;
static const void *fp[K];
static char *names[K];
static double us[K];
static long cnt[K];
static int nk, graphs;
static int slot(const void *f) {
  for (int i = 0; i < nk; i++) if (fp[i] == f) return i;
  if (nk == K) return K - 1;
  static int (*getName)(const char **, void *);
  if (!getName) getName = dlsym(RTLD_DEFAULT, "cuFuncGetName");
  void *cf = 0; const char *n = "?";
  if (cudaGetFuncBySymbol((cudaFunction_t *)&cf, f) == cudaSuccess && getName) getName(&n, cf);
  fp[nk] = f; names[nk] = strdup(n);
  return nk++;
}
static void freelist(list *l) {
  if (!l) return;
  for (int i = 0; i < l->n; i++) cudaEventDestroy(l->ev[i]);
  free(l);
}
static void settle(list *l) {
  if (!l) return;
  for (int i = 1; i < l->n; i++) {
    float ms;
    if (cudaEventElapsedTime(&ms, l->ev[i - 1], l->ev[i]) == cudaSuccess) { us[l->kern[i]] += ms * 1e3; cnt[l->kern[i]]++; }
  }
  graphs++;
}
static void report(void) {
  fprintf(stderr, "kshim graphs %d\n", graphs);
  for (int i = 0; i < nk; i++) fprintf(stderr, "kshim %12.1f us %8ld  %s\n", us[i], cnt[i], names[i]);
}
static void mark(cudaStream_t s, int k) {
  if (!cap || cap->n == M) return;
  if (cudaEventCreate(&cap->ev[cap->n]) != cudaSuccess) return;
  cudaEventRecordWithFlags(cap->ev[cap->n], s, cudaEventRecordExternal); // a timed node, not an edge
  cap->kern[cap->n++] = k;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t s, enum cudaStreamCaptureMode mode) {
  static cudaError_t (*real)(cudaStream_t, enum cudaStreamCaptureMode);
  static int init;
  if (!real) real = dlsym(RTLD_NEXT, "cudaStreamBeginCapture");
  if (!init) { atexit(report); init = 1; }
  cudaError_t r = real(s, mode);
  if (r == cudaSuccess) { cap = calloc(1, sizeof *cap); capS = s; mark(s, 0); }
  return r;
}
cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g) {
  static cudaError_t (*real)(cudaStream_t, cudaGraph_t *);
  if (!real) real = dlsym(RTLD_NEXT, "cudaStreamEndCapture");
  cudaError_t r = real(s, g);
  if (cap) { if (built && built != launched) freelist(built); built = cap; cap = 0; }
  return r;
}
cudaError_t cudaLaunchKernel(const void *f, dim3 g, dim3 b, void **args, size_t sh, cudaStream_t s) {
  static cudaError_t (*real)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  if (!real) real = dlsym(RTLD_NEXT, "cudaLaunchKernel");
  cudaError_t r = real(f, g, b, args, sh, s);
  if (cap && s == capS) mark(s, slot(f));
  return r;
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t s) {
  static cudaError_t (*real)(cudaGraphExec_t, cudaStream_t);
  if (!real) real = dlsym(RTLD_NEXT, "cudaGraphLaunch");
  // The previous replay is done: llama.cpp synchronises for the logits
  // between tokens. The exec now carries the latest capture's events.
  settle(launched);
  if (launched && launched != built) freelist(launched);
  launched = built;
  return real(e, s);
}
