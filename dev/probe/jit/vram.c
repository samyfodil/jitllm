#include <stdio.h>
#include <cuda.h>
int main(){
  cuInit(0); CUdevice d; cuDeviceGet(&d,0);
  size_t f,t; CUcontext c; cuCtxCreate(&c,0,d);
  cuMemGetInfo(&f,&t);
  printf("after cuCtxCreate: free=%zu MiB total=%zu MiB\n", f>>20, t>>20);
  // largest single allocation
  size_t lo=0, hi=f, best=0; CUdeviceptr p;
  while(lo<=hi){ size_t mid=lo+(hi-lo)/2; if(mid==0)break;
    if(cuMemAlloc(&p,mid)==CUDA_SUCCESS){ cuMemFree(p); best=mid; lo=mid+(1<<20); } else hi=mid-(1<<20); }
  printf("largest single cuMemAlloc: %zu MiB\n", best>>20);
  // total allocatable in 64 MiB chunks
  CUdeviceptr chunks[128]; int n=0; size_t tot=0;
  while(n<128 && cuMemAlloc(&chunks[n],64ull<<20)==CUDA_SUCCESS){ tot+=64ull<<20; n++; }
  printf("total allocatable in 64 MiB chunks: %zu MiB (%d chunks)\n", tot>>20, n);
  cuMemGetInfo(&f,&t); printf("free with all held: %zu MiB\n", f>>20);
  for(int i=0;i<n;i++) cuMemFree(chunks[i]);
  return 0;
}
