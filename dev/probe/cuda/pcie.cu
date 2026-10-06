#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static double med(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

// a compute kernel that reads device memory for a controllable duration
__global__ void burn(const float4* __restrict__ p, size_t n4, float* out, int iters){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  float4 a=make_float4(0,0,0,0);
  for(int r=0;r<iters;r++) for(size_t j=i;j<n4;j+=st){float4 v=p[j];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
  if(a.x==1e30f)out[0]=a.x;
}
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("asyncEngineCount=%d  canMapHost=%d  unifiedAddressing=%d  concurrentKernels=%d\n",
     p.asyncEngineCount,p.canMapHostMemory,p.unifiedAddressing,p.concurrentKernels);
  size_t MAX=256ull<<20;
  void *pinned,*wc; CK(cudaHostAlloc(&pinned,MAX,cudaHostAllocDefault));
  CK(cudaHostAlloc(&wc,MAX,cudaHostAllocWriteCombined));
  void* pageable=malloc(MAX); memset(pageable,1,MAX);
  void* d; CK(cudaMalloc(&d,MAX));
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
  printf("\n-- HtoD, cudaEvent-timed, median of 30 --\n");
  printf("%10s %12s %12s %12s\n","size","pageable","pinned","pinned+WC");
  for(size_t sz : {(size_t)64<<10,(size_t)256<<10,(size_t)1<<20,(size_t)4<<20,(size_t)16<<20,(size_t)64<<20,(size_t)256<<20}){
    std::vector<double> vp,vi,vw;
    for(int i=0;i<30;i++){
      float ms;
      cudaEventRecord(a); CK(cudaMemcpy(d,pageable,sz,cudaMemcpyHostToDevice)); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); vp.push_back(sz/(ms*1e-3)/1e9);
      cudaEventRecord(a); CK(cudaMemcpy(d,pinned,sz,cudaMemcpyHostToDevice));   cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); vi.push_back(sz/(ms*1e-3)/1e9);
      cudaEventRecord(a); CK(cudaMemcpy(d,wc,sz,cudaMemcpyHostToDevice));       cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); vw.push_back(sz/(ms*1e-3)/1e9);
    }
    printf("%8.2f MiB %9.2f GB/s %8.2f GB/s %8.2f GB/s\n", sz/1048576.0, med(vp), med(vi), med(vw));
  }
  printf("\n-- DtoH pinned 256MiB: ");
  { std::vector<double> v; for(int i=0;i<20;i++){float ms;cudaEventRecord(a);CK(cudaMemcpy(pinned,d,MAX,cudaMemcpyDeviceToHost));cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&ms,a,b);v.push_back(MAX/(ms*1e-3)/1e9);} printf("%.2f GB/s\n",med(v)); }

  // ---- OVERLAP: does HtoD run concurrently with a compute kernel? ----
  printf("\n-- overlap test: 64 MiB HtoD vs a compute kernel --\n");
  size_t CB=64ull<<20; float4* cbuf; CK(cudaMalloc(&cbuf,CB)); CK(cudaMemset(cbuf,1,CB));
  float* o; CK(cudaMalloc(&o,4));
  cudaStream_t s1,s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
  size_t n4=CB/16; int iters=6;
  float tk,tc,tboth;
  { burn<<<320,256,0,s1>>>(cbuf,n4,o,iters); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for(int i=0;i<10;i++) burn<<<320,256,0,s1>>>(cbuf,n4,o,iters); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tk,a,b); tk/=10; }
  { cudaEventRecord(a); for(int i=0;i<10;i++) CK(cudaMemcpyAsync(d,pinned,CB,cudaMemcpyHostToDevice,s2)); cudaStreamSynchronize(s2); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tc,a,b); tc/=10; }
  { CK(cudaDeviceSynchronize()); cudaEventRecord(a);
    for(int i=0;i<10;i++){ burn<<<320,256,0,s1>>>(cbuf,n4,o,iters); CK(cudaMemcpyAsync(d,pinned,CB,cudaMemcpyHostToDevice,s2)); }
    CK(cudaDeviceSynchronize()); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tboth,a,b); tboth/=10; }
  printf("  kernel alone      = %.3f ms  (reads %.0f MiB, %.1f GB/s)\n", tk, CB*iters/1048576.0, CB*(double)iters/(tk*1e-3)/1e9);
  printf("  64MiB HtoD alone  = %.3f ms  (%.2f GB/s)\n", tc, CB/(tc*1e-3)/1e9);
  printf("  both concurrent   = %.3f ms   (serial would be %.3f ms; perfect overlap = %.3f ms)\n", tboth, tk+tc, tk>tc?tk:tc);
  printf("  -> transfer hidden: %.0f%%\n", 100.0*(tk+tc-tboth)/tc);

  // ---- how much does a concurrent HtoD steal from kernel bandwidth? ----
  printf("\n-- device-memory BW while a HtoD is in flight --\n");
  printf("  kernel-only effective BW  = %.1f GB/s\n", CB*(double)iters/(tk*1e-3)/1e9);
  printf("  kernel BW during transfer = %.1f GB/s\n", CB*(double)iters/(tboth*1e-3)/1e9);
  return 0;
}
