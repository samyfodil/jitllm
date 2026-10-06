#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
__global__ void bw_read(const float4* __restrict__ p, size_t n4, float* out){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  float4 a=make_float4(0,0,0,0);
  for(;i<n4;i+=st){float4 v=p[i];a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;}
  if(a.x==1e30f)out[0]=a.x;
}
int main(){
  size_t WB=512ull<<20;                 // "resident weights" the kernel streams
  float4* w; CK(cudaMalloc(&w,WB)); CK(cudaMemset(w,1,WB));
  float* o; CK(cudaMalloc(&o,4));
  size_t TB=256ull<<20;                 // transfer buffer
  void* pin; CK(cudaHostAlloc(&pin,TB,cudaHostAllocDefault));
  void* dst; CK(cudaMalloc(&dst,TB));
  cudaStream_t sk,sc; cudaStreamCreate(&sk); cudaStreamCreate(&sc);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  size_t n4=WB/16;
  auto kern=[&](cudaStream_t s){ bw_read<<<320,256,0,s>>>(w,n4,o); };
  float tk;
  kern(sk); CK(cudaDeviceSynchronize());
  cudaEventRecord(a,sk); for(int i=0;i<20;i++) kern(sk); cudaEventRecord(b,sk); CK(cudaEventSynchronize(b));
  cudaEventElapsedTime(&tk,a,b); tk/=20;
  printf("kernel alone: %.3f ms for %zu MiB = %.1f GB/s\n", tk, WB>>20, WB/(tk*1e-3)/1e9);
  // transfer sized to roughly match kernel duration at 10 GB/s
  for(size_t sz : {(size_t)16<<20,(size_t)32<<20,(size_t)64<<20}){
    float tc, tb;
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a,sc); for(int i=0;i<20;i++) CK(cudaMemcpyAsync(dst,pin,sz,cudaMemcpyHostToDevice,sc)); cudaEventRecord(b,sc); CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(&tc,a,b); tc/=20;
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for(int i=0;i<20;i++){ kern(sk); CK(cudaMemcpyAsync(dst,pin,sz,cudaMemcpyHostToDevice,sc)); }
    CK(cudaDeviceSynchronize()); cudaEventRecord(b); CK(cudaEventSynchronize(b));
    cudaEventElapsedTime(&tb,a,b); tb/=20;
    printf("copy %3zu MiB: alone %.3f ms (%.2f GB/s) | concurrent-with-kernel total %.3f ms "
           "(serial %.3f, ideal %.3f) | kernel BW during copy %.1f GB/s (%.0f%% of solo) | copy hidden %.0f%%\n",
      sz>>20, tc, sz/(tc*1e-3)/1e9, tb, tk+tc, tk>tc?tk:tc,
      WB/(tb*1e-3)/1e9, 100.0*tk/tb, 100.0*(tk+tc-tb)/tc);
  }
  return 0;
}
