#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e){printf("CUBLAS ERR line %d st=%d\n",__LINE__,(int)e);exit(1);} }while(0)
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(){
  cublasHandle_t h; CB(cublasCreate(&h));
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  const int K=2048,N=16384,TMAX=1024;
  __half *A,*B,*C16; float *Cf,*Af,*Bf; int8_t *A8,*B8; int32_t* C32;
  CK(cudaMalloc(&A,(size_t)TMAX*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C16,(size_t)TMAX*N*2));
  CK(cudaMalloc(&Af,(size_t)TMAX*K*4)); CK(cudaMalloc(&Bf,(size_t)K*N*4)); CK(cudaMalloc(&Cf,(size_t)TMAX*N*4));
  CK(cudaMalloc(&A8,(size_t)TMAX*K)); CK(cudaMalloc(&B8,(size_t)K*N)); CK(cudaMalloc(&C32,(size_t)TMAX*N*4));
  // fill with 1.0 / 1 so we can verify: C[i][j] should equal K
  { std::vector<__half> ha((size_t)TMAX*K,__float2half(1.f)); CK(cudaMemcpy(A,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
    std::vector<__half> hb((size_t)K*N,__float2half(1.f));   CK(cudaMemcpy(B,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
    std::vector<float> fa((size_t)TMAX*K,1.f); CK(cudaMemcpy(Af,fa.data(),fa.size()*4,cudaMemcpyHostToDevice));
    std::vector<float> fb((size_t)K*N,1.f);    CK(cudaMemcpy(Bf,fb.data(),fb.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemset(A8,1,(size_t)TMAX*K)); CK(cudaMemset(B8,1,(size_t)K*N)); }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto verify16=[&](int T){ std::vector<__half> o((size_t)T*N); CK(cudaMemcpy(o.data(),C16,o.size()*2,cudaMemcpyDeviceToHost));
     return __half2float(o[0]); };
  auto verifyf=[&](int T){ std::vector<float> o((size_t)T*N); CK(cudaMemcpy(o.data(),Cf,o.size()*4,cudaMemcpyDeviceToHost)); return o[0]; };
  auto verify32=[&](int T){ std::vector<int32_t> o((size_t)T*N); CK(cudaMemcpy(o.data(),C32,o.size()*4,cudaMemcpyDeviceToHost)); return (float)o[0]; };

  printf("GPU %s sm_%d%d, %d SMs.  GEMM C[Tx%d] = A[Tx%d]*B[%dx%d] (gemma ffn_up shape); all inputs=1 so C must be %d\n",
    p.name,p.major,p.minor,p.multiProcessorCount,N,K,K,N,K);
  printf("%5s | %-22s | %-22s | %-22s | %-22s\n","T","fp16 in / fp32 acc TC","fp16 in / fp32 acc SIMT","fp32 TF32 TC","int8 in / int32 acc TC");
  for(int T : {1,4,16,32,64,128,256,512,1024}){
    double fl=2.0*T*(double)K*N; float al=1.f,be=0.f; int32_t ial=1,ibe=0;
    auto go=[&](cublasComputeType_t ct,cudaDataType at,cudaDataType bt,cudaDataType cbt,
                const void*aa,const void*bb,void*cc,cublasGemmAlgo_t alg,const void*pa,const void*pb)->double{
      for(int w=0;w<3;w++) if(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,T,K,pa,bb,bt,N,aa,at,K,pb,cc,cbt,N,ct,alg)) return -1;
      CK(cudaDeviceSynchronize()); std::vector<double> v;
      for(int r=0;r<11;r++){ cudaEventRecord(e0);
        for(int k=0;k<20;k++) cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,T,K,pa,bb,bt,N,aa,at,K,pb,cc,cbt,N,ct,alg);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms;cudaEventElapsedTime(&ms,e0,e1); v.push_back(ms/20);} 
      return med(v); };
    CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
    double t1=go(CUBLAS_COMPUTE_32F,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,A,B,C16,CUBLAS_GEMM_DEFAULT_TENSOR_OP,&al,&be); float v1=verify16(T);
    double t4=go(CUBLAS_COMPUTE_32I,CUDA_R_8I,CUDA_R_8I,CUDA_R_32I,A8,B8,C32,CUBLAS_GEMM_DEFAULT_TENSOR_OP,&ial,&ibe); float v4=t4<0?-1:verify32(T);
    double t3=go(CUBLAS_COMPUTE_32F_FAST_TF32,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F,Af,Bf,Cf,CUBLAS_GEMM_DEFAULT_TENSOR_OP,&al,&be); float v3=verifyf(T);
    CB(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
    double t2=go(CUBLAS_COMPUTE_32F_PEDANTIC,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,A,B,C16,CUBLAS_GEMM_DEFAULT,&al,&be); float v2=verify16(T);
    printf("%5d | %7.3fms %6.2f TF/s%s | %7.3fms %6.2f TF/s%s | %7.3fms %6.2f TF/s%s | %7.3fms %6.2f TOP/s%s\n",T,
      t1,fl/(t1*1e-3)/1e12, v1==K?"":"!!", t2,fl/(t2*1e-3)/1e12, v2==K?"":"!!",
      t3,fl/(t3*1e-3)/1e12, v3==K?"":"!!", t4,t4<0?0:fl/(t4*1e-3)/1e12, v4==K?"":"!!");
  }
  printf("(!! = numeric check failed; fp16 accumulate saturates at K=2048 so fp16-acc is excluded)\n");
  return 0;
}
