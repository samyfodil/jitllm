#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  int seed = atoi(argv[1]);           // same seed => byte-identical PTX
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuCtxCreate(&c,0,d);
  char ptx[4096];
  // a kernel whose text depends on seed, with enough work that JIT is non-trivial
  snprintf(ptx,sizeof ptx,
   ".version 7.8\n.target sm_86\n.address_size 64\n"
   ".visible .entry k(.param .u64 px,.param .u64 py){\n"
   " .reg .pred %%p<2>; .reg .f32 %%f<8>; .reg .b32 %%r<6>; .reg .b64 %%rd<8>;\n"
   " ld.param.u64 %%rd1,[px]; ld.param.u64 %%rd2,[py];\n"
   " cvta.to.global.u64 %%rd3,%%rd1; cvta.to.global.u64 %%rd4,%%rd2;\n"
   " mov.u32 %%r1,%%ctaid.x; mov.u32 %%r2,%%ntid.x; mov.u32 %%r3,%%tid.x;\n"
   " mad.lo.s32 %%r4,%%r1,%%r2,%%r3; setp.ge.s32 %%p1,%%r4,%d; @%%p1 bra D;\n"
   " mul.wide.s32 %%rd5,%%r4,4; add.s64 %%rd6,%%rd3,%%rd5; add.s64 %%rd7,%%rd4,%%rd5;\n"
   " ld.global.f32 %%f1,[%%rd6]; ld.global.f32 %%f2,[%%rd7];\n"
   " fma.rn.f32 %%f3,%%f1,0f%08X,%%f2; fma.rn.f32 %%f4,%%f3,0f%08X,%%f1;\n"
   " fma.rn.f32 %%f5,%%f4,0f%08X,%%f2; st.global.f32 [%%rd7],%%f5;\n"
   "D: ret; }\n", 1048576+seed, 0x40000000+seed, 0x40100000+seed, 0x40200000+seed);
  for(int i=0;i<3;i++){
    CUmodule m; double t0=now(); CUresult r=cuModuleLoadData(&m,ptx); double dt=now()-t0;
    printf("  seed=%d load#%d : %8.3f ms  %s\n", seed, i, dt*1000, r?"FAIL":"ok");
    if(!r) cuModuleUnload(m);
  }
  return 0;
}
