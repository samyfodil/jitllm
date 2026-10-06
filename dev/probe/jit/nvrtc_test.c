#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef void* nvrtcProgram;
int main(int argc,char**argv){
  const char* path = argv[1];
  void* h = dlopen(path, RTLD_NOW);
  if(!h){ printf("dlopen %s FAILED: %s\n", path, dlerror()); return 1; }
  int (*nvrtcVersion)(int*,int*) = dlsym(h,"nvrtcVersion");
  int (*nvrtcCreateProgram)(nvrtcProgram*,const char*,const char*,int,const char**,const char**) = dlsym(h,"nvrtcCreateProgram");
  int (*nvrtcCompileProgram)(nvrtcProgram,int,const char**) = dlsym(h,"nvrtcCompileProgram");
  int (*nvrtcGetPTXSize)(nvrtcProgram,size_t*) = dlsym(h,"nvrtcGetPTXSize");
  int (*nvrtcGetPTX)(nvrtcProgram,char*) = dlsym(h,"nvrtcGetPTX");
  int (*nvrtcGetProgramLogSize)(nvrtcProgram,size_t*) = dlsym(h,"nvrtcGetProgramLogSize");
  int (*nvrtcGetProgramLog)(nvrtcProgram,char*) = dlsym(h,"nvrtcGetProgramLog");
  int maj,min; nvrtcVersion(&maj,&min);
  printf("== %s : nvrtc %d.%d ==\n", path, maj, min);
  const char* src =
   "extern \"C\" __global__ void k(float* y, const float* x){ int i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i]*3.5f+1.0f; }\n";
  nvrtcProgram p;
  if(nvrtcCreateProgram(&p,src,"k.cu",0,0,0)){printf("create fail\n");return 1;}
  const char* opts[]={"--gpu-architecture=compute_86"};
  double t0=now();
  int rc=nvrtcCompileProgram(p,1,opts);
  double tc=now()-t0;
  size_t ls; nvrtcGetProgramLogSize(p,&ls);
  if(ls>1){char*l=malloc(ls);nvrtcGetProgramLog(p,l);printf("log: %s\n",l);}
  if(rc){printf("compile rc=%d\n",rc);return 1;}
  size_t ps; nvrtcGetPTXSize(p,&ps); char* ptx=malloc(ps); nvrtcGetPTX(p,ptx);
  printf("nvrtc compile time = %.1f ms, ptx = %zu bytes\n", tc*1000, ps);
  char* vl = strstr(ptx,".version"); if(vl){ char b[32]; sscanf(vl,"%31[^\n]",b); printf("PTX header: %s\n", b); }
  // now try to load it with the installed DRIVER
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuCtxCreate(&c,0,d);
  int dv; cuDriverGetVersion(&dv); printf("driver version = %d\n", dv);
  CUmodule m; t0=now(); CUresult r=cuModuleLoadData(&m,ptx); double tl=now()-t0;
  const char* es; cuGetErrorString(r,&es);
  printf("cuModuleLoadData(nvrtc PTX) -> %s   (%.2f ms)\n", r==0?"OK":es, tl*1000);
  if(r==0){ CUfunction f; printf("  getFunction: %s\n", cuModuleGetFunction(&f,m,"k")==0?"OK":"FAIL"); }
  return 0;
}
