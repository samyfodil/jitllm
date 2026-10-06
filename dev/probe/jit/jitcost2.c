#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
  FILE* f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  char* base=malloc(n+64); fread(base,1,n,f); base[n]=0; fclose(f);
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuCtxCreate(&c,0,d);
  // warm the JIT subsystem
  { CUmodule m; cuModuleLoadData(&m,base); cuModuleUnload(m); }
  double t[25];
  char* buf=malloc(n+64);
  for(int i=0;i<12;i++){
    memcpy(buf,base,n+1);
    // make the PTX text unique without changing its size class: rename the entry symbols
    char* p=buf; static int uniq=0; uniq++;
    while((p=strstr(p,"mv_dyn"))){ p[4]='a'+(uniq%26); p[5]='a'+((uniq/26)%26); p+=6; }
    { char* q=buf; while((q=strstr(q,"mv_gate"))){ q[4]='a'+(uniq%26); q[5]='a'+((uniq/26)%26); q+=7; } }
    CUmodule m; double t0=now(); CUresult r=cuModuleLoadData(&m,buf); t[i]=(now()-t0)*1000;
    if(r){const char*e;cuGetErrorString(r,&e);printf("FAIL %s\n",e);return 1;}
    printf("  unique load #%2d : %8.2f ms\n", i, t[i]);
    cuModuleUnload(m);
  }
  qsort(t,12,sizeof(double),cmp);
  printf("real matvec PTX module (%ld bytes, all kernels): cold-JIT median %.1f ms, min %.1f, max %.1f\n",
     n, t[6], t[0], t[11]);
  // now repeat an identical one (cache hit)
  double w[25]; for(int i=0;i<25;i++){ CUmodule m; double t0=now(); cuModuleLoadData(&m,base); w[i]=(now()-t0)*1000; cuModuleUnload(m); }
  qsort(w,25,sizeof(double),cmp);
  printf("same PTX, cache hit: median %.3f ms\n", w[12]);
  return 0;
}
