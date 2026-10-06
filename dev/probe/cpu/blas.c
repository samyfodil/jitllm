#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
void cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
#define RM 101
#define NT 111
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  int K=2048,N=16384;
  float *A=aligned_alloc(64,(size_t)1024*K*4), *B=aligned_alloc(64,(size_t)K*N*4), *C=aligned_alloc(64,(size_t)1024*N*4);
  for(size_t i=0;i<(size_t)1024*K;i++)A[i]=1.f;
  for(size_t i=0;i<(size_t)K*N;i++)B[i]=1.f;
  int Ts[]={1,4,16,32,64,128,256,512,1024};
  for(int nt=4; nt<=14; nt+=nt==4?2:4){
    openblas_set_num_threads(nt);
    printf("OpenBLAS sgemm threads=%2d (actual %d): ", nt, openblas_get_num_threads());
    for(int i=0;i<9;i++){
      int T=Ts[i]; double fl=2.0*T*K*N, best=1e9;
      for(int r=0;r<7;r++){ double t0=now();
        cblas_sgemm(RM,NT,NT,T,N,K,1.f,A,K,B,N,0.f,C,N);
        double dt=now()-t0; if(dt<best)best=dt; }
      if(T==1||T==128||T==512||T==1024) printf("T=%-4d %5.2f GF/s  ",T,fl/best/1e9);
    }
    printf("\n");
  }
  return 0;
}
