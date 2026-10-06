#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>
#include <sched.h>
static size_t NB; static char* buf; static int NT; static int *cpus;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void* rd(void* a){
  long id=(long)a;
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[id],&s); pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
  size_t chunk=(NB/NT)&~(size_t)4095; const char* p=buf+id*chunk;
  __m256i acc=_mm256_setzero_si256();
  for(int r=0;r<10;r++)
    for(size_t i=0;i<chunk;i+=128){
      acc=_mm256_add_epi32(acc,_mm256_load_si256((const __m256i*)(p+i)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((const __m256i*)(p+i+32)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((const __m256i*)(p+i+64)));
      acc=_mm256_add_epi32(acc,_mm256_load_si256((const __m256i*)(p+i+96)));
    }
  if(_mm256_extract_epi32(acc,0)==0x7fffffff) printf("x");
  return 0;
}
int main(int argc,char**argv){
  NB=(size_t)1024<<20; NT=atoi(argv[1]);
  cpus=malloc(NT*sizeof(int));
  for(int i=0;i<NT;i++) cpus[i]=atoi(argv[2+i]);
  posix_memalign((void**)&buf,4096,NB); memset(buf,1,NB);
  pthread_t th[64];
  double t0=now();
  for(long i=0;i<NT;i++) pthread_create(&th[i],0,rd,(void*)i);
  for(int i=0;i<NT;i++) pthread_join(th[i],0);
  double dt=now()-t0;
  printf("threads=%2d cpus=", NT); for(int i=0;i<NT;i++)printf("%d,",cpus[i]);
  printf("  %.1f GB/s  (%.1f GiB in %.3fs)\n", (double)NB*10/dt/1e9, NB*10/1073741824.0, dt);
  return 0;
}
