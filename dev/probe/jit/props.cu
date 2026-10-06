#include <cuda.h>
#include <stdio.h>
int main(){
  cuInit(0);
  CUdevice d; cuDeviceGet(&d,0);
  char name[256]; cuDeviceGetName(name,256,d);
  int v; cuDriverGetVersion(&v);
  printf("name=%s driverVersion=%d\n",name,v);
  struct { const char*n; CUdevice_attribute a;} at[] = {
   {"CC_MAJOR",CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
   {"CC_MINOR",CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
   {"SM_COUNT",CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
   {"CLOCK_KHZ",CU_DEVICE_ATTRIBUTE_CLOCK_RATE},
   {"MEMCLOCK_KHZ",CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE},
   {"BUSWIDTH",CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
   {"L2_BYTES",CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
   {"SHMEM_PER_SM",CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
   {"REGS_PER_SM",CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR},
   {"UNIFIED_ADDR",CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
   {"CAN_MAP_HOST",CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY},
   {"ASYNC_ENGINES",CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
   {"INTEGRATED",CU_DEVICE_ATTRIBUTE_INTEGRATED},
   {"MAXTHREADS_SM",CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
  };
  for(unsigned i=0;i<sizeof(at)/sizeof(at[0]);i++){int val;cuDeviceGetAttribute(&val,at[i].a,d);printf("%-16s %d\n",at[i].n,val);}
  size_t tot; cuDeviceTotalMem(&tot,d); printf("TOTAL_MEM_MB %zu\n",tot/1048576);
  CUcontext c; cuCtxCreate(&c,0,d);
  size_t f,t; cuMemGetInfo(&f,&t); printf("FREE_MB %zu TOTAL_MB %zu\n",f/1048576,t/1048576);
  return 0;
}
