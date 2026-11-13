#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;}}while(0)
int main(){
 const size_t n=2u<<20;void *mapped,*pinned,*gpu;void* plain=malloc(n);memset(plain,0x5a,n);
 CK(hipSetDevice(0));CK(hipHostMalloc(&mapped,n,hipHostMallocMapped|hipHostMallocPortable));CK(hipHostMalloc(&pinned,n,hipHostMallocPortable));CK(hipMalloc(&gpu,n));
 memcpy(mapped,plain,n);memcpy(pinned,plain,n);
 CK(hipMemcpy(gpu,mapped,n,hipMemcpyHostToDevice));memset(plain,0,n);CK(hipMemcpy(plain,gpu,n,hipMemcpyDeviceToHost));
 for(size_t i=0;i<n;++i)if(((unsigned char*)plain)[i]!=0x5a)return 3;
 CK(hipHostFree(mapped));CK(hipHostFree(pinned));CK(hipFree(gpu));free(plain);
 puts("Data checksum passed. Expected HIP H2D/D2H 2097152 each, mapped staging 2097152 bytes in one call.");
}
