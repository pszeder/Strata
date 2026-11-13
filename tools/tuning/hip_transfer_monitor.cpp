#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint64_t* counts=nullptr;
static thread_local int nesting=0;
struct Range { uintptr_t start; size_t bytes; };
static Range mapped_ranges[512];
static int ranges_lock=0;
static void lock_ranges(){while(__atomic_exchange_n(&ranges_lock,1,__ATOMIC_ACQUIRE)){} }
static void unlock_ranges(){__atomic_store_n(&ranges_lock,0,__ATOMIC_RELEASE);}
static void track_range(void *p,size_t n,bool remove){
 if(!counts)return;
 lock_ranges();int empty=-1;
 for(int i=0;i<512;++i){if(mapped_ranges[i].start==(uintptr_t)p){mapped_ranges[i]={remove?0:(uintptr_t)p,remove?0:n};unlock_ranges();return;}if(!mapped_ranges[i].start&&empty<0)empty=i;}
 if(!remove&&empty>=0)mapped_ranges[empty]={(uintptr_t)p,n};
 else if(!remove)__atomic_fetch_add(counts+10,1ull,__ATOMIC_RELAXED);
 unlock_ranges();
}
extern "C" void* memcpy(void *dst,const void *src,size_t n){
 using F=void*(*)(void*,const void*,size_t);static F real_fn=nullptr;static int resolving=0;
 F f=__atomic_load_n(&real_fn,__ATOMIC_ACQUIRE);
 if(!f){
  if(__atomic_exchange_n(&resolving,1,__ATOMIC_ACQUIRE)==0){f=(F)dlsym(RTLD_NEXT,"memcpy");__atomic_store_n(&real_fn,f,__ATOMIC_RELEASE);__atomic_store_n(&resolving,0,__ATOMIC_RELEASE);}
  if(!f){volatile unsigned char *d=(volatile unsigned char*)dst;const volatile unsigned char *s=(const volatile unsigned char*)src;for(size_t i=0;i<n;++i)d[i]=s[i];return dst;}
 }
 void *ret=f(dst,src,n);
 if(counts&&nesting==0&&n>=(1u<<20)){
  const uintptr_t d=(uintptr_t)dst;bool found=false;
  lock_ranges();for(int i=0;i<512;++i){const auto &r=mapped_ranges[i];if(r.start&&d>=r.start&&d-r.start<=r.bytes&&n<=r.bytes-(d-r.start)){found=true;break;}}unlock_ranges();
  if(found){__atomic_fetch_add(counts+7,(uint64_t)n,__ATOMIC_RELAXED);__atomic_fetch_add(counts+9,1ull,__ATOMIC_RELAXED);}
 }
 return ret;
}
extern "C" hipError_t hipHostMalloc(void **ptr,size_t bytes,unsigned flags){
 using F=hipError_t(*)(void**,size_t,unsigned);static F f=(F)dlsym(RTLD_NEXT,"hipHostMalloc");++nesting;hipError_t e=f(ptr,bytes,flags);--nesting;
 if(e==hipSuccess&&(flags&hipHostMallocMapped))track_range(*ptr,bytes,false);return e;
}
extern "C" hipError_t hipHostFree(void *ptr){
 using F=hipError_t(*)(void*);static F f=(F)dlsym(RTLD_NEXT,"hipHostFree");++nesting;hipError_t e=f(ptr);--nesting;
 if(e==hipSuccess)track_range(ptr,0,true);return e;
}
__attribute__((constructor)) static void setup(){
 const char* path=getenv("STRATA_TRANSFER_COUNTER_FILE");if(!path)return;
 char exe[4096]={};ssize_t n=readlink("/proc/self/exe",exe,sizeof(exe)-1);if(n<0)return;
 if(!strstr(exe,"strata-dsv41")&&!strstr(exe,"maya-ds4")&&!strstr(exe,"strata-transfer-probe"))return;
 int fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0600);if(fd<0)return;
 if(ftruncate(fd,4096)){close(fd);return;}void* p=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
 if(p==MAP_FAILED)return;counts=(uint64_t*)p;counts[0]=0x5354524154414354ull;counts[1]=getpid();counts[8]=2;
}
static void add(hipMemcpyKind k,size_t bytes){if(!counts)return;unsigned i=k==hipMemcpyHostToDevice?2:k==hipMemcpyDeviceToHost?3:k==hipMemcpyDeviceToDevice?4:5;__atomic_fetch_add(counts+i,(uint64_t)bytes,__ATOMIC_RELAXED);__atomic_fetch_add(counts+6,1ull,__ATOMIC_RELAXED);}
extern "C" hipError_t hipMemcpyAsync(void* dst,const void* src,size_t n,hipMemcpyKind kind,hipStream_t stream){
 using F=hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpyAsync");bool top=nesting++==0;hipError_t e=fn(dst,src,n,kind,stream);--nesting;if(top&&e==hipSuccess)add(kind,n);return e;
}
extern "C" hipError_t hipMemcpy(void* dst,const void* src,size_t n,hipMemcpyKind kind){
 using F=hipError_t(*)(void*,const void*,size_t,hipMemcpyKind);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpy");bool top=nesting++==0;hipError_t e=fn(dst,src,n,kind);--nesting;if(top&&e==hipSuccess)add(kind,n);return e;
}
extern "C" hipError_t hipMemcpy2DAsync(void* dst,size_t dp,const void* src,size_t sp,size_t width,size_t height,hipMemcpyKind kind,hipStream_t stream){
 using F=hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind,hipStream_t);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpy2DAsync");bool top=nesting++==0;hipError_t e=fn(dst,dp,src,sp,width,height,kind,stream);--nesting;if(top&&e==hipSuccess)add(kind,width*height);return e;
}
extern "C" hipError_t hipMemcpy2D(void* dst,size_t dp,const void* src,size_t sp,size_t width,size_t height,hipMemcpyKind kind){
 using F=hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpy2D");bool top=nesting++==0;hipError_t e=fn(dst,dp,src,sp,width,height,kind);--nesting;if(top&&e==hipSuccess)add(kind,width*height);return e;
}
extern "C" hipError_t hipMemcpyFromSymbol(void* dst,const void* symbol,size_t count,size_t offset,hipMemcpyKind kind){
 using F=hipError_t(*)(void*,const void*,size_t,size_t,hipMemcpyKind);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpyFromSymbol");bool top=nesting++==0;hipError_t e=fn(dst,symbol,count,offset,kind);--nesting;if(top&&e==hipSuccess)add(kind,count);return e;
}
extern "C" hipError_t hipMemcpyToSymbol(const void* symbol,const void* src,size_t count,size_t offset,hipMemcpyKind kind){
 using F=hipError_t(*)(const void*,const void*,size_t,size_t,hipMemcpyKind);static F fn=(F)dlsym(RTLD_NEXT,"hipMemcpyToSymbol");bool top=nesting++==0;hipError_t e=fn(symbol,src,count,offset,kind);--nesting;if(top&&e==hipSuccess)add(kind,count);return e;
}
