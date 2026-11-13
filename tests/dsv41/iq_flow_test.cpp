#include "ds4_iq_flow.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <string>
#include <numeric>
using namespace strata::ds4;
namespace gg=strata::ds4::gguf;
struct Entry {int e=0,n=0,slot0=0,tok[16]{};};
static void fill(std::vector<unsigned char>& w,int type,uint64_t seed) {
 auto next=[&]{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
 for(size_t i=0;i<w.size();++i)w[i]=unsigned(next());
 size_t bb=ggml_type_size((ggml_type)type);
 for(size_t i=0;i<w.size();i+=bb){auto d=ggml_fp32_to_fp16(.001f+.00001f*(next()%200));std::memcpy(w.data()+i,&d,2);}
}
int main(){
 ggml_cpu_init();constexpr int E=5120,F=2304,W=16,CAP=128,NE=3;unsigned total=0;uint64_t mask=0;
 for(int threads:{1,12}) {
  Pool pool(threads-1,{},2000);auto run=[&](int n,const std::function<void(int)>& fn){pool.run(n,fn);};
  for(auto types:std::vector<std::pair<int,int>>{{16,16},{17,17},{18,18},{22,22},{16,17},{17,16},{22,18}}) {
   gg::IqCpu f;std::string err;if(!gg::iq_cpu_fmt(types.first,types.second,E,F,f,err)){fprintf(stderr,"fmt %s\n",err.c_str());return 2;}
   std::vector<unsigned char> g[NE],u[NE],d[NE];
   for(int e=0;e<NE;++e){g[e].resize(f.gu_row*F);u[e].resize(f.gu_row*F);d[e].resize(f.d_row*E);fill(g[e],types.first,17+e);fill(u[e],types.first,29+e);fill(d[e],types.second,37+e);}
   auto weight=[&](const Entry& en,int r)->const unsigned char*{return r==0?g[en.e].data():r==1?u[en.e].data():d[en.e].data();};
   iq_flow::Flow flow;
   for(int nt:{1,2,3,4,5,6,8,16}) for(int reuse=0;reuse<3;++reuse) {
    std::vector<Entry> entries(NE);for(int e=0;e<NE;++e){auto& en=entries[e];en.e=e;en.n=nt;en.slot0=e==2?CAP-nt:(reuse==2?56:2)+e*(nt+2);for(int t=0;t<nt;++t)en.tok[t]=(t*3+(reuse==0?0:e*5))%W;}
    int ord[NE]={2,0,1};int RB=reuse==2?80:64;float lim=reuse==1?0.f:10.f;
    std::vector<float> x(size_t(W)*E);for(size_t i=0;i<x.size();++i){int k=int((i*73+nt*19)%2001)-1000;x[i]=reuse==0?k*.001f:reuse==1?((i%2)?1.f:-1.f):k*1e-39f;}
    std::vector<unsigned char> a0(size_t(W+CAP)*E*4+128,0xa5),a1=a0;
    std::vector<float> h0(size_t(CAP)*F+32,-123456.5f),h1=h0,y0(size_t(CAP)*E+32,-234567.5f),y1=y0;
    // Alias x and aq on alternate calls, as permitted by the legacy in-place shrink.
    bool alias=reuse==1;if(alias)for(int t=0;t<W;++t)std::memcpy(a1.data()+64+size_t(t)*E*4,x.data()+size_t(t)*E,size_t(E)*4);
    if(alias)a0=a1;
    iq_flow::Flow::reference(f,E,F,W,RB,x.data(),E,a0.data()+64,h0.data()+16,y0.data()+16,entries,ord,lim,weight,run);
    const float* input=alias?(const float*)(a1.data()+64):x.data();
    if(!flow.execute(f,292,E,F,W,CAP,RB,input,E,a1.data()+64,h1.data()+16,y1.data()+16,entries,ord,lim,weight,run,true))return 3;
    if(a0!=a1 || std::memcmp(h0.data(),h1.data(),h0.size()*4) || std::memcmp(y0.data(),y1.data(),y0.size()*4)){fprintf(stderr,"DIFFERENT types=%d/%d nt=%d threads=%d reuse=%d\n",types.first,types.second,nt,threads,reuse);return 4;}
    // Outer guards and untouched slot rows are included in the whole-buffer comparisons above.
    ++total;mask|=1ull<<nt;
   }
   printf("threads=%d types=%d/%d calls=24 NT1..6,8,16 3reuses: xQ,h-floats,hQ,Y,guards and unused slots bit-identical\n",threads,types.first,types.second);fflush(stdout);
  }
 }
 // Q8_K quantization's block independence, including zero, subnormal, tied and signed-zero inputs.
 gg::IqCpu f;std::string err;if(!gg::iq_cpu_fmt(16,16,E,F,f,err))return 5;
 for(int pattern=0;pattern<4;++pattern){std::vector<float>x(F);for(int i=0;i<F;++i)x[i]=pattern==0?0.f:pattern==1?((i%2)?-0.f:0.f):pattern==2?((i%2)?1.f:-1.f):(i%17-8)*1e-39f;std::vector<unsigned char>a(f.h_bytes),b(a.size());gg::iq_cpu_quant(f.d_act,x.data(),F,a.data());for(int k=0;k<F/256;++k)gg::iq_cpu_quant(f.d_act,x.data()+k*256,256,b.data()+k*292);if(a!=b)return 6;}
 printf("RESULT calls=%u nt_mask=0x%llx all exact, block_quant_patterns=4\n",total,(unsigned long long)mask);return 0;
}
