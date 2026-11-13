// Experimental single-NUMA GGUF CPU flow, adapted from Project Maya PR #928
// (MIT, Peasant Smith and contributors), head 9ddb86cf7479a08d2ac129b8d835e7a2644f840c.
// Original row kernels and quantizers are retained. This header is shared by the
// engine port and the standalone full-pipeline numerical test.
#pragma once
#include "ds4_gguf_cpu.hpp"
#include "ds4_pool.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
namespace strata::ds4::iq_flow {
namespace gg = strata::ds4::gguf;
struct Flow {
    std::unique_ptr<std::atomic<int>[]> counters;
    size_t allocated=0;
    unsigned long long calls=0, checked=0, nt_mask=0;
    std::vector<float> cx,ch,cy;
    std::vector<unsigned char> cq,cxq;
    ~Flow() { if(calls) std::fprintf(stderr,"ds4: experimental IQ flow totals calls=%llu checked=%llu nt_mask=0x%llx\n",calls,checked,nt_mask); }
    static void wait_for(std::atomic<int>& flag,int value) {
        if(flag.load(std::memory_order_acquire)>=value) return;
        unsigned spins=0; const auto start=std::chrono::steady_clock::now();
        while(flag.load(std::memory_order_acquire)<value) {
            cpu_relax();
            if((++spins&0xfffff)==0 && std::chrono::steady_clock::now()-start>std::chrono::seconds(5)) {
                std::fprintf(stderr,"ds4: IQ flow dependency timeout; aborting, no expert omission\n"); std::abort();
            }
        }
    }
    template<class Entries,class Weight,class Run>
    static void reference(const gg::IqCpu& f,int E,int FF,int window,int RB,const float* x,size_t ldx,
                          unsigned char* aq,float* h,float* y,const Entries& ents,const int* ord,float limit,
                          Weight weight,Run run) {
        bool seen[16]={};
        for(const auto& en:ents) for(int i=0;i<en.n;++i) if(!seen[en.tok[i]]) {
            int t=en.tok[i];seen[t]=true;gg::iq_cpu_quant(f.gu_act,x+size_t(t)*ldx,E,aq+size_t(t)*E*4);
        }
        const int GB=(FF+RB-1)/RB,DB=(E+RB-1)/RB,ne=int(ents.size());
        std::function<void(int)> gu=[&](int j) {
            const auto& en=ents[size_t(ord?ord[j/GB]:j/GB)];int r0=(j%GB)*RB,r1=std::min(FF,r0+RB);
            const void* ap[16];float* hp[16];
            for(int i=0;i<en.n;++i){ap[i]=aq+size_t(en.tok[i])*E*4;hp[i]=h+size_t(en.slot0+i)*FF;}
            gg::iq_cpu_gate_up(f,weight(en,0),weight(en,1),ap,en.n,hp,r0,r1,limit);
        };
        run(ne*GB,gu);
        for(const auto& en:ents) for(int i=0;i<en.n;++i) {
            int s=en.slot0+i;gg::iq_cpu_quant(f.d_act,h+size_t(s)*FF,FF,aq+size_t(window+s)*E*4);
        }
        std::function<void(int)> dn=[&](int j) {
            const auto& en=ents[size_t(ord?ord[j/DB]:j/DB)];int r0=(j%DB)*RB,r1=std::min(E,r0+RB);
            const void* hp[16];float* yp[16];
            for(int i=0;i<en.n;++i){hp[i]=aq+size_t(window+en.slot0+i)*E*4;yp[i]=y+size_t(en.slot0+i)*E;}
            gg::iq_cpu_down(f,weight(en,2),hp,en.n,yp,r0,r1);
        };
        run(ne*DB,dn);
    }
    template<class Entries,class Weight,class Run>
    bool execute(const gg::IqCpu& f,size_t block_bytes,int E,int FF,int window,int capacity,int RB,
                 const float* x,size_t ldx,unsigned char* aq,float* h,float* y,const Entries& ents,
                 const int* ord,float limit,Weight weight,Run run,bool check) {
        const int nb=FF/256,ne=int(ents.size());
        if(E%256 || FF%256 || nb<1 || nb>64 || window<1 || window>16 || capacity<1 || capacity>128 ||
           block_bytes!=292 || f.h_bytes!=size_t(nb)*block_bytes || f.act_bytes!=size_t(E/256)*block_bytes || ne<1 || RB<1) return false;
        bool seen[16]={},slots[128]={};int ut[16],nut=0;
        for(const auto& en:ents) {
            if(en.n<1 || en.n>16 || en.slot0<0 || en.slot0+en.n>capacity) return false;
            for(int i=0;i<en.n;++i) {
                int t=en.tok[i],s=en.slot0+i;if(t<0 || t>=window || slots[s]) return false;slots[s]=true;
                if(!seen[t]){seen[t]=true;ut[nut++]=t;}
            }
        }
        if(check) {
            cx.resize(size_t(window)*E);ch.resize(size_t(capacity)*FF);cy.resize(size_t(capacity)*E);
            cq.resize(size_t(capacity)*f.h_bytes);cxq.resize(size_t(window)*f.act_bytes);
            for(int j=0;j<nut;++j) std::memcpy(cx.data()+size_t(ut[j])*E,x+size_t(ut[j])*ldx,size_t(E)*4);
        }
        const size_t need=size_t(capacity)*(nb+1)+window;
        if(allocated<need){counters.reset(new std::atomic<int>[need]);allocated=need;}
        for(size_t j=0;j<need;++j)counters[j].store(0,std::memory_order_relaxed);
        auto* B=counters.get();auto* Dn=B+size_t(capacity)*nb;auto* Fx=Dn+capacity;
        const int GB=(FF+RB-1)/RB,DB=(E+RB-1)/RB,Jg=ne*GB;
        int expected[64]={};for(int j=0;j<GB;++j) for(int b=j*RB/256;b<nb && b<=(std::min(FF,(j+1)*RB)-1)/256;++b)++expected[b];
        std::function<void(int)> jobs=[&](int j) {
            if(j<nut){int t=ut[j];gg::iq_cpu_quant(f.gu_act,x+size_t(t)*ldx,E,aq+size_t(t)*E*4);Fx[t].store(1,std::memory_order_release);return;}
            j-=nut;
            if(j<Jg) {
                const auto& en=ents[size_t(ord?ord[j/GB]:j/GB)];int r0=(j%GB)*RB,r1=std::min(FF,r0+RB);
                const void* ap[16];float* hp[16];
                for(int i=0;i<en.n;++i){wait_for(Fx[en.tok[i]],1);ap[i]=aq+size_t(en.tok[i])*E*4;hp[i]=h+size_t(en.slot0+i)*FF;}
                gg::iq_cpu_gate_up(f,weight(en,0),weight(en,1),ap,en.n,hp,r0,r1,limit);
                for(int i=0;i<en.n;++i) {
                    int s=en.slot0+i;for(int b=r0/256;b<nb && b<=(r1-1)/256;++b)
                        if(B[size_t(s)*nb+b].fetch_add(1,std::memory_order_acq_rel)==expected[b]-1) {
                            gg::iq_cpu_quant(f.d_act,h+size_t(s)*FF+size_t(b)*256,256,aq+size_t(window+s)*E*4+size_t(b)*block_bytes);
                            Dn[s].fetch_add(1,std::memory_order_acq_rel);
                        }
                }
            } else {
                j-=Jg;const auto& en=ents[size_t(ord?ord[j/DB]:j/DB)];int r0=(j%DB)*RB,r1=std::min(E,r0+RB);
                const void* hp[16];float* yp[16];
                for(int i=0;i<en.n;++i){int s=en.slot0+i;wait_for(Dn[s],nb);hp[i]=aq+size_t(window+s)*E*4;yp[i]=y+size_t(s)*E;}
                gg::iq_cpu_down(f,weight(en,2),hp,en.n,yp,r0,r1);
            }
        };
        run(nut+Jg+ne*DB,jobs);
        if(!calls) std::fprintf(stderr,"ds4: experimental IQ flow active: single NUMA, Q8_K 292-byte blocks, RB=%d, checker=%d\n",RB,int(check));
        ++calls;for(const auto& en:ents)nt_mask|=1ull<<en.n;
        if(check) {
            for(int j=0;j<nut;++j)std::memcpy(cxq.data()+size_t(ut[j])*f.act_bytes,aq+size_t(ut[j])*E*4,f.act_bytes);
            for(const auto& en:ents)for(int i=0;i<en.n;++i){int s=en.slot0+i;std::memcpy(ch.data()+size_t(s)*FF,h+size_t(s)*FF,size_t(FF)*4);std::memcpy(cy.data()+size_t(s)*E,y+size_t(s)*E,size_t(E)*4);std::memcpy(cq.data()+size_t(s)*f.h_bytes,aq+size_t(window+s)*E*4,f.h_bytes);}
            reference(f,E,FF,window,RB,cx.data(),E,aq,h,y,ents,ord,limit,weight,run);
            bool bad=false;
            for(int j=0;j<nut;++j)bad|=std::memcmp(cxq.data()+size_t(ut[j])*f.act_bytes,aq+size_t(ut[j])*E*4,f.act_bytes)!=0;
            for(const auto& en:ents)for(int i=0;i<en.n;++i){int s=en.slot0+i;bad|=std::memcmp(ch.data()+size_t(s)*FF,h+size_t(s)*FF,size_t(FF)*4)!=0;bad|=std::memcmp(cy.data()+size_t(s)*E,y+size_t(s)*E,size_t(E)*4)!=0;bad|=std::memcmp(cq.data()+size_t(s)*f.h_bytes,aq+size_t(window+s)*E*4,f.h_bytes)!=0;}
            if(bad){std::fprintf(stderr,"ds4: IQ flow exact check FAILED after %llu successful checks; aborting\n",checked);std::abort();}
            ++checked;
        }
        return true;
    }
};
}
