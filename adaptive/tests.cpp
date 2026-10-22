#include "common.hpp"
#include "residual.hpp"
#include <iostream>
#include <limits>
using namespace relic;using namespace relic::adaptive;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv){try {
  if(argc!=2)return 2;
  for(int n:{3,4}){
    const auto cfg=load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg"));
    Simulation baseline(cfg),same(cfg,{});PhysicsParameters p;p.gain_scale=.5;p.force_scale=.4;p.mass_scale=1.4;p.payload_kg=.1;p.friction_Nm=.2;p.damping_scale=2;
    Simulation degraded(cfg,p);Vec q{.1,-.6,.3};if(n==4)q.push_back(.1);
    baseline.reset(q);same.reset(q);baseline.step(20);same.step(20);
    check(baseline.snapshot().observed_rad==same.snapshot().observed_rad,"default controller changed");
    check(baseline.model().actuator_gainprm[0]==200 && degraded.model().actuator_gainprm[0]==100,"isolated gain scaling");
    check(degraded.model().actuator_forcerange[1]==8 && degraded.model().dof_frictionloss[0]==.2,"physical force/friction scaling");
    const int body=mj_name2id(&baseline.model(),mjOBJ_BODY,"forearm");
    check(std::abs(degraded.model().body_mass[body]-baseline.model().body_mass[body]*1.4-.1)<1e-12,"payload/mass");
    for(int b=1;b<degraded.model().nbody;++b){auto v=degraded.model().body_inertia+3*b;check(v[0]+v[1]>=v[2]-1e-12 && v[0]+v[2]>=v[1]-1e-12 && v[1]+v[2]>=v[0]-1e-12,"inertia triangle");}
    const auto before=baseline.snapshot();check(safe_segment(baseline,q,q,true),"safe pose");
    check(before.observed_rad==baseline.snapshot().observed_rad && before.time_s==baseline.snapshot().time_s,"planning state mutated execution");
    auto bad=q;bad[0]=std::numeric_limits<double>::quiet_NaN();check(!safe_segment(baseline,q,bad,true),"nan admitted");
    bad=q;bad[0]+=.006;check(!safe_segment(baseline,q,bad,true),"slew admitted");bad[0]=4;check(!safe_segment(baseline,q,bad,false),"limit admitted");
    bad=q;bad[1]=1.0;bad[2]=0;check(!safe_segment(baseline,bad,bad,false),"ground contact admitted");
    baseline.reset(q);baseline.set_external_torques(Vec(n,0.1));baseline.step(100);baseline.reset(q);const auto reset=baseline.snapshot();
    check(baseline.execution_data().qfrc_applied[0]==0,"disturbance reset");
    check(reset.observed_rad==q && reset.velocity_rad_s==Vec(n,0) && reset.time_s==0,"reset leakage");
    bool rejected=false;try {p.mass_scale=-1;Simulation invalid(cfg,p);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid randomization");
  }

  {auto cfg=load_config(std::filesystem::path(argv[1])/"synthetic_3dof.cfg");Simulation sim(cfg);Vec q{0,-.6,.3},zero(3,0);sim.reset(q);
   Bounds b;b.magnitude.fill(.1);b.rate.fill(.001);
   auto r=guarded(sim,q,q,zero,Vec(3,.05),b);check(!r.fallback && r.applied==Vec(3,.001),"per-joint correction rate");
   auto bad=zero;bad[0]=NAN;r=guarded(sim,q,q,zero,bad,b);check(r.fallback && r.command==q && r.reason=="correction_nonfinite_or_magnitude","invalid correction fallback");
   r=guarded(sim,q,q,zero,Vec(4,0),b);check(r.fallback,"dimension fallback");
   bad=zero;bad[0]=.2;r=guarded(sim,q,q,zero,bad,b);check(r.fallback,"magnitude fallback");
   bool stop=false;try {guarded(sim,q,q,Vec(3,.05),bad,b);}catch(const std::runtime_error&){stop=true;}check(stop,"unsafe fallback rate must terminate");


   const auto f=features(cfg,1,.2,.1,.3,.1,.2,.01,.02,.03);
   check(f.size()==16 && f[3]==.1 && f[5]==.01 && f[6]==.02 && f[7]==.03,"causal feature order");
   MLP m;m.scale.fill(1);m.bounds=b;m.b3=.5;check(std::abs(m.predict(f,1)-std::tanh(.5)*.1)<1e-15,"native inference");
   auto nonfinite=f;nonfinite[0]=NAN;check(!std::isfinite(m.predict(nonfinite,0)),"nonfinite features");
  }
  Random a{42},b{42};for(int i=0;i<100;++i)check(a.next()==b.next(),"seed reproducibility");
  std::cout<<"physical randomization, isolation, reset, limits, sampled paths and defaults passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
