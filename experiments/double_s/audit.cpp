#include <interpolatecpp/motion/double_s_trajectory.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <string_view>
#include <vector>

using namespace interpolatecpp::motion;
struct Case { const char* name; StateParams state; double v=0.5,a=1,j=5; };
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  if (std::string_view(argv[1]) == "--smoke") {
    DoubleSTrajectory p({0,0.8,0,0},TrajectoryBounds(0.5,1,5));
    const auto phases=p.phase_durations();
    return phases.size()==6 && p.duration()>0 &&
      p.evaluate(0).position==0 && p.evaluate(p.duration()).position==0.8 ? 0 : 1;
  }
  std::ofstream out(argv[1]); out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "case,status,duration_s,phase_sum_s,endpoint_error_rad,peak_v_rad_s,peak_a_rad_s2,peak_j_rad_s3,peak_2ms_step_rad,peak_2ms_velocity_rad_s,max_position_jump_rad,max_velocity_jump_rad_s,max_acceleration_jump_rad_s2,max_dq_dt_residual_rad_s\n";
  const std::vector<Case> cases{
    {"zero",{0,0,0,0}}, {"tiny",{0,1e-8,0,0}},
    {"short",{0,0.1,0,0}}, {"negative",{0,-0.1,0,0}},
    {"reversed",{0.1,0,0,0}}, {"near_limit",{2.7,2.79,0,0}},
    {"asymmetric_long",{0,0.8,0,0}}, {"asymmetric_small",{0,0.013,0,0}},
    {"sub_tick",{0,1.1e-6,0,0}}, {"infeasible_velocity",{0,0.1,0.6,0}},
    {"same_position_different_velocity",{0,0,0,0.4}},
    {"very_small_bounds",{0,0.01,0,0},1e-9,1e-9,1e-9}
  };
  int rejected=0, exceptions=0;
  for (const auto& c:cases) {
    try {
      DoubleSTrajectory p(c.state,TrajectoryBounds(c.v,c.a,c.j));
      auto phases=p.phase_durations();
      const double T=p.duration(), ta=phases.at("acceleration"), tv=phases.at("constant_velocity"),
        td=phases.at("deceleration"), j1=phases.at("jerk_acceleration"),j2=phases.at("jerk_deceleration");
      double pv=0,pa=0,pj=0,step=0,qjump=0,vjump=0,ajump=0,derivative_residual=0;
      bool finite=std::isfinite(T) && T>=0;
      auto inspect=[&](double t) {
        auto r=p.evaluate(t);
        for(double x:{r.position,r.velocity,r.acceleration,r.jerk}) finite &= std::isfinite(x);
        pv=std::max(pv,std::abs(r.velocity)); pa=std::max(pa,std::abs(r.acceleration)); pj=std::max(pj,std::abs(r.jerk));
      };
      for(int k=0;k<=20000;++k) inspect(T*k/20000.0);
      std::vector<double> edges{0,j1,ta-j1,ta,ta+tv,ta+tv+j2,ta+tv+td-j2,T-j2,T};
      std::sort(edges.begin(),edges.end());
      for(std::size_t k=1;k<edges.size();++k) {
        const double l=std::max(0.0,edges[k-1]),r=std::min(T,edges[k]);
        if(r-l<1e-9) continue;
        const double h=std::min(1e-6,(r-l)/8);
        const double a=p.evaluate(l+h).acceleration,b=p.evaluate(r-h).acceleration;
        if(a*b<0) inspect(l+h-a*(r-l-2*h)/(b-a));
        const double t=(l+r)/2;
        const double dq=(p.evaluate(t+h).position-p.evaluate(t-h).position)/(2*h);
        derivative_residual=std::max(derivative_residual,std::abs(dq-p.evaluate(t).velocity));
      }
      for(double t:edges) if(t>=0 && t<=T) {
        inspect(t); inspect(std::nextafter(t,0.0)); inspect(std::nextafter(t,T));
        const double h=std::min(1e-10,T*1e-7);
        if(t>h && t+h<T) {
          auto l=p.evaluate(t-h),r=p.evaluate(t+h);
          qjump=std::max(qjump,std::abs(r.position-l.position));
          vjump=std::max(vjump,std::abs(r.velocity-l.velocity));
          ajump=std::max(ajump,std::abs(r.acceleration-l.acceleration));
        }
      }
      if(T <= 40) {
        double prev=p.evaluate(0).position;
        for(std::size_t k=1;k<=static_cast<std::size_t>(std::ceil(T/0.002));++k) {
          double q=p.evaluate(k*0.002).position; step=std::max(step,std::abs(q-prev)); prev=q;
        }
      }
      const double endpoint=std::max(std::abs(p.evaluate(0).position-c.state.q_0),std::abs(p.evaluate(T).position-c.state.q_1));
      const bool ok=finite && endpoint<=1e-12 && pv<=c.v+1e-10 && pa<=c.a+1e-10 && pj<=c.j+1e-10 &&
        qjump<=1e-8 && vjump<=1e-8 && ajump<=1e-8 && derivative_residual<=1e-7 && step<=c.v*0.002+1e-12 && T<=40;
      rejected+=!ok;
      out << c.name << ',' << (ok?"admitted_scalar":"rejected_audit") << ',' << T << ',' << ta+tv+td << ',' << endpoint << ',' << pv << ',' << pa << ',' << pj << ',' << step << ',' << step/0.002 << ',' << qjump << ',' << vjump << ',' << ajump << ',' << derivative_residual << '\n';
    } catch(const std::exception&) { ++exceptions; out << c.name << ",upstream_exception,,,,,,,,,,,,\n"; }
  }
  if(!out) return 1;
  std::cout << "Audit completed: " << rejected << " rejected profiles; " << exceptions << " upstream exceptions.\n";
  std::cout << "Scalar admission does not establish multi-joint synchronization. No profiles sent to actuators.\n";
  return rejected || exceptions ? 2 : 0;
}
