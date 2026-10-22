#pragma once
#include <ruckig/ruckig.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace relic_experiment {
constexpr double dt=0.002;
using Vec=std::vector<double>;
struct Bounds { Vec velocity,acceleration,jerk; };
struct Interval { double lower,upper; };
struct Samples {
  std::vector<Vec> q;
  double duration=0,endpoint_error=0,continuous_v=0,continuous_a=0,continuous_j=0;
  double sampled_v=0,sampled_a=0,sampled_j=0,max_step=0,sync_error=0;
  int solver_result=0;
};
struct Rejection:std::runtime_error {
  std::string category;
  Rejection(std::string kind,std::string reason):std::runtime_error(reason),category(std::move(kind)) {}
};
inline void require(bool condition,const std::string& category,const std::string& message) {
  if(!condition) throw Rejection(category,message);
}
inline bool finite(double x) { return std::isfinite(x); }
inline void validate_request(const Vec& start,const Vec& goal,const Bounds& b,
  const std::vector<Interval>& limits,double step,std::size_t budget,double minimum) {
  const auto n=start.size();
  require((n==3 || n==4) && goal.size()==n && b.velocity.size()==n &&
    b.acceleration.size()==n && b.jerk.size()==n && limits.size()==n,"invalid_input","bad dimensions");
  require(finite(step) && step>0 && budget>0 && budget<=100000 && finite(minimum) && minimum>=0,
    "invalid_input","bad step/budget/minimum duration");
  for(std::size_t j=0;j<n;++j) {
    require(finite(start[j]) && finite(goal[j]) && finite(limits[j].lower) && finite(limits[j].upper) &&
      limits[j].lower<limits[j].upper && start[j]>=limits[j].lower && start[j]<=limits[j].upper &&
      goal[j]>=limits[j].lower && goal[j]<=limits[j].upper,"invalid_input","nonfinite or out-of-range position");
    for(double x:{b.velocity[j],b.acceleration[j],b.jerk[j]})
      require(finite(x) && x>0 && x<1e9,"invalid_input","finite positive constraints below 1e9 required");
  }
}
inline Samples solve(const Vec& start,const Vec& goal,const Bounds& b,const std::vector<Interval>& limits,
  double max_step=0.005,std::size_t budget=20000,double minimum_duration=0) {
  validate_request(start,goal,b,limits,max_step,budget,minimum_duration);
  const auto n=start.size();
  ruckig::Ruckig<0> solver(n,dt);
  ruckig::InputParameter<0> in(n);
  ruckig::Trajectory<0> trajectory(n);
  in.current_position=start; in.target_position=goal;
  in.max_velocity=b.velocity; in.max_acceleration=b.acceleration; in.max_jerk=b.jerk;
  in.synchronization=ruckig::Synchronization::Time;
  in.duration_discretization=ruckig::DurationDiscretization::Discrete;
  if(minimum_duration>0) in.minimum_duration=minimum_duration;
  require(solver.validate_input<false>(in,true,true),"upstream_invalid","upstream validation rejected request");
  const auto result=solver.calculate(in,trajectory);
  if(result==ruckig::Result::ErrorTrajectoryDuration)
    throw Rejection("range_limit","Ruckig ErrorTrajectoryDuration (-101)");
  require(result==ruckig::Result::Working,"solver_error","Ruckig result "+std::to_string(static_cast<int>(result)));
  Samples out; out.solver_result=static_cast<int>(result); out.duration=trajectory.get_duration();
  require(finite(out.duration) && out.duration>=0,"numerical_failure","invalid duration");
  require(out.duration<=7000,"range_limit","duration outside upstream guaranteed range");
  require(out.duration+1e-10>=minimum_duration,"numerical_failure","minimum duration not respected");
  const double tick_count=std::round(out.duration/dt);
  require(std::abs(tick_count*dt-out.duration)<=1e-10,"numerical_failure","duration not on 2ms grid");
  require(tick_count<=static_cast<double>(budget),"resource_limit","command tick budget exceeded");
  const auto ticks=static_cast<std::size_t>(tick_count);
  const auto profiles=trajectory.get_profiles();
  require(profiles.size()==1,"numerical_failure","unexpected sections");
  // Audit every constant-jerk polynomial analytically, including velocity extrema.
  // Rest-to-rest inputs need no brake phase; reject any unexpected brake.
  for(std::size_t j=0;j<n;++j) {
    const auto& p=profiles[0][j];
    require(p.brake.duration==0,"numerical_failure","unexpected brake phase");
    if(start[j]!=goal[j]) {
      out.sync_error=std::max(out.sync_error,std::abs(p.t_sum.back()-out.duration));
      require(finite(p.t_sum.back()) && out.sync_error<=1e-9,"numerical_failure","joint completion not synchronized");
    }
    double phase_time=0;
    for(std::size_t k=0;k<7;++k) {
      const double h=p.t[k],jerk=p.j[k];
      require(finite(h) && h>=0 && finite(jerk),"numerical_failure","invalid phase");
      phase_time+=h;
      require(finite(p.t_sum[k]) && std::abs(phase_time-p.t_sum[k])<=1e-10,"numerical_failure","phase time mismatch");
      const auto check=[&](double t) {
        const auto [q,v,a]=ruckig::integrate(t,p.p[k],p.v[k],p.a[k],jerk);
        require(finite(q) && finite(v) && finite(a),"numerical_failure","nonfinite polynomial");
        require(q>=limits[j].lower-1e-10 && q<=limits[j].upper+1e-10,"joint_limit","continuous position limit");
        require(std::abs(v)<=b.velocity[j]+1e-10 && std::abs(a)<=b.acceleration[j]+1e-10 &&
          std::abs(jerk)<=b.jerk[j]+1e-10,"numerical_failure","continuous derivative bound");
        out.continuous_v=std::max(out.continuous_v,std::abs(v));
        out.continuous_a=std::max(out.continuous_a,std::abs(a));
        if(h>0) out.continuous_j=std::max(out.continuous_j,std::abs(jerk));
      };
      check(0); check(h);
      if(jerk!=0) {const double t=-p.a[k]/jerk; if(t>0 && t<h) check(t);}
      const auto [q,v,a]=ruckig::integrate(h,p.p[k],p.v[k],p.a[k],jerk);
      require(std::abs(q-p.p[k+1])<=1e-9 && std::abs(v-p.v[k+1])<=1e-9 &&
        std::abs(a-p.a[k+1])<=1e-9,"numerical_failure","discontinuous phase boundary");
    }
    const auto ext=p.get_position_extrema();
    require(finite(ext.min) && finite(ext.max) && ext.min>=limits[j].lower-1e-10 &&
      ext.max<=limits[j].upper+1e-10,"joint_limit","position extrema outside limits");
  }
  Vec q(n),v(n),a(n),jerk(n); std::size_t section=0;
  double previous_time=-1;
  for(std::size_t k=0;k<=ticks;++k) {
    const double t=static_cast<double>(k)*dt;
    require(t>previous_time,"numerical_failure","nonmonotonic timestamp"); previous_time=t;
    trajectory.at_time(t,q,v,a,jerk,section);
    for(std::size_t j=0;j<n;++j) {
      require(finite(q[j]) && finite(v[j]) && finite(a[j]) && finite(jerk[j]),"numerical_failure","nonfinite sample");
      require(q[j]>=limits[j].lower && q[j]<=limits[j].upper,"joint_limit","sample outside exact joint interval");
      require(std::abs(v[j])<=b.velocity[j]+1e-10 && std::abs(a[j])<=b.acceleration[j]+1e-10 &&
        std::abs(jerk[j])<=b.jerk[j]+1e-10,"numerical_failure","sample derivative bound");
      if(k==0 || k==ticks) {
        out.endpoint_error=std::max(out.endpoint_error,std::abs(q[j]-(k==0?start[j]:goal[j])));
        require(std::abs(v[j])<=1e-8 && std::abs(a[j])<=1e-10,"numerical_failure","endpoint not at rest");
      }
    }
    out.q.push_back(q);
  }
  require(out.endpoint_error<=1e-8,"numerical_failure","endpoint tolerance exceeded");
  // Constant extension before/after the actual samples includes boundary stencils.
  // No endpoint replacement: all stored commands are direct at_time results.
  std::vector<Vec> padded(3,out.q.front());
  padded.insert(padded.end(),out.q.begin(),out.q.end());
  padded.insert(padded.end(),3,out.q.back());
  for(std::size_t k=3;k<padded.size();++k) for(std::size_t j=0;j<n;++j) {
    const double step=padded[k][j]-padded[k-1][j];
    const double vfd=step/dt;
    const double afd=(padded[k][j]-2*padded[k-1][j]+padded[k-2][j])/(dt*dt);
    const double jfd=(padded[k][j]-3*padded[k-1][j]+3*padded[k-2][j]-padded[k-3][j])/(dt*dt*dt);
    out.max_step=std::max(out.max_step,std::abs(step));
    out.sampled_v=std::max(out.sampled_v,std::abs(vfd)); out.sampled_a=std::max(out.sampled_a,std::abs(afd));
    out.sampled_j=std::max(out.sampled_j,std::abs(jfd));
    require(std::abs(step)<=max_step+1e-12,"step_limit","command step limit exceeded");
    require(std::abs(vfd)<=b.velocity[j]+1e-9 && std::abs(afd)<=b.acceleration[j]+1e-7 &&
      std::abs(jfd)<=b.jerk[j]+1e-5,"numerical_failure","sampled finite-difference bound");
  }
  return out;
}
}
