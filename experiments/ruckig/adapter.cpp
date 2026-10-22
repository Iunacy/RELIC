#include "adapter.hpp"
#include "relic/feasibility.hpp"
#include <chrono>
#include <limits>

namespace relic_experiment {
using namespace relic;
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
double distance(const std::array<double,3>& a,const std::array<double,3>& b) {
  double d=0;for(std::size_t k=0;k<3;++k)d=std::hypot(d,a[k]-b[k]);return d;
}
bool valid_options(const MotionOptions& o,std::size_t n) {
  if(o.max_velocity_rad_s.size()!=n || !o.max_trajectory_steps || o.max_trajectory_steps>100000) return false;
  for(double x:o.max_velocity_rad_s) if(!finite(x) || x<=0) return false;
  for(double x:{o.max_command_step_rad,o.collision_step_rad,o.timeout_s,o.joint_tolerance_rad,
    o.cartesian_tolerance_m,o.settling_velocity_rad_s,o.settling_dwell_s}) if(!finite(x) || x<=0) return false;
  return o.timeout_s>=dt && o.timeout_s/dt<=1000000;
}
}
Generated generate(const Vec& start,const Vec& goal,const Bounds& b,const std::vector<JointLimit>& limits,
  const MotionOptions& o,double minimum) {
  require(valid_options(o,start.size()),"invalid_input","invalid motion options");
  require(b.velocity.size()==start.size(),"invalid_input","bad velocity dimensions");
  for(std::size_t j=0;j<start.size();++j)
    require(b.velocity[j]<=o.max_velocity_rad_s[j],"invalid_input","Ruckig velocity exceeds RELIC motion contract");
  std::vector<Interval> intervals;
  for(auto x:limits) intervals.push_back({x.lower,x.upper});
  auto audit=solve(start,goal,b,intervals,o.max_command_step_rad,o.max_trajectory_steps,minimum);
  JointTrajectory t; t.timestep_s=dt; t.command_duration_s=audit.duration;t.commands=std::move(audit.q);
  for(std::size_t j=0;j<start.size();++j) t.nominal_velocity_rad_s.push_back(t.command_duration_s?(goal[j]-start[j])/t.command_duration_s:0);
  require(validate_joint_path(t.commands,limits,o.max_command_step_rad)==PathStatus::Valid,"invalid_path","RELIC rejected Ruckig command path");
  return {std::move(t),std::move(audit)};
}
Preflight preflight(Simulation& sim,const JointTrajectory& t,const Vec& start,const Vec& goal,
  const MotionOptions& o,const ContactPolicy& policy,std::size_t budget) {
  Preflight out;
  const auto fail=[&](MotionStatus status,std::string reason) {out.status=status;out.reason=std::move(reason);return out;};
  const auto limits=sim.joint_limits();
  if(!valid_options(o,limits.size()) || !budget || budget>2000000 || start.size()!=limits.size() || goal.size()!=limits.size() ||
    !validate_contact_policy(sim.model(),policy).empty() || sim.model().opt.timestep!=dt || t.timestep_s!=dt)
    return fail(MotionStatus::InvalidInput,"invalid preflight options/model/timestep");
  if(t.commands.empty() || t.commands.size()-1>o.max_trajectory_steps)
    return fail(MotionStatus::ResourceLimit,"command budget exceeded or empty path");
  if(!finite(t.command_duration_s) || std::abs(t.command_duration_s-static_cast<double>(t.commands.size()-1)*dt)>1e-10 ||
    validate_joint_path(t.commands,limits,o.max_command_step_rad)!=PathStatus::Valid)
    return fail(MotionStatus::InvalidPath,"RELIC joint path validation failed");
  for(std::size_t j=0;j<limits.size();++j) {
    if(!finite(start[j]) || !finite(goal[j]) || start[j]<limits[j].lower || start[j]>limits[j].upper ||
      goal[j]<limits[j].lower || goal[j]>limits[j].upper || std::abs(t.commands.front()[j]-start[j])>1e-8 ||
      std::abs(t.commands.back()[j]-goal[j])>1e-8)
      return fail(MotionStatus::InvalidPath,"start/goal mismatch");
    for(std::size_t k=1;k<t.commands.size();++k)
      if(std::abs(t.commands[k][j]-t.commands[k-1][j])/dt>o.max_velocity_rad_s[j]+1e-9)
        return fail(MotionStatus::InvalidPath,"sampled command velocity exceeded");
  }
  try {
    for(std::size_t k=0;k<t.commands.size();++k) {
      long double count=1;
      if(k) for(std::size_t j=0;j<limits.size();++j)
        count=std::max(count,std::ceil(std::abs(static_cast<long double>(t.commands[k][j])-t.commands[k-1][j])/o.collision_step_rad));
      if(!std::isfinite(count) || count>budget-out.samples)
        return fail(MotionStatus::ResourceLimit,"collision subdivision budget exhausted");
      const auto subdivisions=static_cast<std::size_t>(count);
      for(std::size_t s=1;s<=subdivisions;++s) {
        Vec q=t.commands[k];
        if(k && s!=subdivisions) for(std::size_t j=0;j<q.size();++j)
          q[j]=static_cast<double>(static_cast<long double>(t.commands[k-1][j])+
            (static_cast<long double>(t.commands[k][j])-t.commands[k-1][j])*s/subdivisions);
        sim.evaluate_fk(q); ++out.samples;
        if(!numerical_warnings(sim.planning_data()).empty()) return fail(MotionStatus::NumericalWarning,"planning warning");
        auto contacts=inspect_contacts(sim.model(),sim.planning_data(),policy);
        out.contact_observations+=contacts.size();
        if(has_disallowed_contact(contacts)) return fail(MotionStatus::DisallowedContact,"sampled command/subdivision contact");
      }
    }
  } catch(const std::exception& e) {return fail(MotionStatus::SimulationError,e.what());}
  return fail(MotionStatus::Success,"sampled preflight admitted; no continuous collision guarantee");
}
Run execute(Simulation& sim,const JointTrajectory& t,const Vec& start,const Vec& goal,const MotionOptions& o,
  const ContactPolicy& policy,TraceWriter* trace,std::size_t budget,const Observer& observer) {
  Run out;
  const auto preflight_start=Clock::now(); out.admission=preflight(sim,t,start,goal,o,policy,budget);
  out.preflight_wall_ms=elapsed(preflight_start);out.status=out.admission.status;
  if(out.status!=MotionStatus::Success) return out;
  const auto execution_start=Clock::now();
  const auto finish=[&](MotionStatus status) {
    out.status=status;out.execution_wall_ms=elapsed(execution_start);
    if(out.reason.empty())out.reason=motion_status_name(status);
    out.settling_s=std::max(0.0,out.execution_s-out.executed_command_s);return out;
  };
  Vec previous_velocity,previous_acceleration(goal.size(),0);
  bool have_acceleration=false;
  const auto target=sim.evaluate_fk(goal).position_m;
  auto capture=[&](bool physics) {
    auto health=execution_health(sim.model(),sim.execution_data());
    const auto snap=sim.snapshot();out.execution_s=snap.time_s;
    out.final_joint_error=0;out.final_joint_errors.clear();
    out.final_cartesian_error=distance(snap.tool.position_m,target);
    for(std::size_t j=0;j<goal.size();++j) {
      const double error=goal[j]-snap.observed_rad[j];out.final_joint_errors.push_back(error);
      out.final_joint_error=std::max(out.final_joint_error,std::abs(error));
      out.peak_tracking_error=std::max(out.peak_tracking_error,std::abs(snap.command_rad[j]-snap.observed_rad[j]));
      out.observed_v=std::max(out.observed_v,std::abs(snap.velocity_rad_s[j]));
      const double force=snap.actuator_torque_Nm[j];out.torque_peak=std::max(out.torque_peak,std::abs(force));
      if(!finite(force) || !finite(out.final_cartesian_error)) health=MotionStatus::SimulationError;
      const auto& m=sim.model();
      if(physics && m.actuator_forcelimited[j]) {
        const double lo=m.actuator_forcerange[2*j],hi=m.actuator_forcerange[2*j+1];
        const double eps=1e-8*std::max({1.0,std::abs(lo),std::abs(hi)});
        if(force<=lo+eps || force>=hi-eps) ++out.saturated_joint_ticks;
      }
      if(physics && !previous_velocity.empty()) {
        const double acceleration=(snap.velocity_rad_s[j]-previous_velocity[j])/dt;
        out.observed_sampled_a=std::max(out.observed_sampled_a,std::abs(acceleration));
        if(have_acceleration) out.observed_sampled_j=std::max(out.observed_sampled_j,std::abs(acceleration-previous_acceleration[j])/dt);
        previous_acceleration[j]=acceleration;
      }
    }
    if(physics) {previous_velocity=snap.velocity_rad_s;have_acceleration=true;}
    const auto contacts=inspect_contacts(sim.model(),sim.execution_data(),policy);
    out.contact_observations+=contacts.size();
    if(health==MotionStatus::Success && has_disallowed_contact(contacts))health=MotionStatus::DisallowedContact;
    return health;
  };
  const auto max_ticks=static_cast<std::size_t>(std::floor(static_cast<long double>(o.timeout_s)/dt+1e-12L));
  std::size_t ticks=0,dwell=0;
  try {
    sim.reset(start);out.started=true;previous_velocity=sim.snapshot().velocity_rad_s;
    auto health=capture(false);if(health!=MotionStatus::Success)return finish(health);
    if(trace)trace->append(sim.snapshot());
    if(observer)observer(sim,0,"initial",t.commands.size()==1);
    for(std::size_t k=1;k<t.commands.size();++k) {
      if(ticks>=max_ticks)return finish(MotionStatus::Timeout);
      sim.set_position_commands(t.commands[k]);health=capture(false);
      if(k==1)out.first_command_did_not_teleport=sim.snapshot().observed_rad==start;
      if(health!=MotionStatus::Success)return finish(health);
      if(observer)observer(sim,ticks+1,"pre",false);
      sim.step();++ticks;out.executed_command_s=static_cast<double>(ticks)*dt;
      health=capture(true);if(trace)trace->append(sim.snapshot());
      if(observer)observer(sim,ticks,"post",false);
      if(health!=MotionStatus::Success)return finish(health);
    }
    out.command_completed=true;out.command_end_joint_error=out.final_joint_error;out.command_end_cartesian_error=out.final_cartesian_error;
    // Hold the exact last sampled command. Never replace it with a goal clamp.
    while(ticks<max_ticks) {
      sim.set_position_commands(t.commands.back());health=capture(false);
      if(health!=MotionStatus::Success)return finish(health);
      if(observer)observer(sim,ticks+1,"pre",true);
      sim.step();++ticks;health=capture(true);if(trace)trace->append(sim.snapshot());
      if(observer)observer(sim,ticks,"post",true);
      if(health!=MotionStatus::Success)return finish(health);
      bool acceptable=out.final_joint_error<=o.joint_tolerance_rad && out.final_cartesian_error<=o.cartesian_tolerance_m;
      for(double v:sim.snapshot().velocity_rad_s)acceptable &= std::abs(v)<=o.settling_velocity_rad_s;
      dwell=acceptable?dwell+1:0;
      if(static_cast<double>(dwell)*dt+1e-12>=o.settling_dwell_s) {out.settled=true;return finish(MotionStatus::Success);}
    }
    return finish(MotionStatus::NonConvergence);
  } catch(const std::exception& e) {
    out.reason=e.what();
    return finish(numerical_warnings(sim.execution_data()).empty()?MotionStatus::SimulationError:MotionStatus::NumericalWarning);
  }
}
Samples linear_diagnostics(const JointTrajectory& t) {
  Samples out;out.duration=t.command_duration_s;
  for(double v:t.nominal_velocity_rad_s)out.continuous_v=std::max(out.continuous_v,std::abs(v));
  out.continuous_a=out.continuous_j=t.acceleration_discontinuous?std::numeric_limits<double>::infinity():0;
  std::vector<Vec> padded(3,t.commands.front());padded.insert(padded.end(),t.commands.begin(),t.commands.end());padded.insert(padded.end(),3,t.commands.back());
  for(std::size_t k=3;k<padded.size();++k)for(std::size_t j=0;j<padded[k].size();++j) {
    out.max_step=std::max(out.max_step,std::abs(padded[k][j]-padded[k-1][j]));
    out.sampled_v=std::max(out.sampled_v,std::abs(padded[k][j]-padded[k-1][j])/dt);
    out.sampled_a=std::max(out.sampled_a,std::abs(padded[k][j]-2*padded[k-1][j]+padded[k-2][j])/(dt*dt));
    out.sampled_j=std::max(out.sampled_j,std::abs(padded[k][j]-3*padded[k-1][j]+3*padded[k-2][j]-padded[k-3][j])/(dt*dt*dt));
  }
  return out;
}
}
