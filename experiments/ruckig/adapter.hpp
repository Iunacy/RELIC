#pragma once
#include "profile.hpp"
#include "relic/motion.hpp"

namespace relic_experiment {
struct Generated { relic::JointTrajectory trajectory; Samples audit; };
Generated generate(const Vec&,const Vec&,const Bounds&,const std::vector<relic::JointLimit>&,
  const relic::MotionOptions&,double minimum_duration=0);
struct Preflight {
  relic::MotionStatus status=relic::MotionStatus::InvalidInput;
  std::string reason;
  std::size_t samples=0,contact_observations=0;
};
Preflight preflight(relic::Simulation&,const relic::JointTrajectory&,const Vec& start,const Vec& goal,
  const relic::MotionOptions&,const relic::ContactPolicy& policy = {},std::size_t budget=500000);
struct Run {
  Preflight admission;
  relic::MotionStatus status=relic::MotionStatus::InvalidInput;
  bool started=false,settled=false,command_completed=false;
  bool first_command_did_not_teleport=false;
  std::string reason;
  double execution_s=0,executed_command_s=0,settling_s=0,command_end_joint_error=0,command_end_cartesian_error=0;
  double final_joint_error=0,final_cartesian_error=0,peak_tracking_error=0,torque_peak=0;
  double observed_v=0,observed_sampled_a=0,observed_sampled_j=0;
  Vec final_joint_errors;
  std::size_t saturated_joint_ticks=0,contact_observations=0;
  double preflight_wall_ms=0,execution_wall_ms=0;
};
// Always preflights; no API accepting an unverified admission token.
Run execute(relic::Simulation&,const relic::JointTrajectory&,const Vec&,const Vec&,
  const relic::MotionOptions&,const relic::ContactPolicy& policy = {},relic::TraceWriter* trace=nullptr,
  std::size_t collision_budget=500000);
Samples linear_diagnostics(const relic::JointTrajectory&);
}
