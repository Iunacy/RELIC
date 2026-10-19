#include "relic/motion.hpp"
#include "relic/feasibility.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>

namespace relic {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
bool positive(double x) { return std::isfinite(x) && x > 0; }
bool joints_valid(std::span<const double> q, const std::vector<JointLimit>& limits) {
  if (q.size() != limits.size()) return false;
  for (std::size_t j = 0; j < q.size(); ++j)
    if (!std::isfinite(q[j]) || q[j] < limits[j].lower || q[j] > limits[j].upper) return false;
  return true;
}
std::string option_error(std::size_t n, double dt, const MotionOptions& o) {
  if (!positive(dt) || o.max_velocity_rad_s.size() != n) return "invalid timestep or velocity dimensions";
  for (double v : o.max_velocity_rad_s) if (!positive(v)) return "velocities must be finite and positive";
  for (double x : {o.max_command_step_rad,o.collision_step_rad,o.timeout_s,o.joint_tolerance_rad,
                  o.cartesian_tolerance_m,o.settling_velocity_rad_s,o.settling_dwell_s})
    if (!positive(x)) return "motion constraints must be finite and positive";
  if (o.max_trajectory_steps < 1 || o.max_trajectory_steps > 100000 || o.timeout_s < dt ||
      static_cast<long double>(o.timeout_s)/dt > 1000000)
    return "trajectory cap must be 1..100000; timeout must permit 1..1000000 physics ticks";
  return {};
}
std::string request_error(const PositionIkRequest& r, const std::vector<JointLimit>& limits) {
  if (!joints_valid(r.current_rad,limits) || !positive(r.position_tolerance_m)) return "invalid start or IK tolerance";
  for (double x : r.position_world_m) if (!std::isfinite(x)) return "nonfinite world target";
  return {};
}
double position_error(const std::array<double,3>& a, const std::array<double,3>& b) {
  double error = 0;
  for (std::size_t k = 0; k < 3; ++k) error = std::hypot(error,a[k]-b[k]);
  return error;
}
}

const char* motion_status_name(MotionStatus s) noexcept {
  switch (s) {
#define RELIC_STATUS(x) case MotionStatus::x: return #x
    RELIC_STATUS(Success); RELIC_STATUS(InvalidInput); RELIC_STATUS(IkFailure); RELIC_STATUS(SearchIncomplete);
    RELIC_STATUS(NoFeasibleCandidate); RELIC_STATUS(InvalidCandidate); RELIC_STATUS(InvalidPath);
    RELIC_STATUS(ResourceLimit); RELIC_STATUS(DisallowedContact); RELIC_STATUS(ObservedJointLimit);
    RELIC_STATUS(NumericalWarning); RELIC_STATUS(SimulationError); RELIC_STATUS(Timeout);
    RELIC_STATUS(NonConvergence); RELIC_STATUS(TraceFailure);
#undef RELIC_STATUS
  }
  return "Unknown";
}

TrajectoryResult make_joint_trajectory(std::span<const double> start, std::span<const double> goal,
  const std::vector<JointLimit>& limits, double dt, const MotionOptions& o) {
  TrajectoryResult out;
  out.reason = option_error(limits.size(),dt,o);
  if (!out.reason.empty()) return out;
  for (const auto lim : limits) if (!std::isfinite(lim.lower) || !std::isfinite(lim.upper) || lim.lower >= lim.upper) {
    out.reason = "invalid joint constraints"; return out;
  }
  if (limits.empty() || !joints_valid(start,limits) || !joints_valid(goal,limits)) {
    out.reason = "start/destination must be finite and inside model joint intervals"; return out;
  }
  long double needed = 0;
  for (std::size_t j = 0; j < start.size(); ++j) {
    const long double delta = std::abs(static_cast<long double>(goal[j])-start[j]);
    const long double bound = std::min(static_cast<long double>(o.max_command_step_rad),static_cast<long double>(o.max_velocity_rad_s[j])*dt);
    if (bound <= 0) { out.status = MotionStatus::ResourceLimit; out.reason = "unrepresentable commanded increment"; return out; }
    needed = std::max(needed,std::ceil(delta/bound));
  }
  if (!std::isfinite(needed) || needed > o.max_trajectory_steps) {
    out.status = MotionStatus::ResourceLimit; out.reason = "trajectory exceeds configured step budget before allocation"; return out;
  }
  std::size_t steps = static_cast<std::size_t>(needed);
  std::size_t retries = 0;
  for (; steps <= o.max_trajectory_steps && retries < 8; ++steps, ++retries) {
    JointTrajectory t; t.timestep_s = dt; t.command_duration_s = static_cast<double>(steps)*dt;
    t.acceleration_discontinuous = steps != 0;
    t.commands.reserve(steps+1); t.commands.emplace_back(start.begin(),start.end());
    for (std::size_t k = 1; k <= steps; ++k) {
      std::vector<double> q(start.size());
      for (std::size_t j = 0; j < q.size(); ++j) q[j] = k == steps ? goal[j] :
        static_cast<double>(static_cast<long double>(start[j])+(static_cast<long double>(goal[j])-start[j])*k/steps);
      t.commands.push_back(std::move(q));
    }
    bool velocities_ok = true;
    for (std::size_t k = 1; k < t.commands.size(); ++k) for (std::size_t j = 0; j < start.size(); ++j)
      velocities_ok &= std::abs(t.commands[k][j]-t.commands[k-1][j])/dt <= o.max_velocity_rad_s[j];
    if (validate_joint_path(t.commands,limits,o.max_command_step_rad) != PathStatus::Valid || !velocities_ok) continue;
    for (std::size_t j = 0; j < start.size(); ++j)
      t.nominal_velocity_rad_s.push_back(steps ? (goal[j]-start[j])/t.command_duration_s : 0);
    out.status = MotionStatus::Success; out.reason = "bounded commands; acceleration discontinuous at velocity changes";
    out.trajectory = std::move(t); return out;
  }
  out.status = MotionStatus::ResourceLimit; out.reason = "roundoff-safe interpolation exceeds step budget"; return out;
}

std::vector<NumericalWarning> numerical_warnings(const mjData& d) {
  std::vector<NumericalWarning> out;
  for (int i = 0; i < mjNWARNING; ++i) if (d.warning[i].number)
    out.push_back({i,d.warning[i].number,d.warning[i].lastinfo});
  return out;
}
MotionStatus execution_health(const mjModel& m, const mjData& d) {
  if (!numerical_warnings(d).empty()) return MotionStatus::NumericalWarning;
  if (!std::isfinite(d.time)) return MotionStatus::SimulationError;
  for (int j = 0; j < m.njnt; ++j) {
    const double q = d.qpos[m.jnt_qposadr[j]], v = d.qvel[m.jnt_dofadr[j]];
    if (!std::isfinite(q) || !std::isfinite(v)) return MotionStatus::SimulationError;
    if (m.jnt_limited[j] && (q < m.jnt_range[2*j] || q > m.jnt_range[2*j+1])) return MotionStatus::ObservedJointLimit;
  }
  return MotionStatus::Success;
}

PreflightResult preflight_candidate(Simulation& sim, const PositionIkRequest& r, const IkCandidate& candidate,
  const MotionOptions& o, const ContactPolicy& policy, std::size_t budget) {
  PreflightResult out; auto& a = out.assessment; a.candidate = candidate;
  auto fail = [&](MotionStatus s, std::string why) { a.status = s; a.reason = std::move(why); return out; };
  const auto limits = sim.joint_limits();
  auto error = option_error(limits.size(),sim.model().opt.timestep,o);
  if (error.empty()) error = request_error(r,limits);
  if (error.empty()) error = validate_contact_policy(sim.model(),policy);
  if (!error.empty()) return fail(MotionStatus::InvalidInput,error);
  if (budget < 1) return fail(MotionStatus::ResourceLimit,"collision sample budget exhausted before candidate evaluation");
  if (!joints_valid(candidate.joints_rad,limits)) return fail(MotionStatus::InvalidCandidate,"candidate joints invalid");
  try {
    const auto fk = sim.evaluate_fk(candidate.joints_rad);
    const double residual = position_error(fk.position_m,r.position_world_m);
    if (!std::isfinite(residual) || residual > r.position_tolerance_m)
      return fail(MotionStatus::InvalidCandidate,"candidate failed independent FK residual check");
    a.candidate.fk_residual_m = residual;
    auto path = make_joint_trajectory(r.current_rad,candidate.joints_rad,limits,sim.model().opt.timestep,o);
    if (path.status != MotionStatus::Success) return fail(path.status,path.reason);
    const auto& commands = path.trajectory.commands;
    for (std::size_t k = 0; k < commands.size(); ++k) {
      long double subdivisions = 1;
      if (k) for (std::size_t j = 0; j < limits.size(); ++j)
        subdivisions = std::max(subdivisions,std::ceil(std::abs(static_cast<long double>(commands[k][j])-commands[k-1][j])/o.collision_step_rad));
      if (!std::isfinite(subdivisions) || subdivisions > budget-a.collision_samples)
        return fail(MotionStatus::ResourceLimit,"aggregate collision inspection budget exhausted");
      const auto count = static_cast<std::size_t>(subdivisions);
      for (std::size_t sample = 1; sample <= count; ++sample) {
        if (a.collision_samples >= budget) return fail(MotionStatus::ResourceLimit,"collision sample budget exhausted");
        std::vector<double> q = commands[k];
        if (k && sample != count) for (std::size_t j = 0; j < q.size(); ++j)
          q[j] = static_cast<double>(static_cast<long double>(commands[k-1][j])+
            (static_cast<long double>(commands[k][j])-commands[k-1][j])*sample/count);
        if (!joints_valid(q,limits)) return fail(MotionStatus::InvalidPath,"collision subdivision outside joint interval");
        sim.evaluate_fk(q); ++a.collision_samples;
        if (!numerical_warnings(sim.planning_data()).empty()) return fail(MotionStatus::NumericalWarning,"planning warning during sampled preflight");
        auto contacts = inspect_contacts(sim.model(),sim.planning_data(),policy);
        if (!contacts.empty() && (a.contacts.empty() || has_disallowed_contact(contacts))) {
          a.contacts = contacts;
          a.contact_sample_time_s = k ? (static_cast<double>(k-1)+static_cast<double>(sample)/static_cast<double>(count))*path.trajectory.timestep_s : 0;
        }
        if (has_disallowed_contact(contacts)) return fail(MotionStatus::DisallowedContact,"disallowed contact on sampled command path");
      }
    }
    long double cost = 0;
    for (std::size_t j = 0; j < limits.size(); ++j) {
      const long double delta = static_cast<long double>(candidate.joints_rad[j])-r.current_rad[j]; cost += delta*delta;
    }
    a.squared_displacement_rad2 = static_cast<double>(cost);
    if (!std::isfinite(a.squared_displacement_rad2)) return fail(MotionStatus::ResourceLimit,"unrepresentable ranking cost");
    a.status = MotionStatus::Success; a.reason = "joint, FK, bounded path and sampled contacts admitted";
    out.trajectory = std::move(path.trajectory); return out;
  } catch (const std::exception& e) { return fail(MotionStatus::SimulationError,e.what()); }
}
bool prefer_motion_candidate(const CandidateAssessment& a, const CandidateAssessment& b) {
  if (a.status != MotionStatus::Success) return false;
  if (b.status != MotionStatus::Success) return true;
  if (a.squared_displacement_rad2 != b.squared_displacement_rad2) return a.squared_displacement_rad2 < b.squared_displacement_rad2;
  if (a.candidate.fk_residual_m != b.candidate.fk_residual_m) return a.candidate.fk_residual_m < b.candidate.fk_residual_m;
  return a.candidate.joints_rad < b.candidate.joints_rad;
}

MotionPlan plan_cartesian_motion(Simulation& sim, const PositionIkRequest& r, const MotionOptions& o,
  const SearchOptions& search, const ContactPolicy& policy) {
  const auto started = Clock::now(); MotionPlan out;
  const auto model_limits = sim.joint_limits();
  auto finish = [&](MotionStatus s, std::string why) {
    out.status = s; out.reason = std::move(why); out.planning_wall_ms = milliseconds(started); return out;
  };
  auto error = option_error(sim.joint_limits().size(),sim.model().opt.timestep,o);
  if (error.empty()) error = request_error(r,sim.joint_limits());
  if (error.empty()) error = validate_contact_policy(sim.model(),policy);
  if (!error.empty()) return finish(MotionStatus::InvalidInput,error);
  if (search.wrist_budgets.empty() || search.wrist_budgets.size() > 8 || search.total_wrist_budget < 1 ||
      search.total_wrist_budget > 32768 || search.max_unique_candidates < 1 || search.max_unique_candidates > 16384 ||
      search.max_collision_samples < 1 || search.max_collision_samples > 2000000)
    return finish(MotionStatus::InvalidInput,"invalid bounded search/resource constraints");
  std::size_t previous = 0;
  for (auto b : search.wrist_budgets) {
    if (b <= previous || b > 4096) return finish(MotionStatus::InvalidInput,"wrist stages must increase within 1..4096");
    previous = b;
  }
  const bool adaptive = sim.config().dof == 4 && r.wrist_search.mode == WristMode::Sampled;
  std::map<std::vector<double>,std::size_t> checked;
  std::optional<std::size_t> best;
  std::size_t used_wrist_budget = 0;
  for (auto budget : search.wrist_budgets) {
    if (!adaptive) budget = 1;
    if (budget > search.total_wrist_budget-used_wrist_budget)
      return finish(MotionStatus::ResourceLimit,"total repeated IK wrist computation budget exhausted");
    used_wrist_budget += budget;
    auto request = r; request.wrist_search.budget = budget;
    const auto t = Clock::now(); const auto ik = solve_position_ik(sim,request);
    SearchAttempt attempt; attempt.ik_wall_ms = milliseconds(t); out.ik_wall_ms += attempt.ik_wall_ms;
    attempt.wrist_budget = budget; attempt.ik_status = ik.status; attempt.reason = ik.reason;
    attempt.coverage = ik.coverage; attempt.kinematic_rejections = ik.rejected;
    attempt.returned_candidates = ik.candidates.size();
    const auto preflight_start = Clock::now();
    for (const auto& candidate : ik.candidates) {
      const bool cacheable = joints_valid(candidate.joints_rad,model_limits);
      if (cacheable && checked.contains(candidate.joints_rad)) { ++attempt.reused_candidates; continue; }
      if (out.assessments.size() >= search.max_unique_candidates) {
        out.attempts.push_back(attempt); out.preflight_wall_ms += milliseconds(preflight_start);
        return finish(MotionStatus::ResourceLimit,"unique candidate preflight budget exhausted; no partial optimum selected");
      }
      auto preflight = preflight_candidate(sim,r,candidate,o,policy,search.max_collision_samples-out.total_collision_samples);
      out.total_collision_samples += preflight.assessment.collision_samples; ++attempt.newly_checked_candidates;
      const std::size_t index = out.assessments.size();
      if (cacheable) checked.emplace(candidate.joints_rad,index);
      out.assessments.push_back(std::move(preflight.assessment));
      if (out.assessments.back().status == MotionStatus::Success &&
          (!best || prefer_motion_candidate(out.assessments.back(),out.assessments[*best]))) {
        best = index; out.trajectory = std::move(preflight.trajectory);
      }
      if (out.assessments.back().status == MotionStatus::ResourceLimit || out.assessments.back().status == MotionStatus::NumericalWarning ||
          out.assessments.back().status == MotionStatus::SimulationError) {
        out.attempts.push_back(attempt); out.preflight_wall_ms += milliseconds(preflight_start);
        return finish(out.assessments.back().status,out.assessments.back().reason);
      }
    }
    out.preflight_wall_ms += milliseconds(preflight_start); out.attempts.push_back(attempt);
    if (best) {
      out.selected = out.assessments[*best].candidate; out.selected_cost_rad2 = out.assessments[*best].squared_displacement_rad2;
      out.command_fk = sim.evaluate_fk(out.selected->joints_rad);
      return finish(MotionStatus::Success,"minimum ordinary squared displacement among sampled motion-feasible candidates at first successful stage; residual then lexicographic ties");
    }
    if (ik.status == IkStatus::InvalidInput) return finish(MotionStatus::InvalidInput,ik.reason);
    if (ik.status == IkStatus::ResourceLimit) return finish(MotionStatus::ResourceLimit,ik.reason);
    if (ik.status != IkStatus::Success && ik.status != IkStatus::NoCandidateWithinSearch)
      return finish(MotionStatus::IkFailure,std::string(ik_status_name(ik.status))+": "+ik.reason);
    if (!adaptive || attempt.coverage.wrist_continuum_covered) break;
  }
  return finish(out.assessments.empty() && adaptive ? MotionStatus::SearchIncomplete : MotionStatus::NoFeasibleCandidate,
    out.assessments.empty() ? "no IK candidate in configured search; see scope/coverage" : "all evaluated IK candidates failed motion preflight; search scope remains bounded");
}

ExecutionResult execute_cartesian_motion(Simulation& sim, const PositionIkRequest& request, const MotionOptions& o,
  const SearchOptions& search, const ContactPolicy& policy, TraceWriter* trace) {
  const auto started = Clock::now(); ExecutionResult out; out.requested_position_m = request.position_world_m;
  auto finish = [&](MotionStatus status, std::string why) {
    out.status = status; out.reason = std::move(why); out.total_wall_ms = milliseconds(started); return out;
  };
  out.plan = plan_cartesian_motion(sim,request,o,search,policy);
  if (out.plan.status != MotionStatus::Success) return finish(out.plan.status,out.plan.reason);
  const auto execution_start = Clock::now();
  const auto& commands = out.plan.trajectory.commands;
  const auto& goal = out.plan.selected->joints_rad;
  const auto& m = sim.model(); const double dt = m.opt.timestep;
  out.command_duration_s = out.plan.trajectory.command_duration_s;
  out.saturation_samples.assign(goal.size(),0); out.peak_actuator_abs_Nm.assign(goal.size(),0);
  std::size_t ticks = 0, dwell_ticks = 0;
  const auto max_ticks = static_cast<std::size_t>(std::floor(static_cast<long double>(o.timeout_s)/dt+1e-12L));
  auto capture = [&](bool physics_tick) {
    out.warnings = numerical_warnings(sim.execution_data());
    auto health = execution_health(m,sim.execution_data());
    out.final = sim.snapshot(); out.simulated_duration_s = out.final->time_s;
    out.final_pose_authoritative = health != MotionStatus::NumericalWarning && health != MotionStatus::SimulationError;
    out.final_cartesian_error_m = position_error(out.final->tool.position_m,request.position_world_m);
    if (!std::isfinite(out.final_cartesian_error_m) && health != MotionStatus::NumericalWarning) health = MotionStatus::SimulationError;
    out.final_joint_error_rad.clear();
    for (std::size_t j = 0; j < goal.size(); ++j) {
      out.final_joint_error_rad.push_back(goal[j]-out.final->observed_rad[j]);
      out.peak_tracking_error_rad = std::max(out.peak_tracking_error_rad,std::abs(out.final->command_rad[j]-out.final->observed_rad[j]));
      const double force = out.final->actuator_torque_Nm[j];
      if (!std::isfinite(force) && health != MotionStatus::NumericalWarning) health = MotionStatus::SimulationError;
      out.peak_actuator_abs_Nm[j] = std::max(out.peak_actuator_abs_Nm[j],std::abs(force));
      const auto aid = static_cast<int>(j);
      if (physics_tick && m.actuator_forcelimited[aid]) {
        const double lo = m.actuator_forcerange[2*aid], hi = m.actuator_forcerange[2*aid+1];
        const double epsilon = 1e-8*std::max({1.0,std::abs(lo),std::abs(hi)});
        if (force <= lo+epsilon || force >= hi-epsilon) ++out.saturation_samples[j];
      }
    }
    const auto contacts = inspect_contacts(m,sim.execution_data(),policy);
    for (const auto& c : contacts) {
      ++out.contact_events_total;
      if (out.contacts.size() < 4096) out.contacts.push_back({out.final->time_s,c});
    }
    if (health == MotionStatus::Success && has_disallowed_contact(contacts)) health = MotionStatus::DisallowedContact;
    out.final_pose_authoritative = health != MotionStatus::NumericalWarning && health != MotionStatus::SimulationError;
    return health;
  };
  std::string trace_error;
  auto append_trace = [&](const Snapshot& s) {
    try { if (trace) trace->append(s); return true; }
    catch (const std::exception& e) { trace_error = e.what(); return false; }
  };
  auto done = [&](MotionStatus status, std::string why) {
    out.execution_wall_ms = milliseconds(execution_start);
    out.settling_duration_s = std::max(0.0,out.simulated_duration_s-out.executed_command_duration_s);
    return finish(status,std::move(why));
  };
  try {
    sim.reset(request.current_rad); out.execution_started = true; out.initial = sim.snapshot();
    auto health = capture(false);
    if (health != MotionStatus::Success) return done(health,"invalid observed state immediately after approved reset");
    if (!append_trace(*out.initial)) return done(MotionStatus::TraceFailure,trace_error);
    for (std::size_t k = 1; k < commands.size(); ++k) {
      if (ticks >= max_ticks) return done(MotionStatus::Timeout,"timeout during commanded movement");
      sim.set_position_commands(commands[k]);
      if (k == 1) out.first_command_before_step = sim.snapshot();
      health = capture(false);
      if (health != MotionStatus::Success) return done(health,"execution health/contact failure after command");
      sim.step(); ++ticks; out.executed_command_duration_s = static_cast<double>(ticks)*dt;
      health = capture(true);
      if (health != MotionStatus::Success) { append_trace(*out.final); return done(health,"execution health/contact failure after physics tick"); }
      if (!append_trace(*out.final)) return done(MotionStatus::TraceFailure,trace_error);
    }
    while (ticks < max_ticks) {
      sim.set_position_commands(goal);
      const auto command_health = capture(false);
      if (command_health != MotionStatus::Success) return done(command_health,"execution health/contact failure after settling command");
      sim.step(); ++ticks;
      const auto health_now = capture(true);
      if (health_now != MotionStatus::Success) { append_trace(*out.final); return done(health_now,"execution health/contact failure during settling"); }
      if (!append_trace(*out.final)) return done(MotionStatus::TraceFailure,trace_error);
      bool acceptable = out.final_cartesian_error_m <= o.cartesian_tolerance_m;
      for (std::size_t j = 0; j < goal.size(); ++j) acceptable &= std::abs(out.final_joint_error_rad[j]) <= o.joint_tolerance_rad &&
        std::abs(out.final->velocity_rad_s[j]) <= o.settling_velocity_rad_s;
      dwell_ticks = acceptable ? dwell_ticks+1 : 0;
      out.accepted_settling_dwell_s = static_cast<double>(dwell_ticks)*dt;
      if (out.accepted_settling_dwell_s+1e-12 >= o.settling_dwell_s) {
        out.settled = true; return done(MotionStatus::Success,"observed position, joint error and speed satisfy sampled settling dwell");
      }
    }
    return done(MotionStatus::NonConvergence,"movement commanded but settling criteria not met before timeout");
  } catch (const std::exception& e) {
    const auto health = capture(false);
    if (health == MotionStatus::NumericalWarning) return done(health,e.what());
    return done(MotionStatus::SimulationError,e.what());
  }
}
}
