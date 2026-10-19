#include "relic/motion.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <numbers>

namespace {
using namespace relic;
double uniform(std::mt19937_64& rng, double lo, double hi) { return lo+(hi-lo)*static_cast<double>(rng() >> 11)*0x1.0p-53; }
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: relic_motion_benchmark EXAMPLES REPORTS_DIRECTORY");
    const std::filesystem::path reports(argv[2]); if (!std::filesystem::is_directory(reports)) throw std::runtime_error("reports directory must exist");
    std::cout << std::setprecision(12)
      << "dof,case_id,kind,reachable,ik_success,plan_feasible,execution_started,status,attempts,budgets,returned_candidates,unique_candidates,reused_candidates,rejected_candidates,collision_rejections,preflight_samples,ik_ms,preflight_ms,planning_ms,execution_ms,total_ms,command_s,simulated_s,settling_s,cartesian_error_m,peak_joint_tracking_rad,saturated_joint_ticks,warnings,observed_contacts\n";
    for (int n : {3,4}) {
      Simulation sim(load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg")));
      std::mt19937_64 rng(0x52454c494343ULL);
      std::map<std::string,std::size_t> counts;
      const int total = n == 3 ? 26 : 29;
      for (int id = 0; id < total; ++id) {
        MotionOptions o; o.max_velocity_rad_s.assign(static_cast<std::size_t>(n),0.5);
        PositionIkRequest r; r.current_rad.assign(static_cast<std::size_t>(n),0);
        std::vector<double> q(static_cast<std::size_t>(n),0); std::string kind; bool reachable = true;
        if (id < 12) {
          kind = "interior"; q = {uniform(rng,-0.8,0.8),uniform(rng,-0.65,-0.15),uniform(rng,0.1,0.9)};
          r.current_rad = {uniform(rng,-0.4,0.4),uniform(rng,-0.25,0),uniform(rng,0,0.2)};
          if (n == 4) { q.push_back(uniform(rng,-1,1)); r.current_rad.push_back(uniform(rng,-0.2,0.2)); }
        } else if (id < 20) {
          kind = "full_joint_range";
          for (std::size_t j = 0; j < q.size(); ++j) { const auto lim = sim.config().joint_limits_rad[j]; q[j] = uniform(rng,lim.lower,lim.upper); }
        } else if (id == 20) { kind = "zero_displacement"; }
        else if (id == 21) { kind = "collision"; q[2] = std::numbers::pi/2; }
        else if (id == 22) { kind = "outside"; reachable = false; }
        else if (id == 23) { kind = "short_timeout"; q[0] = 0.6; o.timeout_s = 0.01; }
        else if (id == 24) { kind = "strict_tracking"; q = {0.6,-0.3,0.5}; if (n == 4) q.push_back(0); o.timeout_s = 2; o.cartesian_tolerance_m = o.joint_tolerance_rad = 1e-12; }
        else if (id == 25) { kind = "saturation"; q[0] = 1.5; o.max_velocity_rad_s.assign(static_cast<std::size_t>(n),1000); o.max_command_step_rad = 2; }
        else kind = id == 26 ? "adaptive65" : (id == 27 ? "adaptive257" : "incomplete257");
        r.position_world_m = sim.evaluate_fk(q).position_m;
        if (id == 22) r.position_world_m = {5,5,5};
        if (id == 26) r.position_world_m = {0.42693414739579011,0.024212786719060447,0.61208956526453306};
        if (id == 27) r.position_world_m = {-0.20232397716216027,-0.32762171046313787,0.65396120215840337};
        if (id == 28) r.position_world_m = {-0.15656020960157971,-0.55501236942387,0.28358833272306733};
        const auto result = execute_cartesian_motion(sim,r,o); ++counts[motion_status_name(result.status)];
        std::size_t returned = 0, reused = 0, rejected = 0, collisions = 0, saturated = 0; bool ik_success = false;
        std::string budgets;
        for (const auto& a : result.plan.attempts) {
          returned += a.returned_candidates; reused += a.reused_candidates; ik_success |= a.ik_status == IkStatus::Success;
          budgets += std::to_string(a.wrist_budget)+";";
        }
        for (const auto& a : result.plan.assessments) { rejected += a.status != MotionStatus::Success; collisions += a.status == MotionStatus::DisallowedContact; }
        for (auto count : result.saturation_samples) saturated += count;
        const double error = result.final && result.final_pose_authoritative ? result.final_cartesian_error_m : std::numeric_limits<double>::quiet_NaN();
        std::cout << n << ',' << id << ',' << kind << ',' << reachable << ',' << ik_success << ',' << bool(result.plan.selected) << ',' << result.execution_started
          << ',' << motion_status_name(result.status) << ',' << result.plan.attempts.size() << ',' << budgets << ',' << returned << ','
          << result.plan.assessments.size() << ',' << reused << ',' << rejected << ',' << collisions << ',' << result.plan.total_collision_samples << ','
          << result.plan.ik_wall_ms << ',' << result.plan.preflight_wall_ms << ',' << result.plan.planning_wall_ms << ',' << result.execution_wall_ms << ',' << result.total_wall_ms << ','
          << result.command_duration_s << ',' << result.simulated_duration_s << ',' << result.settling_duration_s << ',' << error << ',' << result.peak_tracking_error_rad << ','
          << saturated << ',' << result.warnings.size() << ',' << result.contact_events_total << '\n';
        if (result.status != MotionStatus::Success || id >= 26) {
          std::ofstream evidence(reports/("case-"+std::to_string(n)+"-"+std::to_string(id)+".txt"));
          evidence << "seed=0x52454c494343 case=" << id << " kind=" << kind << '\n';
          write_execution_report(evidence,result);
          if (!evidence) throw std::runtime_error("failed saving case evidence");
        }
      }
      std::cerr << "dof=" << n << " pipeline_cases=" << total;
      for (const auto& [status,count] : counts) std::cerr << ' ' << status << '=' << count;
      std::cerr << '\n';
    }
    std::cerr << "seed=0x52454c494343; no warmups; full-pipeline sample includes diagnostic failures; no tracing in benchmark; timings exclude model construction, FK truth generation and report output.\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
