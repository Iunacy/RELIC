#include "relic/analytical_ik.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

namespace {
using namespace relic;
struct Case { std::string group; std::vector<double> source; std::array<double,3> target; bool reachable; };
constexpr std::uint64_t seed = 0x52454c494342ULL;
double uniform(std::mt19937_64& rng, double lo, double hi) {
  return lo+(hi-lo)*(static_cast<double>(rng() >> 11)*0x1.0p-53);
}
std::vector<Case> corpus(Simulation& sim) {
  std::mt19937_64 rng(seed);
  std::vector<Case> cases;
  const auto& c = sim.config();
  for (const std::string group : {"interior","joint_boundary","workspace_boundary","near_boundary"}) {
    const int count = group == "interior" ? 1000 : 400;
    for (int k = 0; k < count; ++k) {
      std::vector<double> q(static_cast<std::size_t>(c.dof));
      for (std::size_t j = 0; j < q.size(); ++j) {
        const auto lim = c.joint_limits_rad[j]; q[j] = uniform(rng,lim.lower,lim.upper);
      }
      if (group == "joint_boundary") {
        const std::size_t j = static_cast<std::size_t>(k)%q.size();
        const auto lim = c.joint_limits_rad[j]; q[j] = (static_cast<std::size_t>(k)/q.size())%2 ? lim.lower : lim.upper;
      } else if (group == "workspace_boundary" || group == "near_boundary") {


        const double w = c.dof == 4 ? q[3] : 0;
        const auto t = c.tool_frame.position_m;
        const double v = t[1]*std::sin(w)+t[2]*std::cos(w);
        q[1] = uniform(rng,-0.9,0.9);
        q[2] = std::atan2(v,c.link_lengths_m[1]+t[0]);
        if (group == "near_boundary") q[2] += uniform(rng,-1e-5,1e-5);
      }
      cases.push_back({group,q,sim.evaluate_fk(q).position_m,true});
    }
  }
  for (int k = 0; k < 200; ++k) {

    cases.push_back({"unreachable",{}, {uniform(rng,1.5,2.5),uniform(rng,-0.5,0.5),c.pedestal_height_m},false});
  }
  return cases;
}
struct Stats {
  std::size_t total = 0, successes = 0, missed = 0, correct_rejections = 0, false_success = 0;
  std::size_t normalized_candidates = 0;
  std::size_t endpoint_candidates = 0;
  double worst_residual = 0, max_gap = 0;
  std::vector<double> latency_us, best_residuals;
  std::map<std::string,std::size_t> statuses;
};
double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(),values.end());
  const auto index = static_cast<std::size_t>(std::ceil(fraction*static_cast<double>(values.size())))-1;
  return values[std::min(index,values.size()-1)];
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: relic_ik_benchmark EXAMPLES MISSED_CASES.csv");
    std::ofstream misses(argv[2]);
    if (!misses) throw std::runtime_error("cannot create missed-case evidence");
    misses << std::setprecision(17) << "mode,budget,group,case_index,status,target_x,target_y,target_z,source_yaw,source_shoulder,source_elbow,source_wrist\n";
    std::cout << std::setprecision(12)
      << "mode,budget,group,cases,successes,missed_reachable,correct_unreachable,false_success,success_rate,median_us,p95_us,max_candidate_residual_m,median_best_residual_m,max_wrist_gap_rad,normalized_candidates,limit_endpoint_candidates,statuses\n";
    for (int n : {3,4}) {
      auto c = load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg"));
      Simulation sim(c);
      const auto cases = corpus(sim);
      const std::vector<std::pair<std::string,std::size_t>> modes = n == 3 ?
        std::vector<std::pair<std::string,std::size_t>>{{"3dof",1}} :
        std::vector<std::pair<std::string,std::size_t>>{{"4dof_fixed",1},{"4dof_sampled",1},{"4dof_sampled",5},
          {"4dof_sampled",17},{"4dof_sampled",33},{"4dof_sampled",65},{"4dof_sampled",257}};
      for (const auto& [mode,budget] : modes) {
        auto request = [&](const Case& entry) {
          PositionIkRequest r; r.position_world_m = entry.target; r.current_rad.assign(static_cast<std::size_t>(n),0);
          r.wrist_search.budget = budget;
          if (mode == "4dof_fixed") { r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = entry.reachable ? entry.source[3] : 0; }
          return r;
        };
        for (std::size_t k = 0; k < 16; ++k) solve_position_ik(sim,request(cases[k]));
        std::map<std::string,Stats> stats;
        for (std::size_t k = 0; k < cases.size(); ++k) {
          const auto& entry = cases[k]; const auto r = request(entry);
          const auto start = std::chrono::steady_clock::now();
          const auto result = solve_position_ik(sim,r);
          const double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
          auto& s = stats[entry.group]; ++s.total; ++s.statuses[ik_status_name(result.status)]; s.latency_us.push_back(us);
          s.max_gap = std::max(s.max_gap,result.coverage.max_wrist_gap_rad);
          const bool success = result.status == IkStatus::Success;
          if (entry.reachable) { s.successes += success; s.missed += !success; }
          else { s.false_success += success; s.correct_rejections += result.status == IkStatus::Unreachable; }
          if (success) {
            double best = std::numeric_limits<double>::infinity();
            for (const auto& candidate : result.candidates) {
              best = std::min(best,candidate.fk_residual_m); s.worst_residual = std::max(s.worst_residual,candidate.fk_residual_m);
              s.normalized_candidates += candidate.branch.boundary_normalized;
              s.endpoint_candidates += candidate.branch.joint_boundary_reconstructed;
            }
            s.best_residuals.push_back(best);
          } else if (entry.reachable) {
            misses << mode << ',' << budget << ',' << entry.group << ',' << k << ',' << ik_status_name(result.status);
            for (double p : entry.target) misses << ',' << p;
            for (double q : entry.source) misses << ',' << q;
            if (n == 3) misses << ',';
            misses << '\n';
          }
        }
        Stats all;
        for (const auto& [group,s] : stats) if (group != "unreachable") {
          all.total += s.total; all.successes += s.successes; all.missed += s.missed;
          all.normalized_candidates += s.normalized_candidates; all.endpoint_candidates += s.endpoint_candidates;
          all.worst_residual = std::max(all.worst_residual,s.worst_residual); all.max_gap = std::max(all.max_gap,s.max_gap);
          all.latency_us.insert(all.latency_us.end(),s.latency_us.begin(),s.latency_us.end());
          all.best_residuals.insert(all.best_residuals.end(),s.best_residuals.begin(),s.best_residuals.end());
          for (const auto& [status,count] : s.statuses) all.statuses[status] += count;
        }
        stats.emplace("all_reachable",std::move(all));
        for (const auto& [group,s] : stats) {
          std::cout << mode << ',' << budget << ',' << group << ',' << s.total << ',' << s.successes << ',' << s.missed << ','
            << s.correct_rejections << ',' << s.false_success << ','
            << (group == "unreachable" ? 0 : static_cast<double>(s.successes)/static_cast<double>(s.total)) << ','
            << percentile(s.latency_us,0.5) << ',' << percentile(s.latency_us,0.95) << ',' << s.worst_residual << ','
            << percentile(s.best_residuals,0.5) << ',' << s.max_gap << ',' << s.normalized_candidates << ',' << s.endpoint_candidates << ',';
          for (const auto& [status,count] : s.statuses) std::cout << status << ':' << count << ';';
          std::cout << '\n';
        }
      }
    }
    misses.close(); if (!misses) throw std::runtime_error("failed to save missed-case evidence");
    std::cerr << "seed=0x52454c494342; 16 warmups per mode; nearest-rank median/p95; timing includes geometry checks, enumeration and MuJoCo FK; excludes model construction, target generation and CSV output. No motion execution.\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
