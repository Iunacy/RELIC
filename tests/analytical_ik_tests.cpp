#include "relic/analytical_ik.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {
using namespace relic;
std::filesystem::path examples;
std::size_t checks = 0;
void require(bool value, const std::string& why) {
  ++checks;
  if (!value) throw std::runtime_error(why);
}
RobotConfig fixture(int n = 3) { return load_config(examples/ ("synthetic_"+std::to_string(n)+"dof.cfg")); }
PositionIkRequest target(Simulation& sim, const std::vector<double>& q) {
  PositionIkRequest r;
  r.position_world_m = sim.evaluate_fk(q).position_m;
  r.current_rad.assign(q.size(),0);
  return r;
}
std::vector<mjtNum> live(const Simulation& s) {
  std::vector<mjtNum> state(mj_stateSize(&s.model(),mjSTATE_INTEGRATION));
  mj_getState(&s.model(),&s.execution_data(),state.data(),mjSTATE_INTEGRATION);
  return state;
}
void verified(Simulation& sim, const PositionIkRequest& r, const PositionIkResult& out) {
  require(out.status == IkStatus::Success,"expected Success, got "+std::string(ik_status_name(out.status))+": "+out.reason);
  require(!out.candidates.empty(),"success must include candidates");
  Simulation independent(sim.config());
  for (const auto& candidate : out.candidates) {
    require(candidate.joints_rad.size() == r.current_rad.size(),"candidate dimension");
    for (std::size_t i = 0; i < candidate.joints_rad.size(); ++i) {
      const double q = candidate.joints_rad[i];
      const auto lim = sim.config().joint_limits_rad[i];
      require(std::isfinite(q) && q >= lim.lower && q <= lim.upper,"candidate within strict inclusive interval");
    }
    const auto p = independent.evaluate_fk(candidate.joints_rad).position_m;
    double error = 0;
    for (std::size_t i = 0; i < 3; ++i) error = std::hypot(error,p[i]-r.position_world_m[i]);
    require(error <= r.position_tolerance_m,"independent FK residual exceeds tolerance");
    require(std::abs(error-candidate.fk_residual_m) <= 1e-15,"reported residual matches independent FK");
  }
}
void branches() {
  Simulation sim(fixture());
  auto r = target(sim,{0.35,-0.25,0.8});
  auto out = solve_position_ik(sim,r);
  verified(sim,r,out);
  bool plus = false, minus = false;
  for (const auto& x : out.candidates) { plus |= x.branch.elbow_sign == 1; minus |= x.branch.elbow_sign == -1; }
  require(plus && minus,"both elbow branches retained");
  auto c = fixture(); c.joint_limits_rad[2].lower = 0;
  Simulation limited(c);
  out = solve_position_ik(limited,r); verified(limited,r,out);
  for (const auto& x : out.candidates) require(x.branch.elbow_sign >= 0,"limit excludes negative elbow branch");
  c.joint_limits_rad[1] = {-3.2,3.2}; c.joint_limits_rad[0] = {-3.2,3.2}; c.joint_limits_rad[2] = {-1.8,1.8};
  Simulation wide(c); r = target(wide,{0,0,0.8});
  out = solve_position_ik(wide,r); verified(wide,r,out);
  plus = false; minus = false;
  for (const auto& x : out.candidates) { plus |= x.branch.radius_sign == 1; minus |= x.branch.radius_sign == -1; }
  require(plus && minus,"both signed-radius branches retained");

  c.joint_limits_rad[0] = {-0.1,0.1}; Simulation yaw_limited(c);
  out = solve_position_ik(yaw_limited,r); verified(yaw_limited,r,out);
  for (const auto& x : out.candidates) require(x.branch.radius_sign == 1,"yaw bounds exclude negative radius branch");
  c = fixture(); c.joint_limits_rad[0] = {-0.1,0.1}; c.joint_limits_rad[1] = {-0.1,0.1}; c.joint_limits_rad[2] = {-0.1,0.1};
  Simulation all_limited(c); out = solve_position_ik(all_limited,r);
  require(out.status == IkStatus::JointLimitsExcludeCandidates && out.candidates.empty(),"all-branch limit exclusion diagnosed");

  c = fixture(); c.joint_limits_rad[0] = {-7,7}; Simulation turns(c);
  r = target(turns,{0.2,-0.2,0.7}); out = solve_position_ik(turns,r); verified(turns,r,out);
  bool lifted = false;
  for (const auto& x : out.candidates) lifted |= std::abs(x.joints_rad[0]) > 6;
  require(lifted,"admissible multi-turn angle lift enumerated");
  r.candidate_budget = 1; out = solve_position_ik(turns,r);
  require(out.status == IkStatus::ResourceLimit && out.candidates.empty(),"resource failure cannot expose partial candidates");
}
void boundaries() {
  Simulation sim(fixture());
  auto r = target(sim,{0,0,0});
  verified(sim,r,solve_position_ik(sim,r));
  bool reconstructed = false;
  for (int n : {3,4}) {
    Simulation endpoints(fixture(n));
    for (int j = 0; j < n; ++j) for (int side : {0,1}) {
      std::vector<double> q(static_cast<std::size_t>(n),0);
      q[1] = -0.2; q[2] = 0.7;
      const auto lim = endpoints.config().joint_limits_rad[static_cast<std::size_t>(j)];
      q[static_cast<std::size_t>(j)] = side ? lim.lower : lim.upper;
      auto limit_request = target(endpoints,q);
      if (n == 4) { limit_request.wrist_search.mode = WristMode::Fixed; limit_request.wrist_search.fixed_rad = q[3]; }
      const auto result = solve_position_ik(endpoints,limit_request); verified(endpoints,limit_request,result);
      for (const auto& candidate : result.candidates) reconstructed |= candidate.branch.joint_boundary_reconstructed;
    }
  }
  require(reconstructed,"limit endpoint reconstruction is explicitly reported");

  auto regression = r;
  regression.position_world_m = {-0.45667263176786854,-0.16236074389212982,-0.018785751776149516};
  verified(sim,regression,solve_position_ik(sim,regression));
  r.position_world_m[0] += 1e-9; r.position_tolerance_m = 1;
  auto out = solve_position_ik(sim,r);
  require(out.status == IkStatus::Unreachable && out.candidates.empty(),"loose FK tolerance cannot hide an outside-workspace target");
  r.position_world_m = {0,0,sim.config().pedestal_height_m};
  require(solve_position_ik(sim,r).status == IkStatus::Unreachable,"inner workspace hole");
  for (double e : {1e-7,-1e-7,1e-4,-1e-4}) {
    r = target(sim,{0.3,-0.3,e}); verified(sim,r,solve_position_ik(sim,r));
  }

  const double L1 = 0.30, D = 0.31;
  const double vertical = std::sqrt(L1*L1+D*D+2*L1*D*std::cos(1.5));
  r.position_world_m = {0,0,0.20-vertical}; r.current_rad = {0.6,0,0};
  out = solve_position_ik(sim,r); verified(sim,r,out);
  require(out.coverage.singular_family_representatives,"free-yaw family is labeled");
  for (const auto& x : out.candidates) require(x.branch.free_yaw && x.joints_rad[0] == 0.6,"current admissible free yaw retained");
  auto c = fixture(); c.link_lengths_m = {0.25,0.125}; c.tool_frame.position_m = {0.125,0,0};
  c.joint_limits_rad[2] = {-3.2,3.2}; Simulation folded(c);
  r.position_world_m = {0,0,c.pedestal_height_m}; r.current_rad = {0.4,0.2,0};
  out = solve_position_ik(folded,r); verified(folded,r,out);
  for (const auto& x : out.candidates) require(x.branch.free_shoulder && x.branch.free_yaw,"folded equal-length singular family");
  c = fixture(); c.tool_frame.position_m = {-c.link_lengths_m[1],0,0}; Simulation collapsed(c);
  r.current_rad = {0,0,0}; r.position_world_m = {c.link_lengths_m[0],0,c.pedestal_height_m};
  out = solve_position_ik(collapsed,r);
  require(out.status == IkStatus::DegenerateGeometry && !out.coverage.branch_enumeration_complete,"collapsed distal vector diagnosed, not false unreachable");
}
void wrist() {
  Simulation sim(fixture(4));
  auto r = target(sim,{0.3,-0.4,0.8,0.73});
  r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = 0.73;
  auto out = solve_position_ik(sim,r); verified(sim,r,out);
  require(out.coverage.scope == IkScope::FixedWrist && !out.coverage.wrist_continuum_covered,"fixed wrist is conditional coverage");
  for (const auto& x : out.candidates) require(x.joints_rad[3] == 0.73,"specified wrist held exactly");
  r.wrist_search.mode = WristMode::Sampled; r.wrist_search.budget = 1; r.current_rad[3] = 0.37;
  out = solve_position_ik(sim,r);
  require(out.coverage.sampled_wrist_rad == std::vector<double>{0.37},"budget one uses actual requested current wrist");
  const auto p0 = sim.evaluate_fk(std::vector<double>{0,0,0,0}).position_m;
  const auto p1 = sim.evaluate_fk(std::vector<double>{0,0,0,0.5}).position_m;
  require(std::abs(p1[2]-p0[2]) > 0.004,"actual fixture wrist changes tool position");

  const double w = 1.501;
  r = target(sim,{0,0,std::atan2(0.01*std::sin(w),0.30),w});
  r.wrist_search.budget = 1;
  out = solve_position_ik(sim,r);
  require(out.status == IkStatus::NoCandidateWithinSearch && out.candidates.empty(),"reachable target missed by one wrist sample, not proven unreachable");
  r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = 0;
  require(solve_position_ik(sim,r).status == IkStatus::UnreachableAtFixedWrist,"conditional impossibility has separate status");
  r.wrist_search.mode = WristMode::Sampled;
  std::vector<double> previous;
  for (std::size_t budget : {1u,5u,17u,65u}) {
    r.wrist_search.budget = budget; out = solve_position_ik(sim,r);
    require(out.coverage.sampled_wrist_rad.size() <= budget,"wrist evaluations stay within budget");
    require(out.coverage.requested_wrist_budget == budget,"requested budget reported");
    require(out.coverage.max_wrist_gap_rad > 0 && !out.coverage.wrist_continuum_covered,"incomplete search reports angular gap");
    require(std::equal(previous.begin(),previous.end(),out.coverage.sampled_wrist_rad.begin()),"larger budget retains earlier samples");
    previous = out.coverage.sampled_wrist_rad;
    if (budget == 65) verified(sim,r,out);
  }
  auto c = fixture(4); c.tool_frame.position_m = {0.07,0,0}; Simulation axial(c);
  r = target(axial,{0.2,-0.4,0.8,1.1}); r.current_rad[3] = -0.8;
  out = solve_position_ik(axial,r); verified(axial,r,out);
  require(out.coverage.wrist_continuum_covered && out.coverage.sampled_wrist_rad.size() == 1,"axial wrist needs one representative");
  for (const auto& x : out.candidates) require(x.joints_rad[3] == -0.8,"position-independent wrist retains current value");

  c = fixture(); c.tool_frame.position_m = {0.05,0.02,-0.03};
  c.tool_frame.quaternion_wxyz = {0,1,0,0}; Simulation offset(c);
  r = target(offset,{0.2,-0.3,0.7}); verified(offset,r,solve_position_ik(offset,r));
}
void inputs() {
  Simulation sim(fixture(4));
  const auto good = target(sim,{0.3,-0.3,0.7,0.6});
  auto invalid = [&](PositionIkRequest r) {
    auto out = solve_position_ik(sim,r);
    require(out.status == IkStatus::InvalidInput && out.candidates.empty(),"invalid request must fail without candidates");
  };
  for (double x : {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
    for (std::size_t k = 0; k < 3; ++k) { auto r = good; r.position_world_m[k] = x; invalid(r); }
    for (std::size_t k = 0; k < 4; ++k) { auto r = good; r.current_rad[k] = x; invalid(r); }
    auto r = good; r.position_tolerance_m = x; invalid(r);
    r = good; r.wrist_search.fixed_rad = x; invalid(r);
  }
  for (double x : {0.0,-1.0}) { auto r = good; r.position_tolerance_m = x; invalid(r); }
  auto r = good; r.current_rad.pop_back(); invalid(r);
  r = good; r.current_rad.push_back(0); invalid(r);
  r = good; r.current_rad[3] = 3; invalid(r);
  for (std::size_t b : {0u,4097u}) { r = good; r.wrist_search.budget = b; invalid(r); }
  for (std::size_t b : {0u,65537u}) { r = good; r.candidate_budget = b; invalid(r); }
  r = good; r.wrist_search.mode = static_cast<WristMode>(42); invalid(r);
  r = good; r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = 3; invalid(r);
  r = good; r.position_world_m = {std::numeric_limits<double>::max(),std::numeric_limits<double>::max(),0};
  const auto out = solve_position_ik(sim,r);
  require(out.status == IkStatus::Unreachable || out.status == IkStatus::NumericalFailure,"huge finite target fails explicitly");
}
void geometry() {
  for (int n : {3,4}) {
    Simulation sim(fixture(n));
    require(check_analytical_geometry(sim.model(),sim.config()).status == IkStatus::Success,"fixture topology supported");
    auto changed = [&](const std::function<void(mjModel&)>& change) {
      std::unique_ptr<mjModel,ModelDeleter> copy(mj_copyModel(nullptr,&sim.model()));
      require(bool(copy),"independent model clone"); change(*copy);
      const auto out = check_analytical_geometry(*copy,sim.config());
      require(out.status == IkStatus::UnsupportedGeometry && !out.reason.empty(),"unsupported model has explicit diagnostic");
    };
    changed([](mjModel& m) { m.jnt_axis[4] = -1; });
    changed([](mjModel& m) { m.jnt_pos[3] = 0.01; });
    changed([](mjModel& m) { m.body_pos[9] += 0.01; });
    changed([](mjModel& m) { m.body_quat[8] = 0.5; });
    changed([](mjModel& m) { m.site_pos[1] += 0.01; });
    changed([](mjModel& m) { m.body_parentid[3] = 1; });
    changed([](mjModel& m) { m.actuator_trnid[0] = 1; });
    changed([](mjModel& m) { m.qpos0[0] = 0.1; });
    changed([](mjModel& m) { m.jnt_range[0] += 0.1; });
    auto c = sim.config(); c.dof = 2;
    require(check_analytical_geometry(sim.model(),c).status == IkStatus::UnsupportedGeometry,"invalid dimension rejected");
    c = sim.config(); c.link_lengths_m[0] = 0;
    require(check_analytical_geometry(sim.model(),c).status == IkStatus::UnsupportedGeometry,"invalid geometry rejected");
    c = sim.config(); c.joint_limits_rad[0] = {0,0};
    require(check_analytical_geometry(sim.model(),c).status == IkStatus::UnsupportedGeometry,"invalid constraints rejected");
  }
}
void isolation() {
  Simulation sim(fixture(4));
  sim.set_position_commands(std::vector<double>{0.2,-0.3,0.4,0.1}); sim.step(30);
  const auto before = live(sim);
  auto r = target(sim,{0.3,-0.3,0.8,0.5});
  auto a = solve_position_ik(sim,r); verified(sim,r,a);
  for (int repeat = 0; repeat < 10; ++repeat) {
    const auto b = solve_position_ik(sim,r);
    require(b.status == a.status && b.candidates.size() == a.candidates.size(),"repeatable candidate count/status");
    for (std::size_t k = 0; k < a.candidates.size(); ++k)
      require(b.candidates[k].joints_rad == a.candidates[k].joints_rad &&
              b.candidates[k].fk_residual_m == a.candidates[k].fk_residual_m,"repeatable exact candidates/residuals");
    require(live(sim) == before,"successful IK preserves complete live integration state");
  }
  r.position_world_m = {100,100,100}; require(solve_position_ik(sim,r).status == IkStatus::Unreachable,"global outer bound");
  require(live(sim) == before,"failed IK preserves live integration state");
  r.current_rad[0] = std::numeric_limits<double>::quiet_NaN(); solve_position_ik(sim,r);
  require(live(sim) == before,"invalid IK preserves live integration state");

  Simulation three(fixture()); r = target(three,{0,0,std::numbers::pi/2});
  const auto colliding = solve_position_ik(three,r); verified(three,r,colliding);
  require(std::any_of(colliding.candidates.begin(),colliding.candidates.end(),[](const auto& x) { return x.planning_contact_count > 0; }),"contacts remain visible on unranked kinematic candidates");
}
void generated() {
  std::mt19937_64 rng(0x52454c49434bULL);
  std::size_t solved = 0;
  for (int n : {3,4}) {
    Simulation sim(fixture(n));
    for (int sample = 0; sample < 300; ++sample) {
      std::vector<double> q(static_cast<std::size_t>(n));
      for (std::size_t j = 0; j < q.size(); ++j) {
        const auto lim = sim.config().joint_limits_rad[j];
        const double u = static_cast<double>(rng() >> 11)*0x1.0p-53;
        q[j] = lim.lower+u*(lim.upper-lim.lower);
      }
      auto r = target(sim,q);
      for (std::size_t j = 0; j < r.current_rad.size(); ++j) {
        const auto lim = sim.config().joint_limits_rad[j];
        const double u = static_cast<double>(rng() >> 11)*0x1.0p-53;
        r.current_rad[j] = lim.lower+u*(lim.upper-lim.lower);
      }
      if (n == 4) { r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = q[3]; }
      const auto out = solve_position_ik(sim,r); verified(sim,r,out); ++solved;
    }
  }
  std::cout << "generated_solved=" << solved << " seed=0x52454c49434b\n";
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: analytical_ik_tests CASE EXAMPLES");
    examples = argv[2];
    const std::pair<const char*,void(*)()> cases[]{
      {"branches",branches},{"boundaries",boundaries},{"wrist",wrist},{"inputs",inputs},
      {"geometry",geometry},{"isolation",isolation},{"generated",generated}};
    for (const auto& [name,run] : cases) if (name == std::string(argv[1])) {
      run(); std::cout << "PASS " << name << " checks=" << checks << '\n'; return 0;
    }
    throw std::runtime_error("unknown test case");
  } catch (const std::exception& e) { std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n'; return 1; }
}
