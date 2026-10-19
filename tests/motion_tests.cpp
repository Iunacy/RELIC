#include "relic/motion.hpp"
#include "relic/feasibility.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace {
using namespace relic;
std::filesystem::path examples; std::size_t checks = 0;
void require(bool x, const std::string& why) { ++checks; if (!x) throw std::runtime_error(why); }
RobotConfig config(int n = 3) { return load_config(examples/("synthetic_"+std::to_string(n)+"dof.cfg")); }
MotionOptions options(int n = 3) { MotionOptions o; o.max_velocity_rad_s.assign(static_cast<std::size_t>(n),0.5); return o; }
PositionIkRequest request(Simulation& sim, std::vector<double> goal, std::vector<double> start = {}) {
  PositionIkRequest r; r.position_world_m = sim.evaluate_fk(goal).position_m;
  r.current_rad = start.empty() ? std::vector<double>(goal.size(),0) : std::move(start); return r;
}
std::vector<mjtNum> live(const Simulation& sim) {
  std::vector<mjtNum> state(mj_stateSize(&sim.model(),mjSTATE_INTEGRATION));
  mj_getState(&sim.model(),&sim.execution_data(),state.data(),mjSTATE_INTEGRATION); return state;
}
void success(const ExecutionResult& r, const MotionOptions& o) {
  require(r.status == MotionStatus::Success,std::string("expected Success, got ")+motion_status_name(r.status)+": "+r.reason);
  require(r.settled && r.execution_started && r.final && r.final_pose_authoritative,"measured settled execution required");
  require(r.final_cartesian_error_m <= o.cartesian_tolerance_m,"observed Cartesian acceptance");
  for (double x : r.final_joint_error_rad) require(std::abs(x) <= o.joint_tolerance_rad,"observed joint acceptance");
  require(r.warnings.empty(),"successful run has no warnings");
}
void trajectory() {
  Simulation sim(config()); auto o = options();
  const std::vector<double> a{0,0,0}, b{0.56,-0.24,0.36};
  const auto t = make_joint_trajectory(a,b,sim.joint_limits(),0.002,o);
  require(t.status == MotionStatus::Success,"valid trajectory");
  require(t.trajectory.commands.front() == a && t.trajectory.commands.back() == b,"exact endpoints");
  require(validate_joint_path(t.trajectory.commands,sim.joint_limits(),o.max_command_step_rad) == PathStatus::Valid,"unchanged validator admits every command");
  for (std::size_t k = 1; k < t.trajectory.commands.size(); ++k) for (std::size_t j = 0; j < a.size(); ++j)
    require(std::abs(t.trajectory.commands[k][j]-t.trajectory.commands[k-1][j])/0.002 <= o.max_velocity_rad_s[j],"binary64 velocity bound");
  require(t.trajectory.acceleration_discontinuous,"acceleration discontinuity explicit");
  require(make_joint_trajectory(a,a,sim.joint_limits(),0.002,o).trajectory.commands.size() == 1,"zero displacement has no moving ticks");
  o.max_trajectory_steps = 1;
  require(make_joint_trajectory(a,b,sim.joint_limits(),0.002,o).status == MotionStatus::ResourceLimit,"step budget before allocation");
  o = options(); o.max_velocity_rad_s[0] = std::numeric_limits<double>::denorm_min();
  require(make_joint_trajectory(a,b,sim.joint_limits(),0.002,o).status == MotionStatus::ResourceLimit,"tiny velocity cannot overflow step count");
  o = options(); auto wrong = b; wrong[2] = 3;
  require(make_joint_trajectory(a,wrong,sim.joint_limits(),0.002,o).status == MotionStatus::InvalidInput,"invalid command rejected");
}
void adaptive() {
  Simulation sim(config(4)); auto o = options(4);
  auto r = request(sim,{0.4,-0.4,0.7,0.4}); const auto before = live(sim);
  auto plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::Success && plan.attempts.size() == 1 && plan.attempts[0].wrist_budget == 17,"immediate stage-17 success");
  r.position_world_m = {0.42693414739579011,0.024212786719060447,0.61208956526453306};
  plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::Success && plan.attempts.size() == 2 && plan.attempts.back().wrist_budget == 65,"reachable case escalates to 65");
  r.position_world_m = {-0.20232397716216027,-0.32762171046313787,0.65396120215840337};
  plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::Success && plan.attempts.size() == 3 && plan.attempts.back().wrist_budget == 257,"optional 257 escalation");
  SearchOptions search; search.wrist_budgets = {17,65}; search.total_wrist_budget = 82;
  plan = plan_cartesian_motion(sim,r,o,search);
  require(plan.status == MotionStatus::SearchIncomplete && !plan.selected,"configured two-stage incomplete miss");
  search = {}; search.total_wrist_budget = 82;
  require(plan_cartesian_motion(sim,r,o,search).status == MotionStatus::ResourceLimit,"257 requires explicit computation budget");
  r.position_world_m = {-0.15656020960157971,-0.55501236942387,0.28358833272306733};
  plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::SearchIncomplete && !plan.attempts.back().coverage.wrist_continuum_covered,"largest-budget miss remains incomplete");
  require(live(sim) == before,"all adaptive preflight preserves execution state");
}
void selection() {
  Simulation sim(config()); auto o = options(); auto r = request(sim,{0.35,-0.25,0.8});
  auto plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::Success && plan.selected,"a motion-feasible branch selected");
  for (const auto& a : plan.assessments) if (a.status == MotionStatus::Success)
    require(plan.selected_cost_rad2 <= a.squared_displacement_rad2,"minimum displacement only among feasible candidates");
  const auto again = plan_cartesian_motion(sim,r,o);
  require(again.selected->joints_rad == plan.selected->joints_rad,"repeatable selection");
  CandidateAssessment a,b; a.status = b.status = MotionStatus::Success;
  a.squared_displacement_rad2 = b.squared_displacement_rad2 = 1; a.candidate.joints_rad = {-1,0,0}; b.candidate.joints_rad = {1,0,0};
  require(prefer_motion_candidate(a,b) && !prefer_motion_candidate(b,a),"lexicographic exact tie");
  a.candidate.fk_residual_m = 1e-8;
  require(prefer_motion_candidate(b,a),"residual breaks equal-cost tie first");
  a.status = MotionStatus::DisallowedContact; a.squared_displacement_rad2 = 0;
  require(!prefer_motion_candidate(a,b),"hard rejection cannot win ranking");
  IkCandidate malformed; malformed.joints_rad = {0,0};
  require(preflight_candidate(sim,r,malformed,o,{},1000).assessment.status == MotionStatus::InvalidCandidate,"candidate dimension check");
  malformed.joints_rad = {0,0,0};
  require(preflight_candidate(sim,r,malformed,o,{},1000).assessment.status == MotionStatus::InvalidCandidate,"candidate authoritative FK check");
  require(preflight_candidate(sim,r,*plan.selected,o,{},1).assessment.status == MotionStatus::ResourceLimit,"collision work bounded");
  SearchOptions tiny; tiny.max_unique_candidates = 1;
  require(plan_cartesian_motion(sim,r,o,tiny).status == MotionStatus::ResourceLimit,"no partial ranking on candidate budget exhaustion");

  r = request(sim,{-2.7,-0.2,0.6},{2.7,0,0}); plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::Success && plan.selected_cost_rad2 > 29,"no wrapped angular distance across stops");
}
void contacts() {
  Simulation sim(config()); auto o = options(); auto r = request(sim,{0,0,std::numbers::pi/2});
  const auto before = live(sim); const auto plan = plan_cartesian_motion(sim,r,o);
  require(plan.status == MotionStatus::NoFeasibleCandidate && !plan.selected,"joint-valid collision-invalid candidates rejected");
  bool found = false;
  for (const auto& a : plan.assessments) if (a.status == MotionStatus::DisallowedContact) {
    found = true; require(!a.contacts.empty(),"contact evidence retained");
    require(!a.contacts[0].geom_a.empty() && std::isfinite(a.contacts[0].distance_m),"geom identity and distance");
  }
  require(found && live(sim) == before,"contact preflight does not move live arm");
  Simulation four(config(4)); auto four_request = request(four,{0,0,std::numbers::pi/2,0});
  const auto denied_all = plan_cartesian_motion(four,four_request,options(4));
  std::size_t reused = 0, newly_checked = 0;
  for (const auto& attempt : denied_all.attempts) { reused += attempt.reused_candidates; newly_checked += attempt.newly_checked_candidates; }
  require(denied_all.status == MotionStatus::NoFeasibleCandidate && reused > 0 && newly_checked == denied_all.assessments.size(),"adaptive collision rejection reuses identical prior assessments");
  const auto denied = execute_cartesian_motion(sim,r,o);
  require(!denied.execution_started && live(sim) == before,"denied plan never resets or moves arm");
  sim.evaluate_fk(std::vector<double>{0,0,std::numbers::pi/2});
  auto observed = inspect_contacts(sim.model(),sim.planning_data(),{}); ContactPolicy allow;
  for (const auto& c : observed) allow.allowed_pairs.emplace_back(c.geom_a,c.geom_b);
  require(!allow.allowed_pairs.empty(),"known actual MuJoCo contact fixture");
  auto admitted = inspect_contacts(sim.model(),sim.planning_data(),allow);
  require(!has_disallowed_contact(admitted) && admitted[0].classification == ContactClass::ExplicitlyAllowed,"explicit unordered allowlist classification");
  std::swap(allow.allowed_pairs[0].first,allow.allowed_pairs[0].second);
  require(!has_disallowed_contact(inspect_contacts(sim.model(),sim.planning_data(),allow)),"allowlist pair is unordered");
  r.current_rad = {0,0,std::numbers::pi/2}; o.timeout_s = 0.002; o.settling_dwell_s = 0.02;
  const auto allowed_run = execute_cartesian_motion(sim,r,o,{},allow);
  require(allowed_run.execution_started && !allowed_run.contacts.empty(),"actual observed contacts recorded when explicitly permitted");
}
void execution() {
  for (int n : {3,4}) {
    Simulation sim(config(n)); auto o = options(n);
    for (double start_yaw : {0.0,-0.3,0.25}) {
      std::vector<double> goal{0.5,-0.3,0.6}, start{start_yaw,-0.1,0.1};
      if (n == 4) { goal.push_back(0.3); start.push_back(0.1); }
      auto r = request(sim,goal,start); std::ostringstream csv; TraceWriter trace(csv,sim.config());
      const auto run = execute_cartesian_motion(sim,r,o,{}, {},&trace); success(run,o);
      require(run.first_command_before_step && run.first_command_before_step->observed_rad == start,"first issued actuator command cannot teleport qpos");
      require(run.peak_tracking_error_rad > 1e-4,"actual actuator tracking delay measured");
      require(run.final->observed_rad != run.final->command_rad,"observed joints are independent of commands");
      require(csv.str().find("_observed_rad") != std::string::npos && csv.str().size() > 1000,"usable trace saved");
    }
    std::vector<double> zero(static_cast<std::size_t>(n),0); auto r = request(sim,zero);
    const auto run = execute_cartesian_motion(sim,r,o); success(run,o);
    require(run.command_duration_s == 0 && !run.plan.trajectory.acceleration_discontinuous,"zero-displacement command only settles");
    if (n == 4) {
      r = request(sim,{0.2,-0.4,0.8,0.7}); r.wrist_search.mode = WristMode::Fixed; r.wrist_search.fixed_rad = 0.7;
      const auto fixed = execute_cartesian_motion(sim,r,o); success(fixed,o);
      require(fixed.plan.attempts.size() == 1 && fixed.plan.selected->joints_rad[3] == 0.7,"fixed-wrist execution preserves requested angle");
    }
  }
}
void failures() {
  Simulation sim(config()); auto o = options(); auto r = request(sim,{0.6,-0.3,0.5});
  sim.set_position_commands(std::vector<double>{-0.1,0,0}); sim.step(5); const auto before = live(sim);
  auto invalid = [&](const MotionOptions& bad) {
    const auto run = execute_cartesian_motion(sim,r,bad);
    require(run.status == MotionStatus::InvalidInput && !run.execution_started && live(sim) == before,"invalid constraints preserve live state");
  };
  for (double x : {0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    auto bad = o; bad.collision_step_rad = x; invalid(bad);
    bad = o; bad.max_velocity_rad_s[0] = x; invalid(bad);
    bad = o; bad.settling_dwell_s = x; invalid(bad);
  }
  auto bad = o; bad.max_velocity_rad_s.pop_back(); invalid(bad);
  bad = o; bad.timeout_s = std::numeric_limits<double>::max(); invalid(bad);
  bad = o; bad.max_trajectory_steps = 100001; invalid(bad);
  auto malformed = r; malformed.current_rad[0] = 4;
  require(!execute_cartesian_motion(sim,malformed,o).execution_started && live(sim) == before,"bad start never resets");
  malformed = r; malformed.position_world_m = {5,5,5};
  require(execute_cartesian_motion(sim,malformed,o).status == MotionStatus::IkFailure && live(sim) == before,"proven unreachable diagnostic");
  ContactPolicy wrong; wrong.allowed_pairs.emplace_back("missing","ground");
  require(execute_cartesian_motion(sim,r,o,{},wrong).status == MotionStatus::InvalidInput,"bad allowlist rejected");
  o.timeout_s = 0.01; auto run = execute_cartesian_motion(sim,r,o);
  require(run.status == MotionStatus::Timeout && run.execution_started && run.final && run.final->time_s <= 0.010000001,"partial-motion timeout measured");
  o = options(); o.timeout_s = 2; o.cartesian_tolerance_m = 1e-12; o.joint_tolerance_rad = 1e-12;
  run = execute_cartesian_motion(sim,r,o);
  require(run.status == MotionStatus::NonConvergence && !run.settled && run.final_cartesian_error_m > 1e-12,"unmet settling acceptance fails explicitly");
  o = options(); std::ostringstream csv; TraceWriter trace(csv,sim.config()); csv.setstate(std::ios::badbit);
  run = execute_cartesian_motion(sim,r,o,{}, {},&trace);
  require(run.status == MotionStatus::TraceFailure && run.simulated_duration_s == 0,"trace failure stops before physics");
  Simulation four(config(4)); auto r4 = request(four,{0,0,0,0});
  r4.position_world_m = {-0.26883530825051655,0.41461810528088366,-0.028854115890642261};
  const auto contact_run = execute_cartesian_motion(four,r4,options(4));
  require(contact_run.plan.status == MotionStatus::Success && contact_run.status == MotionStatus::DisallowedContact &&
    !contact_run.settled && !contact_run.contacts.empty(),"actual ground contact despite admitted commanded path fails explicitly");
  require(four.snapshot().observed_rad == contact_run.final->observed_rad,"contact failure preserves observed diagnostic state");
  four.evaluate_fk(contact_run.plan.selected->joints_rad);
  const int hub = mj_name2id(&four.model(),mjOBJ_GEOM,"wrist_hub");
  const double clearance = four.planning_data().geom_xpos[3*hub+2]-four.model().geom_size[3*hub];
  std::cout << "admitted_command_wrist_clearance_m=" << clearance << " observed_contact_distance_m=" << contact_run.contacts.back().contact.distance_m << '\n';
}
void diagnostics() {
  Simulation sim(config()); auto o = options(); auto r = request(sim,{1.5,0,0});
  o.max_velocity_rad_s.assign(3,1000); o.max_command_step_rad = 2;
  auto run = execute_cartesian_motion(sim,r,o); success(run,o);
  require(run.saturation_samples[0] > 0 && run.peak_actuator_abs_Nm[0] >= 19.999,"actual unchanged actuator saturation observed");
  const auto before = live(sim);
  std::unique_ptr<mjData,DataDeleter> scratch(mj_makeData(&sim.model()));
  mj_resetData(&sim.model(),scratch.get()); scratch->ctrl[0] = std::numeric_limits<double>::quiet_NaN();
  mj_step(&sim.model(),scratch.get());
  require(!numerical_warnings(*scratch).empty() && execution_health(sim.model(),*scratch) == MotionStatus::NumericalWarning,"actual independent MuJoCo bad-control warning mapped to failure");
  mj_resetData(&sim.model(),scratch.get()); scratch->warning[mjWARN_CNSTRFULL].number = 1;
  require(execution_health(sim.model(),*scratch) == MotionStatus::NumericalWarning,"non-BAD warning also fatal");
  mj_resetData(&sim.model(),scratch.get()); scratch->qpos[0] = 3;
  require(execution_health(sim.model(),*scratch) == MotionStatus::ObservedJointLimit,"observed violation distinct from command admission");
  require(live(sim) == before,"diagnostic tests never mutate live simulation");
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: relic_motion_tests CASE EXAMPLES");
    examples = argv[2];
    const std::pair<const char*,void(*)()> cases[]{{"trajectory",trajectory},{"adaptive",adaptive},{"selection",selection},
      {"contacts",contacts},{"execution",execution},{"failures",failures},{"diagnostics",diagnostics}};
    for (const auto& [name,run] : cases) if (std::string(argv[1]) == name) { run(); std::cout << "PASS " << name << " checks=" << checks << '\n'; return 0; }
    throw std::runtime_error("unknown test group");
  } catch (const std::exception& e) { std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n'; return 1; }
}
