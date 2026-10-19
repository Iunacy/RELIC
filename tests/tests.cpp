#include "relic/simulation.hpp"
#include "relic/feasibility.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {
std::filesystem::path examples;
int checks = 0;
void require(bool condition, const std::string& what) {
  ++checks;
  if (!condition) throw std::runtime_error(what);
}
void near(double a, double b, double tol = 1e-12) {
  require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tol,
          "expected " + std::to_string(a) + " ~= " + std::to_string(b));
}
void same_pose(const relic::Pose& a, const relic::Pose& b) {
  for (int i = 0; i < 3; ++i) near(a.position_m[i], b.position_m[i]);
  for (int i = 0; i < 4; ++i) near(a.quaternion_wxyz[i], b.quaternion_wxyz[i]);
}
template<class F> void rejects(F action, const std::string& message) {
  ++checks;
  try { action(); }
  catch (const std::exception& e) {
    if (std::string(e.what()).find(message) != std::string::npos) return;
    throw std::runtime_error("wrong rejection: " + std::string(e.what()));
  }
  throw std::runtime_error("expected rejection containing: " + message);
}
relic::RobotConfig fixture(int dof = 3) {
  return relic::load_config(examples / ("synthetic_" + std::to_string(dof) + "dof.cfg"));
}
std::vector<mjtNum> integration(const relic::Simulation& sim) {
  std::vector<mjtNum> state(mj_stateSize(&sim.model(), mjSTATE_INTEGRATION));
  mj_getState(&sim.model(), &sim.execution_data(), state.data(), mjSTATE_INTEGRATION);
  return state;
}

void models() {
  for (int n : {3, 4}) {
    const auto c = fixture(n);
    relic::Simulation sim(c);
    const auto& m = sim.model();
    require(m.nq == n && m.nv == n && m.njnt == n && m.nu == n, "joint/actuator counts");
    near(m.opt.timestep, 0.002);
    near(m.opt.gravity[2], -9.81);
    const auto limits = sim.joint_limits();
    for (int i = 0; i < n; ++i) {
      near(limits[i].lower, c.joint_limits_rad[i].lower);
      near(limits[i].upper, c.joint_limits_rad[i].upper);
      require(m.jnt_limited[i] && m.actuator_ctrllimited[i] && m.actuator_forcelimited[i], "limits enabled");
      for (int k = 0; k < 3; ++k) {
        const int axis = i == 0 ? 2 : (i == 3 ? 0 : 1);
        near(m.jnt_axis[3*i+k], k == axis ? 1 : 0);
      }
    }
    const int link = mj_name2id(&m, mjOBJ_GEOM, "link_1");
    require(m.geom_contype[link] != 0 && m.geom_conaffinity[link] != 0, "link collisions enabled");
    sim.step(10);
    require(sim.snapshot().time_s > 0, "both variants simulate");
  }
}

void fk() {
  for (int n : {3, 4}) {
    auto c = fixture(n);
    c.tool_frame = {{0.06, 0, 0}, {1, 0, 0, 0}};
    relic::Simulation sim(c);
    std::vector<double> q(n, 0);
    const auto zero = sim.evaluate_fk(q);
    near(zero.position_m[0], c.link_lengths_m[0] + c.link_lengths_m[1] + 0.06);
    near(zero.position_m[1], 0);
    near(zero.position_m[2], c.pedestal_height_m);
    near(zero.quaternion_wxyz[0], 1);
    q[0] = std::numbers::pi/2;
    const auto yaw = sim.evaluate_fk(q);
    near(yaw.position_m[0], 0);
    near(yaw.position_m[1], zero.position_m[0]);
    near(yaw.quaternion_wxyz[0], std::sqrt(0.5));
    near(yaw.quaternion_wxyz[3], std::sqrt(0.5));
    q[0] = 0; q[2] = std::numbers::pi/2;
    const auto elbow = sim.evaluate_fk(q);
    near(elbow.position_m[0], c.link_lengths_m[0]);
    near(elbow.position_m[2], c.pedestal_height_m - c.link_lengths_m[1] - 0.06);
    require(sim.planning_data().ncon > 0, "instantaneous collision capability remains available");
    q[0] = 0.35; q[1] = -0.25; q[2] = 0.4;
    if (n == 4) q[3] = 0.6;
    const auto pose = sim.evaluate_fk(q);
    for (int i = 0; i < 20; ++i) same_pose(pose, sim.evaluate_fk(q));
    relic::Simulation independent(c);
    same_pose(pose, independent.evaluate_fk(q));
    same_pose(zero, sim.evaluate_fk(std::vector<double>(n, 0)));
  }
  auto c = fixture(4);
  c.tool_frame = {{0, 0.1, 0}, {std::sqrt(0.5), 0, 0, std::sqrt(0.5)}};
  relic::Simulation sim(c);
  const auto pose = sim.evaluate_fk(std::vector<double>{0, 0, 0, std::numbers::pi/2});
  near(pose.position_m[0], c.link_lengths_m[0] + c.link_lengths_m[1]);
  near(pose.position_m[1], 0);
  near(pose.position_m[2], c.pedestal_height_m + 0.1);
  near(pose.quaternion_wxyz[0], 0.5); near(pose.quaternion_wxyz[1], 0.5);
  near(pose.quaternion_wxyz[2], -0.5); near(pose.quaternion_wxyz[3], 0.5);
}

void validation() {
  const auto good = fixture();
  auto bad = [&](auto mutate, const std::string& diagnostic) {
    auto c = good; mutate(c);
    rejects([&] { relic::Simulation sim(c); }, diagnostic);
  };
  for (double invalid : {0.0, -0.3, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    bad([&](auto& c) { c.link_lengths_m[0] = invalid; }, "link_lengths_m");
    bad([&](auto& c) { c.pedestal_height_m = invalid; }, "pedestal_height_m");
  }
  bad([](auto& c) { c.dof = 5; }, "dof");
  bad([](auto& c) { c.id = "bad,\"id"; }, "id");
  bad([](auto& c) { c.joint_limits_rad.pop_back(); }, "count");
  bad([](auto& c) { c.joint_limits_rad[0] = {1, 2}; }, "including zero");
  bad([](auto& c) { c.joint_limits_rad[0] = {1, -1}; }, "lower < upper");
  bad([](auto& c) { c.joint_limits_rad[0].upper = std::numeric_limits<double>::infinity(); }, "finite");
  bad([](auto& c) { c.tool_frame.quaternion_wxyz = {0, 0, 0, 0}; }, "unit length");
  bad([](auto& c) { c.tool_frame.quaternion_wxyz[1] = std::numeric_limits<double>::quiet_NaN(); }, "finite");
  bad([](auto& c) { c.tool_frame.position_m[2] = std::numeric_limits<double>::infinity(); }, "finite");
  relic::Simulation sim(good);
  const auto before = integration(sim);
  for (const auto& q : std::vector<std::vector<double>>{{0, 0}, {3, 0, 0}, {0, 0, -2},
       {0, std::numeric_limits<double>::quiet_NaN(), 0}, {0, 0, std::numeric_limits<double>::infinity()}}) {
    rejects([&] { sim.reset(q); }, "joint state");
    rejects([&] { sim.set_position_commands(q); }, "joint state");
    rejects([&] { sim.evaluate_fk(q); }, "joint state");
    require(integration(sim) == before, "rejected input preserves live state");
  }
  sim.evaluate_fk(std::vector<double>{-2.8, -1.2, -1.8});
  sim.evaluate_fk(std::vector<double>{2.8, 1.2, 1.8});
  rejects([&] { sim.step(0); }, "step count");
  rejects([&] { sim.step(1000001); }, "step count");
}

void feasibility() {
  using relic::PathStatus;
  const std::vector<relic::JointLimit> limits{{-1.0, 1.0}, {-2.0, 2.0}};
  const auto validate_path = [](const std::vector<std::vector<double>>& path,
                                const std::vector<relic::JointLimit>& path_limits,
                                double max_step) {
    return relic::validate_joint_path(path, path_limits, max_step);
  };
  const auto status = [&](const std::vector<std::vector<double>>& path) {
    return validate_path(path, limits, 0.5);
  };

  require(status({{0.0, 0.0}, {0.5, -0.5}}) == PathStatus::Valid,
          "valid path accepts inclusive joint limits and step threshold");
  require(status({{1.0, -2.0}}) == PathStatus::Valid,
          "single waypoint accepts joint-limit boundaries");
  require(status({}) == PathStatus::EmptyPath, "empty path is rejected");
  require(validate_path({{0.0, 0.0}}, {}, 0.5) == PathStatus::InvalidConstraints,
          "empty joint limits are invalid");
  require(validate_path({{0.0, 0.0}}, limits, 0.0) == PathStatus::InvalidConstraints,
          "zero step threshold is invalid");
  require(validate_path({{0.0, 0.0}}, limits, -0.5) == PathStatus::InvalidConstraints,
          "negative step threshold is invalid");
  require(validate_path({{0.0, 0.0}}, limits,
                        std::numeric_limits<double>::infinity()) == PathStatus::InvalidConstraints,
          "infinite step threshold is invalid");
  require(validate_path({{0.0, 0.0}}, limits,
                        std::numeric_limits<double>::quiet_NaN()) == PathStatus::InvalidConstraints,
          "NaN step threshold is invalid");
  require(validate_path({{0.0, 0.0}}, {{1.0, -1.0}}, 0.5) ==
              PathStatus::InvalidConstraints,
          "reversed joint limits are invalid");
  require(validate_path({{0.0, 0.0}},
                        {{-std::numeric_limits<double>::infinity(), 1.0}}, 0.5) ==
              PathStatus::InvalidConstraints,
          "non-finite joint limits are invalid");
  require(status({{0.0}}) == PathStatus::WrongJointCount,
          "waypoint joint count is checked");
  require(status({{0.0, 0.0}, {0.0}}) == PathStatus::WrongJointCount,
          "joint count is checked on later waypoints");
  require(status({{0.0, 0.0},
                  {std::numeric_limits<double>::quiet_NaN(), 0.0}}) ==
              PathStatus::NonFiniteJoint,
          "non-finite angles are rejected");
  require(status({{0.0, 0.0}, {1.1, 0.0}}) == PathStatus::JointLimitViolation,
          "joint upper limits are enforced");
  require(status({{0.0, 0.0}, {-1.1, 0.0}}) == PathStatus::JointLimitViolation,
          "joint lower limits are enforced");
  require(status({{0.0, 0.0}, {0.0, -0.6}}) == PathStatus::ExcessiveJointStep,
          "step is checked for each joint");
  require(status({{0.0, 0.0}, {0.0, 0.0}, {0.6, 0.0}}) ==
              PathStatus::ExcessiveJointStep,
          "step is checked between consecutive waypoints");
}

void config() {
  std::ifstream input(examples / "synthetic_3dof.cfg");
  const std::string base{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  struct Temporary {
    std::filesystem::path path = "relic_bad_config_test.cfg";
    ~Temporary() { std::error_code error; std::filesystem::remove(path, error); }
  } temporary;
  auto invalid = [&](const std::string& contents, const std::string& message) {
    { std::ofstream out(temporary.path); out << contents; }
    rejects([&] { relic::load_config(temporary.path); }, message);
  };
  invalid(base + "\ndof = 3\n", "duplicate");
  invalid(base + "\nunknown = 1\n", "unknown");
  invalid(base + "\nno equals sign\n", "expected key");
  invalid("id = incomplete\n", "missing");
  auto replace = [&](const std::string& from, const std::string& to) {
    auto text = base; text.replace(text.find(from), from.size(), to); return text;
  };
  invalid(replace("provenance = synthetic", "provenance = official"), "only synthetic");
  invalid(replace("dof = 3", "dof = 3.5"), "dof");
  invalid(replace("0.30 0.25", "0.3 0.25 junk"), "expected 2 numbers");
  invalid(replace("0.30 0.25", "0.3"), "expected 2 numbers");
  invalid(replace("0.30 0.25", "1e999 0.2"), "expected 2 numbers");
  rejects([&] { relic::load_config(examples / "does-not-exist.cfg"); }, "cannot open");
  require(fixture(4).dof == 4, "four-joint file loads");
}

void isolation() {
  relic::Simulation sim(fixture(4));
  sim.set_position_commands(std::vector<double>{0.2, -0.1, 0.2, 0.3});
  sim.step(100);
  const auto before = integration(sim);
  const auto pose = sim.snapshot().tool;
  require(&sim.execution_data() != &sim.planning_data(), "distinct mjData objects");
  require(sim.execution_data().qpos != sim.planning_data().qpos, "distinct state storage");
  for (int i = 0; i < 30; ++i) sim.evaluate_fk(std::vector<double>{-0.5, 0.3, -0.2, -0.1});
  require(before == integration(sim), "FK preserves entire execution integration state");
  same_pose(pose, sim.snapshot().tool);
  near(sim.planning_data().time, 0);
  const double planning_q = sim.planning_data().qpos[0];
  rejects([&] { sim.evaluate_fk(std::vector<double>{99, 0, 0, 0}); }, "joint state");
  near(planning_q, sim.planning_data().qpos[0]);
}

void reset() {
  for (int n : {3, 4}) {
    relic::Simulation sim(fixture(n));
    const std::vector<double> start(n, 0), target(n, 0.1);
    const auto initial = integration(sim);
    auto run = [&] {
      std::ostringstream out;
      relic::TraceWriter trace(out, sim.config());
      sim.set_position_commands(target);
      for (int i = 0; i < 100; ++i) { sim.step(); trace.append(sim.snapshot()); }
      return out.str();
    };
    const auto first = run();
    near(sim.snapshot().time_s, 0.2);
    same_pose(sim.snapshot().tool, sim.evaluate_fk(sim.snapshot().observed_rad));
    sim.reset(start);
    require(integration(sim) == initial, "reset restores all integration fields");
    for (double v : sim.snapshot().velocity_rad_s) near(v, 0);
    require(run() == first, "reset then replay is byte-identical");
    sim.reset(start);
    sim.set_position_commands(target);
    sim.step(100);
    relic::Simulation other(fixture(n));
    other.set_position_commands(target);
    for (int i = 0; i < 100; ++i) other.step();
    for (int i = 0; i < n; ++i) near(sim.snapshot().observed_rad[i], other.snapshot().observed_rad[i]);
  }
}

void actuators() {
  for (int n : {3, 4}) {
    relic::Simulation sim(fixture(n));
    const auto& m = sim.model();
    for (int i = 0; i < n; ++i) {
      const int aid = i;
      require(m.actuator_trnid[2*aid] == i && m.actuator_trntype[aid] == mjTRN_JOINT, "actuator mapping");
      near(m.actuator_gear[6*aid], 1);
      near(m.actuator_ctrlrange[2*aid], sim.joint_limits()[i].lower);
      near(m.actuator_ctrlrange[2*aid+1], sim.joint_limits()[i].upper);
      sim.reset(std::vector<double>(n, 0));
      std::vector<double> target(n, 0); target[i] = 0.15;
      sim.set_position_commands(target);
      require(sim.snapshot().observed_rad == std::vector<double>(n, 0), "control does not teleport qpos");
      near(sim.execution_data().ctrl[aid], 0.15);
      require(sim.snapshot().actuator_torque_Nm[i] > 0, "nonzero actuator force before step");
      sim.step(400);
      const double observed = sim.snapshot().observed_rad[i];
      require(observed > 0.05 && std::abs(observed - 0.15) < 0.08, "actual actuator-driven motion approaches target");
      for (int w = 0; w < mjNWARNING; ++w) require(sim.execution_data().warning[w].number == 0, "no physics warnings");
    }
  }
}

void lifetime() {
  static_assert(!std::is_copy_constructible_v<relic::Simulation>);
  static_assert(!std::is_move_constructible_v<relic::Simulation>);
  for (int i = 0; i < 100; ++i) {
    relic::Simulation sim(fixture(i % 2 ? 3 : 4));
    sim.step(2);
    require(sim.execution_data().qpos != sim.planning_data().qpos, "independent allocations");
  }
  auto c = fixture();
  c.link_lengths_m[0] = 1e-300;
  rejects([&] { relic::Simulation sim(c); }, "MJCF compile failed");
  relic::Simulation recovered(fixture()); recovered.step();
  require(recovered.snapshot().time_s > 0, "recovery after constructor exception");
}

void trace() {
  for (int n : {3, 4}) {
    relic::Simulation sim(fixture(n));
    std::ostringstream out;
    relic::TraceWriter writer(out, sim.config());
    writer.append(sim.snapshot());
    sim.step(); writer.append(sim.snapshot());
    std::istringstream input(out.str());
    std::string header, row;
    std::getline(input, header);
    require(header.find("time_s,pose_frame,tool_x_m") != std::string::npos, "time/frame/units schema");
    require(header.find("j" + std::to_string(n-1) + "_actuator_torque_Nm") != std::string::npos, "all joints exported");
    int rows = 0;
    while (std::getline(input, row)) {
      require(std::count(row.begin(), row.end(), ',') == 12 + 4*n, "CSV column count");
      require(row.find(sim.config().id + ",synthetic," + mj_versionString()) == 0, "configuration and provenance metadata");
      require(row.find(",world,") != std::string::npos, "world frame");
      std::istringstream cells(row);
      std::string cell;
      for (int i = 0; i < 4; ++i) std::getline(cells, cell, ',');
      near(std::stod(cell), rows * 0.002);
      ++rows;
    }
    require(rows == 2, "two sample rows");
    auto malformed = sim.snapshot(); malformed.observed_rad.pop_back();
    rejects([&] { writer.append(malformed); }, "joint count");
    malformed = sim.snapshot(); malformed.tool.position_m[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { writer.append(malformed); }, "non-finite");
    std::ostringstream broken; broken.setstate(std::ios::badbit);
    rejects([&] { relic::TraceWriter fail(broken, sim.config()); }, "CSV write failed");
    out.setstate(std::ios::badbit);
    rejects([&] { writer.append(sim.snapshot()); }, "CSV write failed");
  }
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: relic_tests CASE EXAMPLES_DIR");
    examples = argv[2];
    const std::vector<std::pair<std::string, std::function<void()>>> cases{
      {"models", models}, {"fk", fk}, {"validation", validation}, {"config", config},
      {"isolation", isolation}, {"reset", reset}, {"actuators", actuators},
      {"lifetime", lifetime}, {"trace", trace}, {"feasibility", feasibility}};
    for (const auto& [name, run] : cases) {
      if (name == argv[1]) { run(); std::cout << "PASS " << name << " (" << checks << " checks)\n"; return 0; }
    }
    throw std::runtime_error("unknown test case");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << " (after " << checks << " checks)\n";
    return 1;
  }
}
