#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <filesystem>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace relic {

struct JointLimit { double lower; double upper; };
struct Pose {
  std::array<double, 3> position_m{};
  std::array<double, 4> quaternion_wxyz{};
};

struct RobotConfig {
  std::string id;
  int dof = 3;
  std::array<double, 2> link_lengths_m{0.30, 0.25};
  double pedestal_height_m = 0.20;
  std::vector<JointLimit> joint_limits_rad;
  Pose tool_frame{{0.06, 0, 0}, {1, 0, 0, 0}};
};

void validate(const RobotConfig& config);
RobotConfig load_config(const std::filesystem::path& path);
std::string make_mjcf(const RobotConfig& config);

struct Snapshot {
  double time_s = 0;
  std::vector<double> command_rad;
  std::vector<double> observed_rad;
  std::vector<double> velocity_rad_s;
  std::vector<double> actuator_torque_Nm;
  Pose tool;
  int contact_count = 0;
};

struct ModelDeleter { void operator()(mjModel* p) const noexcept; };
struct DataDeleter { void operator()(mjData* p) const noexcept; };



struct PhysicsParameters {
  double gain_scale = 1, force_scale = 1, damping_scale = 1;
  double friction_Nm = 0, mass_scale = 1, payload_kg = 0;
};

class Simulation {
 public:
  explicit Simulation(RobotConfig config);
  Simulation(RobotConfig config, const PhysicsParameters& physics);
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  Simulation(Simulation&&) = delete;
  Simulation& operator=(Simulation&&) = delete;

  const RobotConfig& config() const noexcept { return config_; }
  std::vector<JointLimit> joint_limits() const;
  void reset(std::span<const double> q_rad);
  void set_position_commands(std::span<const double> q_rad);

  void set_external_torques(std::span<const double> torque_Nm);
  void step(std::size_t count = 1);
  Snapshot snapshot() const;
  Pose evaluate_fk(std::span<const double> q_rad);

  const mjModel& model() const noexcept { return *model_; }
  const mjData& execution_data() const noexcept { return *execution_; }
  const mjData& planning_data() const noexcept { return *planning_; }

 private:
  void check_joints(std::span<const double> q_rad) const;
  void initialize(mjData& data, std::span<const double> q_rad);
  Pose tool_pose(const mjData& data) const;
  RobotConfig config_;
  std::unique_ptr<mjModel, ModelDeleter> model_;
  std::unique_ptr<mjData, DataDeleter> execution_;
  std::unique_ptr<mjData, DataDeleter> planning_;
  std::vector<int> joint_ids_, qpos_addresses_, dof_addresses_, actuator_ids_;
  int tool_site_ = -1;
};

class TraceWriter {
 public:
  TraceWriter(std::ostream& out, const RobotConfig& config);
  void append(const Snapshot& state);
 private:
  std::ostream& out_;
  RobotConfig config_;
};

}
