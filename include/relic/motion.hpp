#pragma once
#include "relic/analytical_ik.hpp"
#include "relic/contact_inspection.hpp"
#include <optional>

namespace relic {
enum class MotionStatus {
  Success, InvalidInput, IkFailure, SearchIncomplete, NoFeasibleCandidate,
  InvalidCandidate, InvalidPath, ResourceLimit, DisallowedContact,
  ObservedJointLimit, NumericalWarning, SimulationError, Timeout, NonConvergence, TraceFailure
};
const char* motion_status_name(MotionStatus) noexcept;
struct MotionOptions {
  std::vector<double> max_velocity_rad_s;
  double max_command_step_rad = 0.005;
  double collision_step_rad = 0.005;
  std::size_t max_trajectory_steps = 20000;
  double timeout_s = 10;
  double joint_tolerance_rad = 0.01;
  double cartesian_tolerance_m = 0.005;
  double settling_velocity_rad_s = 0.01;
  double settling_dwell_s = 0.25;
};
struct SearchOptions {
  std::vector<std::size_t> wrist_budgets{17,65,257};
  std::size_t total_wrist_budget = 339;
  std::size_t max_unique_candidates = 2048;
  std::size_t max_collision_samples = 500000;
};
struct JointTrajectory {
  double timestep_s = 0;
  std::vector<std::vector<double>> commands;
  std::vector<double> nominal_velocity_rad_s;
  double command_duration_s = 0;
  bool acceleration_discontinuous = false;
};
struct TrajectoryResult {
  MotionStatus status = MotionStatus::InvalidInput;
  std::string reason;
  JointTrajectory trajectory;
};
TrajectoryResult make_joint_trajectory(std::span<const double> start, std::span<const double> goal,
  const std::vector<JointLimit>& limits, double timestep_s, const MotionOptions&);

struct CandidateAssessment {
  IkCandidate candidate;
  MotionStatus status = MotionStatus::InvalidCandidate;
  std::string reason;
  double squared_displacement_rad2 = 0;
  std::size_t collision_samples = 0;
  std::vector<ContactFinding> contacts;
  double contact_sample_time_s = 0;
};
struct PreflightResult {
  CandidateAssessment assessment;
  JointTrajectory trajectory;
};
PreflightResult preflight_candidate(Simulation&, const PositionIkRequest&, const IkCandidate&,
  const MotionOptions&, const ContactPolicy&, std::size_t collision_sample_budget);

bool prefer_motion_candidate(const CandidateAssessment&, const CandidateAssessment&);

struct SearchAttempt {
  std::size_t wrist_budget = 0;
  IkStatus ik_status = IkStatus::InvalidInput;
  std::string reason;
  IkCoverage coverage;
  IkRejections kinematic_rejections;
  std::size_t returned_candidates = 0, reused_candidates = 0, newly_checked_candidates = 0;
  double ik_wall_ms = 0;
};
struct MotionPlan {
  MotionStatus status = MotionStatus::InvalidInput;
  std::string reason;
  std::vector<SearchAttempt> attempts;
  std::vector<CandidateAssessment> assessments;
  std::optional<IkCandidate> selected;
  JointTrajectory trajectory;
  Pose command_fk;
  double selected_cost_rad2 = 0;
  std::size_t total_collision_samples = 0;
  double ik_wall_ms = 0, preflight_wall_ms = 0, planning_wall_ms = 0;
};
MotionPlan plan_cartesian_motion(Simulation&, const PositionIkRequest&, const MotionOptions&,
  const SearchOptions& = {}, const ContactPolicy& = {});

struct NumericalWarning { int index = 0, count = 0, last_info = 0; };
std::vector<NumericalWarning> numerical_warnings(const mjData&);

MotionStatus execution_health(const mjModel&, const mjData&);
struct ExecutionContactEvent { double time_s = 0; ContactFinding contact; };
struct ExecutionResult {
  MotionStatus status = MotionStatus::InvalidInput;
  std::string reason;
  std::array<double,3> requested_position_m{};
  MotionPlan plan;
  bool execution_started = false, settled = false, final_pose_authoritative = false;
  std::optional<Snapshot> initial, first_command_before_step, final;
  std::vector<double> final_joint_error_rad;
  double final_cartesian_error_m = 0, peak_tracking_error_rad = 0;
  double command_duration_s = 0, executed_command_duration_s = 0, simulated_duration_s = 0, settling_duration_s = 0;
  double accepted_settling_dwell_s = 0, execution_wall_ms = 0, total_wall_ms = 0;
  std::vector<std::size_t> saturation_samples;
  std::vector<double> peak_actuator_abs_Nm;
  std::vector<NumericalWarning> warnings;
  std::vector<ExecutionContactEvent> contacts;
  std::size_t contact_events_total = 0;
};

ExecutionResult execute_cartesian_motion(Simulation&, const PositionIkRequest&, const MotionOptions&,
  const SearchOptions& = {}, const ContactPolicy& = {}, TraceWriter* = nullptr);
void write_execution_report(std::ostream&, const ExecutionResult&);
}
