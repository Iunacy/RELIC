#pragma once

#include "relic/simulation.hpp"

#include <cstddef>
#include <string>

namespace relic {

enum class IkStatus {
  Success, InvalidInput, UnsupportedGeometry, Unreachable, UnreachableAtFixedWrist,
  NoCandidateWithinSearch, JointLimitsExcludeCandidates, DegenerateGeometry,
  FkValidationFailed, ResourceLimit, NumericalFailure
};
const char* ik_status_name(IkStatus status) noexcept;

enum class WristMode { Fixed, Sampled };
struct WristSearchOptions {
  WristMode mode = WristMode::Sampled;
  double fixed_rad = 0;


  std::size_t budget = 33;
};
struct PositionIkRequest {
  std::array<double, 3> position_world_m{};
  double position_tolerance_m = 1e-8;
  std::vector<double> current_rad;
  WristSearchOptions wrist_search;
  std::size_t candidate_budget = 16384;
};

struct IkBranch {
  int radius_sign = 0;
  int elbow_sign = 0;
  double wrist_rad = 0;
  bool free_yaw = false;
  bool free_shoulder = false;

  bool boundary_normalized = false;
  bool joint_boundary_reconstructed = false;
};
struct IkCandidate {
  std::vector<double> joints_rad;
  double fk_residual_m = 0;
  IkBranch branch;
  int planning_contact_count = 0;
};
enum class IkScope { ThreeJoint, FixedWrist, SampledWrist, PositionIndependentWrist };
struct IkCoverage {
  IkScope scope = IkScope::ThreeJoint;
  std::size_t requested_wrist_budget = 0;
  std::vector<double> sampled_wrist_rad;
  double max_wrist_gap_rad = 0;
  bool wrist_continuum_covered = false;
  bool branch_enumeration_complete = false;
  bool singular_family_representatives = false;
};
struct IkRejections {
  std::size_t geometric_wrist_misses = 0;
  std::size_t joint_limit_branches = 0;
  std::size_t fk_candidates = 0;
  std::size_t degenerate_wrist_samples = 0;
  std::size_t boundary_normalizations = 0;
};
struct PositionIkResult {
  IkStatus status = IkStatus::InvalidInput;
  std::string reason;
  IkCoverage coverage;
  IkRejections rejected;



  std::vector<IkCandidate> candidates;
};

struct IkGeometryCheck {
  IkStatus status = IkStatus::UnsupportedGeometry;
  std::string reason;
};

IkGeometryCheck check_analytical_geometry(const mjModel&, const RobotConfig&);



PositionIkResult solve_position_ik(Simulation&, const PositionIkRequest&);

}
