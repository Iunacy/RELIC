#pragma once

#include "relic/simulation.hpp"

#include <vector>

namespace relic {
enum class PathStatus {
    Valid,
    EmptyPath,
    InvalidConstraints,
    WrongJointCount,
    NonFiniteJoint,
    JointLimitViolation,
    ExcessiveJointStep
};

PathStatus validate_joint_path(
    const std::vector<std::vector<double>>& waypoints,
    const std::vector<JointLimit>& limits,
    double max_joint_step_rad
);

}
