#include "relic/feasibility.hpp"

#include <cmath>

namespace relic {
PathStatus validate_joint_path(
    const std::vector<std::vector<double>>& waypoints,
    const std::vector<JointLimit>& limits,
    double max_joint_step_rad
) {
    if (!std::isfinite(max_joint_step_rad) || max_joint_step_rad <= 0.0 ||
        limits.empty()) {
        return PathStatus::InvalidConstraints;
    }
    for (const JointLimit& limit : limits) {
        if (!std::isfinite(limit.lower) || !std::isfinite(limit.upper) ||
            limit.lower >= limit.upper) {
            return PathStatus::InvalidConstraints;
        }
    }
    if (waypoints.empty()) {
        return PathStatus::EmptyPath;
    }

    const std::size_t joint_count = limits.size();
    for (std::size_t waypoint_index = 0; waypoint_index < waypoints.size();
         ++waypoint_index) {
        const std::vector<double>& waypoint = waypoints[waypoint_index];
        if (waypoint.size() != joint_count) {
            return PathStatus::WrongJointCount;
        }

        for (std::size_t joint_index = 0; joint_index < joint_count;
             ++joint_index) {
            const double angle = waypoint[joint_index];
            if (!std::isfinite(angle)) {
                return PathStatus::NonFiniteJoint;
            }

            const JointLimit& limit = limits[joint_index];
            if (angle < limit.lower || angle > limit.upper) {
                return PathStatus::JointLimitViolation;
            }

            if (waypoint_index > 0) {
                const double previous_angle =
                    waypoints[waypoint_index - 1][joint_index];
                if (std::abs(angle - previous_angle) > max_joint_step_rad) {
                    return PathStatus::ExcessiveJointStep;
                }
            }
        }
    }

    return PathStatus::Valid;
}
}
