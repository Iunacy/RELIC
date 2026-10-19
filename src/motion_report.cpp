#include "relic/motion.hpp"
#include <iomanip>
#include <locale>

namespace relic {
void write_execution_report(std::ostream& out, const ExecutionResult& r) {
  out.imbue(std::locale::classic()); out << std::setprecision(17);
  auto values = [&](const char* name, const auto& v) { out << name << '='; for (auto x : v) out << x << ' '; out << '\n'; };
  out << "status=" << motion_status_name(r.status) << " reason=" << r.reason << '\n';
  values("requested_world_m",r.requested_position_m);
  out << "execution_started=" << r.execution_started << " settled=" << r.settled
      << " final_pose_authoritative=" << r.final_pose_authoritative << '\n';
  if (r.plan.selected) {
    const auto& c = *r.plan.selected; const auto& b = c.branch;
    values("selected_rad",c.joints_rad); values("command_fk_world_m",r.plan.command_fk.position_m);
    out << "selected_cost_rad2=" << r.plan.selected_cost_rad2 << " fk_residual_m=" << c.fk_residual_m
        << " radius_sign=" << b.radius_sign << " elbow_sign=" << b.elbow_sign << " wrist_rad=" << b.wrist_rad
        << " free_yaw=" << b.free_yaw << " free_shoulder=" << b.free_shoulder
        << " boundary_normalized=" << b.boundary_normalized << " joint_boundary_reconstructed=" << b.joint_boundary_reconstructed << '\n';
  }
  for (const auto& a : r.plan.attempts) {
    out << "search budget=" << a.wrist_budget << " ik_status=" << ik_status_name(a.ik_status)
        << " scope=" << static_cast<int>(a.coverage.scope) << " wrist_continuum_covered=" << a.coverage.wrist_continuum_covered
        << " branch_enumeration_complete=" << a.coverage.branch_enumeration_complete
        << " max_wrist_gap_rad=" << a.coverage.max_wrist_gap_rad << " candidates=" << a.returned_candidates
        << " reused=" << a.reused_candidates << " newly_checked=" << a.newly_checked_candidates << " ik_ms=" << a.ik_wall_ms
        << " reason=" << a.reason << '\n';
    values("sampled_wrist_rad",a.coverage.sampled_wrist_rad);
    out << "kinematic_rejections geometric_samples=" << a.kinematic_rejections.geometric_wrist_misses
        << " joint_branches=" << a.kinematic_rejections.joint_limit_branches << " fk_candidates=" << a.kinematic_rejections.fk_candidates
        << " degenerate_samples=" << a.kinematic_rejections.degenerate_wrist_samples << '\n';
  }
  for (std::size_t i = 0; i < r.plan.assessments.size(); ++i) {
    const auto& a = r.plan.assessments[i];
    out << "candidate=" << i << " status=" << motion_status_name(a.status) << " cost_rad2=" << a.squared_displacement_rad2
        << " fk_residual_m=" << a.candidate.fk_residual_m << " samples=" << a.collision_samples << " reason=" << a.reason << '\n';
    for (const auto& c : a.contacts) out << "preflight_contact candidate=" << i << " time_s=" << a.contact_sample_time_s
      << " geoms=" << c.geom_a << ':' << c.geom_b << " ids=" << c.geom_a_id << ':' << c.geom_b_id
      << " distance_m=" << c.distance_m << " classification=" << static_cast<int>(c.classification)
      << " exclude=" << c.exclude << " constraint_address=" << c.constraint_address << '\n';
  }
  if (r.initial) values("initial_observed_rad",r.initial->observed_rad);
  if (r.first_command_before_step) {
    values("first_command_rad",r.first_command_before_step->command_rad);
    values("first_command_observed_before_step_rad",r.first_command_before_step->observed_rad);
  }
  if (r.final) {
    values("final_command_rad",r.final->command_rad); values("final_observed_rad",r.final->observed_rad);
    values("final_velocity_rad_s",r.final->velocity_rad_s); values("final_observed_world_m",r.final->tool.position_m);
    values("final_joint_error_rad",r.final_joint_error_rad);
    out << "final_cartesian_error_m=" << r.final_cartesian_error_m << " peak_tracking_error_rad=" << r.peak_tracking_error_rad << '\n';
  }
  out << "command_duration_s=" << r.command_duration_s << " executed_command_duration_s=" << r.executed_command_duration_s
      << " simulated_duration_s=" << r.simulated_duration_s << " settling_duration_s=" << r.settling_duration_s
      << " accepted_settling_dwell_s=" << r.accepted_settling_dwell_s << '\n';
  out << "ik_wall_ms=" << r.plan.ik_wall_ms << " preflight_wall_ms=" << r.plan.preflight_wall_ms
      << " planning_wall_ms=" << r.plan.planning_wall_ms << " execution_wall_ms=" << r.execution_wall_ms
      << " total_pipeline_wall_ms=" << r.total_wall_ms << " total_preflight_samples=" << r.plan.total_collision_samples << '\n';
  values("nominal_command_velocity_rad_s",r.plan.trajectory.nominal_velocity_rad_s);
  out << "acceleration_discontinuous=" << r.plan.trajectory.acceleration_discontinuous << " acceleration_bound=none jerk_bound=none\n";
  values("saturation_tick_counts",r.saturation_samples); values("peak_actuator_abs_Nm",r.peak_actuator_abs_Nm);
  std::vector<double> saturated_s;
  for (auto count : r.saturation_samples) saturated_s.push_back(static_cast<double>(count)*r.plan.trajectory.timestep_s);
  values("saturation_observed_s",saturated_s);
  for (const auto& w : r.warnings) out << "warning index=" << w.index << " count=" << w.count << " last_info=" << w.last_info << '\n';
  out << "warning_count=" << r.warnings.size() << " observed_contact_events_total=" << r.contact_events_total
      << " observed_contact_events_saved=" << r.contacts.size() << '\n';
  for (const auto& e : r.contacts) out << "observed_contact time_s=" << e.time_s << " geoms=" << e.contact.geom_a << ':' << e.contact.geom_b
    << " ids=" << e.contact.geom_a_id << ':' << e.contact.geom_b_id << " distance_m=" << e.contact.distance_m
    << " classification=" << static_cast<int>(e.contact.classification) << " exclude=" << e.contact.exclude << '\n';
  out << "Contact classes: 0=environment-only 1=explicitly-allowed 2=disallowed-robot. Scope: 0=3-joint 1=fixed-wrist 2=sampled-wrist 3=axial-wrist.\n"
      << "Finite samples do not establish continuous collision safety. Pedestal/yaw masks, adjacent filtering and noncolliding tool site omit collisions. Synthetic simulation only.\n";
}
}
