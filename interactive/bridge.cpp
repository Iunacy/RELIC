#include "residual.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
using namespace relic;
using namespace relic::adaptive;
std::string finite_array(std::span<const double> values) {
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (std::size_t j = 0; j < values.size(); ++j) {
    if (j)
      out << ',';
    if (std::isfinite(values[j]))
      out << values[j];
    else
      out << "null";
  }
  out << ']';
  return out.str();
}
int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--scene") {
      const auto cfg = load_config(argv[2]);
      Simulation sim(cfg);
      const std::filesystem::path out = argv[3];
      std::filesystem::create_directories(out);
      std::ofstream xml(out / "model.xml");
      xml << make_mjcf(cfg);
      std::ofstream scene(out / "scene.json");
      scene << "{\"target_left\":"
            << array(sim.evaluate_fk(Vec{-.55, -.65, .65, 0}).position_m)
            << ",\"target_center\":"
            << array(sim.evaluate_fk(Vec{0, -.75, .8, 0}).position_m)
            << ",\"target_right\":"
            << array(sim.evaluate_fk(Vec{.55, -.65, .65, 0}).position_m) << "}";
      return 0;
    }
    if (argc != 11)
      throw std::invalid_argument(
          "CONFIG OUTPUT CONTROLLER BOUNDS MODEL X Y Z DEGRADED INJECT_TICK");
    const auto cfg = load_config(argv[1]);
    const std::filesystem::path output = argv[2];
    std::filesystem::create_directories(output);
    const std::string controller = argv[3];
    if (controller != "conventional" && controller != "integral" &&
        controller != "mlp")
      throw std::invalid_argument("unknown controller");
    auto bounds = Bounds::load(argv[4]);
    const int inject = std::stoi(argv[10]);
    PhysicsParameters physics;
    const std::string degradation = argv[9];
    if (degradation != "0" && degradation != "1")
      throw std::invalid_argument("invalid degradation");
    if (degradation == "1") {
      physics.gain_scale = .4;
      physics.damping_scale = 2;
      physics.friction_Nm = .15;
    }
    Simulation sim(cfg, physics), nominal(cfg);
    {
      std::ofstream xml(output / "model.xml");
      xml << make_mjcf(cfg);
    }
    PositionIkRequest request;
    request.current_rad = {0, -.5, .5, 0};
    for (int j = 0; j < 3; ++j)
      request.position_world_m[j] = std::stod(argv[6 + j]);
    MotionOptions options;
    options.max_velocity_rad_s.assign(4, .4);
    const auto plan = plan_cartesian_motion(sim, request, options);
    if (plan.status != MotionStatus::Success) {
      std::ofstream result(output / "result.json");
      result.exceptions(std::ios::badbit | std::ios::failbit);
      result << "{\"status\":\"" << motion_status_name(plan.status)
             << "\",\"success\":false,\"reason\":\"planning_rejected\","
                "\"trace\":[]}";
      return 2;
    }
    std::optional<MLP> mlp;
    std::string invalid = "none";
    if (controller == "mlp")
      try {
        mlp = MLP::load(argv[5]);
        if (mlp->embodiment != 0 && mlp->embodiment != 4)
          throw std::runtime_error("unsupported_embodiment");
        for (int j = 0; j < 4; ++j)
          if (mlp->bounds.magnitude[j] != bounds.magnitude[j] ||
              mlp->bounds.rate[j] != bounds.rate[j])
            throw std::runtime_error("bounds_mismatch");
      } catch (const std::exception &) {
        mlp.reset();
        invalid = "model_load_failure";
      }
    sim.reset(request.current_rad);
    std::ofstream trace_file(output / "trace.jsonl"), csv(output / "trace.csv");
    trace_file.exceptions(std::ios::badbit | std::ios::failbit);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    TraceWriter writer(csv, cfg);
    const auto &commands = plan.trajectory.commands;
    const std::size_t moving = commands.size() - 1, total = moving + 1000;
    Vec correction(4, 0), older(4, 0), previous_error(4, 0), integral(4, 0),
        zero(4, 0);
    bool withdrawing = false;
    std::string why = "none", status = "Success", reason = "completed";
    std::size_t fallbacks = 0, samples = 0;
    double squared = 0, peak = 0, command_error = 0;
    auto failure_observation = [&](const Snapshot &observed) {
      std::ofstream event(output / "failure_observation.json");
      event << "{\"observed\":" << finite_array(observed.observed_rad)
            << ",\"velocity\":" << finite_array(observed.velocity_rad_s)
            << ",\"tool\":" << finite_array(observed.tool.position_m)
            << ",\"contacts\":" << observed.contact_count << "}\n";
    };
    Snapshot last = sim.snapshot();
    const double dt = sim.model().opt.timestep;
    writer.append(last);
    for (std::size_t k = 1; k <= total; ++k) {
      const auto pre = sim.snapshot();
      if (execution_health(sim.model(), sim.execution_data()) !=
              MotionStatus::Success ||
          has_disallowed_contact(
              inspect_contacts(sim.model(), sim.execution_data(), {}))) {
        failure_observation(pre);
        status = "Stopped";
        reason = "observed_health_or_contact";
        break;
      }
      const Vec desired = commands[std::min(k, moving)];
      const Vec velocity =
          k <= moving ? plan.trajectory.nominal_velocity_rad_s : zero;
      Vec candidate(4, 0);
      if (controller == "integral") {
        nominal.evaluate_fk(desired);
        for (int j = 0; j < 4; ++j) {
          const double raw = .1 * velocity[j] +
                             nominal.planning_data().qfrc_bias[j] / 200. +
                             previous_error[j] + integral[j];
          candidate[j] =
              std::clamp(raw, -bounds.magnitude[j], bounds.magnitude[j]);
          const double delta = 4 * previous_error[j] * dt;
          if ((std::abs(raw) < bounds.magnitude[j] &&
               std::abs(candidate[j] - correction[j]) <= bounds.rate[j]) ||
              delta * (raw - correction[j]) < 0)
            integral[j] = std::clamp(integral[j] + delta, -.2, .2);
        }
      }
      if (mlp)
        for (int j = 0; j < 4; ++j)
          candidate[j] =
              mlp->predict(features(cfg, j, desired[j], velocity[j], 0,
                                    pre.observed_rad[j], pre.velocity_rad_s[j],
                                    previous_error[j], older[j], correction[j]),
                           j);
      if (controller == "mlp" && inject >= 0 &&
          k >= static_cast<std::size_t>(inject))
        invalid = "injected_invalid_output";
      const auto guarded_command =
          guarded(sim, desired, pre.command_rad, correction, candidate, bounds,
                  withdrawing ? why : invalid);
      if (guarded_command.fallback && !withdrawing) {
        withdrawing = true;
        why = guarded_command.reason;
      }
      if (guarded_command.stopped) {
        status = "WithdrawalInfeasible";
        reason = guarded_command.reason;
        break;
      }
      if (guarded_command.fallback)
        ++fallbacks;
      sim.set_position_commands(guarded_command.command);
      try {
        sim.step();
      } catch (const std::exception &) {
        status = "Stopped";
        reason = "physics_failure_last_valid_trace";
        failure_observation(sim.snapshot());
        break;
      }
      const auto now = sim.snapshot();
      if (execution_health(sim.model(), sim.execution_data()) !=
              MotionStatus::Success ||
          has_disallowed_contact(
              inspect_contacts(sim.model(), sim.execution_data(), {})) ||
          !safe_segment(sim, pre.observed_rad, now.observed_rad, false)) {
        failure_observation(now);
        status = "Stopped";
        reason = "observed_health_contact_or_path";
        break;
      }
      writer.append(now);
      last = now;
      correction = guarded_command.applied;
      older = previous_error;
      for (int j = 0; j < 4; ++j) {
        previous_error[j] = desired[j] - now.observed_rad[j];
        peak = std::max(peak, std::abs(previous_error[j]));
        if (k <= moving) {
          squared += previous_error[j] * previous_error[j];
          ++samples;
        }
      }
      Pose target;
      target.position_m = request.position_world_m;
      if (k == moving)
        command_error = distance(now.tool, target);
      trace_file << std::setprecision(17) << "{\"t\":" << now.time_s
                 << ",\"desired\":" << array(desired)
                 << ",\"observed\":" << array(now.observed_rad)
                 << ",\"command\":" << array(now.command_rad)
                 << ",\"velocity\":" << array(now.velocity_rad_s)
                 << ",\"effort\":" << array(now.actuator_torque_Nm)
                 << ",\"correction\":" << array(correction)
                 << ",\"tool\":" << array(now.tool.position_m)
                 << ",\"contacts\":" << now.contact_count
                 << ",\"fallback_reason\":\"" << guarded_command.reason
                 << "\",\"fallback\":"
                 << (guarded_command.fallback ? "true" : "false")
                 << ",\"phase\":\"" << (k <= moving ? "command" : "settling")
                 << "\"}\n";
    }
    trace_file.close();
    csv.close();
    Pose target;
    target.position_m = request.position_world_m;
    const double error = distance(last.tool, target);
    double speed = 0;
    for (double v : last.velocity_rad_s)
      speed = std::max(speed, std::abs(v));
    const bool success = status == "Success" && error <= .005 && speed < .02;
    if (status == "Success" && !success) {
      status = "NonConvergence";
      reason = "final_pose_or_velocity_tolerance";
    }
    std::ofstream result(output / "result.json");
    result.exceptions(std::ios::badbit | std::ios::failbit);
    result << std::setprecision(17) << "{\"status\":\"" << status
           << "\",\"reason\":\"" << reason
           << "\",\"success\":" << (success ? "true" : "false")
           << ",\"source_commit\":\"" << RELIC_INTERACTIVE_COMMIT
           << "\",\"initial_q\":" << array(request.current_rad)
           << ",\"goal_q\":" << array(plan.selected->joints_rad)
           << ",\"target_m\":" << array(request.position_world_m)
           << ",\"metrics\":{\"rmse_rad\":"
           << (samples ? std::sqrt(squared / samples) : 0)
           << ",\"peak_rad\":" << peak << ",\"command_end_m\":" << command_error
           << ",\"final_m\":" << error << ",\"simulation_s\":" << last.time_s
           << ",\"fallbacks\":" << fallbacks << "}}\n";
    result.close();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
