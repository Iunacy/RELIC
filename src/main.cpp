#include "relic/simulation.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  try {
    std::filesystem::path config_path, csv_path, mjcf_path;
    std::size_t steps = 1000;
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg(argv[i]);
      if (arg == "--help") {
        std::cout << "relic_cli --config FILE --csv FILE [--steps 1..1000000] [--mjcf FILE]\n"
                  << "SYNTHETIC simulation only. Times: seconds from reset; angles: radians; world: metres.\n";
        return 0;
      }
      if (arg != "--config" && arg != "--csv" && arg != "--steps" && arg != "--mjcf")
        throw std::invalid_argument("unknown option: " + std::string(arg));
      if (++i == argc) throw std::invalid_argument("missing value for " + std::string(arg));
      if (arg == "--config") config_path = argv[i];
      else if (arg == "--csv") csv_path = argv[i];
      else if (arg == "--mjcf") mjcf_path = argv[i];
      else {
        const std::string_view value(argv[i]);
        const auto [end, error] = std::from_chars(value.data(), value.data()+value.size(), steps);
        if (error != std::errc{} || end != value.data()+value.size() || steps < 1 || steps > 1000000)
          throw std::invalid_argument("--steps: expected integer 1..1000000");
      }
    }
    if (config_path.empty() || csv_path.empty())
      throw std::invalid_argument("required: --config FILE --csv FILE (see --help)");
    auto same_path = [](const auto& a, const auto& b) {
      return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
        (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a, b));
    };
    if (same_path(config_path, csv_path) || (!mjcf_path.empty() &&
        (same_path(config_path, mjcf_path) || same_path(csv_path, mjcf_path))))
      throw std::invalid_argument("configuration, CSV and MJCF paths must be distinct");
    const auto config = relic::load_config(config_path);
    relic::Simulation simulation(config);
    std::vector<double> target;
    for (int i = 0; i < config.dof; ++i)
      target.push_back(0.2 * (i % 2 ? config.joint_limits_rad[i].lower : config.joint_limits_rad[i].upper));
    const auto target_pose = simulation.evaluate_fk(target);
    std::cout.imbue(std::locale::classic());
    std::cout << std::setprecision(10) << "SYNTHETIC " << config.id << " MuJoCo=" << mj_versionString()
              << " dof=" << config.dof << " dt_s=" << simulation.model().opt.timestep
              << " nq=" << simulation.model().nq << " nu=" << simulation.model().nu << '\n';
    std::cout << "hypothetical target tool position_world_m=";
    for (double x : target_pose.position_m) std::cout << x << ' ';
    std::cout << " execution_time_s=" << simulation.snapshot().time_s << '\n';
    const auto limits = simulation.joint_limits();
    for (int i = 0; i < config.dof; ++i)
      std::cout << "j" << i << " limits_rad=[" << limits[i].lower << ',' << limits[i].upper
                << "] command_rad=" << target[i] << '\n';
    if (!mjcf_path.empty()) {
      std::ofstream xml(mjcf_path, std::ios::binary);
      xml << relic::make_mjcf(config);
      xml.close();
      if (!xml) throw std::runtime_error("cannot write MJCF: " + mjcf_path.string());
    }
    std::ofstream csv(csv_path, std::ios::binary);
    if (!csv) throw std::runtime_error("cannot open CSV: " + csv_path.string());
    relic::TraceWriter trace(csv, config);
    simulation.set_position_commands(target);
    trace.append(simulation.snapshot());
    for (std::size_t i = 0; i < steps; ++i) {
      simulation.step();
      trace.append(simulation.snapshot());
    }
    csv.close();
    if (!csv) throw std::runtime_error("CSV flush/close failed");
    const auto final = simulation.snapshot();
    std::cout << "final_time_s=" << final.time_s << " contacts=" << final.contact_count << '\n';
    for (int i = 0; i < config.dof; ++i)
      std::cout << "j" << i << " commanded_rad=" << final.command_rad[i]
                << " observed_rad=" << final.observed_rad[i]
                << " error_rad=" << final.command_rad[i] - final.observed_rad[i] << '\n';
    std::cout << "rows=" << steps+1 << " CSV=" << csv_path.string()
              << "\nNo path-feasibility, collision-safety or harvesting-success claim.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RELIC error: " << error.what() << '\n';
    return 1;
  }
}
