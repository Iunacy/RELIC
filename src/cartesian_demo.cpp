#include "relic/motion.hpp"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
double number(const std::string& s) {
  double x = 0; const auto [end,err] = std::from_chars(s.data(),s.data()+s.size(),x);
  if (err != std::errc{} || end != s.data()+s.size()) throw std::invalid_argument("expected numeric value: "+s);
  return x;
}
std::size_t integer(const std::string& s) {
  std::size_t x = 0; const auto [end,err] = std::from_chars(s.data(),s.data()+s.size(),x);
  if (err != std::errc{} || end != s.data()+s.size()) throw std::invalid_argument("expected nonnegative integer: "+s);
  return x;
}
std::vector<std::string> split(const std::string& s) {
  std::istringstream input(s); std::string word; std::vector<std::string> out;
  while (std::getline(input,word,',')) out.push_back(word);
  if (s.empty() || s.back() == ',') throw std::invalid_argument("empty numeric list element");
  return out;
}
std::vector<double> numbers(const std::string& s) { std::vector<double> out; for (const auto& x : split(s)) out.push_back(number(x)); return out; }
bool same_path(const std::filesystem::path& a, const std::filesystem::path& b) {
  return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
    (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a,b));
}
}
int main(int argc, char** argv) {
  try {
    std::map<std::string,std::string> args; relic::ContactPolicy policy;
    const std::vector<std::string> options{"--config","--start","--target","--csv","--result","--velocity","--command-step",
      "--collision-step","--max-steps","--timeout","--joint-tol","--position-tol","--settle-speed","--dwell","--ik-tol",
      "--budgets","--wrist-work","--candidate-cap","--collision-budget","--fixed-wrist","--allow-contact"};
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "relic_cartesian_demo --config FILE --start 0,0,0[,0] --target X,Y,Z --csv FILE [--result FILE]\n"
          << "World metres; radians; seconds. Configurable options:\n"
          << "--velocity CSV (default 0.5/joint) --command-step 0.005 --collision-step 0.005 --max-steps 20000\n"
          << "--timeout 10 --joint-tol 0.01 --position-tol 0.005 --settle-speed 0.01 --dwell 0.25 --ik-tol 1e-8\n"
          << "--budgets 17,65,257 --wrist-work 339 --candidate-cap 2048 --collision-budget 500000\n"
          << "--fixed-wrist RAD (optional) --allow-contact geomA:geomB (repeatable; default none)\n"
          << "Linear commands have no acceleration/jerk bound. Synthetic simulation; finite sampled contacts only.\n";
        return 0;
      }
      if (std::find(options.begin(),options.end(),key) == options.end()) throw std::invalid_argument("unknown option: "+key);
      if (++i == argc) throw std::invalid_argument("missing option value");
      if (key == "--allow-contact") {
        const std::string value = argv[i]; const auto colon = value.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("contact pair requires geomA:geomB");
        policy.allowed_pairs.emplace_back(value.substr(0,colon),value.substr(colon+1));
      } else if (!args.emplace(key,argv[i]).second) throw std::invalid_argument("duplicate option: "+key);
    }
    for (const auto key : {"--config","--start","--target","--csv"}) if (!args.contains(key)) throw std::invalid_argument(std::string("required ")+key);
    const std::filesystem::path config = args.at("--config"), csv = args.at("--csv");
    if (same_path(config,csv) || (args.contains("--result") &&
        (same_path(config,args.at("--result")) || same_path(csv,args.at("--result")))))
      throw std::invalid_argument("configuration, trace and result paths must be distinct");
    relic::Simulation sim(relic::load_config(config));
    relic::PositionIkRequest request; request.current_rad = numbers(args.at("--start"));
    const auto target = numbers(args.at("--target")); if (target.size() != 3) throw std::invalid_argument("target requires three coordinates");
    std::copy(target.begin(),target.end(),request.position_world_m.begin());
    relic::MotionOptions motion; motion.max_velocity_rad_s.assign(static_cast<std::size_t>(sim.config().dof),0.5);
    relic::SearchOptions search;
    if (args.contains("--velocity")) motion.max_velocity_rad_s = numbers(args.at("--velocity"));
    const std::pair<const char*,double*> doubles[]{
      {"--command-step",&motion.max_command_step_rad},{"--collision-step",&motion.collision_step_rad},{"--timeout",&motion.timeout_s},
      {"--joint-tol",&motion.joint_tolerance_rad},{"--position-tol",&motion.cartesian_tolerance_m},
      {"--settle-speed",&motion.settling_velocity_rad_s},{"--dwell",&motion.settling_dwell_s},{"--ik-tol",&request.position_tolerance_m}};
    for (const auto& [name,value] : doubles) if (args.contains(name)) *value = number(args.at(name));
    const std::pair<const char*,std::size_t*> integers[]{
      {"--max-steps",&motion.max_trajectory_steps},{"--wrist-work",&search.total_wrist_budget},
      {"--candidate-cap",&search.max_unique_candidates},{"--collision-budget",&search.max_collision_samples}};
    for (const auto& [name,value] : integers) if (args.contains(name)) *value = integer(args.at(name));
    if (args.contains("--budgets")) { search.wrist_budgets.clear(); for (const auto& x : split(args.at("--budgets"))) search.wrist_budgets.push_back(integer(x)); }
    if (args.contains("--fixed-wrist")) { request.wrist_search.mode = relic::WristMode::Fixed; request.wrist_search.fixed_rad = number(args.at("--fixed-wrist")); }
    std::ofstream output(csv,std::ios::binary), result_file;
    if (!output) throw std::runtime_error("cannot open execution trace");
    if (args.contains("--result")) { result_file.open(args.at("--result"),std::ios::binary); if (!result_file) throw std::runtime_error("cannot open result file"); }
    relic::TraceWriter trace(output,sim.config());
    const auto result = relic::execute_cartesian_motion(sim,request,motion,search,policy,&trace);
    relic::write_execution_report(std::cout,result);
    if (args.contains("--result")) { relic::write_execution_report(result_file,result); result_file.close(); if (!result_file) throw std::runtime_error("result flush failed"); }
    output.close(); if (!output) throw std::runtime_error("trace flush failed");
    return result.status == relic::MotionStatus::Success ? 0 : 2;
  } catch (const std::exception& e) { std::cerr << "VIPER Cartesian error: " << e.what() << '\n'; return 1; }
}
