#include "relic/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>

namespace relic {
namespace {
std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::vector<double> numbers(const std::string& value, const std::string& key,
                            std::size_t count) {
  std::istringstream input(value);
  input.imbue(std::locale::classic());
  std::vector<double> result;
  double x;
  while (input >> x) result.push_back(x);
  if (!input.eof() || result.size() != count)
    throw std::invalid_argument(key + ": expected " + std::to_string(count) + " numbers");
  return result;
}
}

void validate(const RobotConfig& c) {
  if (c.id.empty() || c.id.size() > 80 ||
      !std::all_of(c.id.begin(), c.id.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
      }))
    throw std::invalid_argument("id: use 1..80 ASCII letters, digits, underscores or hyphens");
  if (c.dof != 3 && c.dof != 4) throw std::invalid_argument("dof: expected 3 or 4");
  for (double length : c.link_lengths_m)
    if (!std::isfinite(length) || length <= 0)
      throw std::invalid_argument("link_lengths_m: each length must be finite and positive");
  if (!std::isfinite(c.pedestal_height_m) || c.pedestal_height_m <= 0)
    throw std::invalid_argument("pedestal_height_m: must be finite and positive");
  if (c.joint_limits_rad.size() != static_cast<std::size_t>(c.dof))
    throw std::invalid_argument("joint_limits_rad: count must equal dof");
  for (std::size_t i = 0; i < c.joint_limits_rad.size(); ++i) {
    const auto [lo, hi] = c.joint_limits_rad[i];
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi || lo > 0 || hi < 0)
      throw std::invalid_argument("joint_limits_rad[" + std::to_string(i) +
                                  "]: finite lower < upper required, including zero for reset");
  }
  for (double x : c.tool_frame.position_m)
    if (!std::isfinite(x)) throw std::invalid_argument("tool_position_m: must be finite");
  double norm2 = 0;
  for (double x : c.tool_frame.quaternion_wxyz) norm2 += x*x;
  if (!std::isfinite(norm2) || std::abs(norm2 - 1) > 1e-9)
    throw std::invalid_argument("tool_quaternion_wxyz: must be finite and unit length (1e-9)");
}

RobotConfig load_config(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open configuration: " + path.string());
  std::map<std::string, std::string> fields;
  const std::vector<std::string> keys{"id", "provenance", "dof", "link_lengths_m",
    "pedestal_height_m", "joint_limits_rad", "tool_position_m", "tool_quaternion_wxyz"};
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    line = trim(line.substr(0, line.find('#')));
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos)
      throw std::invalid_argument("configuration line " + std::to_string(line_no) + ": expected key = value");
    const auto key = trim(line.substr(0, eq));
    if (std::find(keys.begin(), keys.end(), key) == keys.end())
      throw std::invalid_argument("unknown configuration key: " + key);
    if (!fields.emplace(key, trim(line.substr(eq + 1))).second)
      throw std::invalid_argument("duplicate configuration key: " + key);
  }
  if (input.bad()) throw std::runtime_error("error reading configuration: " + path.string());
  for (const auto& key : keys)
    if (!fields.contains(key)) throw std::invalid_argument("missing configuration key: " + key);
  if (fields.at("provenance") != "synthetic")
    throw std::invalid_argument("provenance: this batch accepts only synthetic configurations");
  RobotConfig c;
  c.id = fields.at("id");
  if (fields.at("dof") == "3") c.dof = 3;
  else if (fields.at("dof") == "4") c.dof = 4;
  else throw std::invalid_argument("dof: expected 3 or 4");
  const auto lengths = numbers(fields.at("link_lengths_m"), "link_lengths_m", 2);
  std::copy(lengths.begin(), lengths.end(), c.link_lengths_m.begin());
  c.pedestal_height_m = numbers(fields.at("pedestal_height_m"), "pedestal_height_m", 1)[0];
  const auto limits = numbers(fields.at("joint_limits_rad"), "joint_limits_rad", 2*c.dof);
  for (int i = 0; i < c.dof; ++i) c.joint_limits_rad.push_back({limits[2*i], limits[2*i+1]});
  const auto pos = numbers(fields.at("tool_position_m"), "tool_position_m", 3);
  std::copy(pos.begin(), pos.end(), c.tool_frame.position_m.begin());
  const auto quat = numbers(fields.at("tool_quaternion_wxyz"), "tool_quaternion_wxyz", 4);
  std::copy(quat.begin(), quat.end(), c.tool_frame.quaternion_wxyz.begin());
  validate(c);
  return c;
}
}
