#include "relic/simulation.hpp"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace relic {
namespace {
void checked_write(std::ostream& out, const std::string& text) {
  out << text;
  if (!out) throw std::runtime_error("CSV write failed");
}
}

TraceWriter::TraceWriter(std::ostream& out, const RobotConfig& config) : out_(out), config_(config) {
  validate(config_);
  std::ostringstream row;
  row << "config_id,provenance,mujoco_version,time_s,pose_frame,tool_x_m,tool_y_m,tool_z_m,"
      << "tool_qw,tool_qx,tool_qy,tool_qz,contact_count";
  for (int i = 0; i < config_.dof; ++i)
    row << ",j" << i << "_command_rad,j" << i << "_observed_rad,j" << i
        << "_velocity_rad_s,j" << i << "_actuator_torque_Nm";
  row << '\n';
  checked_write(out_, row.str());
}

void TraceWriter::append(const Snapshot& s) {
  const auto n = static_cast<std::size_t>(config_.dof);
  if (s.command_rad.size() != n || s.observed_rad.size() != n ||
      s.velocity_rad_s.size() != n || s.actuator_torque_Nm.size() != n)
    throw std::invalid_argument("CSV snapshot joint count does not match configuration");
  auto finite = [](auto& values) {
    for (double x : values)
      if (!std::isfinite(x)) throw std::invalid_argument("CSV snapshot contains non-finite values");
  };
  if (!std::isfinite(s.time_s) || s.time_s < 0 || s.contact_count < 0)
    throw std::invalid_argument("CSV snapshot has invalid time or contact count");
  finite(s.command_rad); finite(s.observed_rad); finite(s.velocity_rad_s);
  finite(s.actuator_torque_Nm); finite(s.tool.position_m); finite(s.tool.quaternion_wxyz);
  std::ostringstream row;
  row.imbue(std::locale::classic());
  row << std::setprecision(17) << config_.id << ",synthetic," << mj_versionString()
      << ',' << s.time_s << ",world";
  for (double x : s.tool.position_m) row << ',' << x;
  for (double x : s.tool.quaternion_wxyz) row << ',' << x;
  row << ',' << s.contact_count;
  for (std::size_t i = 0; i < n; ++i)
    row << ',' << s.command_rad[i] << ',' << s.observed_rad[i] << ','
        << s.velocity_rad_s[i] << ',' << s.actuator_torque_Nm[i];
  row << '\n';
  checked_write(out_, row.str());
}
}
