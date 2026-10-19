#include "relic/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace relic {
namespace {
struct SpecDeleter { void operator()(mjSpec* p) const noexcept { mj_deleteSpec(p); } };
constexpr std::array<const char*, 4> joint_names{"yaw", "shoulder", "elbow", "wrist"};
int named_id(const mjModel* model, mjtObj type, const std::string& name) {
  const int id = mj_name2id(model, type, name.c_str());
  if (id < 0) throw std::runtime_error("generated model missing object: " + name);
  return id;
}
}

void ModelDeleter::operator()(mjModel* p) const noexcept { mj_deleteModel(p); }
void DataDeleter::operator()(mjData* p) const noexcept { mj_deleteData(p); }

std::string make_mjcf(const RobotConfig& c) {
  validate(c);
  std::ostringstream xml;
  xml.imbue(std::locale::classic());
  xml << std::setprecision(17);
  xml << "<mujoco model=\"" << c.id << "\">\n"
      << "<!-- SYNTHETIC: all geometry, inertia, limits and actuator parameters. -->\n"
      << "<compiler angle=\"radian\"/>\n"
      << "<option timestep=\"0.002\" gravity=\"0 0 -9.81\" integrator=\"implicitfast\"/>\n"
      << "<default><joint type=\"hinge\" limited=\"true\" damping=\"1\" armature=\"0.01\"/>"
      << "<geom density=\"600\" contype=\"1\" conaffinity=\"1\"/></default>\n"
      << "<worldbody><geom name=\"ground\" type=\"plane\" size=\"2 2 0.1\"/>\n"
      << "<geom name=\"pedestal\" type=\"cylinder\" contype=\"0\" conaffinity=\"0\" size=\"0.04 "
      << c.pedestal_height_m / 2 << "\" pos=\"0 0 " << c.pedestal_height_m / 2 << "\"/>\n";
  auto joint = [&](int i, const char* axis) {
    const auto limit = c.joint_limits_rad[i];
    xml << "<joint name=\"" << joint_names[i] << "\" axis=\"" << axis
        << "\" range=\"" << limit.lower << ' ' << limit.upper << "\"/>\n";
  };
  auto link = [&](int i) {
    xml << "<geom name=\"link_" << i+1 << "\" type=\"capsule\" fromto=\"0 0 0 "
        << c.link_lengths_m[i] << " 0 0\" size=\""
        << std::min(0.018, c.link_lengths_m[i] / 10) << "\"/>\n";
  };
  xml << "<body name=\"yaw_body\" pos=\"0 0 " << c.pedestal_height_m << "\">\n";
  joint(0, "0 0 1");
  xml << "<geom name=\"yaw_hub\" type=\"sphere\" size=\"0.025\" contype=\"0\" conaffinity=\"0\"/>\n"
      << "<body name=\"upper_arm\">\n";
  joint(1, "0 1 0");
  link(0);
  xml << "<body name=\"forearm\" pos=\"" << c.link_lengths_m[0] << " 0 0\">\n";
  joint(2, "0 1 0");
  link(1);
  xml << "<body name=\"distal_mount\" pos=\"" << c.link_lengths_m[1] << " 0 0\">\n";
  if (c.dof == 4) {
    joint(3, "1 0 0");
    xml << "<geom name=\"wrist_hub\" type=\"sphere\" size=\"0.02\"/>\n";
  }
  xml << "<site name=\"tool\" size=\"0.005\" pos=\"";
  for (double x : c.tool_frame.position_m) xml << x << ' ';
  xml << "\" quat=\"";
  for (double x : c.tool_frame.quaternion_wxyz) xml << x << ' ';
  xml << "\"/>\n</body></body></body></body></worldbody>\n<actuator>\n";
  for (int i = 0; i < c.dof; ++i) {
    const auto limit = c.joint_limits_rad[i];
    xml << "<position name=\"drive_" << joint_names[i] << "\" joint=\"" << joint_names[i]
        << "\" gear=\"1\" kp=\"200\" kv=\"20\" ctrllimited=\"true\" ctrlrange=\""
        << limit.lower << ' ' << limit.upper
        << "\" forcelimited=\"true\" forcerange=\"-20 20\"/>\n";
  }
  xml << "</actuator></mujoco>\n";
  return xml.str();
}

Simulation::Simulation(RobotConfig config) : config_(std::move(config)) {
  if (mj_version() != mjVERSION_HEADER)
    throw std::runtime_error("MuJoCo header/runtime version mismatch");
  const auto xml = make_mjcf(config_);
  char error[2048]{};
  std::unique_ptr<mjSpec, SpecDeleter> spec(mj_parseXMLString(xml.c_str(), nullptr, error, sizeof(error)));
  if (!spec) throw std::runtime_error(std::string("MJCF parse failed: ") + error);
  model_.reset(mj_compile(spec.get(), nullptr));
  if (!model_) throw std::runtime_error(std::string("MJCF compile failed: ") + mjs_getError(spec.get()));
  execution_.reset(mj_makeData(model_.get()));
  if (!execution_) throw std::runtime_error("MuJoCo execution data allocation failed");
  planning_.reset(mj_makeData(model_.get()));
  if (!planning_) throw std::runtime_error("MuJoCo planning data allocation failed");
  if (model_->njnt != config_.dof || model_->nq != config_.dof ||
      model_->nv != config_.dof || model_->nu != config_.dof)
    throw std::runtime_error("generated model joint/actuator dimensions do not match configuration");
  for (int i = 0; i < config_.dof; ++i) {
    const int jid = named_id(model_.get(), mjOBJ_JOINT, joint_names[i]);
    const int aid = named_id(model_.get(), mjOBJ_ACTUATOR, std::string("drive_") + joint_names[i]);
    if (model_->jnt_type[jid] != mjJNT_HINGE ||
        model_->actuator_trntype[aid] != mjTRN_JOINT ||
        model_->actuator_trnid[2*aid] != jid)
      throw std::runtime_error("generated actuator transmission mismatch");
    joint_ids_.push_back(jid);
    qpos_addresses_.push_back(model_->jnt_qposadr[jid]);
    dof_addresses_.push_back(model_->jnt_dofadr[jid]);
    actuator_ids_.push_back(aid);
  }
  tool_site_ = named_id(model_.get(), mjOBJ_SITE, "tool");
  const std::vector<double> zero(config_.dof, 0);
  initialize(*execution_, zero);
  initialize(*planning_, zero);
}

std::vector<JointLimit> Simulation::joint_limits() const {
  std::vector<JointLimit> result;
  for (int jid : joint_ids_)
    result.push_back({model_->jnt_range[2*jid], model_->jnt_range[2*jid+1]});
  return result;
}

void Simulation::check_joints(std::span<const double> q) const {
  if (q.size() != joint_ids_.size()) throw std::invalid_argument("joint state: count must equal dof");
  for (std::size_t i = 0; i < q.size(); ++i) {
    const int jid = joint_ids_[i];
    if (!std::isfinite(q[i]) || q[i] < model_->jnt_range[2*jid] || q[i] > model_->jnt_range[2*jid+1])
      throw std::invalid_argument(std::string("joint state ") + joint_names[i] +
                                  ": expected finite radians within inclusive model limits");
  }
}

void Simulation::initialize(mjData& data, std::span<const double> q) {
  mj_resetData(model_.get(), &data);
  for (std::size_t i = 0; i < q.size(); ++i) {
    data.qpos[qpos_addresses_[i]] = q[i];
    data.ctrl[actuator_ids_[i]] = q[i];
  }
  mj_forward(model_.get(), &data);
}

void Simulation::reset(std::span<const double> q) {
  check_joints(q);
  initialize(*execution_, q);
}

void Simulation::set_position_commands(std::span<const double> q) {
  check_joints(q);
  for (std::size_t i = 0; i < q.size(); ++i) execution_->ctrl[actuator_ids_[i]] = q[i];
  mj_forward(model_.get(), execution_.get());
}

void Simulation::step(std::size_t count) {
  if (count == 0 || count > 1000000) throw std::invalid_argument("step count: expected 1..1000000");
  for (std::size_t i = 0; i < count; ++i) {
    mj_step(model_.get(), execution_.get());
    for (int warning : {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC, mjWARN_BADCTRL})
      if (execution_->warning[warning].number)
        throw std::runtime_error("MuJoCo numerical warning during stepping; reset required");
  }
  mj_forward(model_.get(), execution_.get());
}

Pose Simulation::tool_pose(const mjData& data) const {
  Pose result;
  std::copy_n(data.site_xpos + 3*tool_site_, 3, result.position_m.begin());
  mju_mat2Quat(result.quaternion_wxyz.data(), data.site_xmat + 9*tool_site_);
  return result;
}

Snapshot Simulation::snapshot() const {
  Snapshot s;
  s.time_s = execution_->time;
  s.tool = tool_pose(*execution_);
  s.contact_count = execution_->ncon;
  for (std::size_t i = 0; i < joint_ids_.size(); ++i) {
    s.command_rad.push_back(execution_->ctrl[actuator_ids_[i]]);
    s.observed_rad.push_back(execution_->qpos[qpos_addresses_[i]]);
    s.velocity_rad_s.push_back(execution_->qvel[dof_addresses_[i]]);
    s.actuator_torque_Nm.push_back(execution_->actuator_force[actuator_ids_[i]]);
  }
  return s;
}

Pose Simulation::evaluate_fk(std::span<const double> q) {
  check_joints(q);
  initialize(*planning_, q);
  return tool_pose(*planning_);
}
}
