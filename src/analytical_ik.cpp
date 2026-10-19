#include "relic/analytical_ik.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace relic {
namespace {
using Real = long double;
constexpr Real pi = std::numbers::pi_v<Real>;
constexpr Real tau = 2*pi;
constexpr Real eps = std::numeric_limits<double>::epsilon();
constexpr const char* joints[]{"yaw", "shoulder", "elbow", "wrist"};
constexpr const char* bodies[]{"yaw_body", "upper_arm", "forearm", "distal_mount"};

bool equal(Real a, Real b) {
  return std::isfinite(a) && std::isfinite(b) &&
         std::abs(a-b) <= 8*eps*std::max({Real(1), std::abs(a), std::abs(b)});
}
bool identity(const mjtNum* q) {
  return equal(q[0], 1) && equal(q[1], 0) && equal(q[2], 0) && equal(q[3], 0);
}
bool valid_joints(std::span<const double> q, const std::vector<JointLimit>& limits) {
  if (q.size() != limits.size()) return false;
  for (std::size_t i = 0; i < q.size(); ++i)
    if (!std::isfinite(q[i]) || q[i] < limits[i].lower || q[i] > limits[i].upper) return false;
  return true;
}
void add_unique(std::vector<double>& values, double x, std::size_t budget) {
  if (values.size() < budget && std::find(values.begin(), values.end(), x) == values.end())
    values.push_back(x);
}
std::vector<double> wrist_samples(const PositionIkRequest& r, const RobotConfig& c) {
  if (c.dof == 3) return {0};
  if (r.wrist_search.mode == WristMode::Fixed) return {r.wrist_search.fixed_rad};
  if (c.tool_frame.position_m[1] == 0 && c.tool_frame.position_m[2] == 0)
    return {r.current_rad[3]};
  const auto lim = c.joint_limits_rad[3];
  const auto budget = r.wrist_search.budget;
  std::vector<double> values;
  add_unique(values, r.current_rad[3], budget);
  add_unique(values, 0, budget);
  add_unique(values, lim.lower, budget);
  add_unique(values, lim.upper, budget);


  for (std::size_t denominator = 2; values.size() < budget && denominator <= 16384; denominator *= 2) {
    for (std::size_t numerator = 1; numerator < denominator && values.size() < budget; numerator += 2) {
      const Real t = Real(numerator)/Real(denominator);
      add_unique(values, static_cast<double>((1-t)*lim.lower+t*lim.upper), budget);
    }
  }
  return values;
}
double wrist_gap(std::vector<double> values, JointLimit lim) {
  values.push_back(lim.lower); values.push_back(lim.upper);
  std::sort(values.begin(), values.end());
  Real gap = 0;
  for (std::size_t i = 1; i < values.size(); ++i)
    gap = std::max(gap, Real(values[i])-values[i-1]);
  return static_cast<double>(gap);
}



struct AngleValue { double value; bool boundary_reconstructed = false; };
std::vector<AngleValue> angle_lifts(Real angle, JointLimit lim, std::size_t budget) {



  const Real roundoff = 64*eps*(1+std::max({std::abs(angle),std::abs(Real(lim.lower)),std::abs(Real(lim.upper))}));
  const Real first = std::ceil((Real(lim.lower)-angle-roundoff)/tau);
  const Real last = std::floor((Real(lim.upper)-angle+roundoff)/tau);
  if (first > last) return {};
  if (last-first+1 > Real(budget) || std::abs(first) > 0x1p50L || std::abs(last) > 0x1p50L)
    throw std::length_error("angle lift enumeration exceeds resource/precision budget");
  std::vector<AngleValue> values;
  for (Real k = first; k <= last; k += 1) {
    const Real lifted = angle+k*tau;
    double q = static_cast<double>(lifted);
    bool boundary = false;
    if (lifted < lim.lower && Real(lim.lower)-lifted <= roundoff) { q = lim.lower; boundary = true; }
    if (lifted > lim.upper && lifted-Real(lim.upper) <= roundoff) { q = lim.upper; boundary = true; }
    if (std::isfinite(q) && q >= lim.lower && q <= lim.upper) values.push_back({q,boundary});
  }
  return values;
}
bool same_configuration(const std::vector<double>& a, const std::vector<double>& b) {
  return a == b;
}
}

const char* ik_status_name(IkStatus s) noexcept {
  switch (s) {
    case IkStatus::Success: return "Success";
    case IkStatus::InvalidInput: return "InvalidInput";
    case IkStatus::UnsupportedGeometry: return "UnsupportedGeometry";
    case IkStatus::Unreachable: return "Unreachable";
    case IkStatus::UnreachableAtFixedWrist: return "UnreachableAtFixedWrist";
    case IkStatus::NoCandidateWithinSearch: return "NoCandidateWithinSearch";
    case IkStatus::JointLimitsExcludeCandidates: return "JointLimitsExcludeCandidates";
    case IkStatus::DegenerateGeometry: return "DegenerateGeometry";
    case IkStatus::FkValidationFailed: return "FkValidationFailed";
    case IkStatus::ResourceLimit: return "ResourceLimit";
    case IkStatus::NumericalFailure: return "NumericalFailure";
  }
  return "Unknown";
}

IkGeometryCheck check_analytical_geometry(const mjModel& m, const RobotConfig& c) {
  auto fail = [](const char* reason) { return IkGeometryCheck{IkStatus::UnsupportedGeometry, reason}; };
  try { validate(c); } catch (const std::exception& e) { return fail(e.what()); }
  if (m.nq != c.dof || m.nv != c.dof || m.njnt != c.dof || m.nu != c.dof || m.nbody != 5)
    return fail("expected generated 3/4 hinge serial chain, five bodies and one actuator per joint");
  for (int i = 0; i < 4; ++i) {
    const int body = mj_name2id(&m, mjOBJ_BODY, bodies[i]);
    if (body != i+1 || m.body_parentid[body] != i || !identity(m.body_quat+4*body))
      return fail("body hierarchy or body rotation differs from synthetic arm");
    const double x = i >= 2 ? c.link_lengths_m[static_cast<std::size_t>(i-2)] : 0;
    const double z = i == 0 ? c.pedestal_height_m : 0;
    if (!equal(m.body_pos[3*body], x) || !equal(m.body_pos[3*body+1], 0) || !equal(m.body_pos[3*body+2], z))
      return fail("body translation differs from configured geometry");
  }
  for (int i = 0; i < c.dof; ++i) {
    const int j = mj_name2id(&m, mjOBJ_JOINT, joints[i]);
    const std::string drive = std::string("drive_")+joints[i];
    const int a = mj_name2id(&m, mjOBJ_ACTUATOR, drive.c_str());
    if (j != i || a != i || m.jnt_type[j] != mjJNT_HINGE || m.jnt_bodyid[j] != i+1 ||
        m.jnt_qposadr[j] != i || m.jnt_dofadr[j] != i || !equal(m.qpos0[i], 0) ||
        !m.jnt_limited[j] || m.actuator_trntype[a] != mjTRN_JOINT || m.actuator_trnid[2*a] != j)
      return fail("joint ordering, reference, limits or actuator transmission unsupported");
    for (int k = 0; k < 3; ++k) {
      const int axis = i == 0 ? 2 : (i == 3 ? 0 : 1);
      if (!equal(m.jnt_axis[3*j+k], k == axis ? 1 : 0) || !equal(m.jnt_pos[3*j+k], 0))
        return fail("joint axis sign or anchor differs from synthetic arm");
    }
    const auto lim = c.joint_limits_rad[static_cast<std::size_t>(i)];
    if (!equal(m.jnt_range[2*j], lim.lower) || !equal(m.jnt_range[2*j+1], lim.upper))
      return fail("model and configuration joint limits differ");
  }
  const int site = mj_name2id(&m, mjOBJ_SITE, "tool");
  if (site < 0 || m.site_bodyid[site] != 4) return fail("tool site must belong to distal mount");
  for (int k = 0; k < 3; ++k)
    if (!equal(m.site_pos[3*site+k], c.tool_frame.position_m[static_cast<std::size_t>(k)]))
      return fail("tool translation differs from configuration");
  return {IkStatus::Success, "supported synthetic yaw/shoulder/elbow[/roll] geometry"};
}

PositionIkResult solve_position_ik(Simulation& sim, const PositionIkRequest& request) {
  PositionIkResult out;
  auto finish = [&](IkStatus status, std::string reason) {
    out.status = status; out.reason = std::move(reason);
    if (status != IkStatus::Success) out.candidates.clear();
    return out;
  };
  const auto& c = sim.config();
  const auto geometry = check_analytical_geometry(sim.model(), c);
  if (geometry.status != IkStatus::Success) return finish(geometry.status, geometry.reason);
  if (!std::isfinite(request.position_tolerance_m) || request.position_tolerance_m <= 0 ||
      !valid_joints(request.current_rad, c.joint_limits_rad) ||
      !std::isfinite(request.wrist_search.fixed_rad) || request.wrist_search.budget < 1 ||
      request.wrist_search.budget > 4096 || request.candidate_budget < 1 || request.candidate_budget > 65536 ||
      (request.wrist_search.mode != WristMode::Fixed && request.wrist_search.mode != WristMode::Sampled))
    return finish(IkStatus::InvalidInput, "invalid tolerance, current joints, wrist options or candidate budget");
  for (double p : request.position_world_m)
    if (!std::isfinite(p)) return finish(IkStatus::InvalidInput, "world target must contain finite metres");
  if (c.dof == 4 && request.wrist_search.mode == WristMode::Fixed &&
      (request.wrist_search.fixed_rad < c.joint_limits_rad[3].lower ||
       request.wrist_search.fixed_rad > c.joint_limits_rad[3].upper))
    return finish(IkStatus::InvalidInput, "fixed wrist outside inclusive model limits");

  auto& coverage = out.coverage;
  const bool axial_wrist = c.dof == 4 && c.tool_frame.position_m[1] == 0 && c.tool_frame.position_m[2] == 0;
  coverage.scope = c.dof == 3 ? IkScope::ThreeJoint : (request.wrist_search.mode == WristMode::Fixed ?
    IkScope::FixedWrist : (axial_wrist ? IkScope::PositionIndependentWrist : IkScope::SampledWrist));
  coverage.wrist_continuum_covered = c.dof == 3 || axial_wrist;
  coverage.requested_wrist_budget = c.dof == 3 ? 0 : request.wrist_search.budget;
  const bool bounded = coverage.scope == IkScope::SampledWrist;
  const auto wrists = wrist_samples(request, c);
  if (bounded) coverage.max_wrist_gap_rad = wrist_gap(wrists, c.joint_limits_rad[3]);

  const Real x = request.position_world_m[0], y = request.position_world_m[1];
  const Real d = Real(c.pedestal_height_m)-request.position_world_m[2];
  const Real rho = std::hypot(x,y), distance = std::hypot(rho,d);
  const Real L1 = c.link_lengths_m[0], A = Real(c.link_lengths_m[1])+c.tool_frame.position_m[0];
  const Real ty = c.tool_frame.position_m[1], tz = c.tool_frame.position_m[2];

  const Real outer_bound = L1+std::hypot(A,std::hypot(ty,tz));
  if (!std::isfinite(distance) || !std::isfinite(outer_bound))
    return finish(IkStatus::NumericalFailure, "unrepresentable geometric scale");
  if (distance > outer_bound+16*eps*outer_bound)
    return finish(IkStatus::Unreachable, "outside conservative all-wrist outer radius");

  bool any_geometry = false;
  std::size_t enumerated = 0;
  try {
    for (double wrist : wrists) {
      const std::size_t wrist_candidates_begin = out.candidates.size();
      if (c.dof == 4) coverage.sampled_wrist_rad.push_back(wrist);
      const Real u = ty*std::cos(Real(wrist))-tz*std::sin(Real(wrist));
      const Real v = ty*std::sin(Real(wrist))+tz*std::cos(Real(wrist));
      const Real D = std::hypot(A,v), beta = std::atan2(v,A);
      bool normalized = false;
      if (D == 0) { ++out.rejected.degenerate_wrist_samples; continue; }
      Real radial_gap = rho-std::abs(u);
      if (radial_gap < 0) {
        if (-radial_gap > 16*eps*std::max(rho,std::abs(u))) {
          ++out.rejected.geometric_wrist_misses; continue;
        }
        radial_gap = 0; normalized = true;
      }
      const Real radius = std::sqrt(radial_gap*(rho+std::abs(u)));
      const Real scale = std::max({L1,D,radius,std::abs(d)});
      const Real a = L1/scale, b = D/scale, r = radius/scale, z = d/scale;
      const Real denominator = 2*a*b;
      if (denominator == 0 || !std::isfinite(denominator)) {
        ++out.rejected.degenerate_wrist_samples; continue;
      }
      Real cosine = (r*r+z*z-a*a-b*b)/denominator;
      const Real cosine_error = 16*eps*(r*r+z*z+a*a+b*b)/denominator;
      if (!std::isfinite(cosine)) return finish(IkStatus::NumericalFailure, "nonfinite planar inverse");
      if (std::abs(cosine) > 1) {
        if (std::abs(cosine)-1 > cosine_error) {
          ++out.rejected.geometric_wrist_misses; continue;
        }
        cosine = std::copysign(Real(1),cosine); normalized = true;
      }
      if (normalized) ++out.rejected.boundary_normalizations;
      any_geometry = true;
      const Real elbow = std::acos(cosine);
      for (int radius_sign : {1,-1}) {
        if (radius == 0 && radius_sign == -1) continue;
        const Real signed_radius = radius_sign*radius;
        const bool free_yaw = rho == 0 && u == 0;
        const Real yaw = free_yaw ? Real(request.current_rad[0]) : std::atan2(y,x)-std::atan2(u,signed_radius);
        for (int elbow_sign : {1,-1}) {
          if (elbow == 0 && elbow_sign == -1) continue;
          const Real phi = elbow_sign*elbow;
          const bool free_shoulder = radius == 0 && d == 0 && L1 == D;

          const Real shoulder = free_shoulder ? Real(request.current_rad[1]) :
            std::atan2(d,signed_radius)-std::atan2(D*std::sin(phi),L1+D*std::cos(phi));
          coverage.singular_family_representatives |= free_yaw || free_shoulder;
          const auto yaws = free_yaw ? std::vector<AngleValue>{{request.current_rad[0],false}} :
            angle_lifts(yaw,c.joint_limits_rad[0],request.candidate_budget);
          const auto shoulders = free_shoulder ? std::vector<AngleValue>{{request.current_rad[1],false}} :
            angle_lifts(shoulder,c.joint_limits_rad[1],request.candidate_budget);
          const auto elbows = angle_lifts(phi+beta,c.joint_limits_rad[2],request.candidate_budget);
          if (yaws.empty() || shoulders.empty() || elbows.empty()) { ++out.rejected.joint_limit_branches; continue; }
          for (const auto q0 : yaws) for (const auto q1 : shoulders) for (const auto q2 : elbows) {
            if (++enumerated > request.candidate_budget)
              return finish(IkStatus::ResourceLimit, "candidate enumeration budget exhausted; no partial solutions exposed");
            std::vector<double> q{q0.value,q1.value,q2.value};
            if (c.dof == 4) q.push_back(wrist);
            if (!valid_joints(q,c.joint_limits_rad)) { ++out.rejected.joint_limit_branches; continue; }


            bool duplicate = false;
            for (std::size_t old = wrist_candidates_begin; old < out.candidates.size(); ++old)
              duplicate |= same_configuration(q,out.candidates[old].joints_rad);
            if (duplicate) continue;
            const auto fk = sim.evaluate_fk(q).position_m;
            bool warning = false;
            for (const auto& w : sim.planning_data().warning) warning |= w.number != 0;
            double residual = 0;
            for (int k = 0; k < 3; ++k) residual = std::hypot(residual,fk[static_cast<std::size_t>(k)]-request.position_world_m[static_cast<std::size_t>(k)]);
            if (warning || !std::isfinite(residual) || residual > request.position_tolerance_m) { ++out.rejected.fk_candidates; continue; }
            out.candidates.push_back({std::move(q),residual,
              {radius == 0 ? 0 : radius_sign, elbow == 0 ? 0 : elbow_sign, wrist,free_yaw,free_shoulder,normalized,
               q0.boundary_reconstructed || q1.boundary_reconstructed || q2.boundary_reconstructed},
              sim.planning_data().ncon});
          }
        }
      }
    }
  } catch (const std::length_error& e) { return finish(IkStatus::ResourceLimit,e.what()); }
    catch (const std::exception& e) { return finish(IkStatus::NumericalFailure,e.what()); }
  coverage.branch_enumeration_complete = out.rejected.degenerate_wrist_samples == 0;
  if (!out.candidates.empty())
    return finish(IkStatus::Success, bounded ? "FK-verified candidates from bounded wrist samples; continuum coverage incomplete" :
      "FK-verified kinematic candidates; no path/contact filtering or ranking performed");
  if (out.rejected.degenerate_wrist_samples)
    return finish(IkStatus::DegenerateGeometry,"collapsed distal geometry at one or more wrist samples; coverage incomplete");
  if (bounded) return finish(IkStatus::NoCandidateWithinSearch,"bounded wrist search exhausted; NOT proven unreachable");
  if (out.rejected.fk_candidates) return finish(IkStatus::FkValidationFailed,"joint-admissible candidates failed authoritative FK/warning checks");
  if (any_geometry) return finish(IkStatus::JointLimitsExcludeCandidates,"joint limits exclude branches in the reported fixed-wrist/3-joint scope");
  return finish(c.dof == 4 && !axial_wrist ? IkStatus::UnreachableAtFixedWrist : IkStatus::Unreachable, c.dof == 4 && !axial_wrist ?
    "geometrically unreachable AT THE FIXED WRIST; unrestricted wrist not evaluated" :
    "outside analytical position workspace");
}
}
