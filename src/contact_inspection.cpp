#include "relic/contact_inspection.hpp"
#include <algorithm>

namespace relic {
std::string validate_contact_policy(const mjModel& m, const ContactPolicy& policy) {
  if (policy.allowed_pairs.size() > 128) return "contact allowlist exceeds 128 pairs";
  for (const auto& [a,b] : policy.allowed_pairs)
    if (a == b || mj_name2id(&m,mjOBJ_GEOM,a.c_str()) < 0 || mj_name2id(&m,mjOBJ_GEOM,b.c_str()) < 0)
      return "allowlist requires two distinct existing named geoms";
  return {};
}
std::vector<ContactFinding> inspect_contacts(const mjModel& m, const mjData& d, const ContactPolicy& policy) {
  std::vector<ContactFinding> out;
  for (int i = 0; i < d.ncon; ++i) {
    const auto& c = d.contact[i];
    ContactFinding f;
    f.geom_a_id = c.geom[0]; f.geom_b_id = c.geom[1]; f.distance_m = c.dist;
    f.exclude = c.exclude; f.constraint_address = c.efc_address;
    auto name = [&](int id) {
      const char* n = id >= 0 ? mj_id2name(&m,mjOBJ_GEOM,id) : nullptr;
      return n ? std::string(n) : std::string("<unnamed-or-flex>");
    };
    f.geom_a = name(f.geom_a_id); f.geom_b = name(f.geom_b_id);

    const bool robot = f.geom_a_id < 0 || f.geom_b_id < 0 ||
      m.geom_bodyid[f.geom_a_id] != 0 || m.geom_bodyid[f.geom_b_id] != 0;
    const bool allowed = std::any_of(policy.allowed_pairs.begin(),policy.allowed_pairs.end(),[&](const auto& p) {
      return (p.first == f.geom_a && p.second == f.geom_b) || (p.first == f.geom_b && p.second == f.geom_a);
    });
    f.classification = allowed ? ContactClass::ExplicitlyAllowed :
      (robot ? ContactClass::DisallowedRobotContact : ContactClass::EnvironmentOnly);
    out.push_back(std::move(f));
  }
  return out;
}
bool has_disallowed_contact(const std::vector<ContactFinding>& contacts) {
  return std::any_of(contacts.begin(),contacts.end(),[](const auto& f) {
    return f.classification == ContactClass::DisallowedRobotContact;
  });
}
}
