#pragma once
#include "relic/simulation.hpp"
#include <string>
#include <utility>

namespace relic {
struct ContactPolicy {
  std::vector<std::pair<std::string,std::string>> allowed_pairs;
};
enum class ContactClass { EnvironmentOnly, ExplicitlyAllowed, DisallowedRobotContact };
struct ContactFinding {
  std::string geom_a, geom_b;
  int geom_a_id = -1, geom_b_id = -1;
  double distance_m = 0;
  int exclude = 0, constraint_address = -1;
  ContactClass classification = ContactClass::DisallowedRobotContact;
};
std::string validate_contact_policy(const mjModel&, const ContactPolicy&);
std::vector<ContactFinding> inspect_contacts(const mjModel&, const mjData&, const ContactPolicy&);
bool has_disallowed_contact(const std::vector<ContactFinding>&);
}
