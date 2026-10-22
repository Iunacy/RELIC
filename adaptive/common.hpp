#pragma once
#include "relic/motion.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace relic::adaptive {
using Vec=std::vector<double>;
constexpr double command_step_bound=0.005, collision_step=0.002;
struct Random {
  std::uint64_t state;
  std::uint64_t next() { auto z=(state+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31); }
  double uniform(double lo,double hi) {return lo+(hi-lo)*static_cast<double>(next()>>11)*0x1.0p-53;}
};
inline double distance(const Pose& a,const Pose& b) {
  double x=0;for(int j=0;j<3;++j)x=std::hypot(x,a.position_m[j]-b.position_m[j]);return x;
}
inline std::string array(std::span<const double> v) {
  std::ostringstream s;s<<std::setprecision(17)<<'[';
  for(std::size_t j=0;j<v.size();++j){if(j)s<<',';s<<v[j];}s<<']';return s.str();
}


inline bool safe_segment(Simulation& sim,const Vec& from,const Vec& to,bool enforce_step) {
  const auto limits=sim.joint_limits();if(from.size()!=limits.size() || to.size()!=limits.size())return false;
  std::size_t samples=1;
  for(std::size_t j=0;j<to.size();++j) {
    if(!std::isfinite(to[j]) || !std::isfinite(from[j]) || to[j]<limits[j].lower || to[j]>limits[j].upper ||
      from[j]<limits[j].lower || from[j]>limits[j].upper)return false;
    const auto& model=sim.model();
    if(enforce_step && model.actuator_ctrllimited[j] &&
       (to[j]<model.actuator_ctrlrange[2*j] || to[j]>model.actuator_ctrlrange[2*j+1]))return false;
    const double delta=std::abs(to[j]-from[j]);
    if(enforce_step && delta>command_step_bound+1e-12)return false;
    samples=std::max(samples,static_cast<std::size_t>(std::ceil(delta/collision_step)));
  }
  if(samples>10000)return false;
  for(std::size_t k=0;k<=samples;++k) {
    Vec q(to.size());for(std::size_t j=0;j<q.size();++j)q[j]=from[j]+(to[j]-from[j])*static_cast<double>(k)/static_cast<double>(samples);
    sim.evaluate_fk(q);
    if(!numerical_warnings(sim.planning_data()).empty() || has_disallowed_contact(inspect_contacts(sim.model(),sim.planning_data(),{})))return false;
  }
  return true;
}
}
