#include "profile.hpp"
#include "qualification_api.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>

using namespace relic_experiment;
int relic_experiment::qualify(const std::filesystem::path& path) {
  std::ofstream out(path); out.imbue(std::locale::classic()); out<<std::setprecision(17);
  out<<"dof,case,status,duration_s,endpoint_error_rad,sync_error_s,continuous_v,continuous_a,continuous_j,sampled_v,sampled_a,sampled_j,max_step_rad,reason\n";
  int failures=0;
  for(std::size_t n:{3u,4u}) {
    const std::vector<std::string> cases{"zero","tiny","tiny_offset","short","long","negative","reversed","asymmetric","near_limit","restrictive","nan","infinity","bad_dimensions","zero_bound","negative_bound","infinite_bound","resource_limit"};
    for(const auto& name:cases) {
      Vec start(n,0),goal(n,0.1); Bounds b{Vec(n,0.5),Vec(n,1),Vec(n,5)};
      std::vector<Interval> limits(n,{-2.8,2.8});
      if(name=="zero") goal=start;
      if(name=="tiny") goal.assign(n,1e-8);
      if(name=="tiny_offset") {start={0,-0.3,0.6};if(n==4)start.push_back(0.1);goal=start;for(double& q:goal)q+=1e-8;}
      if(name=="long") goal.assign(n,2.5);
      if(name=="negative") goal.assign(n,-0.3);
      if(name=="reversed") {start.assign(n,0.3);goal.assign(n,-0.2);}
      if(name=="asymmetric") {goal={0.8,0.013,0};if(n==4)goal.push_back(-0.21);}
      if(name=="near_limit") {start.assign(n,2.7);goal.assign(n,2.79);}
      if(name=="restrictive") {goal.assign(n,0.001);b={Vec(n,0.0001),Vec(n,0.001),Vec(n,0.01)};}
      if(name=="nan") goal[0]=std::numeric_limits<double>::quiet_NaN();
      if(name=="infinity") start[0]=std::numeric_limits<double>::infinity();
      if(name=="bad_dimensions") goal.pop_back();
      if(name=="zero_bound") b.jerk[0]=0;
      if(name=="negative_bound") b.acceleration[0]=-1;
      if(name=="infinite_bound") b.velocity[0]=std::numeric_limits<double>::infinity();
      if(name=="resource_limit") b.velocity.assign(n,1e-9);
      const bool expected_invalid=name=="nan" || name=="infinity" || name=="bad_dimensions" ||
        name=="zero_bound" || name=="negative_bound" || name=="infinite_bound";
      try {
        const auto s=solve(start,goal,b,limits);
        if(expected_invalid || name=="resource_limit") ++failures;
        out<<n<<','<<name<<','<<(s.endpoint_error>1e-10?"qualified_with_endpoint_tolerance":"qualified")<<','<<s.duration<<','<<s.endpoint_error<<','<<s.sync_error<<','<<s.continuous_v<<','<<s.continuous_a<<','<<s.continuous_j<<','<<s.sampled_v<<','<<s.sampled_a<<','<<s.sampled_j<<','<<s.max_step<<",\n";
      } catch(const Rejection& e) {
        const bool expected=(expected_invalid && e.category=="invalid_input") ||
          (name=="resource_limit" && (e.category=="resource_limit" || e.category=="range_limit"));
        if(!expected) {++failures; std::cerr<<n<<' '<<name<<": "<<e.category<<": "<<e.what()<<'\n';}
        out<<n<<','<<name<<','<<e.category<<",,,,,,,,,,,"<<e.what()<<'\n';
      } catch(const std::exception& e) {++failures;std::cerr<<e.what()<<'\n';}
    }
  }
  // Raw upstream validation and output-dimension behavior with safe fixed storage.
  ruckig::Ruckig<3> solver(dt); ruckig::InputParameter<3> in;
  in.current_position={0,0,0};in.target_position={0.1,0.1,0.1};
  in.max_velocity={0.5,0.5,0.5};in.max_acceleration={1,1,1};in.max_jerk={5,5,5};
  in.target_velocity[0]=0.6;
  if(solver.validate_input<false>(in,true,true)) ++failures;
  out<<"3,infeasible_target_velocity,upstream_rejected,,,,,,,,,,,validate_input returned false\n";
  // Dynamic output vector mismatch has an explicit safe upstream exception.
  ruckig::Trajectory<0> dynamic_trajectory(3);Vec wrong(2),v(3),a(3),j(3);std::size_t section=0;
  bool dimensions_rejected=false;
  try{dynamic_trajectory.at_time(0,wrong,v,a,j,section);}catch(const ruckig::RuckigError&){dimensions_rejected=true;}
  if(!dimensions_rejected)++failures;
  out<<"3,bad_output_dimensions,upstream_rejected,,,,,,,,,,,at_time threw before profile access\n";
  if(!out) ++failures;
  std::cout<<"Qualification: "<<failures<<" unexpected failures\n";
  return failures?1:0;
}
