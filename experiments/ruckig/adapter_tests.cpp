#include "adapter.hpp"
#include <iostream>
#include <sstream>

using namespace relic_experiment;
using namespace relic;
namespace {
void check(bool value,const char* reason) {if(!value)throw std::runtime_error(reason);}
Vec state(const Simulation& sim) {
  Vec s(static_cast<std::size_t>(mj_stateSize(&sim.model(),mjSTATE_INTEGRATION)));
  mj_getState(&sim.model(),&sim.execution_data(),s.data(),mjSTATE_INTEGRATION);return s;
}
}
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  try {
    for(std::size_t n:{3u,4u}) {
      Simulation sim(load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg")));
      Vec start{0,-0.1,0.1},goal{0.5,-0.3,0.6};if(n==4){start.push_back(0.1);goal.push_back(0.3);}
      MotionOptions o;o.max_velocity_rad_s.assign(n,0.5);o.collision_step_rad=0.0001;
      Bounds b{Vec(n,0.5),Vec(n,1),Vec(n,5)};
      auto generated=generate(start,goal,b,sim.joint_limits(),o);
      check(generated.trajectory.timestep_s==dt,"2ms adapter contract");
      sim.reset(start);sim.step(3);const auto before=state(sim);
      auto admission=preflight(sim,generated.trajectory,start,goal,o);
      check(admission.status==MotionStatus::Success,"valid path preflight");
      check(admission.samples>generated.trajectory.commands.size(),"inter-command collision subdivisions required");
      check(state(sim)==before,"planning cannot mutate execution state");
      auto denied=execute(sim,generated.trajectory,start,goal,o,{},nullptr,1);
      check(!denied.started && denied.status==MotionStatus::ResourceLimit && state(sim)==before,"budget rejection must not reset or execute");
      auto malformed=generated.trajectory;malformed.commands[1][0]=4;
      check(preflight(sim,malformed,start,goal,o).status==MotionStatus::InvalidPath,"invalid command rejected");
      malformed=generated.trajectory;malformed.timestep_s=0.001;
      check(preflight(sim,malformed,start,goal,o).status==MotionStatus::InvalidInput,"wrong tick rejected");
      ContactPolicy bad;bad.allowed_pairs={{"missing","ground"}};
      check(preflight(sim,generated.trajectory,start,goal,o,bad).status==MotionStatus::InvalidInput,"policy validated");
      auto contact_goal=goal;contact_goal[0]=0;contact_goal[1]=1;contact_goal[2]=0;
      auto collision=generate(start,contact_goal,b,sim.joint_limits(),o);
      check(preflight(sim,collision.trajectory,start,contact_goal,o).status==MotionStatus::DisallowedContact,"actual contact rejected");
      bool step_rejected=false;
      try {auto tight=o;tight.max_command_step_rad=1e-9;(void)generate(start,goal,b,sim.joint_limits(),tight);}
      catch(const Rejection& e){step_rejected=e.category=="step_limit";}
      check(step_rejected,"small command-step rejection");
      std::ostringstream trace;TraceWriter writer(trace,sim.config());
      const auto run=execute(sim,generated.trajectory,start,goal,o,{},&writer);
      check(run.started && run.command_completed && run.settled && run.status==MotionStatus::Success,"actual Ruckig execution must settle");
      check(run.first_command_did_not_teleport && run.peak_tracking_error>1e-4,"actual position actuators have tracking lag");
      check(trace.str().find("_observed_rad")!=std::string::npos && trace.str().size()>1000,"unchanged TraceWriter used");
      const auto linear=make_joint_trajectory(start,goal,sim.joint_limits(),dt,o);
      check(linear.status==MotionStatus::Success,"production linear generation");
      const auto linear_run=execute(sim,linear.trajectory,start,goal,o);
      check(linear_run.started && linear_run.settled,"actual linear execution must settle");
      auto strict=o;strict.joint_tolerance_rad=strict.cartesian_tolerance_m=1e-12;strict.timeout_s=3;
      check(execute(sim,generated.trajectory,start,goal,strict).status==MotionStatus::NonConvergence,"strict accuracy failure retained");
      auto short_timeout=o;short_timeout.timeout_s=0.002;
      const auto timed=execute(sim,generated.trajectory,start,goal,short_timeout);
      check(timed.status==MotionStatus::Timeout && !timed.command_completed,"partial command timeout explicit");
      const auto first=execute(sim,generated.trajectory,start,goal,o),second=execute(sim,generated.trajectory,start,goal,o);
      check(first.execution_s==second.execution_s && first.final_joint_errors==second.final_joint_errors,"documented resets reproducible");
    }
    std::cout<<"adapter/preflight/MuJoCo integration checks passed\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
