#include "adapter.hpp"
#include "qualification_api.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <sstream>

using namespace relic_experiment;
using namespace relic;
namespace {
std::string num(double value) {std::ostringstream s;s.imbue(std::locale::classic());s<<std::setprecision(17)<<value;return s.str();}
std::string csv(std::string value) {std::string q="\"";for(char c:value){if(c=='\"')q+='\"';q+=c;}return q+'\"';}
void row(std::ostream& o,const std::vector<std::string>& values) {
  for(std::size_t k=0;k<values.size();++k){if(k)o<<',';o<<values[k];}o<<'\n';
}
double ms(std::chrono::steady_clock::time_point t) {return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
struct Case {std::string name;Vec start,goal;bool restrictive=false,strict=false,timeout=false;};
}
int main(int argc,char** argv) {
  if(argc!=3){std::cerr<<"usage: relic_ruckig_benchmark EXAMPLES OUTPUT_DIR\n";return 2;}
  const std::filesystem::path output(argv[2]);std::filesystem::create_directories(output/"traces");
  if(qualify(output/"qualification.csv")!=0) {std::cerr<<"Qualification failed; no execution attempted\n";return 1;}
  std::ofstream report(output/"comparison.csv"),runtime(output/"runtime.csv"),joints(output/"joints.csv"),tradeoff(output/"tradeoff.csv"),summary(output/"summary.csv");
  report.imbue(std::locale::classic());runtime.imbue(std::locale::classic());joints.imbue(std::locale::classic());
  row(report,{"dof","case","method","generation_status","preflight_status","execution_status","started","command_completed","settled","command_s","executed_command_s","execution_s","settling_s","endpoint_error_rad","sync_error_s","reference_v_peak_rad_s","reference_a_peak_rad_s2","reference_j_peak_rad_s3","command_sampled_v_rad_s","command_sampled_a_rad_s2","command_sampled_j_rad_s3","command_max_step_rad","reference_vaj_compliant","command_end_joint_error_rad","command_end_cartesian_error_m","final_joint_error_rad","final_cartesian_error_m","peak_tracking_error_rad","observed_v_rad_s","observed_sampled_a_rad_s2","observed_sampled_j_rad_s3","peak_torque_Nm","saturated_joint_ticks","preflight_samples","preflight_contact_observations","execution_contact_observations","velocity_limit","acceleration_limit","jerk_limit","joint_tolerance_rad","cartesian_tolerance_m","timeout_s","endpoint_tolerance_used","reason"});
  row(runtime,{"dof","case","method","generation_wall_ms","preflight_wall_ms","execution_wall_ms"});
  row(joints,{"dof","case","method","joint","start_rad","goal_rad","final_error_rad"});
  row(tradeoff,{"dof","case","method","command_s","ruckig_reference_command_s","delta_vs_reference_s","execution_s","command_end_cartesian_error_m","final_cartesian_error_m","peak_tracking_error_rad","status"});
  row(summary,{"dof","method","status","count"});
  std::map<std::tuple<std::size_t,std::string,std::string>,std::size_t> counts;
  std::size_t rows=0,executed=0,settled=0;int unexpected=0;
  for(std::size_t n:{3u,4u}) {
    auto with_wrist=[&](Vec q,double wrist){if(n==4)q.push_back(wrist);return q;};
    const Vec base=with_wrist({0,-0.3,0.6},0.1);
    const Vec nominal=with_wrist({0.6,-0.2,0.4},0.35);
    auto tiny=base;for(double& q:tiny)q+=1e-8;
    std::vector<Case> cases{
      {"zero",base,base},{"tiny",base,tiny},
      {"short",base,with_wrist({0.03,-0.29,0.58},0.14)},
      {"nominal",base,nominal},
      {"long",with_wrist({-1.5,-0.7,1.2},-1),with_wrist({1.6,-0.2,0.2},1.1)},
      {"negative",base,with_wrist({-0.6,-0.45,0.8},-0.4)},
      {"reversed",nominal,base},
      {"asymmetric",base,with_wrist({0.9,-0.299,0.6},0.03)},
      {"near_limit",with_wrist({2.7,-0.3,0.6},1.9),with_wrist({2.79,-0.29,0.58},1.99)},
      {"restrictive",base,with_wrist({0.08,-0.28,0.56},0.13),true},
      {"contact_rejection",base,with_wrist({0,1,0},0)},
      {"strict_settling",base,nominal,false,true},
      {"timeout",base,nominal,false,false,true}
    };
    auto config=load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg"));
    for(const auto& c:cases) {
      MotionOptions original;original.max_velocity_rad_s.assign(n,c.restrictive?0.05:0.5);
      original.collision_step_rad=0.0005;
      if(c.strict){original.joint_tolerance_rad=original.cartesian_tolerance_m=1e-12;original.timeout_s=4;}
      if(c.timeout)original.timeout_s=0.05;
      Bounds b{original.max_velocity_rad_s,Vec(n,c.restrictive?0.1:1),Vec(n,c.restrictive?0.5:5)};
      double reference=0;
      try {Simulation ref(config);reference=generate(c.start,c.goal,b,ref.joint_limits(),original).trajectory.command_duration_s;}
      catch(const std::exception& e){std::cerr<<"Unexpected reference failure "<<c.name<<": "<<e.what()<<'\n';return 1;}
      for(const std::string method:{"linear_native","ruckig","linear_matched","ruckig_slow"}) {
        Simulation sim(config);auto options=original;JointTrajectory trajectory;Samples audit;Run run;
        std::string generation="Success",reason;double generation_ms=0;
        const auto generation_start=std::chrono::steady_clock::now();
        try {
          if(method=="ruckig" || method=="ruckig_slow") {
            auto g=generate(c.start,c.goal,b,sim.joint_limits(),options,method=="ruckig_slow"?2*reference:0);
            trajectory=std::move(g.trajectory);audit=std::move(g.audit);
          } else {
            if(method=="linear_matched" && reference>0) for(std::size_t j=0;j<n;++j) {
              const double displacement=std::abs(c.goal[j]-c.start[j]);
              if(displacement>0)options.max_velocity_rad_s[j]=std::min(options.max_velocity_rad_s[j],displacement/reference*(1+1e-12));
            }
            auto g=make_joint_trajectory(c.start,c.goal,sim.joint_limits(),dt,options);
            require(g.status==MotionStatus::Success,"linear_generation",g.reason);
            trajectory=std::move(g.trajectory);audit=linear_diagnostics(trajectory);
          }
          generation_ms=ms(generation_start);
          const std::string key=std::to_string(n)+"_"+c.name+"_"+method;
          std::ofstream trace_file(output/"traces"/(key+".csv"));TraceWriter trace(trace_file,config);
          run=execute(sim,trajectory,c.start,c.goal,options,{},&trace);
          reason=run.started?run.reason:run.admission.reason;
        } catch(const Rejection& e) {generation=e.category;reason=e.what();generation_ms=ms(generation_start);++unexpected;}
        catch(const std::exception& e){generation="Exception";reason=e.what();++unexpected;}
        ++rows;executed+=run.started;settled+=run.settled;
        ++counts[{n,method,motion_status_name(run.status)}];
        const auto expected=c.name=="contact_rejection"?MotionStatus::DisallowedContact:
          (c.timeout || (c.name=="long" && method=="ruckig_slow"))?MotionStatus::Timeout:
          c.strict?MotionStatus::NonConvergence:MotionStatus::Success;
        if(run.status!=expected) {++unexpected;std::cerr<<n<<' '<<c.name<<' '<<method<<" expected "<<motion_status_name(expected)<<" got "<<motion_status_name(run.status)<<'\n';}
        const double na=std::numeric_limits<double>::quiet_NaN();
        row(report,{std::to_string(n),c.name,method,generation,motion_status_name(run.admission.status),motion_status_name(run.status),
          std::to_string(run.started),std::to_string(run.command_completed),std::to_string(run.settled),num(audit.duration),num(run.executed_command_s),num(run.execution_s),num(run.settling_s),num(audit.endpoint_error),num(audit.sync_error),num(audit.continuous_v),num(audit.continuous_a),num(audit.continuous_j),num(audit.sampled_v),num(audit.sampled_a),num(audit.sampled_j),num(audit.max_step),
          std::to_string(generation=="Success" && (!trajectory.acceleration_discontinuous)),
          num(run.command_completed?run.command_end_joint_error:na),num(run.command_completed?run.command_end_cartesian_error:na),
          num(run.started?run.final_joint_error:na),num(run.started?run.final_cartesian_error:na),num(run.started?run.peak_tracking_error:na),num(run.started?run.observed_v:na),num(run.started?run.observed_sampled_a:na),num(run.started?run.observed_sampled_j:na),num(run.started?run.torque_peak:na),std::to_string(run.saturated_joint_ticks),std::to_string(run.admission.samples),std::to_string(run.admission.contact_observations),std::to_string(run.contact_observations),num(b.velocity[0]),num(b.acceleration[0]),num(b.jerk[0]),num(options.joint_tolerance_rad),num(options.cartesian_tolerance_m),num(options.timeout_s),std::to_string(audit.endpoint_error>1e-10),csv(reason)});
        row(runtime,{std::to_string(n),c.name,method,num(generation_ms),num(run.preflight_wall_ms),num(run.execution_wall_ms)});
        row(tradeoff,{std::to_string(n),c.name,method,num(audit.duration),num(reference),num(audit.duration-reference),
          num(run.started?run.execution_s:na),num(run.command_completed?run.command_end_cartesian_error:na),num(run.started?run.final_cartesian_error:na),num(run.started?run.peak_tracking_error:na),motion_status_name(run.status)});
        for(std::size_t j=0;j<n;++j)row(joints,{std::to_string(n),c.name,method,std::to_string(j),num(c.start[j]),num(c.goal[j]),num(run.started?run.final_joint_errors[j]:na)});
      }
    }
  }
  for(const auto& [key,count]:counts) {
    const auto& [n,method,status]=key;row(summary,{std::to_string(n),method,status,std::to_string(count)});
  }
  std::cout<<rows<<" comparisons; "<<executed<<" actuator executions; "<<settled<<" settled; "<<unexpected<<" unexpected failures\n";
  return unexpected || !report || !runtime || !joints || !tradeoff || !summary ? 1:0;
}
