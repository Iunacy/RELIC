#include "adapter.hpp"
#include "qualification_api.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>

using namespace relic;
using namespace relic_experiment;
namespace {
void check(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
bool at_limit(double x,double lo,double hi) {
  const double eps=1e-8*std::max({1.0,std::abs(lo),std::abs(hi)});
  return x<=lo+eps || x>=hi-eps;
}
void format(std::ostream& o) {o.imbue(std::locale::classic());o<<std::setprecision(17);}
void saturation_checks(const Simulation& sim) {
  const auto& m=sim.model();
  std::unique_ptr<mjData,DataDeleter> d(mj_makeData(&m));
  for(double sign:{-1.0,0.0,1.0}) {
    mj_resetData(&m,d.get());
    d->ctrl[0]=sign;
    mj_forward(&m,d.get());
    const double f=d->actuator_force[0];
    check(at_limit(f,m.actuator_forcerange[0],m.actuator_forcerange[1])==(sign!=0),"saturation detector positive/negative control");
    check(std::abs(f-sign*20)<1e-10,"force clamp +/-20 Nm");
  }
}
struct Recorder {
  const JointTrajectory& t;
  const Vec& start;
  std::ostringstream csv;
  std::size_t rows=0;
  double peak=0;
  Recorder(const JointTrajectory& path,const Vec& q):t(path),start(q) {
    format(csv);
    csv<<"tick,stage,phase,command_completed,sim_time_s,reference_time_s,joint,joint_name,q_des_rad,v_des_fd_rad_s,a_des_fd_rad_s2,q_actual_rad,v_actual_rad_s,error_rad,ctrl_rad,actuator_force_Nm,qfrc_actuator_Nm,servo_unclipped_Nm,force_saturated,ctrl_at_limit,force_lower_Nm,force_upper_Nm,ctrl_lower_rad,ctrl_upper_rad,qfrc_bias_Nm,qfrc_passive_Nm\n";
  }
  double q(std::size_t k,std::size_t j) const { return t.commands[std::min(k,t.commands.size()-1)][j]; }
  double v(std::size_t k,std::size_t j) const { return k?(q(k,j)-q(k-1,j))/dt:0; }
  void operator()(const Simulation& sim,std::size_t k,const char* stage,bool settling) {
    const auto& m=sim.model(); const auto& d=sim.execution_data();const auto snap=sim.snapshot();
    const bool pre=std::string(stage)=="pre";
    check(std::abs(d.time-(static_cast<double>(k)-(pre?1:0))*dt)<1e-10,"timestamp alignment");
    check(m.nq==static_cast<int>(start.size()) && m.nv==m.nq && m.nu==m.nq,"dimensions");
    const bool completed=t.commands.size()==1 || (pre?k-1:k)>=t.commands.size()-1;
    for(std::size_t j=0;j<start.size();++j) {
      const auto aid=static_cast<int>(j); const int jid=m.actuator_trnid[2*aid];
      const int qa=m.jnt_qposadr[jid],va=m.jnt_dofadr[jid];
      check(m.jnt_type[jid]==mjJNT_HINGE && m.actuator_trntype[aid]==mjTRN_JOINT &&
        m.actuator_gear[6*aid]==1 && m.actuator_gaintype[aid]==mjGAIN_FIXED &&
        m.actuator_biastype[aid]==mjBIAS_AFFINE,"supported position-servo units/transmission");
      check(snap.observed_rad[j]==d.qpos[qa] && snap.velocity_rad_s[j]==d.qvel[va],"snapshot indexing");
      check(std::abs(d.ctrl[aid]-q(k,j))<1e-14,"reference indexing");
      if(k==0)check(d.qpos[qa]==start[j] && d.qvel[va]==0 && d.time==0,"reset state");
      const double control=std::clamp(d.ctrl[aid],m.actuator_ctrlrange[2*aid],m.actuator_ctrlrange[2*aid+1]);
      const double raw=m.actuator_gainprm[mjNGAIN*aid]*control+m.actuator_biasprm[mjNBIAS*aid]+
        m.actuator_biasprm[mjNBIAS*aid+1]*d.actuator_length[aid]+m.actuator_biasprm[mjNBIAS*aid+2]*d.actuator_velocity[aid];
      check(std::abs(d.actuator_force[aid]-std::clamp(raw,m.actuator_forcerange[2*aid],m.actuator_forcerange[2*aid+1]))<1e-10,"servo/clamp equation");
      check(std::abs(d.qfrc_actuator[va]-d.actuator_force[aid])<1e-10,"gear-one torque mapping");
      const double desired_v=v(k,j),desired_a=k?(desired_v-v(k-1,j))/dt:0;
      const double error=q(k,j)-d.qpos[qa];peak=std::max(peak,std::abs(error));
      for(double x:{d.time,q(k,j),desired_v,desired_a,d.qpos[qa],d.qvel[va],error,d.ctrl[aid],
        d.actuator_force[aid],d.qfrc_actuator[va],raw,d.qfrc_bias[va],d.qfrc_passive[va]})
        check(std::isfinite(x),"finite telemetry");
      csv<<k<<','<<stage<<','<<(settling?"settling":"command")<<','<<completed<<','<<d.time<<','<<static_cast<double>(k)*dt<<','<<j<<','
        <<mj_id2name(&m,mjOBJ_JOINT,jid)<<','<<q(k,j)<<','<<desired_v<<','<<desired_a<<','<<d.qpos[qa]<<','<<d.qvel[va]<<','
        <<error<<','<<d.ctrl[aid]<<','<<d.actuator_force[aid]<<','<<d.qfrc_actuator[va]<<','<<raw<<','
        <<(m.actuator_forcelimited[aid] && at_limit(d.actuator_force[aid],m.actuator_forcerange[2*aid],m.actuator_forcerange[2*aid+1]))<<','
        <<(m.actuator_ctrllimited[aid] && at_limit(d.ctrl[aid],m.actuator_ctrlrange[2*aid],m.actuator_ctrlrange[2*aid+1]))<<','
        <<m.actuator_forcerange[2*aid]<<','<<m.actuator_forcerange[2*aid+1]<<','<<m.actuator_ctrlrange[2*aid]<<','<<m.actuator_ctrlrange[2*aid+1]<<','
        <<d.qfrc_bias[va]<<','<<d.qfrc_passive[va]<<'\n';
      ++rows;
    }
  }
};
}
int main(int argc,char** argv) {
  if(argc!=3) {std::cerr<<"usage: relic_ruckig_diagnostics EXAMPLES OUTPUT_DIR\n";return 2;}
  const std::filesystem::path out(argv[2]);std::filesystem::create_directories(out);
  if(qualify(out/"qualification.csv"))return 1;
  std::ofstream summary(out/"runs.csv"),tests(out/"checks.csv");format(summary);
  summary<<"dof,case,method,repeat,status,command_completed,settled,command_s,execution_s,command_end_cartesian_m,final_cartesian_m,legacy_peak_rad,torque_peak_Nm,saturated_joint_ticks,rows,reason\n";
  tests<<"dof,case,method,check,result\n";
  int failures=0;
  for(std::size_t n:{3u,4u}) {
    const auto config=load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(n)+"dof.cfg"));
    std::ofstream xml(out/("model_"+std::to_string(n)+"dof.xml"));xml<<make_mjcf(config);
    for(const std::string name:{"nominal","zero","fast"}) {
      Vec start{0,-0.3,0.6},goal=name=="zero"?start:Vec{0.6,-0.2,0.4};
      if(n==4){start.push_back(0.1);goal.push_back(name=="zero"?0.1:0.35);}
      MotionOptions options;options.max_velocity_rad_s.assign(n,name=="fast"?2:0.5);options.collision_step_rad=0.0005;
      Bounds bounds{options.max_velocity_rad_s,Vec(n,name=="fast"?20:1),Vec(n,name=="fast"?200:5)};
      Simulation sim(config);saturation_checks(sim);
      auto r=generate(start,goal,bounds,sim.joint_limits(),options);
      for(const std::string method:{"linear_matched","ruckig"}) {
        auto t=r.trajectory;
        if(method=="linear_matched") {
          auto o=options;
          for(std::size_t j=0;j<n;++j)if(goal[j]!=start[j])
            o.max_velocity_rad_s[j]=std::min(o.max_velocity_rad_s[j],std::abs(goal[j]-start[j])/t.command_duration_s*(1+1e-12));
          const auto linear=make_joint_trajectory(start,goal,sim.joint_limits(),dt,o);
          check(linear.status==MotionStatus::Success,"linear generation");t=linear.trajectory;
        }
        check(std::abs(t.command_duration_s-r.trajectory.command_duration_s)<1e-12,"matched duration");
        std::string first;
        Run baseline;
        for(int repeat=0;repeat<2;++repeat) {
          Recorder rec(t,start);
          sim.reset(goal);sim.step(7);
          const auto run=execute(sim,t,start,goal,options,{},nullptr,500000,std::ref(rec));
          const std::string key=std::to_string(n)+"_"+name+"_"+method+"_"+std::to_string(repeat);
          std::ofstream file(out/(key+".csv"));file<<rec.csv.str();
          summary<<n<<','<<name<<','<<method<<','<<repeat<<','<<motion_status_name(run.status)<<','<<run.command_completed<<','<<run.settled<<','
            <<t.command_duration_s<<','<<run.execution_s<<','<<run.command_end_cartesian_error<<','<<run.final_cartesian_error<<','
            <<run.peak_tracking_error<<','<<run.torque_peak<<','<<run.saturated_joint_ticks<<','<<rec.rows<<",\""<<run.reason<<"\"\n";
          bool ok=run.status==MotionStatus::Success && run.command_completed && run.settled && file.good();
          ok &= std::abs(run.peak_tracking_error-rec.peak)<1e-14;
          ok &= rec.rows==n*(1+2*static_cast<std::size_t>(std::llround(run.execution_s/dt)));
          if(repeat==0) {
            first=rec.csv.str();baseline=execute(sim,t,start,goal,options);
            ok &= baseline.status==run.status && baseline.execution_s==run.execution_s &&
              baseline.peak_tracking_error==run.peak_tracking_error && baseline.final_joint_errors==run.final_joint_errors;
          } else ok &= first==rec.csv.str();
          tests<<n<<','<<name<<','<<method<<','<<(repeat?"repeat_byte_identical":"observer_matches_baseline_reset_alignment_finite_units_saturation")<<','<<(ok?"PASS":"FAIL")<<'\n';
          if(!ok){++failures;std::cerr<<key<<": "<<run.reason<<'\n';}
        }
      }
    }
  }
  std::cout<<"24 diagnostic runs; "<<failures<<" failures\n";
  return failures || !summary || !tests?1:0;
}
