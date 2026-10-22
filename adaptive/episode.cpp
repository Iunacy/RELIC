#include "recorder.hpp"
#include "residual.hpp"
#include <optional>
#include <chrono>
#include <iostream>
#ifndef RELIC_SOURCE_COMMIT
#define RELIC_SOURCE_COMMIT "unknown"
#endif
using namespace relic;
using namespace relic::adaptive;
int main(int argc,char** argv) {try {
  if(argc!=10 && argc!=13 && argc!=14)throw std::invalid_argument("usage: relic_episode EXAMPLES OUTPUT SEED FAMILY DOF CATEGORY SEVERITY PARTITION EPISODE [CONTROLLER BOUNDS MODEL_OR_REPLAY]");
  const auto began=std::chrono::steady_clock::now();
  const std::filesystem::path out(argv[2]);std::filesystem::create_directories(out);
  const auto seed=std::stoull(argv[3]),family=std::stoull(argv[4]);const int dof=std::stoi(argv[5]);
  const std::string category=argv[6],partition=argv[8],id=argv[9];const double severity=std::stod(argv[7]);
  if((dof!=3 && dof!=4) || !std::isfinite(severity) || severity<0 || severity>1)throw std::invalid_argument("invalid embodiment/severity");
  std::string controller=argc>=13?argv[10]:"conventional",model_error="none";Bounds bounds;std::optional<MLP> mlp;
  double lead=.1,ki=0,kp_error=0,integral_limit=.05;int inject_tick=-1;
  if(argc==14){std::ifstream control(argv[13]);std::string version; if(!(control>>version>>lead>>ki>>kp_error>>integral_limit>>inject_tick) || version!="RELIC_CONTROL_V1" || !std::isfinite(lead) || !std::isfinite(ki) || !std::isfinite(kp_error) || !std::isfinite(integral_limit) || lead<0 || lead>1 || ki<0 || ki>10 || kp_error<0 || kp_error>5 || integral_limit<0 || integral_limit>.2)throw std::invalid_argument("invalid control configuration");}
  std::vector<Vec> replay;
  if(controller!="conventional")bounds=Bounds::load(argv[11]);
  if(controller=="mlp" || controller=="transfer" || controller=="common_transfer") {try {mlp=MLP::load(argv[12]);
    for(int j=0;j<4;++j)if(mlp->bounds.magnitude[j]!=bounds.magnitude[j] || mlp->bounds.rate[j]!=bounds.rate[j])throw std::runtime_error("model_bounds_mismatch");
    if(controller!="transfer" && controller!="common_transfer" && mlp->embodiment!=0 && mlp->embodiment!=dof)throw std::runtime_error("unsupported_embodiment");
    if((controller=="transfer" || controller=="common_transfer") && !(mlp->embodiment==3 && dof==4))throw std::runtime_error("unsupported_transfer");
  } catch(const std::exception& e){mlp.reset();model_error=e.what();}}
  else if(controller=="replay") {std::ifstream file(argv[12]);std::string version;int n=0;std::size_t ticks=0;
    if(!(file>>version>>n>>ticks) || version!="RELIC_REPLAY_V1" || n!=dof || ticks>10000)model_error="replay_version_or_dimension";
    else {replay.assign(ticks,Vec(dof));for(auto& row:replay)for(double& x:row)if(!(file>>x) || !std::isfinite(x))model_error="replay_nonfinite_or_truncated";
      std::string extra;if(file>>extra)model_error="replay_trailing_data";}}
  else if(controller!="conventional" && controller!="analytical" && controller!="gravity" && controller!="integral" && controller!="zero" && controller!="copy_previous")throw std::invalid_argument("unknown controller");
  Random r{seed},trajectory{family};PhysicsParameters p;
  if(category=="gain" || category=="combined")p.gain_scale=1-severity*r.uniform(0.45,0.65);
  if(category=="force" || category=="combined")p.force_scale=std::max(0.1,0.35-severity*r.uniform(0.15,0.30));
  if(category=="friction" || category=="combined"){p.damping_scale=1+severity*r.uniform(0.5,2);p.friction_Nm=severity*r.uniform(0.05,0.35);}
  if(category=="mass" || category=="combined")p.mass_scale=1+severity*r.uniform(0.25,0.7);
  if(category=="payload" || category=="combined")p.payload_kg=severity*r.uniform(0.03,0.14);
  if(category!="nominal" && category!="recovery" && category!="restored_nominal" && category!="gain" && category!="force" && category!="friction" && category!="mass" && category!="payload" && category!="combined")throw std::invalid_argument("unknown category");

  auto cfg=load_config(std::filesystem::path(argv[1])/("synthetic_"+std::to_string(dof)+"dof.cfg"));Simulation sim(cfg,p);Simulation nominal(cfg);
  Vec start{trajectory.uniform(-0.9,0.9),trajectory.uniform(-0.85,-0.3),trajectory.uniform(0.05,0.65)},goal{
    trajectory.uniform(-0.9,0.9),trajectory.uniform(-0.85,-0.3),trajectory.uniform(0.05,0.65)};
  if(dof==4){start.push_back(trajectory.uniform(-0.7,0.7));goal.push_back(trajectory.uniform(-0.7,0.7));}
  double duration=trajectory.uniform(0.8,1.8);const double vmax=trajectory.uniform(0.45,1.1),amax=trajectory.uniform(1.2,3);
  for(int j=0;j<dof;++j){const double delta=std::abs(goal[j]-start[j]);duration=std::max({duration,1.875*delta/vmax,std::sqrt(5.774*delta/amax)});}
  const double dt=sim.model().opt.timestep;const auto command_ticks=static_cast<std::size_t>(std::ceil(duration/dt));duration=command_ticks*dt;
  const std::size_t total_ticks=command_ticks+500;
  std::vector<Vec> qs(total_ticks+1),vs(total_ticks+1),as(total_ticks+1);
  for(std::size_t k=0;k<=total_ticks;++k){
    const double u=std::min(1.0,k*dt/duration),s=u*u*u*(10+u*(-15+6*u));
    const double v=30*u*u*(1-u)*(1-u)/duration,a=60*u*(1-u)*(1-2*u)/(duration*duration);
    for(int j=0;j<dof;++j){qs[k].push_back(start[j]+(goal[j]-start[j])*s);vs[k].push_back(k<=command_ticks?(goal[j]-start[j])*v:0);as[k].push_back(k<=command_ticks?(goal[j]-start[j])*a:0);}
  }
  std::string status="Success",reason="completed fixed settlement window";std::size_t completed=0,contacts=0;

  for(std::size_t k=0;k<=command_ticks;++k)if(!safe_segment(sim,k?qs[k-1]:start,qs[k],true)){status="InvalidPath";reason="sampled nominal admission failed";break;}
  sim.reset(goal);sim.step(3);sim.reset(start);
  const auto initial=sim.snapshot();auto last_recorded=initial;bool failed_step=false;const auto target=sim.evaluate_fk(goal);
  Vec integral(dof,0),joint_squared(dof,0);std::size_t correction_saturated=0,rate_saturated=0;bool withdrawing=false;std::string withdrawal_reason="none";
  Vec zero(dof,0),previous_error(dof,0),older_error(dof,0),previous_correction(dof,0);std::size_t fallbacks=0;double inference_s=0;std::string last_fallback="none";double squared=0,peak=0,command_error=0;std::size_t samples=0;
  Recorder recorder(out/"telemetry.csv",id);
  recorder.append(sim,0,"initial",start,zero,zero,initial,zero,zero,initial.actuator_torque_Nm,status,false);
  if(status=="Success")for(std::size_t k=1;k<=total_ticks;++k) {
    const auto pre=sim.snapshot();
    if(execution_health(sim.model(),sim.execution_data())!=MotionStatus::Success || has_disallowed_contact(inspect_contacts(sim.model(),sim.execution_data(),{}))) {status="SimulationError";reason="pre_command_observed_health";break;}
    Vec candidate(dof,0);Correction correction;correction.command=qs[k];correction.applied=zero;
    if(controller!="conventional") {
      const auto infer_start=std::chrono::steady_clock::now();
      if(controller=="analytical" || controller=="gravity" || controller=="integral") {
        nominal.evaluate_fk(qs[k]);
        for(int j=0;j<dof;++j){double value=lead*vs[k][j];
          if(controller!="analytical")value+=nominal.planning_data().qfrc_bias[j]/200.;
          if(controller=="integral")value+=kp_error*previous_error[j]+integral[j];
          candidate[j]=std::clamp(value,-bounds.magnitude[j],bounds.magnitude[j]);
          if(std::abs(value)>=bounds.magnitude[j])++correction_saturated;
          if(std::abs(candidate[j]-previous_correction[j])>bounds.rate[j])++rate_saturated;
          if(controller=="integral") {const double delta=ki*previous_error[j]*dt;
            if((std::abs(value)<bounds.magnitude[j] && std::abs(candidate[j]-previous_correction[j])<=bounds.rate[j]) || delta*(value-previous_correction[j])<0)
              integral[j]=std::clamp(integral[j]+delta,-integral_limit,integral_limit);}
        }
      }
      if(controller=="copy_previous")candidate=previous_correction;
      if(mlp)for(int j=0;j<(controller=="common_transfer"?3:dof);++j)candidate[j]=mlp->predict(features(cfg,j,qs[k][j],vs[k][j],as[k][j],pre.observed_rad[j],pre.velocity_rad_s[j],previous_error[j],older_error[j],previous_correction[j]),j);
      if(controller=="replay" && model_error=="none") {if(k>=replay.size())model_error="replay_tick_count";else candidate=replay[k];}
      inference_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-infer_start).count();
      if(inject_tick>=0 && k>=static_cast<std::size_t>(inject_tick))model_error="injected_invalid_output";
      correction=guarded(sim,qs[k],pre.command_rad,previous_correction,candidate,bounds,withdrawing?withdrawal_reason:model_error);
      if(correction.fallback && !withdrawing){withdrawing=true;withdrawal_reason=correction.reason;}
      if(correction.stopped){status="WithdrawalInfeasible";reason=correction.reason;break;}
    } else if(!safe_segment(sim,pre.command_rad,qs[k],true)){status="InvalidPath";reason="conventional_command_or_path";break;}
    if(correction.fallback){++fallbacks;last_fallback=correction.reason;}
    Vec disturbance(dof,0);if(category=="combined" && k>=command_ticks/2 && k<command_ticks/2+50)disturbance[0]=0.2*severity;sim.set_external_torques(disturbance);sim.set_position_commands(correction.command);const auto force=sim.snapshot().actuator_torque_Nm;
    try {sim.step();}catch(const std::exception&){status="NumericalWarning";reason="physics_step_exception_last_recorded_state_only";failed_step=true;break;}
    const auto now=sim.snapshot();
    auto health=execution_health(sim.model(),sim.execution_data());
    const auto findings=inspect_contacts(sim.model(),sim.execution_data(),{});contacts+=findings.size();
    if(health==MotionStatus::SimulationError || health==MotionStatus::NumericalWarning){status=motion_status_name(health);reason="nonfinite_or_warning_last_recorded_state_only";failed_step=true;break;}
    completed=k;
    if(health!=MotionStatus::Success){status=motion_status_name(health);reason="observed execution health";}
    else if(has_disallowed_contact(findings) || !safe_segment(sim,pre.observed_rad,now.observed_rad,false)){status="DisallowedContact";reason="observed sampled path or contact";}
    recorder.append(sim,k,k<=command_ticks?"command":"settling",qs[k],vs[k],as[k],pre,previous_error,correction.applied,force,status,correction.fallback,correction.reason);
    last_recorded=now;older_error=previous_error;previous_correction=correction.applied;
    for(int j=0;j<dof;++j){const double e=qs[k][j]-now.observed_rad[j];previous_error[j]=e;peak=std::max(peak,std::abs(e));if(k<=command_ticks){squared+=e*e;joint_squared[j]+=e*e;++samples;}}
    if(k==command_ticks)command_error=distance(now.tool,target);
    if(status!="Success")break;
  }
  Vec joint_rmse;for(double value:joint_squared)joint_rmse.push_back(command_ticks && completed>=command_ticks?std::sqrt(value/command_ticks):0);
  recorder.close();const auto final=last_recorded;const double final_error=distance(final.tool,target);
  double joint_peak=0,velocity_peak=0;for(int j=0;j<dof;++j){joint_peak=std::max(joint_peak,std::abs(goal[j]-final.observed_rad[j]));velocity_peak=std::max(velocity_peak,std::abs(final.velocity_rad_s[j]));}
  const bool success=status=="Success" && completed==total_ticks && joint_peak<=0.01 && final_error<=0.005 && velocity_peak<=0.01;
  if(status=="Success" && !success){status="NonConvergence";reason="fixed settlement tolerance not met";}
  std::ofstream meta(out/"native.json");meta<<std::setprecision(17);
  const auto& m=sim.model();Vec mass(m.body_mass,m.body_mass+m.nbody),inertia(m.body_inertia,m.body_inertia+3*m.nbody),gains,forces,damping(m.dof_damping,m.dof_damping+m.nv),friction(m.dof_frictionloss,m.dof_frictionloss+m.nv);
  Vec descriptors;for(int j=0;j<dof;++j){const auto f=features(cfg,j,0,0,0,0,0,0,0,0);descriptors.insert(descriptors.end(),f.begin()+8,f.end());}
  for(int j=0;j<dof;++j){gains.push_back(m.actuator_gainprm[mjNGAIN*j]);forces.push_back(m.actuator_forcerange[2*j+1]);}
  meta<<"{\"schema_version\":\"relic.episode.v1\",\"episode\":\""<<id<<"\",\"source_commit\":\""<<RELIC_SOURCE_COMMIT<<"\",\"mujoco_version\":\""<<mj_versionString()<<"\",\"embodiment\":\""<<cfg.id<<"\",\"dof\":"<<dof
    <<",\"seed\":"<<seed<<",\"family\":"<<family<<",\"partition\":\""<<partition<<"\",\"category\":\""<<category<<"\",\"severity\":"<<severity
    <<",\"joint_descriptors\":"<<array(descriptors)<<",\"initial_q_rad\":"<<array(start)<<",\"goal_q_rad\":"<<array(goal)<<",\"goal_world_m\":"<<array(target.position_m)
    <<",\"trajectory\":\"quintic_analytic_v1\",\"controller\":\""<<controller<<"\",\"learned_model\":\""<<(argc>=13?argv[12]:"none")<<"\",\"timestep_s\":"<<dt<<",\"command_ticks\":"<<command_ticks<<",\"planned_ticks\":"<<total_ticks<<",\"completed_ticks\":"<<completed
    <<",\"velocity_limit_rad_s\":"<<vmax<<",\"acceleration_limit_rad_s2\":"<<amax<<",\"physics\":{\"gain_scale\":"<<p.gain_scale<<",\"force_scale\":"<<p.force_scale<<",\"damping_scale\":"<<p.damping_scale<<",\"friction_Nm\":"<<p.friction_Nm<<",\"mass_scale\":"<<p.mass_scale<<",\"payload_kg\":"<<p.payload_kg
    <<",\"payload_location\":\"forearm_COM_sphere_radius_0.025m\",\"body_mass_kg\":"<<array(mass)<<",\"body_inertia_kg_m2\":"<<array(inertia)<<",\"actuator_kp\":"<<array(gains)<<",\"actuator_kv\":"<<20<<",\"actuator_force_upper_Nm\":"<<array(forces)<<",\"joint_damping_Nm_s_rad\":"<<array(damping)<<",\"joint_friction_Nm\":"<<array(friction)<<"}"
    <<",\"outcome\":\""<<status<<"\",\"termination_reason\":\""<<reason<<"\",\"failure_class\":\""<<(success?"none":status)<<"\",\"task_success\":"<<(success?"true":"false")
    <<",\"unrecorded_failed_step\":"<<(failed_step?"true":"false")<<",\"fallbacks\":"<<fallbacks<<",\"last_fallback_reason\":\""<<last_fallback<<"\",\"inference_s\":"<<inference_s<<",\"correction_magnitude_rad\":"<<array(bounds.magnitude)<<",\"correction_rate_rad_per_tick\":"<<array(bounds.rate)<<",\"disturbance\":{\"joint\":0,\"torque_Nm\":"<<(category=="combined"?0.2*severity:0)<<",\"start_tick\":"<<command_ticks/2<<",\"duration_ticks\":50},\"metrics\":{\"joint_rmse_rad\":"<<(samples?std::sqrt(squared/static_cast<double>(samples)):0)<<",\"joint_rmse_by_joint_rad\":"<<array(joint_rmse)<<",\"correction_magnitude_saturations\":"<<correction_saturated<<",\"correction_rate_saturations\":"<<rate_saturated<<",\"peak_error_rad\":"<<peak<<",\"command_end_cartesian_m\":"<<command_error<<",\"final_cartesian_m\":"<<final_error<<",\"final_joint_peak_rad\":"<<joint_peak<<",\"final_velocity_peak_rad_s\":"<<velocity_peak<<",\"contact_observations\":"<<contacts<<"},\"rows\":"<<recorder.rows
    <<",\"wall_s\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<<"}\n";
  if(!meta)throw std::runtime_error("metadata write failed");
  std::ofstream xml(out/"nominal.xml");xml<<make_mjcf(cfg);if(!xml)throw std::runtime_error("model write failed");return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
