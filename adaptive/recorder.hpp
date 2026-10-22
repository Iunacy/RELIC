#pragma once
#include "common.hpp"
#include <fstream>
#include <locale>

namespace relic::adaptive {



class Recorder {
  std::ofstream out;
  std::string episode;
 public:
  std::size_t rows=0;
  Recorder(const std::filesystem::path& file,const std::string& id):out(file),episode(id) {
    out.imbue(std::locale::classic());out<<std::setprecision(17);
    out<<"episode,tick,stage,phase,sim_time_s,reference_time_s,embodiment,joint,joint_name,q_des_rad,v_des_analytic_rad_s,a_des_analytic_rad_s2,q_pre_rad,v_pre_rad_s,previous_error_rad,q_actual_rad,v_actual_rad_s,error_rad,ctrl_rad,correction_rad,actuator_force_Nm,interval_start_force_Nm,qfrc_actuator_Nm,qfrc_applied_Nm,servo_unclipped_Nm,force_saturated,ctrl_lower_rad,ctrl_upper_rad,force_lower_Nm,force_upper_Nm,contact_count,safety_status,fallback,fallback_reason\n";
    if(!out)throw std::runtime_error("recorder open failed");
  }
  void append(const Simulation& sim,std::size_t tick,const char* phase,const Vec& q,const Vec& v,const Vec& a,
      const Snapshot& pre,const Vec& previous_error,const Vec& correction,const Vec& interval_force,const std::string& health,bool fallback,const std::string& fallback_reason="none") {
    const auto& m=sim.model();const auto& d=sim.execution_data();const auto snap=sim.snapshot();
    const auto n=q.size();
    if(n!=static_cast<std::size_t>(m.nu) || v.size()!=n || a.size()!=n || pre.observed_rad.size()!=n ||
       previous_error.size()!=n || correction.size()!=n || interval_force.size()!=n)throw std::runtime_error("recorder dimension mismatch");
    if(std::abs(d.time-static_cast<double>(tick)*m.opt.timestep)>1e-9)throw std::runtime_error("recorder time alignment");
    for(std::size_t j=0;j<n;++j) {
      const int aid=static_cast<int>(j),jid=m.actuator_trnid[2*aid],qa=m.jnt_qposadr[jid],va=m.jnt_dofadr[jid];
      if(m.jnt_type[jid]!=mjJNT_HINGE || m.actuator_trntype[aid]!=mjTRN_JOINT || m.actuator_gear[6*aid]!=1 ||
        m.actuator_gaintype[aid]!=mjGAIN_FIXED || m.actuator_biastype[aid]!=mjBIAS_AFFINE || snap.observed_rad[j]!=d.qpos[qa])
        throw std::runtime_error("unsupported recorder mapping");
      const double ctrl=std::clamp(d.ctrl[aid],m.actuator_ctrlrange[2*aid],m.actuator_ctrlrange[2*aid+1]);
      const double raw=m.actuator_gainprm[mjNGAIN*aid]*ctrl+m.actuator_biasprm[mjNBIAS*aid]+
        m.actuator_biasprm[mjNBIAS*aid+1]*d.actuator_length[aid]+m.actuator_biasprm[mjNBIAS*aid+2]*d.actuator_velocity[aid];
      if(std::abs(d.actuator_force[aid]-std::clamp(raw,m.actuator_forcerange[2*aid],m.actuator_forcerange[2*aid+1]))>1e-8 ||
         std::abs(d.qfrc_actuator[va]-d.actuator_force[aid])>1e-8)throw std::runtime_error("recorder servo equation");
      for(double x:{q[j],v[j],a[j],pre.observed_rad[j],pre.velocity_rad_s[j],previous_error[j],d.qpos[qa],d.qvel[va],
          d.ctrl[aid],correction[j],d.actuator_force[aid],interval_force[j],raw})if(!std::isfinite(x))throw std::runtime_error("nonfinite telemetry");
      const bool saturated=std::abs(d.actuator_force[aid]-m.actuator_forcerange[2*aid])<1e-8 ||
        std::abs(d.actuator_force[aid]-m.actuator_forcerange[2*aid+1])<1e-8;
      out<<episode<<','<<tick<<','<<(tick?"post":"initial")<<','<<phase<<','<<d.time<<','<<tick*m.opt.timestep<<','<<sim.config().id<<','<<j<<','
        <<mj_id2name(&m,mjOBJ_JOINT,jid)<<','<<q[j]<<','<<v[j]<<','<<a[j]<<','<<pre.observed_rad[j]<<','<<pre.velocity_rad_s[j]<<','
        <<previous_error[j]<<','<<d.qpos[qa]<<','<<d.qvel[va]<<','<<q[j]-d.qpos[qa]<<','<<d.ctrl[aid]<<','<<correction[j]<<','
        <<d.actuator_force[aid]<<','<<interval_force[j]<<','<<d.qfrc_actuator[va]<<','<<d.qfrc_applied[va]<<','<<raw<<','<<saturated<<','
        <<m.actuator_ctrlrange[2*aid]<<','<<m.actuator_ctrlrange[2*aid+1]<<','<<m.actuator_forcerange[2*aid]<<','<<m.actuator_forcerange[2*aid+1]<<','
        <<d.ncon<<','<<health<<','<<fallback<<','<<fallback_reason<<'\n';++rows;
    }
    if(!out)throw std::runtime_error("recorder write failed");
  }
  void close(){out.close();if(out.fail())throw std::runtime_error("recorder close failed");}
};
}
