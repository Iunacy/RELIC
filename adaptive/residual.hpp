#pragma once
#include "common.hpp"
#include <fstream>
#include <array>
namespace relic::adaptive {
constexpr std::size_t feature_count=16,hidden_count=32;
using Features=std::array<double,feature_count>;
struct Bounds {
  std::array<double,4> magnitude{},rate{};
  static Bounds load(const std::filesystem::path& path) {
    std::ifstream in(path);std::string version;int n=0;Bounds b;
    if(!(in>>version>>n) || version!="RELIC_BOUNDS_V1" || n!=4)throw std::runtime_error("bounds_version_or_dimension");
    for(int j=0;j<4;++j)if(!(in>>b.magnitude[j]>>b.rate[j]) || !std::isfinite(b.magnitude[j]) || !std::isfinite(b.rate[j]) ||
       b.magnitude[j]<=0 || b.magnitude[j]>.2 || b.rate[j]<=0 || b.rate[j]>.005)throw std::runtime_error("bounds_invalid");
    std::string extra;if(in>>extra)throw std::runtime_error("bounds_trailing_data");return b;
  }
};
inline Features features(const RobotConfig& c,std::size_t j,double desired,double velocity,double acceleration,
  double observed,double observed_v,double error,double older_error,double previous_correction) {
  const double local=j==1?c.link_lengths_m[0]:j==2?c.link_lengths_m[1]:j==3?c.tool_frame.position_m[0]:0;
  const double upstream=j==2?c.link_lengths_m[0]:j==3?c.link_lengths_m[0]+c.link_lengths_m[1]:0;
  return {desired,velocity,acceleration,observed,observed_v,error,older_error,previous_correction,
    j==3?1.:0.,j==1 || j==2?1.:0.,j==0?1.:0.,c.joint_limits_rad.at(j).lower,c.joint_limits_rad.at(j).upper,
    local,upstream,c.dof/4.};
}
struct MLP {
  std::array<double,feature_count> mean{},scale{};
  std::array<double,hidden_count*feature_count> w1{};std::array<double,hidden_count> b1{},b2{},w3{};
  std::array<double,hidden_count*hidden_count> w2{};
  double b3=0;int embodiment=0;Bounds bounds;
  static MLP load(const std::filesystem::path& path) {
    std::ifstream in(path);std::string version,feature_version;int f=0,h1=0,h2=0;MLP m;
    if(!(in>>version>>feature_version>>f>>h1>>h2>>m.embodiment) || version!="RELIC_MLP_V1" || feature_version!="relic.local.v1" ||
      f!=feature_count || h1!=hidden_count || h2!=hidden_count || (m.embodiment!=0 && m.embodiment!=3 && m.embodiment!=4))throw std::runtime_error("model_version_or_dimension");
    auto read=[&](auto& values){for(double& x:values)if(!(in>>x) || !std::isfinite(x) || std::abs(x)>1000)throw std::runtime_error("model_nonfinite_or_range");};
    read(m.bounds.magnitude);read(m.bounds.rate);read(m.mean);read(m.scale);read(m.w1);read(m.b1);read(m.w2);read(m.b2);read(m.w3);
    if(!(in>>m.b3) || !std::isfinite(m.b3) || std::abs(m.b3)>1000)throw std::runtime_error("model_bias");
    for(double x:m.scale)if(x<1e-8)throw std::runtime_error("model_scale");
    for(int j=0;j<4;++j)if(m.bounds.magnitude[j]<=0 || m.bounds.magnitude[j]>.2 || m.bounds.rate[j]<=0 || m.bounds.rate[j]>.005)throw std::runtime_error("model_bounds");
    std::string extra;if(in>>extra)throw std::runtime_error("model_trailing_data");return m;
  }
  double predict(const Features& f,std::size_t joint)const {
    std::array<double,feature_count> x{};std::array<double,hidden_count> a{},b{};
    for(std::size_t i=0;i<feature_count;++i){if(!std::isfinite(f[i]))return NAN;x[i]=(f[i]-mean[i])/scale[i];}
    for(std::size_t i=0;i<hidden_count;++i){double s=b1[i];for(std::size_t j=0;j<feature_count;++j)s+=w1[i*feature_count+j]*x[j];a[i]=std::tanh(s);}
    for(std::size_t i=0;i<hidden_count;++i){double s=b2[i];for(std::size_t j=0;j<hidden_count;++j)s+=w2[i*hidden_count+j]*a[j];b[i]=std::tanh(s);}
    double y=b3;for(std::size_t i=0;i<hidden_count;++i)y+=w3[i]*b[i];return std::tanh(y)*bounds.magnitude.at(joint);
  }
};
struct Correction {
  Vec command,applied;
  bool fallback=false;
  std::string reason="none";
};


inline Correction guarded(Simulation& sim,const Vec& desired,const Vec& previous_command,const Vec& previous_correction,
  const Vec& candidate,const Bounds& bounds,const std::string& invalid="none") {
  Correction result;result.command=desired;result.applied.assign(desired.size(),0);
  std::string why=invalid;
  if(candidate.size()!=desired.size())why="correction_dimension";
  if(why=="none")for(std::size_t j=0;j<candidate.size();++j)
    if(!std::isfinite(candidate[j]) || std::abs(candidate[j])>bounds.magnitude[j]+1e-12){why="correction_nonfinite_or_magnitude";break;}
  if(why=="none") {
    for(std::size_t j=0;j<desired.size();++j){result.applied[j]=std::clamp(candidate[j],previous_correction[j]-bounds.rate[j],previous_correction[j]+bounds.rate[j]);result.command[j]+=result.applied[j];}
    if(!safe_segment(sim,previous_command,result.command,true))why="corrected_command_or_path";
  }
  if(why!="none") {
    result.fallback=true;result.reason=why;result.command=desired;std::fill(result.applied.begin(),result.applied.end(),0);
    for(std::size_t j=0;j<desired.size();++j)if(std::abs(previous_correction[j])>bounds.rate[j]+1e-12)
      throw std::runtime_error("fallback_cannot_meet_correction_rate");
    if(!safe_segment(sim,previous_command,desired,true))throw std::runtime_error("fallback_command_or_path_unsafe");
  }
  return result;
}
}
