#include "residual.hpp"
#include <iostream>
int main(int argc,char** argv){try {
  if(argc!=19)return 2;
  const auto model=relic::adaptive::MLP::load(argv[1]);
  relic::adaptive::Features f{};for(std::size_t i=0;i<f.size();++i)f[i]=std::stod(argv[3+i]);
  std::cout<<std::setprecision(17)<<model.predict(f,std::stoull(argv[2]))<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
