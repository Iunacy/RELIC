#include "qualification_api.hpp"
int main(int argc,char** argv) { return argc==2 ? relic_experiment::qualify(argv[1]) : 2; }
