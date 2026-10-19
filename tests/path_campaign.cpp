#include "relic/feasibility.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using S = relic::PathStatus;
using Path = std::vector<std::vector<double>>;
using Limits = std::vector<relic::JointLimit>;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double big = std::numeric_limits<double>::max();
constexpr double tiny = std::numeric_limits<double>::denorm_min();
std::uint64_t checks = 0, calls = 0;
void require(bool ok, const std::string& message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
const char* name(S s) {
  constexpr const char* names[] = {"Valid", "EmptyPath", "InvalidConstraints",
    "WrongJointCount", "NonFiniteJoint", "JointLimitViolation", "ExcessiveJointStep"};
  return names[static_cast<unsigned>(s)];
}
struct Input { Path path; Limits limits; double step; };
S production(const Input& x) {
  ++calls;
  return relic::validate_joint_path(x.path, x.limits, x.step);
}

std::set<S> oracle(const Input& x) {
  const bool constraints = !x.limits.empty() && std::isfinite(x.step) && x.step > 0 &&
    std::all_of(x.limits.begin(), x.limits.end(), [](auto l) {
      return std::isfinite(l.lower) && std::isfinite(l.upper) && l.lower < l.upper;
    });
  if (!constraints) {
    if (x.path.empty()) return {S::InvalidConstraints, S::EmptyPath};
    return {S::InvalidConstraints};
  }
  if (x.path.empty()) return {S::EmptyPath};
  std::set<S> errors;
  for (const auto& row : x.path) {
    if (row.size() != x.limits.size()) errors.insert(S::WrongJointCount);
  }
  for (std::size_t j = 0; j < x.limits.size(); ++j) {
    for (const auto& row : x.path) {
      if (row.size() != x.limits.size()) continue;
      if (!std::isfinite(row[j])) errors.insert(S::NonFiniteJoint);
      else if (!(row[j] >= x.limits[j].lower && row[j] <= x.limits[j].upper))
        errors.insert(S::JointLimitViolation);
    }
    for (std::size_t w = 1; w < x.path.size(); ++w) {
      const auto& a = x.path[w-1];
      const auto& b = x.path[w];
      if (a.size() != x.limits.size() || b.size() != x.limits.size()) continue;
      if (!std::isfinite(a[j]) || !std::isfinite(b[j])) continue;
      volatile double rounded = b[j] - a[j];
      const double magnitude = rounded < 0 ? -rounded : rounded;
      if (!std::isfinite(magnitude) || magnitude > x.step)
        errors.insert(S::ExcessiveJointStep);
    }
  }
  return errors.empty() ? std::set<S>{S::Valid} : errors;
}
bool mismatch(const Input& x) { return !oracle(x).contains(production(x)); }
void dump(std::ostream& out, const Input& x) {
  out << std::hexfloat << "step=" << x.step << "\nlimits={";
  for (auto l : x.limits) out << '{' << l.lower << ',' << l.upper << "},";
  out << "}\npath={";
  for (const auto& row : x.path) {
    out << '{'; for (double v : row) out << v << ','; out << "},";
  }
  out << "}\nexpected={";
  for (S s : oracle(x)) out << name(s) << ',';
  out << "} actual=" << name(production(x)) << '\n';
}
Input minimize(Input x) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t w = 0; w < x.path.size(); ++w) {
      auto y = x; y.path.erase(y.path.begin() + static_cast<std::ptrdiff_t>(w));
      if (mismatch(y)) { x = std::move(y); changed = true; break; }
    }
    if (changed) continue;
    for (std::size_t j = 0; j < x.limits.size(); ++j) {
      auto y = x; y.limits.erase(y.limits.begin() + static_cast<std::ptrdiff_t>(j));
      for (auto& row : y.path) if (j < row.size())
        row.erase(row.begin() + static_cast<std::ptrdiff_t>(j));
      if (mismatch(y)) { x = std::move(y); changed = true; break; }
    }
  }
  for (std::size_t w = 0; w < x.path.size(); ++w)
    for (std::size_t j = 0; j < x.path[w].size(); ++j)
      for (double v : {0.0, 1.0, -1.0}) {
        auto y = x; y.path[w][j] = v;
        if (mismatch(y)) { x = std::move(y); break; }
      }
  for (double v : {1.0, tiny}) {
    auto y = x; y.step = v; if (mismatch(y)) { x = std::move(y); break; }
  }
  return x;
}
void expect(const std::string& label, const Input& x, S expected) {
  const S actual = production(x);
  require(actual == expected, label + ": expected " + name(expected) + " got " + name(actual));
  require(oracle(x).contains(expected), "oracle disagrees with targeted expectation: " + label);
}
void targeted() {
  std::uint64_t cases = 0;
  auto test = [&](const std::string& label, Path p, Limits l, double t, S s) {
    ++cases; expect(label, {std::move(p), std::move(l), t}, s);
  };
  const Limits one{{-1,1}}, two{{-1,1},{-1,1}};
  test("empty path", {}, one, 1, S::EmptyPath);
  test("empty limits", {{0}}, {}, 1, S::InvalidConstraints);
  test("both empty (observed precedence)", {}, {}, 1, S::InvalidConstraints);
  for (double t : {0.0,-0.0,-1.0,-tiny,nan,inf,-inf})
    test("bad threshold", {{0}}, one, t, S::InvalidConstraints);
  for (auto l : Limits{{nan,1},{-1,nan},{-inf,1},{-1,inf},{inf,inf},
                       {1,-1},{0,0},{-0.0,0.0},{big,-big}})
    test("bad bounds", {{0}}, {l}, 1, S::InvalidConstraints);
  test("bad constraint in final joint", {{0,0}}, {{-1,1},{nan,1}}, 1, S::InvalidConstraints);
  test("singleton", {{0}}, one, 1, S::Valid);
  test("identical", {{0},{0},{0}}, one, tiny, S::Valid);
  for (double v : {-1.0,1.0,std::nextafter(-1.0,0.0),std::nextafter(1.0,0.0),-0.0,0.0})
    test("inclusive/inside bound", {{v}}, one, 1, S::Valid);
  for (double v : {std::nextafter(-1.0,-inf),std::nextafter(1.0,inf)})
    test("one ULP outside", {{v}}, one, 1, S::JointLimitViolation);
  for (double v : {nan,inf,-inf})
    for (std::size_t w : {0u,1u,2u}) for (std::size_t j : {0u,1u}) {
      Path p(3,std::vector<double>(2,0)); p[w][j] = v;
      test("nonfinite position matrix", p, two, 1, S::NonFiniteJoint);
    }
  for (std::size_t w : {0u,1u,2u}) for (std::size_t len : {0u,1u,3u,10000u}) {
    Path p(3,std::vector<double>(2,0)); p[w].assign(len,0);
    test("malformed position matrix", p, two, 1, S::WrongJointCount);
  }
  for (std::size_t w : {0u,1u,2u}) for (std::size_t j : {0u,1u}) {
    Path p(3,std::vector<double>(2,0)); p[w][j] = 2;
    test("out-of-range position matrix", p, two, 1, S::JointLimitViolation);
  }
  for (std::size_t j : {0u,1u}) {
    Path p(3,std::vector<double>(2,0)); p[2][j] = 0.75;
    test("final edge/joint", p, two, 0.5, S::ExcessiveJointStep);
  }
  for (double sign : {-1.0,1.0}) {
    for (double t : {tiny, std::numeric_limits<double>::min(), 0.5, 1.0}) {
      for (double delta : {std::nextafter(t,0.0),t,std::nextafter(t,inf)})
        test("edge threshold ULP", {{0},{sign*delta}}, {{-2,2}}, t,
             delta > t ? S::ExcessiveJointStep : S::Valid);
    }
  }
  test("subnormal bounds", {{-tiny},{0},{tiny}}, {{-tiny,tiny}}, tiny, S::Valid);
  test("signed zero edge", {{-0.0},{0.0}}, one, tiny, S::Valid);
  test("largest threshold", {{0},{big}}, {{-big,big}}, big, S::Valid);
  test("large equal angles", {{big},{big}}, {{-big,big}}, tiny, S::Valid);
  test("small normal threshold", {{0},{tiny}}, one, std::numeric_limits<double>::min(), S::Valid);
  std::feclearexcept(FE_ALL_EXCEPT);
  test("positive subtraction overflow", {{-big},{big}}, {{-big,big}}, big, S::ExcessiveJointStep);
  const bool overflow = std::fetestexcept(FE_OVERFLOW) != 0;
  test("negative subtraction overflow", {{big},{-big}}, {{-big,big}}, big, S::ExcessiveJointStep);
  test("rounding-sensitive candidate contract", {{-std::ldexp(1.0,-54)},{1}}, {{-2,2}}, 1, S::Valid);
  test("endpoint jump", {{-1},{1}}, one, 1, S::ExcessiveJointStep);
  test("waypoints far exceed joints", Path(100000,std::vector<double>{0}), one, 1, S::Valid);
  test("joints far exceed waypoints", {std::vector<double>(100000,0)}, Limits(100000,{-1,1}), 1, S::Valid);
  test("huge count mismatch", {{0}}, Limits(100000,{-1,1}), 1, S::WrongJointCount);
  test("valid rectangular 3 by 2", Path(3,std::vector<double>(2,0)), two, 1, S::Valid);
  test("valid rectangular 2 by 3", Path(2,std::vector<double>(3,0)), Limits(3,{-1,1}), 1, S::Valid);
  const std::array<Input,4> mixed{{
    {{{0},{nan,0}}, one, 1}, {{{2},{nan}},one,1},
    {{{0,0},{0.75,nan}},two,0.5}, {{{0,0},{0,2},{}},two,0.5}}};
  std::size_t mixed_index = 0;
  for (const auto& x : mixed) {
    const auto s = production(x);
    std::cout << "mixed_input=" << mixed_index++ << " observed_status=" << name(s) << '\n';
    require(oracle(x).contains(s), "mixed diagnostic must identify a present violation");
    for (int i = 0; i < 100; ++i) require(production(x) == s, "deterministic mixed errors");
  }
  std::cout << "targeted_cases=" << cases << " mixed_inputs=" << mixed.size()
            << " overflow_flag_observed=" << overflow << '\n';
}

Input generate(std::mt19937_64& rng, std::size_t index) {
  auto pick = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
  const std::size_t joints = 1 + pick(8), rows = 1 + pick(12);
  Input x{Path(rows,std::vector<double>(joints)), Limits(joints,{-2,2}), 0.5};
  for (auto& row : x.path) for (auto& v : row)
    v = static_cast<double>(static_cast<int>(pick(9))-4) / 32;
  const auto w = pick(rows), j = pick(joints);
  switch (index % 16) {
    case 0: case 1: case 2: break;
    case 3: x.path.clear(); break;
    case 4: x.limits.clear(); break;
    case 5: { const double bad[] = {0,-0.0,-1,nan,inf,-inf}; x.step = bad[pick(6)]; break; }
    case 6: { const relic::JointLimit bad[] = {{nan,2},{-2,inf},{2,-2},{1,1},{-inf,2}};
              x.limits[j] = bad[pick(5)]; break; }
    case 7: x.path[w].resize(pick(2) ? joints+1 : joints-1); break;
    case 8: { const double bad[] = {nan,inf,-inf}; x.path[w][j] = bad[pick(3)]; break; }
    case 9: x.path[w][j] = pick(2) ? std::nextafter(2.0,inf) : std::nextafter(-2.0,-inf); break;
    case 10: if (rows == 1) x.path.push_back(x.path[0]);
             x.path[0][j] = -1; x.path[1][j] = 1; break;
    case 11: {
      const double t[] = {tiny, std::numeric_limits<double>::min(),0.5,1,big};
      x.step = t[pick(5)]; x.limits.assign(joints,{-big,big});
      x.path.assign(2,std::vector<double>(joints,0));
      const double v[] = {std::nextafter(x.step,0.0),x.step,std::nextafter(x.step,inf)};
      x.path[1][j] = v[pick(3)]; break;
    }
    case 12: x.limits.assign(joints,{-big,big}); x.step = big;
             x.path.assign(2,std::vector<double>(joints,-big)); x.path[1][j] = big; break;
    case 13: {
      for (auto& row : x.path) for (double& v : row) v = std::bit_cast<double>(rng());
      x.step = std::bit_cast<double>(rng());
      for (auto& l : x.limits) l = {std::bit_cast<double>(rng()),std::bit_cast<double>(rng())};
      break;
    }
    case 14: {
      x.path[w].resize(pick(joints+3));
      if (!x.path[0].empty()) x.path[0][0] = nan;
      else x.path.back().push_back(nan);
      break;
    }
    case 15: x.limits.assign(joints,{-big,big}); x.step = tiny;
             for (auto& row : x.path) for (double& v : row) v = pick(2) ? tiny : -0.0;
             break;
  }
  return x;
}
void fuzz(std::uint64_t seed, std::size_t count, std::size_t start) {
  std::mt19937_64 rng(seed);
  std::size_t valid = 0, invalid = 0, ambiguous = 0, properties = 0;
  std::array<std::size_t,7> statuses{};
  for (std::size_t i = 0; i < start + count; ++i) {
    Input x = generate(rng,i);
    if (i < start) continue;
    const auto allowed = oracle(x);
    const S actual = production(x);
    ++statuses[static_cast<unsigned>(actual)];
    if (allowed.size() > 1) ++ambiguous;
    if (!allowed.contains(actual)) {
      std::ofstream out("batch002-failure.txt");
      out << "seed=" << seed << " index=" << i << "\noriginal:\n";
      dump(out,x); out << "minimized:\n"; dump(out,minimize(x));
      throw std::runtime_error("oracle mismatch seed=" + std::to_string(seed) +
        " index=" + std::to_string(i) + "; see batch002-failure.txt");
    }
    require(allowed.contains(actual), "differential agreement");
    if (actual != S::Valid) { ++invalid; continue; }
    ++valid;
    Input y = x;
    y.path.insert(y.path.begin() + static_cast<std::ptrdiff_t>(y.path.size()/2),
                  x.path[x.path.size()/2]);
    require(production(y) == S::Valid,"duplicate adjacent waypoint"); ++properties;
    y = x; std::reverse(y.path.begin(),y.path.end());
    require(production(y) == S::Valid,"reverse valid path"); ++properties;
    y = x; std::reverse(y.limits.begin(),y.limits.end());
    for (auto& row : y.path) std::reverse(row.begin(),row.end());
    require(production(y) == S::Valid,"joint permutation"); ++properties;
    y = x; y.step = big;
    require(production(y) == S::Valid,"relax threshold"); ++properties;
    y = x; for (auto& l : y.limits) l = {-big,big};
    require(production(y) == S::Valid,"widen intervals"); ++properties;
  }
  std::cout << "seed=" << seed << " start=" << start << " generated=" << count
    << " valid=" << valid << " invalid=" << invalid << " ambiguous=" << ambiguous
    << " metamorphic=" << properties << " mismatches=0\n";
  for (std::size_t i = 0; i < statuses.size(); ++i)
    std::cout << name(static_cast<S>(i)) << '=' << statuses[i] << ' ';
  std::cout << '\n';
}

#ifndef RELIC_STANDALONE
std::vector<unsigned char> bytes(const void* data, std::size_t size) {
  const auto* p = static_cast<const unsigned char*>(data);
  return {p,p+size};
}
struct FrozenData {
  std::vector<unsigned char> object, buffer, arena;
  explicit FrozenData(const mjData& d) : object(bytes(&d,sizeof d)),
    buffer(bytes(d.buffer,d.nbuffer)), arena(bytes(d.arena,d.narena)) {}
  bool same(const mjData& d) const {
    return object == bytes(&d,sizeof d) && buffer == bytes(d.buffer,d.nbuffer) &&
           arena == bytes(d.arena,d.narena);
  }
};
void integration(const std::filesystem::path& examples) {
  std::size_t inputs = 0;
  for (int n : {3,4}) {
    relic::Simulation sim(relic::load_config(examples / ("synthetic_" + std::to_string(n) + "dof.cfg")));
    const auto limits = sim.joint_limits();
    const auto& m = sim.model();
    require(m.nq == n && m.nv == n && m.njnt == n && limits.size() == static_cast<std::size_t>(n),
            "actual MuJoCo joint counts");
    for (int j = 0; j < n; ++j) {
      require(limits[j].lower == m.jnt_range[2*j] && limits[j].upper == m.jnt_range[2*j+1],
              "limits are actual model ranges");
    }
    sim.set_position_commands(std::vector<double>(n,0.1)); sim.step(100);
    std::vector<double> collision(n,0); collision[2] = std::numbers::pi/2;
    sim.evaluate_fk(collision);
    require(sim.planning_data().ncon > 0,"fixture has actual contacts");
    const FrozenData execution(sim.execution_data()), planning(sim.planning_data());
    const auto model_object = bytes(&m,sizeof m), model_buffer = bytes(m.buffer,m.nbuffer);
    auto check = [&](Input x, S s) {
      ++inputs;
      const Input before = x;
      expect("MuJoCo input",x,s);
      require(x.path == before.path && x.step == before.step,"path/threshold unchanged");
      bool same = x.limits.size() == before.limits.size();
      for (std::size_t j = 0; j < x.limits.size(); ++j)
        same = same && x.limits[j].lower == before.limits[j].lower && x.limits[j].upper == before.limits[j].upper;
      require(same,"input limits unchanged");
      require(execution.same(sim.execution_data()),"execution object/buffer/arena unchanged");
      require(planning.same(sim.planning_data()),"planning object/buffer/arena unchanged");
      require(model_object == bytes(&m,sizeof m) && model_buffer == bytes(m.buffer,m.nbuffer),
              "model object/buffer unchanged");
    };
    const auto observed = sim.snapshot().observed_rad;
    check({{observed},limits,0.5},S::Valid);
    check({{observed,observed},limits,tiny},S::Valid);
    std::vector<double> lo,hi;
    for (auto l : limits) { lo.push_back(l.lower); hi.push_back(l.upper); }
    check({{lo},limits,0.5},S::Valid); check({{hi},limits,0.5},S::Valid);
    auto bad = observed; bad.back() = std::nextafter(limits.back().upper,inf);
    check({{bad},limits,0.5},S::JointLimitViolation);
    bad = observed; bad.back() = std::nextafter(limits.back().lower,-inf);
    check({{bad},limits,0.5},S::JointLimitViolation);
    bad = observed; bad.pop_back(); check({{observed,bad},limits,0.5},S::WrongJointCount);
    check({{lo,hi},limits,0.5},S::ExcessiveJointStep);
    check({{collision},limits,0.5},S::Valid);
    require(sim.planning_data().ncon > 0,"Valid does not certify collision safety");
    std::cout << "MuJoCo dof=" << n << " contacts=" << sim.planning_data().ncon
              << " state_preserved=1\n";
  }
  std::cout << "integration_inputs=" << inputs << '\n';
}
#endif

void performance() {
  constexpr std::size_t budget = 128 * 1024 * 1024;
  volatile int sink = 0;
  std::cout << "joints,waypoints,estimated_input_bytes,repeats,median_ms_per_call,waypoints_per_s,joints_per_s\n";
  for (std::size_t j : {3u,4u}) for (std::size_t w : {1u,100u,10000u,100000u,1000000u}) {
    const std::size_t estimate = w * (sizeof(std::vector<double>) + j*sizeof(double) + 32);
    require(estimate < budget,"performance memory budget");
    Input x{Path(w,std::vector<double>(j)),Limits(j,{-2,2}),0.01};
    for (std::size_t i = 0; i < w; ++i) for (double& v : x.path[i])
      v = static_cast<double>(i % 1000) / 100000;
    require(production(x) == S::Valid,"benchmark forces full traversal");
    const std::size_t repeats = std::clamp<std::size_t>(1000000/w,1,100000);
    std::array<double,7> samples{};
    for (double& sample : samples) {
      const auto begin = std::chrono::steady_clock::now();
      for (std::size_t k = 0; k < repeats; ++k)
        sink = static_cast<int>(relic::validate_joint_path(x.path,x.limits,x.step));
      const auto end = std::chrono::steady_clock::now();
      calls += repeats;
      sample = std::chrono::duration<double>(end-begin).count() / static_cast<double>(repeats);
    }
    require(sink == static_cast<int>(S::Valid),"benchmark result");
    std::sort(samples.begin(),samples.end()); const double seconds = samples[3];
    std::cout << j << ',' << w << ',' << estimate << ',' << repeats << ','
      << std::setprecision(9) << seconds*1000 << ',' << static_cast<double>(w)/seconds
      << ',' << static_cast<double>(w*j)/seconds << '\n';
  }
}
}

int main(int argc,char** argv) {
  try {
    require(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
            "binary64 required by campaign oracle");
    require(std::fegetround() == FE_TONEAREST,"campaign requires default round-to-nearest");
    if (argc < 2) throw std::runtime_error("usage: path_campaign targeted|fuzz [seed count start]|integration EXAMPLES|performance");
    const std::string mode = argv[1];
    if (mode == "targeted") targeted();
    else if (mode == "fuzz") {
      const auto seed = argc > 2 ? std::stoull(argv[2],nullptr,0) : 0x52454c494332ULL;
      const auto count = argc > 3 ? std::stoull(argv[3]) : 100000ULL;
      const auto start = argc > 4 ? std::stoull(argv[4]) : 0ULL;
      require(count <= 1000000 && start <= 1000000,"fuzz budget is at most one million cases/start");
      fuzz(seed,static_cast<std::size_t>(count),static_cast<std::size_t>(start));
    }
#ifndef RELIC_STANDALONE
    else if (mode == "integration" && argc == 3) integration(argv[2]);
#endif
    else if (mode == "performance") performance();
    else throw std::runtime_error("unknown mode or invalid arguments");
    std::cout << "PASS " << mode << " checks=" << checks << " validator_calls=" << calls << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << " checks=" << checks << " validator_calls=" << calls << '\n';
    return 1;
  }
}
