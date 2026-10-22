cmake_minimum_required(VERSION 3.24)
project(RELIC VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MUJOCO_ROOT "" CACHE PATH "MuJoCo C SDK root (tested with 3.16.0)")
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h HINTS "${MUJOCO_ROOT}/include" REQUIRED)
find_library(MUJOCO_LIBRARY NAMES mujoco HINTS "${MUJOCO_ROOT}/lib" "${MUJOCO_ROOT}/bin" REQUIRED)
file(STRINGS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" mujoco_version_define
  REGEX "^#define mjVERSION_HEADER ")
if(NOT mujoco_version_define MATCHES "mjVERSION_HEADER 3016000$")
  message(FATAL_ERROR "Batch 001 pins MuJoCo 3.16.0. Select its C SDK using MUJOCO_ROOT.")
endif()
add_library(MuJoCo::MuJoCo UNKNOWN IMPORTED)
set_target_properties(MuJoCo::MuJoCo PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
if(WIN32)
  find_file(MUJOCO_DLL mujoco.dll HINTS "${MUJOCO_ROOT}/bin" REQUIRED)
  set(relic_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}")
  if(CMAKE_CONFIGURATION_TYPES)
    string(APPEND relic_runtime_dir "/$<CONFIG>")
  endif()
  add_custom_target(relic_mujoco_runtime
    COMMAND ${CMAKE_COMMAND} -E make_directory "${relic_runtime_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${MUJOCO_DLL}" "${relic_runtime_dir}/mujoco.dll")
endif()

add_library(relic_sim
    src/config.cpp
    src/simulation.cpp
    src/trace.cpp
    src/analytical_ik.cpp
    src/contact_inspection.cpp
    src/motion.cpp
    src/motion_report.cpp
    src/feasibility.cpp)
target_include_directories(relic_sim PUBLIC include)
target_link_libraries(relic_sim PUBLIC MuJoCo::MuJoCo)
if(MSVC)
  target_compile_options(relic_sim PRIVATE /W4 /permissive-)
else()
  target_compile_options(relic_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

function(relic_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE relic_sim)
  if(WIN32)
    add_dependencies(${target} relic_mujoco_runtime)
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${relic_runtime_dir}")
  endif()
endfunction()

relic_executable(relic_cli src/main.cpp)
relic_executable(relic_cartesian_demo src/cartesian_demo.cpp)
include(CTest)
option(RELIC_RUCKIG_EXPERIMENT "Build isolated Ruckig qualification and MuJoCo benchmark" OFF)
if(RELIC_RUCKIG_EXPERIMENT)
  add_subdirectory(experiments/ruckig)
endif()
option(RELIC_DOUBLE_S_EXPERIMENT "Build isolated pinned InterpolateCpp numerical audit" OFF)
if(RELIC_DOUBLE_S_EXPERIMENT)
  add_subdirectory(experiments/double_s)
endif()
if(BUILD_TESTING)
  relic_executable(relic_motion_tests tests/motion_tests.cpp)
  relic_executable(relic_motion_benchmark tests/motion_benchmark.cpp)
  if(MSVC)
    target_compile_options(relic_motion_tests PRIVATE /W4 /permissive-)
    target_compile_options(relic_cartesian_demo PRIVATE /W4 /permissive-)
    target_compile_options(relic_motion_benchmark PRIVATE /W4 /permissive-)
  else()
    target_compile_options(relic_motion_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(relic_cartesian_demo PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(relic_motion_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  foreach(dof IN ITEMS 3 4)
    add_test(NAME batch003_cartesian_repro_${dof} COMMAND ${CMAKE_COMMAND}
      "-DDEMO=$<TARGET_FILE:relic_cartesian_demo>" "-DDOF=${dof}"
      "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/examples/synthetic_${dof}dof.cfg"
      "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/cartesian_${dof}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cartesian_repro.cmake")
  endforeach()
  foreach(case IN ITEMS trajectory adaptive selection contacts execution failures diagnostics)
    add_test(NAME batch003_motion_${case} COMMAND relic_motion_tests ${case} "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    set_tests_properties(batch003_motion_${case} PROPERTIES TIMEOUT 120)
  endforeach()
  relic_executable(relic_ik_tests tests/analytical_ik_tests.cpp)
  relic_executable(relic_ik_benchmark tests/ik_benchmark.cpp)
  foreach(target IN ITEMS relic_ik_tests relic_ik_benchmark)
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
  endforeach()
  foreach(case IN ITEMS branches boundaries wrist inputs geometry isolation generated)
    add_test(NAME batch003_ik_${case} COMMAND relic_ik_tests ${case} "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    set_tests_properties(batch003_ik_${case} PROPERTIES TIMEOUT 60)
  endforeach()
  relic_executable(relic_path_campaign tests/path_campaign.cpp)
  if(MSVC)
    target_compile_options(relic_path_campaign PRIVATE /W4 /permissive-)
  else()
    target_compile_options(relic_path_campaign PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  add_test(NAME batch002_targeted COMMAND relic_path_campaign targeted)
  add_test(NAME batch002_fuzz COMMAND relic_path_campaign fuzz 0x52454c494332 100000)
  add_test(NAME batch002_integration COMMAND relic_path_campaign integration "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  set_tests_properties(batch002_targeted batch002_fuzz batch002_integration PROPERTIES TIMEOUT 60)
  relic_executable(relic_tests tests/tests.cpp)
  foreach(case IN ITEMS models fk validation feasibility config isolation reset actuators lifetime trace)
    add_test(NAME ${case} COMMAND relic_tests ${case} "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  endforeach()
  foreach(dof IN ITEMS 3 4)
    add_test(NAME cli_repro_${dof} COMMAND ${CMAKE_COMMAND}
      "-DCLI=$<TARGET_FILE:relic_cli>"
      "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/examples/synthetic_${dof}dof.cfg"
      "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/test_trace_${dof}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_repro.cmake")
  endforeach()
endif()

option(RELIC_ADAPTIVE "Build opt-in offline adaptive research pipeline" OFF)
if(RELIC_ADAPTIVE)
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE RELIC_ADAPTIVE_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
  add_subdirectory(adaptive)
endif()
