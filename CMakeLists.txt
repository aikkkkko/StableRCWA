cmake_minimum_required(VERSION 3.16)
project(StableRCWA LANGUAGES CXX)

# 仅头文件库：include/ 加进包含路径即可使用，不需要先编译任何东西。
add_library(stablercwa INTERFACE)
add_library(StableRCWA::stablercwa ALIAS stablercwa)   # 用 add_subdirectory 引入时可写成这个名字
target_include_directories(stablercwa INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(stablercwa INTERFACE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(stablercwa INTERFACE Threads::Threads)
if(MSVC)
  target_compile_options(stablercwa INTERFACE /utf-8)   # 源文件是 UTF-8，含中文注释和提示文字
endif()

option(STABLERCWA_BUILD_EXAMPLES "编译 examples/ 下的示例" ON)
option(STABLERCWA_BUILD_TESTS "编译 tests/ 下的测试" ON)

if(STABLERCWA_BUILD_EXAMPLES)
  foreach(name eig_minimal rcwa_demo conical_demo rcwa_json rcwa_fields bench_accel)
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE stablercwa)
    target_include_directories(${name} PRIVATE examples)
  endforeach()
endif()

if(STABLERCWA_BUILD_TESTS)
  enable_testing()
  file(GLOB test_sources CONFIGURE_DEPENDS tests/test_*.cpp)
  foreach(src ${test_sources})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE stablercwa)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
  # test_eig_qr_parallel 逐位比较新旧两份 eig_qr：必须关掉乘加融合（GCC 在带 FMA 的目标上默认融合，-std=c++17 也一样，
  # 那时同一段源码的两份副本会融合得不一样，与算法对错无关）。MSVC 默认 /fp:precise 不融合，不要加 /fp:contract。
  if(TARGET test_eig_qr_parallel)
    target_compile_options(test_eig_qr_parallel PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-ffp-contract=off>)
    target_compile_definitions(test_eig_qr_parallel PRIVATE EIGQR_TEST_FP_CONTRACT_OFF)
    set_tests_properties(test_eig_qr_parallel PROPERTIES TIMEOUT 1200)
  endif()
  # rcwa_json.hpp 允许包含它之前把 STABLERCWA_RCWA_JSON_CONICAL 或 STABLERCWA_RCWA_JSON_2D 定义为 0，不编入圆锥入射或二维。
  # 场协议程序 examples/rcwa_fields.cpp（连带编入 rcwa_json.hpp 的请求解析）在这两个开关下各编一份：被关掉的模式得到
  # 说明原因的错误，平面入射照常求解、写出文件。请求和平面那一项用的小脚本在配置时写进构建目录；脚本每次先删掉上一次的输出
  # 目录（程序不往已有的目录里写），再要求退出码为 0、应答 ok、fields.npy 写出来了。
  set(optout_dir "${CMAKE_CURRENT_BINARY_DIR}/json_optout")
  file(WRITE "${optout_dir}/conical.json" [=[
{"solve": {"mode": "conical", "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
           "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]}]},
           "M": 5, "wavelength": 1.053, "theta_deg": 30.0, "phi_deg": 20.0, "input": {"psi_deg": 45.0}},
 "sampling": {"z": [-0.1, 0.2, 0.6], "nx": 16, "y": 0.0}, "output_dir": "conical_out", "request_id": "optout-conical"}
]=])
  file(WRITE "${optout_dir}/periodic2d.json" [=[
{"solve": {"mode": "periodic2d", "stack": {"nI": 1.0, "nII": 1.46, "Lx": 0.8, "Ly": 0.8,
           "layers": [{"d": 0.3, "y_edges": [0.0, 0.25, 0.55, 0.8],
                       "x_profiles": [{"edges": [0.0, 0.8], "n_values": [1.0]},
                                      {"edges": [0.0, 0.25, 0.55, 0.8], "n_values": [1.0, 1.8, 1.0]},
                                      {"edges": [0.0, 0.8], "n_values": [1.0]}]}]},
           "M": 2, "My": 2, "points": [{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0}],
           "input": {"psi_deg": 90.0, "delta_deg": 0.0}},
 "sampling": {"z": [0.0, 0.1], "plane": "xz", "nt": 8, "at": 0.4}, "output_dir": "periodic2d_out", "request_id": "optout-periodic2d"}
]=])
  set(optout_planar [=[
{"solve": {"mode": "planar", "polarization": "TE", "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
           "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]}, {"d": 0.115, "n": [1.96, 0.0]}]},
           "M": 5, "wavelength": 1.053, "theta_deg": 30.0},
 "sampling": {"z": [-0.1, 0.2, 0.6], "nx": 16, "y": 0.0}, "output_dir": "OUTPUT_DIR", "request_id": "optout-planar"}
]=])
  file(WRITE "${optout_dir}/run_planar.cmake" [=[
file(REMOVE_RECURSE "${DIR}/${OUT}")
execute_process(COMMAND "${EXE}" "${REQUEST}" WORKING_DIRECTORY "${DIR}"
                RESULT_VARIABLE code OUTPUT_VARIABLE reply ERROR_VARIABLE messages)
if(NOT code EQUAL 0 OR NOT reply MATCHES "^{\"ok\": true, \"version\": 1, \"mode\": \"planar\"" OR NOT EXISTS "${DIR}/${OUT}/fields.npy")
  message(FATAL_ERROR "exit code ${code}\n${reply}\n${messages}")
endif()
]=])
  foreach(variant no_conical no_2d)
    set(target rcwa_fields_${variant})
    add_executable(${target} examples/rcwa_fields.cpp)
    target_link_libraries(${target} PRIVATE stablercwa)
    target_include_directories(${target} PRIVATE examples)
    string(REPLACE "OUTPUT_DIR" "planar_${variant}_out" request "${optout_planar}")
    file(WRITE "${optout_dir}/planar_${variant}.json" "${request}")
    add_test(NAME ${target}_planar
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${target}> -DDIR=${optout_dir} -DREQUEST=planar_${variant}.json
                     -DOUT=planar_${variant}_out -P ${optout_dir}/run_planar.cmake)
  endforeach()
  target_compile_definitions(rcwa_fields_no_conical PRIVATE STABLERCWA_RCWA_JSON_CONICAL=0)
  target_compile_definitions(rcwa_fields_no_2d PRIVATE STABLERCWA_RCWA_JSON_2D=0)
  add_test(NAME rcwa_fields_no_conical_rejects_conical COMMAND rcwa_fields_no_conical conical.json WORKING_DIRECTORY "${optout_dir}")
  add_test(NAME rcwa_fields_no_2d_rejects_periodic2d COMMAND rcwa_fields_no_2d periodic2d.json WORKING_DIRECTORY "${optout_dir}")
  set_tests_properties(rcwa_fields_no_conical_rejects_conical PROPERTIES
                       PASS_REGULAR_EXPRESSION "\"ok\": false, \"error\": \"[^\"]*rcwa1d_conical\\.hpp[^\"]*\", \"where\": \"mode\"")
  set_tests_properties(rcwa_fields_no_2d_rejects_periodic2d PROPERTIES
                       PASS_REGULAR_EXPRESSION "\"ok\": false, \"error\": \"[^\"]*rcwa2d\\.hpp[^\"]*\", \"where\": \"mode\"")
endif()

# standalone/ 里 oneMKL 版本征分解（MatrixEigenValueAndEigenVector_MKL.cpp）的测试。要有 oneMKL 2025.0.1 或更新，
# 所以默认不编（CI 里也不编）。Windows + MSVC：先运行 standalone/install_mkl.bat 下载运行库，然后
#   cmake -S . -B build -DSTABLERCWA_BUILD_STANDALONE_MKL=ON
#   cmake --build build --config Release --target test_standalone_mkl
# 链接的库：Windows 上默认是 standalone/mkl/lib/mkl_rt.lib（对应 oneMKL 2026 及以后的 mkl_rt.3.dll）；其他系统上用
# find_library 找 mkl_rt。别的 MKL 或找不到时用 -DSTABLERCWA_MKL_RT_LIB=<导入库或共享库的路径> 指定。Windows 上有
# standalone/mkl/bin 时，编完会把里面的 DLL 拷到测试程序旁边；没有时运行库要在 PATH 里。
option(STABLERCWA_BUILD_STANDALONE_MKL "编译 standalone/ 里 oneMKL 版本征分解的测试（需要 oneMKL 2025.0.1 或更新）" OFF)
if(STABLERCWA_BUILD_STANDALONE_MKL)
  if(WIN32)
    set(STABLERCWA_MKL_RT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/standalone/mkl/lib/mkl_rt.lib" CACHE FILEPATH
        "oneMKL 单动态库 mkl_rt 的导入库（或共享库）")
  else()   # standalone/mkl/lib/mkl_rt.lib 是 Windows 的导入库，这里用不上
    find_library(STABLERCWA_MKL_RT_LIB NAMES mkl_rt DOC "oneMKL 单动态库 mkl_rt（libmkl_rt.so / libmkl_rt.dylib）")
  endif()
  if(NOT STABLERCWA_MKL_RT_LIB OR NOT EXISTS "${STABLERCWA_MKL_RT_LIB}")
    message(FATAL_ERROR "找不到 oneMKL 的 mkl_rt（STABLERCWA_MKL_RT_LIB = ${STABLERCWA_MKL_RT_LIB}）；"
                        "用 -DSTABLERCWA_MKL_RT_LIB=<导入库或共享库的路径> 指定")
  endif()
  enable_testing()
  add_executable(test_standalone_mkl standalone/tests/test_standalone_mkl.cpp)
  target_link_libraries(test_standalone_mkl PRIVATE stablercwa "${STABLERCWA_MKL_RT_LIB}")
  add_test(NAME test_standalone_mkl COMMAND test_standalone_mkl)
  if(WIN32 AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/standalone/mkl/bin")   # 运行库也可以在 PATH 里
    add_custom_command(TARGET test_standalone_mkl POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/standalone/mkl/bin"
                               "$<TARGET_FILE_DIR:test_standalone_mkl>")
  endif()
endif()

# oneMKL 后端（include/stablercwa/mkl/mkl_backend.hpp：本征分解、稠密乘法、线性求解交给 MKL，经三个钩子接进 RCWA 求解器）
# 的测试、测速程序与带 MKL 的 JSON 命令行程序。要有 oneMKL 2025.0.1 或更新，所以默认不编（CI 里也不编）；库本身不受这个选项影响。
# 目标 stablercwa_mkl 链接 mkl_rt 并定义 STABLERCWA_WITH_MKL=1，于是 rcwa_json.hpp 接受 "backend": "mkl"；
# 自己的程序 target_link_libraries(... stablercwa_mkl) 即可。同一个程序里包含 rcwa_json.hpp 的源文件要么都链接 stablercwa_mkl，
# 要么都不链接（这个宏改变内联函数的定义）。
# Windows + MSVC：先运行 standalone/install_mkl.bat 下载运行库，然后
#   cmake -S . -B build -DSTABLERCWA_WITH_MKL=ON
#   cmake --build build --config Release --target test_rcwa_mkl bench_mkl_rcwa rcwa_json_mkl
# 链接的库：Windows 上默认是 standalone/mkl/lib/mkl_rt.lib（对应 oneMKL 2026 及以后的 mkl_rt.3.dll）；其他系统上用
# find_library 找 mkl_rt（例如 oneAPI 或 pip 装的 libmkl_rt.so），找不到时用 -DSTABLERCWA_MKL_RT_LIB=<路径> 指定。
# 只在 Windows + MSVC 上测过。Windows 上有 standalone/mkl/bin 时，编完把里面的 DLL 拷到程序旁边（内容没变的不再拷）。
option(STABLERCWA_WITH_MKL "编译 oneMKL 后端的测试与测速程序（需要 oneMKL 2025.0.1 或更新）" OFF)
if(STABLERCWA_WITH_MKL)
  if(WIN32)
    set(STABLERCWA_MKL_RT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/standalone/mkl/lib/mkl_rt.lib" CACHE FILEPATH
        "oneMKL 单动态库 mkl_rt 的导入库（或共享库）")
  else()   # standalone/mkl/lib/mkl_rt.lib 是 Windows 的导入库，这里用不上
    find_library(STABLERCWA_MKL_RT_LIB NAMES mkl_rt DOC "oneMKL 单动态库 mkl_rt（libmkl_rt.so / libmkl_rt.dylib）")
  endif()
  if(NOT STABLERCWA_MKL_RT_LIB OR NOT EXISTS "${STABLERCWA_MKL_RT_LIB}")
    message(FATAL_ERROR "找不到 oneMKL 的 mkl_rt（STABLERCWA_MKL_RT_LIB = ${STABLERCWA_MKL_RT_LIB}）；"
                        "用 -DSTABLERCWA_MKL_RT_LIB=<导入库或共享库的路径> 指定")
  endif()
  add_library(stablercwa_mkl INTERFACE)
  target_link_libraries(stablercwa_mkl INTERFACE stablercwa "${STABLERCWA_MKL_RT_LIB}")
  target_compile_definitions(stablercwa_mkl INTERFACE STABLERCWA_WITH_MKL=1)
  enable_testing()
  add_executable(test_rcwa_mkl tests/mkl/test_rcwa_mkl.cpp)
  target_link_libraries(test_rcwa_mkl PRIVATE stablercwa_mkl)
  target_include_directories(test_rcwa_mkl PRIVATE examples tests)
  add_test(NAME test_rcwa_mkl COMMAND test_rcwa_mkl)
  set(mkl_targets test_rcwa_mkl)
  if(STABLERCWA_BUILD_EXAMPLES)
    add_executable(bench_mkl_rcwa examples/bench_mkl_rcwa.cpp)
    target_link_libraries(bench_mkl_rcwa PRIVATE stablercwa_mkl)
    target_include_directories(bench_mkl_rcwa PRIVATE examples tests)
    # 与 rcwa_json 同一份源文件，只是多了 "backend": "mkl"；程序名不同，默认的 rcwa_json 照旧不依赖 MKL。
    add_executable(rcwa_json_mkl examples/rcwa_json.cpp)
    target_link_libraries(rcwa_json_mkl PRIVATE stablercwa_mkl)
    target_include_directories(rcwa_json_mkl PRIVATE examples)
    list(APPEND mkl_targets bench_mkl_rcwa rcwa_json_mkl)
  endif()
  file(GLOB mkl_dlls "${CMAKE_CURRENT_SOURCE_DIR}/standalone/mkl/bin/*.dll")   # 配置时的内容；运行库也可以在 PATH 里
  if(WIN32 AND mkl_dlls)
    foreach(target ${mkl_targets})
      add_custom_command(TARGET ${target} POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy_if_different ${mkl_dlls} "$<TARGET_FILE_DIR:${target}>")
    endforeach()
  endif()
endif()
