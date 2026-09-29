cmake_minimum_required(VERSION 3.24)
project(cuteafd_native LANGUAGES C CXX)

# Single-config serving builds should optimize the native host path too.
# Respect an explicitly selected Debug/RelWithDebInfo or multi-config generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Native build configuration" FORCE)
endif()

option(CUTEAFD_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT "Build full RTX backbone expert kernels alongside dSpark" OFF)
option(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT "Build dual RTX TP2 encoder expert kernels" OFF)
option(CUTEAFD_V41_TP2_COMPACT_EXPERIMENT "Use experimental native compact TP2 experts at capacities 1/16" OFF)
option(CUTEAFD_V41_SPARK_COMPACT_EXPERIMENT "Use experimental compact Spark experts through two live rows" OFF)
# Opt-in replicated-group Spark TP2/TP3 expert shards. Empty (the default)
# compiles/links no new object or symbol, so every release default behaves
# exactly as the historical Spark TP4 shard. Set to a semicolon list of
# `tp2`/`tp3` on the SM121 expert build to pre-compile those AOT capacity
# variants ahead of runtime. (The exporter always adds an additive
# `spark_tp_degree` manifest field, so byte identity of exported JSON is not
# claimed either way.)
set(CUTEAFD_V41_SPARK_TP_ROLES "" CACHE STRING
  "Extra replicated-group Spark expert TP roles to build (tp2;tp3); empty builds only TP4")
set(CUTEAFD_EXPERT_FAMILIES "" CACHE STRING
  "Extra SM121 routed-expert kernel families as FAMILY:ROLE (e.g. dsv4f:spark)")
option(CUTEAFD_V41_VOCAB_ROW_EXPERIMENT "Use experimental single-row FP32 vocabulary projection" OFF)
option(CUTEAFD_ENABLE_V41_EXL3_AOT "Build loadable mixed EXL3 expert modules" OFF)
option(CUTEAFD_ENABLE_V41_NVFP4_AOT "Build W4A4 ModelOpt NVFP4 expert AOT kernels" OFF)
option(CUTEAFD_ENABLE_V41_EXPERT_AOT "Build native V4.1 expert AOT kernels" OFF)
option(CUTEAFD_ENABLE_V41_NARROW_AOT "Use candidate V4.1 narrow projection scheduling" OFF)
option(CUTEAFD_ENABLE_V41_HC_LAGGED_AOT "Build candidate native lagged mHC begin" OFF)
option(CUTEAFD_ENABLE_V41_FP8_AOT "Build native V4.1 coordinator FP8 AOT kernels" OFF)
option(CUTEAFD_ENABLE_DSV4_AOT "Build the DeepSeek V4 coordinator programs" OFF)
option(CUTEAFD_ENABLE_GLM_AOT "Build the GLM 5.x coordinator programs (same table as DeepSeek V4)" OFF)
option(CUTEAFD_ENABLE_MIMO_AOT "Build the MiMo V2 coordinator programs (same table as DeepSeek V4)" OFF)
set(CUTEAFD_MIMO_GEOMETRIES "mimo" CACHE STRING
  "MiMo program geometries with CUTEAFD_ENABLE_MIMO_AOT: mimo (V2 Flash), mimop (V2.6 Pro)")
option(CUTEAFD_ENABLE_GLMF_AOT "Build the GLM 5.3 Flash coordinator programs (same table as DeepSeek V4)" OFF)
option(CUTEAFD_ENABLE_QWEN4_AOT "Build the Qwen 3.8 Flash Next coordinator programs (same table as DeepSeek V4)" OFF)
set(CUTEAFD_DSV4_GEOMETRY "flash,pro" CACHE STRING "DeepSeek V4 program geometries, comma-separated: flash, pro")
set(CUTEAFD_DSV4_DECODE_ROWS "64" CACHE STRING "DeepSeek V4 decode program capacity (below 96 keeps mHC on its decode route)")
set(CUTEAFD_DSV4_PREFILL_ROWS "4096" CACHE STRING "DeepSeek V4 prefill program capacity")
set(CUTEAFD_DSV4_MAX_CONTEXT "131072" CACHE STRING "DeepSeek V4 cache extent compiled into programs")
if((CUTEAFD_ENABLE_V41_HC_LAGGED_AOT OR CUTEAFD_ENABLE_V41_NARROW_AOT) AND NOT CUTEAFD_ENABLE_V41_FP8_AOT)
  message(FATAL_ERROR "Candidate mHC/narrow AOT requires V4.1 FP8 AOT")
endif()
option(CUTEAFD_ENABLE_V41_ATTENTION_AOT "Build candidate direct V4.1 attention AOT kernels" OFF)
option(CUTEAFD_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(CUTEAFD_ENABLE_SPARKINFER_AOT "Build Spark-side SparkInfer AOT kernels" OFF)
option(CUTEAFD_ENABLE_DS4_FLASH_AOT
  "Build DeepSeek-V4-Flash Spark expert-TP4 SparkInfer AOT kernels" OFF)
option(CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT
  "Build coordinator SparkInfer AOT kernels" OFF)
option(CUTEAFD_ENABLE_B12X_AOT
  "Deprecated alias for CUTEAFD_ENABLE_SPARKINFER_AOT" OFF)
option(CUTEAFD_ENABLE_B12X_COORDINATOR_AOT
  "Deprecated alias for CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT" OFF)
option(CUTEAFD_ENABLE_W8A16_AOT "Build coordinator row-major W8A16 Triton AOT kernels" OFF)
option(CUTEAFD_ENABLE_FLASHINFER_PACKED_MLA
  "Build the exact grouped-split SM120 packed-FP8 MLA kernel" OFF)
option(CUTEAFD_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
option(CUTEAFD_ENABLE_XGRAMMAR "Build constrained-decoding grammar support" ON)
set(CUTEAFD_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(CUTEAFD_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer"
  CACHE PATH "Pinned SparkInfer source tree used by every SparkInfer-backed AOT export")
set(CUTEAFD_SPARKINFER_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer.lock.json"
  CACHE FILEPATH "Provenance lock for CUTEAFD_SPARKINFER_SOURCE_DIR")
set(CUTEAFD_XGRAMMAR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar"
  CACHE PATH "Pinned XGrammar source tree")
set(CUTEAFD_XGRAMMAR_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar.lock.json"
  CACHE FILEPATH "Provenance lock for CUTEAFD_XGRAMMAR_SOURCE_DIR")

if(CUTEAFD_ENABLE_B12X_AOT)
  message(DEPRECATION
    "CUTEAFD_ENABLE_B12X_AOT is deprecated; use CUTEAFD_ENABLE_SPARKINFER_AOT")
  set(CUTEAFD_ENABLE_SPARKINFER_AOT ON)
endif()
if(CUTEAFD_ENABLE_B12X_COORDINATOR_AOT)
  message(DEPRECATION
    "CUTEAFD_ENABLE_B12X_COORDINATOR_AOT is deprecated; "
    "use CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT")
  set(CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUTEAFD_NATIVE_SOURCES src/cuteafd_native.cc)

if(CUTEAFD_ENABLE_XGRAMMAR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/include/xgrammar/compiler.h" OR
     NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" OR
     NOT EXISTS "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson/picojson.h")
    message(FATAL_ERROR
      "Pinned XGrammar source is incomplete under ${CUTEAFD_XGRAMMAR_SOURCE_DIR}; "
      "initialize third_party/xgrammar and its nested submodules")
  endif()
  set(CUTEAFD_XGRAMMAR_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-xgrammar-source.py")
  if(NOT EXISTS "${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "XGrammar source verifier was not found: ${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CUTEAFD_XGRAMMAR_VERIFY_SCRIPT}"
      --source "${CUTEAFD_XGRAMMAR_SOURCE_DIR}"
      --lock "${CUTEAFD_XGRAMMAR_LOCK_FILE}"
    RESULT_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_RESULT
    OUTPUT_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_OUTPUT
    ERROR_VARIABLE CUTEAFD_XGRAMMAR_VERIFY_ERROR
  )
  if(NOT CUTEAFD_XGRAMMAR_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "XGrammar source verification failed:\n${CUTEAFD_XGRAMMAR_VERIFY_ERROR}")
  endif()
  string(STRIP "${CUTEAFD_XGRAMMAR_VERIFY_OUTPUT}" CUTEAFD_XGRAMMAR_VERIFY_OUTPUT)
  message(STATUS "${CUTEAFD_XGRAMMAR_VERIFY_OUTPUT}")
  file(GLOB_RECURSE CUTEAFD_XGRAMMAR_SOURCES
    CONFIGURE_DEPENDS
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
  )
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
  # Extend the pinned schema factory in a generated translation unit. The
  # anonymous upstream parser is reused without changing the vendored tree.
  file(READ "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/json_schema_converter.cc" CUTEAFD_SCHEMA_SOURCE)
  set(CUTEAFD_SCHEMA_FACTORY "switch (json_format) {\n    case JSONFormat::kJSON:")
  string(FIND "${CUTEAFD_SCHEMA_SOURCE}" "${CUTEAFD_SCHEMA_FACTORY}" CUTEAFD_SCHEMA_FACTORY_AT)
  if(CUTEAFD_SCHEMA_FACTORY_AT EQUAL -1)
    message(FATAL_ERROR "Pinned XGrammar schema factory no longer matches the V4.1 extension")
  endif()
  string(REPLACE "#include \"json_schema_converter.h\""
    "#include \"json_schema_converter.h\"\n#include \"xgrammar_v41_tools.h\""
    CUTEAFD_SCHEMA_SOURCE "${CUTEAFD_SCHEMA_SOURCE}")
  string(REPLACE "${CUTEAFD_SCHEMA_FACTORY}"
    "switch (json_format) {\n    case static_cast<JSONFormat>(5): {\n      V41ToolCallingConverter converter(ref_resolver);\n      return converter.Convert(spec);\n    }\n    case JSONFormat::kJSON:"
    CUTEAFD_SCHEMA_SOURCE "${CUTEAFD_SCHEMA_SOURCE}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc" "${CUTEAFD_SCHEMA_SOURCE}")
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/json_schema_converter\\.cc$")
  # Correct UTF-8 range tails and Unicode negation in a generated copy. Source
  # provenance is checked above; the patcher additionally verifies its anchors.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_unicode_fsm.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/prepare-xgrammar-unicode.py")
  execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/prepare-xgrammar-unicode.py"
    --source "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp/grammar_functor.cc"
    --replacement "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_unicode_fsm.inc"
    --output "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc"
    COMMAND_ERROR_IS_FATAL ANY)
  list(FILTER CUTEAFD_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/grammar_functor\\.cc$")
  list(APPEND CUTEAFD_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc")
  list(APPEND CUTEAFD_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_regular.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_v41_tools.cc")
  add_library(cuteafd_xgrammar STATIC ${CUTEAFD_XGRAMMAR_SOURCES})
  target_include_directories(cuteafd_xgrammar PRIVATE
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(cuteafd_xgrammar
    PUBLIC "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/include"
  )
  target_include_directories(cuteafd_xgrammar SYSTEM PUBLIC
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include"
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  )
  target_compile_definitions(cuteafd_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  )
  find_package(Threads REQUIRED)
  target_link_libraries(cuteafd_xgrammar PUBLIC Threads::Threads)
  list(APPEND CUTEAFD_NATIVE_SOURCES src/xgrammar_adapter.cc)
endif()

set(CUTEAFD_BUILD_SPARKINFER_AOT OFF)
# FAMILY:fp8 expert families (exact FP8 packages, cmake/fp8_moe.cmake).
set(CUTEAFD_FP8_EXPERT_FAMILIES OFF)
if(CUTEAFD_EXPERT_FAMILIES MATCHES ":fp8")
  set(CUTEAFD_FP8_EXPERT_FAMILIES ON)
endif()
if(CUTEAFD_ENABLE_SPARKINFER_AOT OR CUTEAFD_ENABLE_DS4_FLASH_AOT OR
   CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT OR CUTEAFD_ENABLE_W8A16_AOT OR
   CUTEAFD_ENABLE_V41_EXPERT_AOT OR CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT OR
   CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT OR CUTEAFD_ENABLE_V41_EXL3_AOT OR CUTEAFD_ENABLE_V41_NVFP4_AOT OR CUTEAFD_ENABLE_V41_FP8_AOT OR CUTEAFD_ENABLE_V41_ATTENTION_AOT OR CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR
   CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT OR CUTEAFD_FP8_EXPERT_FAMILIES)
  set(CUTEAFD_BUILD_SPARKINFER_AOT ON)
endif()
if(CUTEAFD_BUILD_SPARKINFER_AOT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/__init__.py")
    message(FATAL_ERROR
      "SparkInfer was not found under CUTEAFD_SPARKINFER_SOURCE_DIR="
      "${CUTEAFD_SPARKINFER_SOURCE_DIR}. Initialize third_party/sparkinfer or "
      "select an explicit pinned source tree.")
  endif()
  set(CUTEAFD_SPARKINFER_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-sparkinfer-source.py")
  set(CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_pinned_sparkinfer.py")
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "SparkInfer source verifier was not found: ${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}")
  endif()
  if(NOT EXISTS "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}")
    message(FATAL_ERROR
      "SparkInfer Python bootstrap was not found: "
      "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}")
  endif()
  set(CUTEAFD_SPARKINFER_VERIFY_COMMAND
    "${Python3_EXECUTABLE}" "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
    --source "${CUTEAFD_SPARKINFER_SOURCE_DIR}"
    --lock "${CUTEAFD_SPARKINFER_LOCK_FILE}"
  )
  set(CUTEAFD_SPARKINFER_PYTHON_ENV
    "CUTEAFD_SPARKINFER_SOURCE_DIR=${CUTEAFD_SPARKINFER_SOURCE_DIR}"
    "CUTEAFD_SPARKINFER_LOCK_FILE=${CUTEAFD_SPARKINFER_LOCK_FILE}"
    # Persisted objects are executable-only ExternalBinaryModules. AOT export
    # needs the live compiler IR consumed by export_to_c(), so never substitute
    # a runtime disk-cache hit in these build-time Python processes.
    "B12X_COMPILE_DISK_CACHE=0"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${CUTEAFD_SPARKINFER_SOURCE_DIR}"
  )
  execute_process(
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    RESULT_VARIABLE CUTEAFD_SPARKINFER_VERIFY_RESULT
    OUTPUT_VARIABLE CUTEAFD_SPARKINFER_VERIFY_OUTPUT
    ERROR_VARIABLE CUTEAFD_SPARKINFER_VERIFY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT CUTEAFD_SPARKINFER_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Pinned SparkInfer verification failed:\n${CUTEAFD_SPARKINFER_VERIFY_ERROR}")
  endif()
  message(STATUS "${CUTEAFD_SPARKINFER_VERIFY_OUTPUT}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CUTEAFD_SPARKINFER_LOCK_FILE}"
    "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
  )
  set(CUTEAFD_SPARKINFER_PROVENANCE_INPUTS
    "${CUTEAFD_SPARKINFER_LOCK_FILE}"
    "${CUTEAFD_SPARKINFER_VERIFY_SCRIPT}"
    "${CUTEAFD_SPARKINFER_PYTHON_BOOTSTRAP}"
  )
  file(GLOB_RECURSE CUTEAFD_SPARKINFER_EXPORT_INPUTS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.py"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.cu"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.cuh"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.h"
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/b12x/*.hpp"
  )
  list(APPEND CUTEAFD_SPARKINFER_EXPORT_INPUTS
    "${CUTEAFD_SPARKINFER_SOURCE_DIR}/pyproject.toml"
  )
  add_custom_target(cuteafd_verify_sparkinfer_source
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMENT "Verifying pinned SparkInfer source"
    VERBATIM
  )
endif()

if(CUTEAFD_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND CUTEAFD_NATIVE_SOURCES
    cuda/kernels/v41_expert_pack.cu
    cuda/kernels/v41_exl3_wire.cu
    cuda/kernels/v41_route_reduce.cu
    cuda/kernels/engram.cu
    cuda/kernels/v41_dspark.cu
    cuda/kernels/v41_hc.cu
    cuda/kernels/v41_router.cu
    cuda/kernels/dsv4_dspark.cu
    cuda/kernels/glm_dflash.cu
    cuda/kernels/glm_mla_prefill.cu
    cuda/kernels/mimo_dflash.cu
    cuda/kernels/fp8_block_dequant.cu
    cuda/kernels/v41_dspark_cache.cu
    cuda/kernels/v41_dspark_attention.cu
    cuda/kernels/v41_compressor.cu
    cuda/kernels/v41_vision.cu
    cuda/kernels/v41_kv.cu
    cuda/kernels/v41_sparse_attention.cu
    cuda/kernels/v41_peer_copy.cu
    cuda/kernels/v41_index_scores.cu
    cuda/kernels/v41_index_topk.cu
    cuda/kernels/v41_candidate_blocks.cu
    cuda/kernels/v41_attention_ops.cu
    cuda/kernels/norm.cu
    cuda/kernels/mlp.cu
    cuda/kernels/residual.cu
    cuda/kernels/kv_cache.cu
    cuda/kernels/mla_indexing.cu
    cuda/kernels/router.cu
    cuda/kernels/linear.cu
    cuda/kernels/attention.cu
    cuda/kernels/embedding.cu
    cuda/kernels/sampling.cu
    cuda/kernels/v41_sampling_gpu.cu
    cuda/kernels/nibbles.cu
    cuda/kernels/nvfp4_payload.cu
    cuda/kernels/nvfp4_scale.cu
    cuda/kernels/ds4_flash_direct.cu
  )
  # SM120 builds use the family-specific target (sm_120f); every live AOT
  # export and hand kernel is qualified with it.
  if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120")
    set(CUTEAFD_CUDA_ARCHITECTURES "120f")
  endif()
endif()

set(CUTEAFD_BUILD_FLASHINFER_PACKED_MLA OFF)
if(CUTEAFD_ENABLE_FLASHINFER_PACKED_MLA OR CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT OR CUTEAFD_ENABLE_W8A16_AOT)
  set(CUTEAFD_BUILD_FLASHINFER_PACKED_MLA ON)
endif()

if(CUTEAFD_BUILD_FLASHINFER_PACKED_MLA)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_FLASHINFER_PACKED_MLA=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent / 'data' / 'include')"
    OUTPUT_VARIABLE CUTEAFD_FLASHINFER_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT EXISTS "${CUTEAFD_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh")
    message(FATAL_ERROR
      "FlashInfer SM120 sparse-MLA headers were not found under ${CUTEAFD_FLASHINFER_INCLUDE_DIR}")
  endif()
  list(APPEND CUTEAFD_NATIVE_SOURCES
    cuda/kernels/packed_fp8_mla_exact.cu
  )
  set_source_files_properties(cuda/kernels/packed_fp8_mla_exact.cu PROPERTIES
    COMPILE_OPTIONS "-use_fast_math;-O3"
  )
  if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120")
    set(CUTEAFD_CUDA_ARCHITECTURES "120f")
  endif()
endif()

if(CUTEAFD_ENABLE_SPARKINFER_AOT)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_SPARKINFER_AOT=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(CUTEAFD_B12X_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/b12x_aot")
  set(CUTEAFD_B12X_AOT_REGIMES 1 2 4 8 16 32 64 128 256)
  set(CUTEAFD_B12X_AOT_TOPK8_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(CUTEAFD_B12X_AOT_OBJECTS)
  set(CUTEAFD_B12X_AOT_HEADERS)
  list(APPEND CUTEAFD_B12X_AOT_OBJECTS
    "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.o"
    "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.o"
  )
  foreach(regime IN LISTS CUTEAFD_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND CUTEAFD_B12X_AOT_OBJECTS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS CUTEAFD_B12X_AOT_REGIMES)
    list(APPEND CUTEAFD_B12X_AOT_OBJECTS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.o"
    )
  endforeach()
  list(APPEND CUTEAFD_B12X_AOT_HEADERS
    "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.h"
    "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.h"
    "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS CUTEAFD_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND CUTEAFD_B12X_AOT_HEADERS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS CUTEAFD_B12X_AOT_REGIMES)
    list(APPEND CUTEAFD_B12X_AOT_HEADERS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${CUTEAFD_B12X_AOT_OBJECTS}
      ${CUTEAFD_B12X_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      --output-dir "${CUTEAFD_B12X_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting Spark-side SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_b12x_aot_export
    DEPENDS
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${CUTEAFD_B12X_AOT_OBJECTS}
      ${CUTEAFD_B12X_AOT_HEADERS}
  )
  add_dependencies(cuteafd_b12x_aot_export cuteafd_verify_sparkinfer_source)
  set_source_files_properties(${CUTEAFD_B12X_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND CUTEAFD_NATIVE_SOURCES
    cuda/kernels/b12x_direct.cu
    ${CUTEAFD_B12X_AOT_OBJECTS}
  )

  set(CUTEAFD_B12X_M1_PARITY_AOT_REGIMES 2 3 4 5 6 7 8)
  set(CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS)
  set(CUTEAFD_B12X_M1_PARITY_AOT_HEADERS
    "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot_config.h"
  )
  foreach(regime IN LISTS CUTEAFD_B12X_M1_PARITY_AOT_REGIMES)
    list(APPEND CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.o"
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.o"
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.o"
    )
    list(APPEND CUTEAFD_B12X_M1_PARITY_AOT_HEADERS
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.h"
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.h"
      "${CUTEAFD_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS}
      ${CUTEAFD_B12X_M1_PARITY_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      --output-dir "${CUTEAFD_B12X_AOT_DIR}"
      --target-sms 48
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ordered Spark W4A16 M=2..8 parity candidates"
    VERBATIM
  )
  add_custom_target(cuteafd_b12x_m1_parity_aot_export
    DEPENDS
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${CUTEAFD_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS}
      ${CUTEAFD_B12X_M1_PARITY_AOT_HEADERS}
  )
  add_dependencies(
    cuteafd_b12x_m1_parity_aot_export
    cuteafd_verify_sparkinfer_source
  )
  set_source_files_properties(${CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND CUTEAFD_NATIVE_SOURCES ${CUTEAFD_B12X_M1_PARITY_AOT_OBJECTS})

endif()

if(CUTEAFD_ENABLE_DS4_FLASH_AOT)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_DS4_FLASH_AOT=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(CUTEAFD_DS4_FLASH_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_flash_aot")
  set(CUTEAFD_DS4_FLASH_AOT_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(CUTEAFD_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES 3 5 6 7 9 10 11 12)
  set(CUTEAFD_DS4_FLASH_AOT_OBJECTS
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.o"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.o")
  set(CUTEAFD_DS4_FLASH_AOT_HEADERS
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.h"
    "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS CUTEAFD_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND CUTEAFD_DS4_FLASH_AOT_OBJECTS
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.o"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.o"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.o"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND CUTEAFD_DS4_FLASH_AOT_HEADERS
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.h"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.h"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.h"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  foreach(regime IN LISTS CUTEAFD_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES)
    list(APPEND CUTEAFD_DS4_FLASH_AOT_OBJECTS
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND CUTEAFD_DS4_FLASH_AOT_HEADERS
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${CUTEAFD_DS4_FLASH_AOT_OBJECTS}
      ${CUTEAFD_DS4_FLASH_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${CUTEAFD_DS4_FLASH_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Flash Spark expert-TP4 AOT kernel"
    VERBATIM
  )
  add_custom_target(cuteafd_ds4_flash_aot_export
    DEPENDS
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${CUTEAFD_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${CUTEAFD_DS4_FLASH_AOT_OBJECTS}
      ${CUTEAFD_DS4_FLASH_AOT_HEADERS}
  )
  add_dependencies(cuteafd_ds4_flash_aot_export cuteafd_verify_sparkinfer_source)

  # Pro experts are trellis-only. Keep their generated symbols, resident
  # geometry, and native bridge separate from the Flash FP4-compatible path.
  set(CUTEAFD_DS4_PRO_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_pro_aot")
  set(CUTEAFD_DS4_PRO_AOT_OBJECTS
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.o"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.o")
  set(CUTEAFD_DS4_PRO_AOT_HEADERS
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.h"
    "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot_config.h")
  foreach(regime IN LISTS CUTEAFD_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND CUTEAFD_DS4_PRO_AOT_OBJECTS
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.o")
    list(APPEND CUTEAFD_DS4_PRO_AOT_HEADERS
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${CUTEAFD_DS4_PRO_AOT_OBJECTS}
      ${CUTEAFD_DS4_PRO_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${CUTEAFD_DS4_PRO_AOT_DIR}"
      --target-sms 48
      --model-variant pro
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Pro EXL3-only Spark expert-TP4 AOT kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_ds4_pro_aot_export
    DEPENDS
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${CUTEAFD_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${CUTEAFD_DS4_PRO_AOT_OBJECTS}
      ${CUTEAFD_DS4_PRO_AOT_HEADERS}
  )
  add_dependencies(cuteafd_ds4_pro_aot_export cuteafd_verify_sparkinfer_source)

  set_source_files_properties(${CUTEAFD_DS4_FLASH_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  set_source_files_properties(${CUTEAFD_DS4_PRO_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND CUTEAFD_NATIVE_SOURCES
    cuda/kernels/ds4_flash_aot.cu
    cuda/kernels/ds4_pro_aot.cu
    ${CUTEAFD_DS4_FLASH_AOT_OBJECTS}
    ${CUTEAFD_DS4_PRO_AOT_OBJECTS}
  )
endif()

if(CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR
      "CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(CUTEAFD_B12X_COORDINATOR_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/b12x_coordinator_aot"
  )
  set(CUTEAFD_B12X_COORDINATOR_AOT_OBJECTS
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.o"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.o"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.o"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.o"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.o"
  )
  set(CUTEAFD_B12X_COORDINATOR_AOT_HEADERS
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.h"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.h"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.h"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.h"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.h"
    "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${CUTEAFD_B12X_COORDINATOR_AOT_OBJECTS}
      ${CUTEAFD_B12X_COORDINATOR_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      --output-dir "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_b12x_coordinator_aot_export
    DEPENDS
      "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${CUTEAFD_B12X_COORDINATOR_AOT_OBJECTS}
      ${CUTEAFD_B12X_COORDINATOR_AOT_HEADERS}
  )
  add_dependencies(
    cuteafd_b12x_coordinator_aot_export
    cuteafd_verify_sparkinfer_source
  )
  set_source_files_properties(${CUTEAFD_B12X_COORDINATOR_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND CUTEAFD_NATIVE_SOURCES
    cuda/kernels/b12x_coordinator.cu
    ${CUTEAFD_B12X_COORDINATOR_AOT_OBJECTS}
  )
endif()

if(CUTEAFD_ENABLE_W8A16_AOT)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_W8A16_AOT=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(CUTEAFD_W8A16_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/w8a16_aot")
  set(CUTEAFD_W8A16_AOT_HEADER
    "${CUTEAFD_W8A16_AOT_DIR}/w8a16_row_major_aot.h"
  )
  set(CUTEAFD_W8A16_AOT_SOURCE
    "${CUTEAFD_W8A16_AOT_DIR}/w8a16_row_major_aot.cc"
  )
  set(CUTEAFD_W8A16_PACKED_O_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/w8a16_packed_o_aot"
  )
  set(CUTEAFD_W8A16_PACKED_O_AOT_OBJECTS
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.o"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.o"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.o"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.o"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.o"
  )
  set(CUTEAFD_W8A16_PACKED_O_AOT_HEADERS
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.h"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.h"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.h"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.h"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.h"
    "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${CUTEAFD_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${CUTEAFD_W8A16_AOT_HEADER}"
      "${CUTEAFD_W8A16_AOT_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      --output-dir "${CUTEAFD_W8A16_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_triton_prefill.py"
    COMMENT "Exporting coordinator row-major W8A16 AOT kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_w8a16_aot_export
    DEPENDS
      "${CUTEAFD_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${CUTEAFD_W8A16_AOT_HEADER}"
      "${CUTEAFD_W8A16_AOT_SOURCE}"
  )
  add_custom_command(
    OUTPUT
      "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${CUTEAFD_W8A16_PACKED_O_AOT_OBJECTS}
      ${CUTEAFD_W8A16_PACKED_O_AOT_HEADERS}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      --output-dir "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_cute_packed_prefill.py"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS}
      ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator packed W8A16 O-projection AOT kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_w8a16_packed_o_aot_export
    DEPENDS
      "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${CUTEAFD_W8A16_PACKED_O_AOT_OBJECTS}
      ${CUTEAFD_W8A16_PACKED_O_AOT_HEADERS}
  )
  add_dependencies(
    cuteafd_w8a16_packed_o_aot_export
    cuteafd_verify_sparkinfer_source
  )
  set_source_files_properties("${CUTEAFD_W8A16_AOT_SOURCE}" PROPERTIES
    GENERATED TRUE
  )
  set_source_files_properties(${CUTEAFD_W8A16_PACKED_O_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND CUTEAFD_NATIVE_SOURCES
    "${CUTEAFD_W8A16_AOT_SOURCE}"
    cuda/kernels/w8a16_packed.cu
    ${CUTEAFD_W8A16_PACKED_O_AOT_OBJECTS}
  )
endif()

if(CUTEAFD_ENABLE_SPARKINFER_AOT OR CUTEAFD_ENABLE_DS4_FLASH_AOT OR
   CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT OR CUTEAFD_ENABLE_W8A16_AOT OR
   CUTEAFD_ENABLE_V41_EXPERT_AOT OR CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT OR
   CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT OR CUTEAFD_ENABLE_V41_EXL3_AOT OR CUTEAFD_ENABLE_V41_NVFP4_AOT OR CUTEAFD_ENABLE_V41_FP8_AOT OR CUTEAFD_ENABLE_V41_ATTENTION_AOT OR
   CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT OR CUTEAFD_FP8_EXPERT_FAMILIES)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE CUTEAFD_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(CUTEAFD_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${CUTEAFD_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

if(CUTEAFD_ENABLE_V41_EXPERT_AOT)
  include(cmake/v41_experts.cmake)
endif()
if(NOT CUTEAFD_V41_SPARK_TP_ROLES STREQUAL "")
  include(cmake/v41_spark_tp_experts.cmake)
endif()
if(NOT CUTEAFD_EXPERT_FAMILIES STREQUAL "")
  include(cmake/expert_families.cmake)
endif()
if(CUTEAFD_FP8_EXPERT_FAMILIES)
  include(cmake/fp8_moe.cmake)
endif()
if(CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT)
  include(cmake/dsv4_programs.cmake)
endif()
if(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT)
  include(cmake/v41_local_experts.cmake)
endif()
if(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT)
  include(cmake/v41_tp2_experts.cmake)
  include(cmake/v41_dspark_tp2_experts.cmake)
endif()
if(CUTEAFD_ENABLE_V41_FP8_AOT)
  include(cmake/v41_fp8.cmake)
  include(cmake/v41_router.cmake)
endif()
if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
  include(cmake/v41_attention.cmake)
endif()
if(CUTEAFD_ENABLE_V41_NVFP4_AOT)
  include(cmake/v41_nvfp4_experts.cmake)
endif()
if(CUTEAFD_ENABLE_V41_EXL3_AOT)
  include(cmake/v41_exl3.cmake)
endif()
add_library(cuteafd_native SHARED ${CUTEAFD_NATIVE_SOURCES})
if(CUTEAFD_ENABLE_V41_EXL3_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_exl3_export)
endif()
if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_HAVE_V41_ATTENTION_AOT=1)
  add_dependencies(cuteafd_native cuteafd_v41_attention_export cuteafd_v41_attention_heads32_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_ATTENTION_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_FP8_AOT)
  # Packed FP8->BF16 conversion requires the SM120 architecture-specific ISA.
  # Keep the normal target's portable image as a fallback in the same fatbin.
  set_property(SOURCE cuda/kernels/v41_sparse_attention.cu APPEND PROPERTY
    COMPILE_OPTIONS "--generate-code=arch=compute_120a,code=sm_120a")
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_HAVE_V41_HC_AOT=1 CUTEAFD_HAVE_V41_ROUTER_AOT=1)
  add_dependencies(cuteafd_native cuteafd_v41_fp8_export cuteafd_v41_index_export cuteafd_v41_router_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_FP8_DIR}" "${CUTEAFD_V41_INDEX_DIR}" "${CUTEAFD_V41_ROUTER_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_EXPERT_DIR}")
endif()
if(CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS)
  add_dependencies(cuteafd_native ${CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS})
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_V41_SPARK_TP_EXPERT_INCLUDE_DIRS})
endif()
if(CUTEAFD_ENABLE_DSV4_AOT OR CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT)
  add_dependencies(cuteafd_native cuteafd_dsv4_programs_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_DSV4_DIR}")
  # Whole archive: the table in dsv4_programs.cc names every program symbol.
  target_link_libraries(cuteafd_native PRIVATE "-Wl,--whole-archive" "${CUTEAFD_DSV4_ARCHIVE}" "-Wl,--no-whole-archive")
endif()
if(CUTEAFD_EXPERT_FAMILY_TARGETS)
  add_dependencies(cuteafd_native ${CUTEAFD_EXPERT_FAMILY_TARGETS})
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_EXPERT_FAMILY_INCLUDE_DIRS})
endif()
if(CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_local_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_LOCAL_EXPERT_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT)
  add_dependencies(cuteafd_native cuteafd_v41_tp2_experts_export cuteafd_v41_dspark_tp2_experts_export)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_V41_TP2_EXPERT_DIR}" "${CUTEAFD_V41_DSPARK_TP2_EXPERT_DIR}")
endif()
if(CUTEAFD_ENABLE_V41_NVFP4_AOT)
  foreach(nvfp4_role IN LISTS CUTEAFD_V41_NVFP4_ROLES)
    add_dependencies(cuteafd_native cuteafd_v41_nvfp4_${nvfp4_role}_export)
  endforeach()
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_V41_NVFP4_INCLUDE_DIRS})
endif()
target_include_directories(cuteafd_native PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(CUTEAFD_ENABLE_XGRAMMAR)
  target_link_libraries(cuteafd_native PRIVATE cuteafd_xgrammar)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp")
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_XGRAMMAR=1)
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_XGRAMMAR=0)
endif()
if(CUTEAFD_ENABLE_CUDA)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_CUDA=1)
  if(CUTEAFD_V41_VOCAB_ROW_EXPERIMENT)
    target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_V41_VOCAB_ROW_EXPERIMENT=1)
  endif()
  target_link_libraries(cuteafd_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(cuteafd_native PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_CUDA=0)
endif()
if(CUTEAFD_BUILD_FLASHINFER_PACKED_MLA)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_FLASHINFER_INCLUDE_DIR}")
endif()
if(CUTEAFD_ENABLE_SPARKINFER_AOT)
  add_dependencies(
    cuteafd_native
    cuteafd_b12x_aot_export
    cuteafd_b12x_m1_parity_aot_export
  )
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_B12X_AOT=1)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_B12X_AOT_DIR}")
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_B12X_AOT=0)
endif()
if(CUTEAFD_ENABLE_DS4_FLASH_AOT)
  add_dependencies(
    cuteafd_native
    cuteafd_ds4_flash_aot_export
    cuteafd_ds4_pro_aot_export
  )
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_DS4_FLASH_AOT=1)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_DS4_FLASH_AOT_DIR}")
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_DS4_PRO_AOT_DIR}")
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_DS4_FLASH_AOT=0)
endif()
if(CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT)
  add_dependencies(cuteafd_native cuteafd_b12x_coordinator_aot_export)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_B12X_COORDINATOR_AOT=1)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_B12X_COORDINATOR_AOT_DIR}")
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_B12X_COORDINATOR_AOT=0)
endif()
if(CUTEAFD_ENABLE_W8A16_AOT)
  add_dependencies(cuteafd_native cuteafd_w8a16_aot_export)
  add_dependencies(cuteafd_native cuteafd_w8a16_packed_o_aot_export)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_W8A16_AOT=1)
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_W8A16_AOT_DIR}")
  target_include_directories(cuteafd_native PRIVATE "${CUTEAFD_W8A16_PACKED_O_AOT_DIR}")
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_W8A16_AOT=0)
endif()
if(CUTEAFD_ENABLE_SPARKINFER_AOT OR CUTEAFD_ENABLE_DS4_FLASH_AOT OR
   CUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT OR CUTEAFD_ENABLE_W8A16_AOT OR
   CUTEAFD_ENABLE_V41_EXPERT_AOT OR CUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT OR
   CUTEAFD_ENABLE_V41_TP2_EXPERT_AOT OR CUTEAFD_ENABLE_V41_EXL3_AOT OR CUTEAFD_ENABLE_V41_NVFP4_AOT OR CUTEAFD_ENABLE_V41_FP8_AOT OR CUTEAFD_ENABLE_V41_ATTENTION_AOT OR
   CUTEAFD_ENABLE_GLM_AOT OR CUTEAFD_ENABLE_MIMO_AOT OR CUTEAFD_ENABLE_GLMF_AOT OR CUTEAFD_ENABLE_QWEN4_AOT)
  target_link_libraries(cuteafd_native PUBLIC "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET cuteafd_native APPEND PROPERTY BUILD_RPATH "${CUTEAFD_B12X_AOT_RUNTIME_DIR}")
endif()
if(CUTEAFD_ENABLE_RDMA)
  find_path(CUTEAFD_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(CUTEAFD_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT CUTEAFD_IBVERBS_INCLUDE_DIR OR NOT CUTEAFD_IBVERBS_LIBRARY)
    message(FATAL_ERROR "CUTEAFD_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_RDMA=1)
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_IBVERBS_INCLUDE_DIR})
  target_link_libraries(cuteafd_native PUBLIC ${CUTEAFD_IBVERBS_LIBRARY})
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_RDMA=0)
endif()
if(CUTEAFD_ENABLE_NCCL)
  if(NOT CUTEAFD_ENABLE_CUDA)
    message(FATAL_ERROR "CUTEAFD_ENABLE_NCCL=ON requires CUTEAFD_ENABLE_CUDA=ON")
  endif()
  find_path(CUTEAFD_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(CUTEAFD_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_NCCL=1)
  target_include_directories(cuteafd_native PRIVATE ${CUTEAFD_NCCL_INCLUDE_DIR})
  target_link_libraries(cuteafd_native PUBLIC ${CUTEAFD_NCCL_LIBRARY})
else()
  target_compile_definitions(cuteafd_native PRIVATE CUTEAFD_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(cuteafd_native_selftest tests/native_selftest.cc)
target_link_libraries(cuteafd_native_selftest PRIVATE cuteafd_native)
# The self-test intentionally uses assertions for both the ABI call and its
# result check. Keep those expressions live in Release builds; defining
# NDEBUG would otherwise skip allocations and make the following copy probe
# dereference an uninitialized buffer.
if(MSVC)
  target_compile_options(cuteafd_native_selftest PRIVATE /UNDEBUG)
else()
  target_compile_options(cuteafd_native_selftest PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME cuteafd_native_selftest COMMAND cuteafd_native_selftest)

if(CUTEAFD_ENABLE_XGRAMMAR)
  add_executable(cuteafd_xgrammar_regular_selftest tests/xgrammar_regular_selftest.cc)
  target_link_libraries(cuteafd_xgrammar_regular_selftest PRIVATE cuteafd_xgrammar)
  target_include_directories(cuteafd_xgrammar_regular_selftest PRIVATE
    "${CUTEAFD_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME cuteafd_xgrammar_regular_selftest COMMAND cuteafd_xgrammar_regular_selftest)
  add_executable(cuteafd_xgrammar_selftest tests/xgrammar_selftest.cc)
  target_link_libraries(cuteafd_xgrammar_selftest PRIVATE cuteafd_native)
  target_compile_definitions(cuteafd_xgrammar_selftest PRIVATE
    "CUTEAFD_XGRAMMAR_TEST_TOKENIZER=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/xgrammar_tiny_tokenizer.json\""
  )
  if(MSVC)
    target_compile_options(cuteafd_xgrammar_selftest PRIVATE /UNDEBUG)
  else()
    target_compile_options(cuteafd_xgrammar_selftest PRIVATE -UNDEBUG)
  endif()
  add_test(NAME cuteafd_xgrammar_selftest COMMAND cuteafd_xgrammar_selftest)
endif()

if(CUTEAFD_ENABLE_CUDA)
  add_executable(cuteafd_v41_peer_copy_selftest tests/v41_peer_copy_selftest.cc)
  target_link_libraries(cuteafd_v41_peer_copy_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_peer_copy_selftest COMMAND cuteafd_v41_peer_copy_selftest)
  set_tests_properties(cuteafd_v41_peer_copy_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_route_reduce_planes_selftest
    tests/v41_route_reduce_planes_selftest.cc)
  target_link_libraries(cuteafd_v41_route_reduce_planes_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_route_reduce_planes_selftest
    COMMAND cuteafd_v41_route_reduce_planes_selftest)
  set_tests_properties(cuteafd_v41_route_reduce_planes_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_expert_pack_tp3_selftest
    tests/v41_expert_pack_tp3_selftest.cc)
  target_link_libraries(cuteafd_v41_expert_pack_tp3_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_expert_pack_tp3_selftest
    COMMAND cuteafd_v41_expert_pack_tp3_selftest)
  set_tests_properties(cuteafd_v41_expert_pack_tp3_selftest PROPERTIES SKIP_RETURN_CODE 77)
  # Geometry-driven packer check: every native extent, including the padded TP4
  # 576 -> 640 storage, in one parameterized device test.
  add_executable(cuteafd_v41_expert_pack_geometry_selftest
    tests/v41_expert_pack_geometry_selftest.cc)
  target_link_libraries(cuteafd_v41_expert_pack_geometry_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_expert_pack_geometry_selftest
    COMMAND cuteafd_v41_expert_pack_geometry_selftest)
  set_tests_properties(cuteafd_v41_expert_pack_geometry_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cuteafd_v41_fp4_attention_selftest tests/v41_fp4_attention_selftest.cc)
  target_link_libraries(cuteafd_v41_fp4_attention_selftest PRIVATE cuteafd_native CUDA::cudart)
  if(CUTEAFD_ENABLE_V41_ATTENTION_AOT)
    target_compile_definitions(cuteafd_v41_fp4_attention_selftest PRIVATE CUTEAFD_TEST_ATTENTION_AOT=1)
  endif()
  add_test(NAME cuteafd_v41_fp4_attention_selftest COMMAND cuteafd_v41_fp4_attention_selftest)
  add_executable(cuteafd_cuda_inplace_rope_selftest tests/cuda_inplace_rope_selftest.cc)
  target_link_libraries(cuteafd_cuda_inplace_rope_selftest PRIVATE cuteafd_native)
  add_test(NAME cuteafd_cuda_inplace_rope_selftest COMMAND cuteafd_cuda_inplace_rope_selftest)
  set_tests_properties(cuteafd_cuda_inplace_rope_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_checked_argmax_selftest tests/cuda_checked_argmax_selftest.cc)
  target_link_libraries(cuteafd_cuda_checked_argmax_selftest PRIVATE cuteafd_native)
  add_test(NAME cuteafd_cuda_checked_argmax_selftest COMMAND cuteafd_cuda_checked_argmax_selftest)
  set_tests_properties(cuteafd_cuda_checked_argmax_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_stream_query_selftest tests/cuda_stream_query_selftest.cc)
  target_link_libraries(cuteafd_cuda_stream_query_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_cuda_stream_query_selftest COMMAND cuteafd_cuda_stream_query_selftest)
  set_tests_properties(cuteafd_cuda_stream_query_selftest PROPERTIES TIMEOUT 15)
  add_executable(cuteafd_cuda_peer_selftest tests/cuda_peer_selftest.cc)
  target_link_libraries(cuteafd_cuda_peer_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_cuda_peer_selftest COMMAND cuteafd_cuda_peer_selftest)
  set_tests_properties(cuteafd_cuda_peer_selftest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
  add_executable(cuteafd_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(cuteafd_cuda_selftest PRIVATE cuteafd_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME cuteafd_cuda_selftest COMMAND cuteafd_cuda_selftest)
  # Chunk-1 GPU target-sampler (K1) device tests. They link the shipped native
  # library and therefore also pin the exported C ABI of v41_sampling_gpu.h.
  # `.cu` because the subnormal/FTZ canary defines a probe kernel compiled with
  # the same CUDA flags as the shipped sampler objects.
  add_executable(cuteafd_v41_sampling_selftest tests/v41_sampling_selftest.cu)
  set_source_files_properties(tests/v41_sampling_selftest.cu PROPERTIES LANGUAGE CUDA)
  # The FTZ canary compiles device code in this target, so it must target the
  # same architecture as the kernels it canaries; without this the executable
  # silently defaults to sm_75 and its probe kernel never launches.
  set_target_properties(cuteafd_v41_sampling_selftest PROPERTIES
    CUDA_ARCHITECTURES "${CUTEAFD_CUDA_ARCHITECTURES}")
  target_link_libraries(cuteafd_v41_sampling_selftest PRIVATE cuteafd_native CUDA::cudart)
  add_test(NAME cuteafd_v41_sampling_selftest COMMAND cuteafd_v41_sampling_selftest)
  set_tests_properties(cuteafd_v41_sampling_selftest PROPERTIES SKIP_RETURN_CODE 77)
endif()
