# Exact FP8 routed-expert packages (FAMILY:fp8 entries of CUTEAFD_EXPERT_FAMILIES,
# for example mimo:fp8 or glm:fp8): the checkpoint's E4M3 experts with FP32
# 128x128 block scales, run by b12x fp8_moe programs. Each entry builds
# fp8-FAMILY/ next to the native library: tp1 in the SM120 coordinator build
# (RTX local / MTP layers), tp4 and tp2 in the SM121 Spark build. The daemon
# resolves <libdir>/fp8/fp8-FAMILY/tp<world>; the artifact scripts install
# the packages there.
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120")
  set(CUTEAFD_FP8_MOE_ROLE coordinator)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "121")
  set(CUTEAFD_FP8_MOE_ROLE spark)
else()
  message(FATAL_ERROR "FP8 expert packages require a single native SM120 or SM121 target")
endif()
set(CUTEAFD_FP8_MOE_CAPACITIES "1,16,80,256,1024,4096" CACHE STRING "FP8 expert package capacities")
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUTEAFD_FP8_MOE_CUDA_INCLUDE)
set(CUTEAFD_FP8_MOE_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/package_fp8_moe_aot.py")
set(CUTEAFD_FP8_MOE_MANIFESTS)
foreach(entry IN LISTS CUTEAFD_EXPERT_FAMILIES)
  if(NOT entry MATCHES ":fp8$")
    continue()
  endif()
  if(NOT entry MATCHES "^(mimo|glm|glmf):fp8$")
    message(FATAL_ERROR "FP8 expert family ${entry} must be (mimo|glm|glmf):fp8")
  endif()
  set(geometry "${CMAKE_MATCH_1}")
  set(package "${CMAKE_CURRENT_BINARY_DIR}/fp8/fp8-${geometry}")
  set(stamp "${CMAKE_CURRENT_BINARY_DIR}/fp8_moe_${geometry}.stamp")
  file(GENERATE OUTPUT "${stamp}" CONTENT "role=${CUTEAFD_FP8_MOE_ROLE}|capacities=${CUTEAFD_FP8_MOE_CAPACITIES}\n")
  add_custom_command(
    OUTPUT "${package}/manifest.json"
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${package}"
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${CUTEAFD_FP8_MOE_TOOL}" build
      --role "${CUTEAFD_FP8_MOE_ROLE}" --geometry "${geometry}" --capacities "${CUTEAFD_FP8_MOE_CAPACITIES}"
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/fp8_moe_exports" --output "${package}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${CUTEAFD_FP8_MOE_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>" --runtime "${CUTEAFD_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${CUTEAFD_FP8_MOE_TOOL}" "${stamp}"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Building exact FP8 expert package fp8-${geometry} (${CUTEAFD_FP8_MOE_ROLE})"
    VERBATIM)
  list(APPEND CUTEAFD_FP8_MOE_MANIFESTS "${package}/manifest.json")
endforeach()
add_custom_target(cuteafd_fp8_moe_packages ALL DEPENDS ${CUTEAFD_FP8_MOE_MANIFESTS})
