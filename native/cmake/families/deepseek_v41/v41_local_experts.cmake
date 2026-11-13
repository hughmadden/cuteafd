# Full RTX backbone experts coexist with the ordinary dSpark variant table.
if(NOT CUTEAFD_ENABLE_V41_EXPERT_AOT OR NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL "coordinator")
  message(FATAL_ERROR "Local V4.1 experts require the SM120 or SM89 coordinator expert build")
endif()
set(CUTEAFD_V41_LOCAL_EXPERT_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_local_experts")
set(CUTEAFD_V41_LOCAL_EXPERT_OBJECTS)
set(CUTEAFD_V41_LOCAL_EXPERT_HEADERS)
foreach(rows IN LISTS CUTEAFD_V41_TARGET_ROWS)
  set(stem "${CUTEAFD_V41_LOCAL_EXPERT_DIR}/v41_rtx_backbone_m${rows}")
  list(APPEND CUTEAFD_V41_LOCAL_EXPERT_OBJECTS "${stem}.o")
  list(APPEND CUTEAFD_V41_LOCAL_EXPERT_HEADERS "${stem}.h")
endforeach()
add_custom_command(
  OUTPUT "${CUTEAFD_V41_LOCAL_EXPERT_DIR}/v41_experts.json"
    "${CUTEAFD_V41_LOCAL_EXPERT_DIR}/v41_local_expert_variants.h"
    ${CUTEAFD_V41_LOCAL_EXPERT_OBJECTS} ${CUTEAFD_V41_LOCAL_EXPERT_HEADERS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
    --output-dir "${CUTEAFD_V41_LOCAL_EXPERT_DIR}" --role rtx_backbone
    --rows ${CUTEAFD_V41_TARGET_ROWS_ARG} --width 192 --standard-names
  DEPENDS "${CUTEAFD_V41_CAPACITY_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_capacities.py" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_spark_tp3_launch_geometry.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_experts_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/expert_grouping.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting full-width RTX backbone expert kernels"
  VERBATIM
)
add_custom_target(cuteafd_v41_local_experts_export DEPENDS
  "${CUTEAFD_V41_LOCAL_EXPERT_DIR}/v41_local_expert_variants.h"
  "${CUTEAFD_V41_LOCAL_EXPERT_DIR}/v41_experts.json"
  ${CUTEAFD_V41_LOCAL_EXPERT_OBJECTS} ${CUTEAFD_V41_LOCAL_EXPERT_HEADERS})
add_dependencies(cuteafd_v41_local_experts_export cuteafd_verify_sparkinfer_source)
set_source_files_properties(${CUTEAFD_V41_LOCAL_EXPERT_OBJECTS} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES ${CUTEAFD_V41_LOCAL_EXPERT_OBJECTS} families/deepseek_v41/src/v41_local_experts.cc)
