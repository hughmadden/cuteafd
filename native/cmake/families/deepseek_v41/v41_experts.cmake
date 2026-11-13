# The export executes on the target architecture and consumes b12x's own planner.
if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "V4.1 expert AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120" OR CUTEAFD_CUDA_ARCHITECTURES STREQUAL "120f")
  set(CUTEAFD_V41_EXPERT_ROLE coordinator)
  set(CUTEAFD_V41_EXPERT_INPUT_FORMAT bf16)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "121")
  set(CUTEAFD_V41_EXPERT_ROLE spark)
  set(CUTEAFD_V41_EXPERT_INPUT_FORMAT fp8_k32)
elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "89")
  # SM89 (Ada) coordinators: promoted-MMA slices over the same resident experts,
  # scratch and route planes; quantizer grids and the SM cap are sized for the
  # largest supported SM89 card (python/tools/lib/sm89_cards.py).
  set(CUTEAFD_V41_EXPERT_ROLE coordinator)
  set(CUTEAFD_V41_EXPERT_INPUT_FORMAT bf16)
else()
  message(FATAL_ERROR "V4.1 expert AOT requires one native SM120, SM121 or SM89 target")
endif()
set(CUTEAFD_V41_EXPERT_SLICE_WIDTH "" CACHE STRING
  "Experimental fused-slice width (64, 128, 192); empty keeps qualified serving backend")
set(CUTEAFD_V41_EXPERT_ATOMIC_MIN_CAPACITY "" CACHE STRING
  "Experimental direct token accumulation threshold: empty, 256, 1024 or 4096")
set(CUTEAFD_V41_EXPERT_EXPORT_SCRIPT export_b12x_v41_experts_aot.py)
set(CUTEAFD_V41_EXPERT_EXPORT_ARGS
  --role "${CUTEAFD_V41_EXPERT_ROLE}" --input-format "${CUTEAFD_V41_EXPERT_INPUT_FORMAT}")
if(NOT CUTEAFD_V41_EXPERT_SLICE_WIDTH STREQUAL "")
  if(NOT CUTEAFD_V41_EXPERT_SLICE_WIDTH MATCHES "^(64|128|192|[0-9]+:(64|128|192)(,[0-9]+:(64|128|192))*)$")
    message(FATAL_ERROR "Experimental expert slices require valid widths or a capacity:width map")
  endif()
  set(CUTEAFD_V41_EXPERT_EXPORT_SCRIPT export_b12x_slices_aot.py)
  set(CUTEAFD_V41_EXPERT_EXPORT_ARGS --width "${CUTEAFD_V41_EXPERT_SLICE_WIDTH}"
    --rows "${CUTEAFD_V41_TARGET_ROWS_ARG}" --role "${CUTEAFD_V41_EXPERT_ROLE}")
endif()
if(NOT CUTEAFD_V41_EXPERT_ATOMIC_MIN_CAPACITY STREQUAL "")
  if(NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL "spark" OR
      CUTEAFD_V41_EXPERT_SLICE_WIDTH STREQUAL "" OR
      NOT CUTEAFD_V41_EXPERT_ATOMIC_MIN_CAPACITY MATCHES "^(256|1024|4096)$")
    message(FATAL_ERROR "Direct token accumulation requires Spark slices and a valid threshold")
  endif()
  list(APPEND CUTEAFD_V41_EXPERT_EXPORT_ARGS
    --atomic-min-capacity "${CUTEAFD_V41_EXPERT_ATOMIC_MIN_CAPACITY}")
endif()
if(CUTEAFD_V41_SPARK_COMPACT_EXPERIMENT)
  if(NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL "spark")
    message(FATAL_ERROR "Spark compact experiment requires the SM121 expert build")
  endif()
  list(APPEND CUTEAFD_V41_EXPERT_EXPORT_ARGS --compact-live-rows 2)
  if(NOT CUTEAFD_V41_EXPERT_SLICE_WIDTH STREQUAL "")
    list(APPEND CUTEAFD_V41_EXPERT_EXPORT_ARGS --compact-max-capacity 16)
  endif()
endif()
set(CUTEAFD_V41_EXPERT_ROWS ${CUTEAFD_V41_TARGET_ROWS})
if(CUTEAFD_V41_EXPERT_ROLE STREQUAL "coordinator" AND CUTEAFD_V41_EXPERT_SLICE_WIDTH STREQUAL "")
  # Draft flattened rows have independent legacy and wide covers.
  set(CUTEAFD_V41_EXPERT_ROWS ${CUTEAFD_V41_DRAFT_ROWS})
  list(JOIN CUTEAFD_V41_EXPERT_ROWS "," CUTEAFD_V41_EXPERT_ROWS_ARG)
  list(APPEND CUTEAFD_V41_EXPERT_EXPORT_ARGS --rows "${CUTEAFD_V41_EXPERT_ROWS_ARG}")
endif()
set(CUTEAFD_V41_EXPERT_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_experts")
set(CUTEAFD_V41_EXPERT_OBJECTS)
set(CUTEAFD_V41_EXPERT_HEADERS
  "${CUTEAFD_V41_EXPERT_DIR}/v41_expert_input_quant.h"
  "${CUTEAFD_V41_EXPERT_DIR}/v41_input_quant_dispatch.h")
list(APPEND CUTEAFD_V41_EXPERT_OBJECTS "${CUTEAFD_V41_EXPERT_DIR}/v41_expert_input_quant.o")
foreach(rows IN LISTS CUTEAFD_V41_EXPERT_ROWS)
  if(CUTEAFD_V41_EXPERT_SLICE_WIDTH STREQUAL "")
    set(stem "${CUTEAFD_V41_EXPERT_DIR}/v41_${CUTEAFD_V41_EXPERT_ROLE}_m${rows}")
  else()
    set(slice_width "${CUTEAFD_V41_EXPERT_SLICE_WIDTH}")
    if(slice_width MATCHES ":")
      if(NOT slice_width MATCHES "(^|,)${rows}:(64|128|192)(,|$)")
        message(FATAL_ERROR "Slice width map is missing capacity ${rows}")
      endif()
      set(slice_width "${CMAKE_MATCH_2}")
    endif()
    set(stem "${CUTEAFD_V41_EXPERT_DIR}/v41_slices_m${rows}_w${slice_width}")
  endif()
  list(APPEND CUTEAFD_V41_EXPERT_OBJECTS "${stem}.o")
  list(APPEND CUTEAFD_V41_EXPERT_HEADERS "${stem}.h")
endforeach()
add_custom_command(
  OUTPUT "${CUTEAFD_V41_EXPERT_DIR}/v41_experts.json"
    "${CUTEAFD_V41_EXPERT_DIR}/v41_expert_variants.h"
    ${CUTEAFD_V41_EXPERT_OBJECTS} ${CUTEAFD_V41_EXPERT_HEADERS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/${CUTEAFD_V41_EXPERT_EXPORT_SCRIPT}"
    --output-dir "${CUTEAFD_V41_EXPERT_DIR}" ${CUTEAFD_V41_EXPERT_EXPORT_ARGS}
  DEPENDS "${CUTEAFD_V41_CAPACITY_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_capacities.py" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/${CUTEAFD_V41_EXPERT_EXPORT_SCRIPT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_experts_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_spark_tp3_launch_geometry.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/expert_grouping.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/mxfp4_layouts.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/mxfp4-layouts.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/spark_packages.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/spark-package-ranges.json"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting native V4.1 expert kernels and scratch layouts"
  VERBATIM
)
add_custom_target(cuteafd_v41_experts_export DEPENDS
  "${CUTEAFD_V41_EXPERT_DIR}/v41_expert_variants.h"
  "${CUTEAFD_V41_EXPERT_DIR}/v41_experts.json"
  ${CUTEAFD_V41_EXPERT_OBJECTS} ${CUTEAFD_V41_EXPERT_HEADERS})
add_dependencies(cuteafd_v41_experts_export cuteafd_verify_sparkinfer_source)
set_source_files_properties(${CUTEAFD_V41_EXPERT_OBJECTS} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES ${CUTEAFD_V41_EXPERT_OBJECTS} shared/src/v41_experts.cc)
