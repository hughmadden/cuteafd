# Spark TP2/TP3/TP6 expert shards (native FP8 K32 family, SM121).
#
# This option is empty for a bare native configure, so such a build compiles and
# links no extra object or symbol and behaves exactly as the historical Spark TP4
# shard. The release path is universal: `./build.sh` resolves the default
# `tp2;tp3;tp6` (subset only via CUTEAFD_RELEASE_SPARK_TP_ROLES) and passes it
# through scripts/build/build-release-artifacts.sh, so each listed role is exported
# ahead of runtime and compiled as one distinct symbol family per degree into the
# same libcuteafd_native.so.
#
# The exporter runs on the target SM121 device and pre-compiles capacities
# from CUTEAFD_V41_TARGET_ROWS ahead of runtime. The TP degree is a plan-time role
# property; live row counts are never part of the AOT compile key. TP6 is pure
# tensor parallelism over the official intermediate 2304: six 384-wide ranks
# with no storage padding (kernel_intermediate == intermediate == 384).
if(NOT CUTEAFD_ENABLE_V41_EXPERT_AOT OR NOT CUTEAFD_V41_EXPERT_ROLE STREQUAL "spark")
  message(FATAL_ERROR "Spark TP2/TP3/TP6 experts require the native SM121 Spark expert build")
endif()

set(CUTEAFD_V41_SPARK_TP_SELECTED)
foreach(tp IN LISTS CUTEAFD_SPARK_TP_ROLES)
  if(tp MATCHES "^tp[2-8]$")
    string(REGEX REPLACE "^tp" "" world "${tp}")
    set(artifacts_var "CUTEAFD_NATIVE_ARTIFACTS_v41_${world}")
    set(role "spark_tp${world}")
    if(NOT role IN_LIST ${artifacts_var})
      continue() # Adjacent-width objects are built by shared/mxfp4_widths.cmake.
    endif()
    if(tp IN_LIST CUTEAFD_V41_SPARK_TP_SELECTED)
      message(FATAL_ERROR "CUTEAFD_SPARK_TP_ROLES lists ${tp} more than once")
    endif()
    list(APPEND CUTEAFD_V41_SPARK_TP_SELECTED "${tp}")
  else()
    message(FATAL_ERROR "CUTEAFD_SPARK_TP_ROLES accepts tp2 through tp8, got ${tp}")
  endif()
endforeach()

set(CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS)
set(CUTEAFD_V41_SPARK_TP_EXPERT_INCLUDE_DIRS)
set(CUTEAFD_V41_SPARK_TP_EXPERT_ROWS ${CUTEAFD_V41_TARGET_ROWS})
set(CUTEAFD_V41_SPARK_TP_EXPERT_ROWS_ARG "${CUTEAFD_V41_TARGET_ROWS_ARG}")
# Derive defaults on every configure; cached maps can retain retired capacities.
# Explicit CUTEAFD_V41_SPARK_TP{2,3,6,8}_SLICE_WIDTH_OVERRIDE values select a
# scalar 64/128/192 or full capacity:width map. The exporter validates coverage
# and tiling, so invalid explicit overrides still fail rather than being repaired.
set(default_width_map)
foreach(rows IN LISTS CUTEAFD_V41_TARGET_ROWS)
  list(APPEND default_width_map "${rows}:192")
endforeach()
list(JOIN default_width_map "," default_width_map)
foreach(world 2 3 6 8)
  set(width_var "CUTEAFD_V41_SPARK_TP${world}_SLICE_WIDTH")
  set(${width_var} "${default_width_map}")
  if(DEFINED ${width_var}_OVERRIDE)
    set(${width_var} "${${width_var}_OVERRIDE}")
  endif()
endforeach()

foreach(tp IN LISTS CUTEAFD_V41_SPARK_TP_SELECTED)
  if(tp STREQUAL "tp2")
    set(role spark_tp2)
    set(width_map "${CUTEAFD_V41_SPARK_TP2_SLICE_WIDTH}")
    set(wrapper_src families/deepseek_v41/src/v41_spark_tp2_experts.cc)
    set(variant_header v41_spark_tp2_expert_variants.h)
  elseif(tp STREQUAL "tp3")
    set(role spark_tp3)
    set(width_map "${CUTEAFD_V41_SPARK_TP3_SLICE_WIDTH}")
    set(wrapper_src families/deepseek_v41/src/v41_spark_tp3_experts.cc)
    set(variant_header v41_spark_tp3_expert_variants.h)
  elseif(tp STREQUAL "tp8")
    set(role spark_tp8)
    set(width_map "${CUTEAFD_V41_SPARK_TP8_SLICE_WIDTH}")
    set(wrapper_src families/deepseek_v41/src/v41_spark_tp8_experts.cc)
    set(variant_header v41_spark_tp8_expert_variants.h)
  else()
    set(role spark_tp6)
    set(width_map "${CUTEAFD_V41_SPARK_TP6_SLICE_WIDTH}")
    set(wrapper_src families/deepseek_v41/src/v41_spark_tp6_experts.cc)
    set(variant_header v41_spark_tp6_expert_variants.h)
  endif()
  # Content-stable per-role stamp. The output stems are constant
  # `v41_{role}_m{capacity}`, so a width change must still invalidate an existing
  # export. `file(GENERATE)` rewrites the stamp only when its content changes,
  # giving Make generators (which track dependency timestamps rather than command
  # strings) the same re-export trigger Ninja gets from the changed command line,
  # while an unchanged configure leaves the timestamp alone and does not churn the
  # AOT objects. The atomic threshold is part of the stamped identity so it cannot
  # be lost when a width map is overridden.
  set(width_stamp "${CMAKE_CURRENT_BINARY_DIR}/v41_${role}_width.stamp")
  file(GENERATE OUTPUT "${width_stamp}"
    CONTENT "role=${role}\nwidth=${width_map}\natomic_min_capacity=none\n")
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/v41_${role}_experts")
  set(objects)
  set(headers)
  foreach(rows IN LISTS CUTEAFD_V41_SPARK_TP_EXPERT_ROWS)
    set(stem "${dir}/v41_${role}_m${rows}")
    list(APPEND objects "${stem}.o")
    list(APPEND headers "${stem}.h")
  endforeach()
  set(json "${dir}/v41_experts.json")
  add_custom_command(
    OUTPUT "${json}" "${dir}/${variant_header}" ${objects} ${headers}
    COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
      ${CUTEAFD_EXPORT_LAUNCHER} "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      --output-dir "${dir}" --role "${role}"
      --rows "${CUTEAFD_V41_SPARK_TP_EXPERT_ROWS_ARG}"
      --width "${width_map}"
      --standard-names
    DEPENDS "${CUTEAFD_V41_CAPACITY_SOURCE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_capacities.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_slices_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/v41_spark_tp3_launch_geometry.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_v41_experts_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_src}"
      "${width_stamp}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/expert_grouping.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/spark_packages.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/spark-package-ranges.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/lib/mxfp4_layouts.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build/mxfp4-layouts.json"
      ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ${role} Spark expert kernels"
    VERBATIM
  )
  add_custom_target(cuteafd_v41_${role}_experts_export DEPENDS
    "${dir}/${variant_header}" "${json}" ${objects} ${headers})
  add_dependencies(cuteafd_v41_${role}_experts_export cuteafd_verify_sparkinfer_source)
  set_source_files_properties(${objects} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_V41_SPARK_TP_EXPERT_INCLUDE_DIRS "${dir}")
  list(APPEND CUTEAFD_V41_SPARK_TP_EXPERT_TARGETS "cuteafd_v41_${role}_experts_export")
  list(APPEND CUTEAFD_NATIVE_SOURCES ${objects} "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_src}")
endforeach()
