# Weight-variant row programs: steering y -= A (B^T y) (k_max == 1 and
# k_max <= 32, behind cuteafd_steer_rows) and the FP32 delta add y += d
# (cuteafd_add_rows). All shape-free SparkInfer exports.
if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "Steer-rows AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120(f|a)?$")
  set(CUTEAFD_STEER_ROWS_CAPABILITY 120)
elseif(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^121(a)?$")
  set(CUTEAFD_STEER_ROWS_CAPABILITY 121)
else()
  message(FATAL_ERROR "Steer-rows AOT requires SM120 or SM121")
endif()
set(CUTEAFD_STEER_ROWS_DIR "${CMAKE_CURRENT_BINARY_DIR}/steer_rows")
set(CUTEAFD_STEER_ROWS_OUTPUTS)
foreach(stem steer_rows_one steer_rows_general add_rows_f32)
  foreach(ext json h o)
    list(APPEND CUTEAFD_STEER_ROWS_OUTPUTS "${CUTEAFD_STEER_ROWS_DIR}/${stem}.${ext}")
  endforeach()
  set_source_files_properties("${CUTEAFD_STEER_ROWS_DIR}/${stem}.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_STEER_ROWS_DIR}/${stem}.o")
endforeach()
add_custom_command(
  OUTPUT ${CUTEAFD_STEER_ROWS_OUTPUTS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_steer_rows_aot.py"
    --output-dir "${CUTEAFD_STEER_ROWS_DIR}" --capability ${CUTEAFD_STEER_ROWS_CAPABILITY}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_steer_rows_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting weight-variant row programs offline"
  VERBATIM)
add_custom_target(cuteafd_steer_rows_export DEPENDS ${CUTEAFD_STEER_ROWS_OUTPUTS})
add_dependencies(cuteafd_steer_rows_export cuteafd_verify_sparkinfer_source)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/steer_rows_aot.cc)
