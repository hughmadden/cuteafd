# DeepSeek V4 coordinator programs (SM120): b12x.integration.cuteafd exports
# launched through the generic table in src/dsv4_programs.cc.
if(NOT CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120")
  message(FATAL_ERROR "DeepSeek V4 coordinator programs require the SM120 build")
endif()
set(CUTEAFD_DSV4_DIR "${CMAKE_CURRENT_BINARY_DIR}/dsv4_programs")
set(CUTEAFD_DSV4_EXPORT_ARGS --geometry "${CUTEAFD_DSV4_GEOMETRY}"
  --decode-rows "${CUTEAFD_DSV4_DECODE_ROWS}" --prefill-rows "${CUTEAFD_DSV4_PREFILL_ROWS}"
  --max-context "${CUTEAFD_DSV4_MAX_CONTEXT}")
set(stamp "${CMAKE_CURRENT_BINARY_DIR}/dsv4_programs.stamp")
file(GENERATE OUTPUT "${stamp}" CONTENT "${CUTEAFD_DSV4_EXPORT_ARGS}\n")
# The program list lives in the exporter; objects are collected into one
# archive so CMake need not know each stem.
set(CUTEAFD_DSV4_ARCHIVE "${CUTEAFD_DSV4_DIR}/libcuteafd_dsv4_programs.a")
add_custom_command(
  OUTPUT "${CUTEAFD_DSV4_DIR}/dsv4_programs.h" "${CUTEAFD_DSV4_DIR}/dsv4_programs.json"
    "${CUTEAFD_DSV4_ARCHIVE}"
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CUTEAFD_DSV4_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_dsv4_aot.py"
    --output-dir "${CUTEAFD_DSV4_DIR}" ${CUTEAFD_DSV4_EXPORT_ARGS}
  COMMAND sh -c "${CMAKE_AR} qcs '${CUTEAFD_DSV4_ARCHIVE}' '${CUTEAFD_DSV4_DIR}'/*.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_dsv4_aot.py" "${stamp}"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting DeepSeek V4 coordinator programs"
  VERBATIM)
add_custom_target(cuteafd_dsv4_programs_export DEPENDS
  "${CUTEAFD_DSV4_DIR}/dsv4_programs.h" "${CUTEAFD_DSV4_ARCHIVE}")
add_dependencies(cuteafd_dsv4_programs_export cuteafd_verify_sparkinfer_source)
set_source_files_properties("${CUTEAFD_DSV4_DIR}/dsv4_programs.h" PROPERTIES GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES src/dsv4_programs.cc)
