# Both official gate geometries share runtime-row AOT entry points.
set(CUTEAFD_V41_ROUTER_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_router")
set(CUTEAFD_V41_ROUTER_OUTPUTS "${CUTEAFD_V41_ROUTER_DIR}/v41_router.json" "${CUTEAFD_V41_ROUTER_DIR}/v41_router_dispatch.h")
set(CUTEAFD_V41_ROUTER_OBJECTS)
foreach(experts IN ITEMS 128 384)
  list(APPEND CUTEAFD_V41_ROUTER_OBJECTS "${CUTEAFD_V41_ROUTER_DIR}/v41_router_e${experts}.o")
  list(APPEND CUTEAFD_V41_ROUTER_OUTPUTS "${CUTEAFD_V41_ROUTER_DIR}/v41_router_e${experts}.h")
endforeach()
list(APPEND CUTEAFD_V41_ROUTER_OUTPUTS ${CUTEAFD_V41_ROUTER_OBJECTS})
add_custom_command(
  OUTPUT ${CUTEAFD_V41_ROUTER_OUTPUTS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_router_aot.py"
    --output-dir "${CUTEAFD_V41_ROUTER_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_router_aot.py"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  VERBATIM)
add_custom_target(cuteafd_v41_router_export DEPENDS ${CUTEAFD_V41_ROUTER_OUTPUTS})
add_dependencies(cuteafd_v41_router_export cuteafd_verify_sparkinfer_source)
set_source_files_properties(${CUTEAFD_V41_ROUTER_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND CUTEAFD_NATIVE_SOURCES ${CUTEAFD_V41_ROUTER_OBJECTS} families/deepseek_v41/src/v41_router_scores.cc)
