if(NOT CUTEAFD_ENABLE_CUDA)
  message(FATAL_ERROR "Native audio AOT requires CUDA")
endif()
if(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^120(f|a)?$")
  set(CUTEAFD_AUDIO_CAPABILITY 120)
elseif(CUTEAFD_CUDA_ARCHITECTURES MATCHES "^121(a)?$")
  set(CUTEAFD_AUDIO_CAPABILITY 121)
else()
  message(FATAL_ERROR "Native audio AOT requires SM120 or SM121")
endif()
set(CUTEAFD_AUDIO_DIR "${CMAKE_CURRENT_BINARY_DIR}/audio_support")
set(CUTEAFD_AUDIO_STEMS
  layer_norm_1024_1_1 rms_norm_1024_1_1 sum_square_1024_1_1
  softmax_1024_1_1 rvq_select_1024_1_1 frame_960_1_1 magnitude_481_1_1 log_128_1_1
  im2col_384_3_1 im2col_3072_3_2 im2col_2048_2_2
  bias_1024_1_1 bias_4096_1_1 gelu_1024_1_1 gelu_16384_1_1
  bias_gelu_1024_1_1 bias_gelu_4096_1_1 silu_product_4096_1_1 add_1024_1_1
  rope_pack_1024_1_1 pack_heads_1024_1_1 unpack_heads_1024_1_1 speech_add_1024_1_1)
set(CUTEAFD_AUDIO_OUTPUTS
  "${CUTEAFD_AUDIO_DIR}/audio_support.json"
  "${CUTEAFD_AUDIO_DIR}/audio_support.cc"
  "${CUTEAFD_AUDIO_DIR}/cuteafd_audio_support_internal.h"
  "${CUTEAFD_AUDIO_DIR}/audio_tables.h"
  "${CUTEAFD_AUDIO_DIR}/audio_identity.h")
foreach(stem IN LISTS CUTEAFD_AUDIO_STEMS)
  foreach(ext h o)
    list(APPEND CUTEAFD_AUDIO_OUTPUTS "${CUTEAFD_AUDIO_DIR}/audio_${stem}.${ext}")
  endforeach()
  set_source_files_properties("${CUTEAFD_AUDIO_DIR}/audio_${stem}.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND CUTEAFD_NATIVE_SOURCES "${CUTEAFD_AUDIO_DIR}/audio_${stem}.o")
endforeach()
add_custom_command(
  OUTPUT ${CUTEAFD_AUDIO_OUTPUTS}
  COMMAND ${CUTEAFD_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${CUTEAFD_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_audio_aot.py"
    --output-dir "${CUTEAFD_AUDIO_DIR}" --capability ${CUTEAFD_AUDIO_CAPABILITY}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/aot/export_b12x_audio_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/cuda/audio_runtime.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/include/cuteafd_audio.h"
    ${CUTEAFD_SPARKINFER_PROVENANCE_INPUTS} ${CUTEAFD_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting qualification-only FP32 audio support offline"
  VERBATIM)
set_source_files_properties("${CUTEAFD_AUDIO_DIR}/audio_support.cc" PROPERTIES GENERATED TRUE)
add_custom_target(cuteafd_audio_export DEPENDS ${CUTEAFD_AUDIO_OUTPUTS})
add_dependencies(cuteafd_audio_export cuteafd_verify_sparkinfer_source)
list(APPEND CUTEAFD_NATIVE_SOURCES shared/cuda/audio_runtime.cu "${CUTEAFD_AUDIO_DIR}/audio_support.cc")
