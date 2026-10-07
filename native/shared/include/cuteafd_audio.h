#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Qualification numerics1: checkpoint matrices expanded to FP32, FP32 RVQ,
 * native cuFFT/cuBLAS/CuTe. Not a claim of mixed-BF16 qualification. Every
 * weight offset is256-byte aligned within the admitted resident arena. */
#define CUTEAFD_AUDIO_ABI 1
#define CUTEAFD_AUDIO_NO_OFFSET UINT64_MAX

typedef struct {
  uint64_t q, qb, k, v, vb, o, ob;
  uint64_t norm1, norm1b, norm2, norm2b, fc1, fc1b, fc2, fc2b;
} cuteafd_audio_codec_block;
typedef struct {
  uint64_t norm1, norm2, q, qb, k, kb, v, vb, o, gate, up, down;
} cuteafd_audio_patch_block;
typedef struct {
  uint32_t abi_version, numerics, max_samples, output_width;
  uint64_t weight_bytes;
  uint64_t conv1, conv1b, conv2, conv2b, downsample;
  uint64_t norm, normb, downnorm, downnormb;
  cuteafd_audio_codec_block codec[24];
  uint64_t codebooks[20], speech[20];
  cuteafd_audio_patch_block patch[6];
  uint64_t patch_norm, projection1, projection2;
} cuteafd_audio_spec;
typedef struct {
  uint64_t weights, scratch, blas_workspace, fft_workspace;
  uint64_t device_allocations, encodes;
} cuteafd_audio_ledger;

/* Plan query creates no device context/allocation. Its conservative FFT bound
 * must be admitted before create; create checks the real cuFFT workspace and
 * fails closed if it exceeds that bound. Owner is serialized on the SAME
 * dedicated encoder thread as vision. All work drains on error and destroy. */
int32_t cuteafd_audio_required(const cuteafd_audio_spec*, cuteafd_audio_ledger*);
int32_t cuteafd_audio_create(const cuteafd_audio_spec*, int32_t device,
                            uint64_t admitted_bytes, void** owner);
/* All four table pointers NULL select the attested build-time tables. A full
 * non-NULL set is a qualification-only oracle override; partial sets fail.
 * Upload synchronizes before returning, so host inputs may then be released. */
int32_t cuteafd_audio_upload(void*, const void* weights, uint64_t weight_bytes,
                            const float* hann, const float* mel_filterbank,
                            const float* codec_rotary, const float* patch_rotary);
/* Host PCM24000Hz mono finiteF32,481..max_samples. Host outputs: F32 LM rows,
 * optional int32 RVQ[frames,20]. Extents are exact; no encode-time CUDA
 * allocation, library-plan creation, graph capture or kernel resolution. */
/* Probe-only observer gets borrowed device FP32 rows after a stream drain:
 * 0 mel, 1/2 post-GELU convs, 3..26 codec blocks, 27 tokenizer norm,
 * 28 pre-RVQ, 29 speech sum, 30 local-transformer norm, 31 projection.
 * segment_frame is the segment's global mel offset (zero for patch stages).
 * It must not retain data after returning and must propagate failures. */
typedef int32_t (*cuteafd_audio_observer)(void*, int32_t stage, int32_t segment_frame,
                                        const void* data, int32_t rows, int32_t width);
int32_t cuteafd_audio_encode(void*, const float* pcm, uint32_t samples,
                            float* rows, uint64_t row_bytes,
                            int32_t* codes, uint64_t code_bytes,
                            cuteafd_audio_observer observer, void* observer_context);
int32_t cuteafd_audio_get_ledger(void*, cuteafd_audio_ledger*);
int32_t cuteafd_audio_backend(char* out, uint64_t capacity);
int32_t cuteafd_audio_destroy(void*);
#ifdef __cplusplus
}
#endif
