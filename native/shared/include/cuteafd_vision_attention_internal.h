#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int32_t cuteafd_vision_attention_create(int32_t head_dim, void** owner);
int32_t cuteafd_vision_attention_destroy(void* owner);
int32_t cuteafd_vision_attention_launch(void* owner, void* q, void* k, void* v,
    void* out, void* lse, void* cu_seqlens, float scale, void* stream);
#ifdef __cplusplus
}
#endif
