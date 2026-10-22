// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../../../native/shared/cuda/expert_pack_layout.h"

struct host_buffer { void *ptr; size_t bytes; uint64_t flags; };
static char *event_path;
static int quiet_events;
void fixture_quiet_events(int quiet) { quiet_events = quiet; }
static int pack_status, drain_status, drain_calls, drain_after;
/* Per thread, as CUDA's current device: two GPU loader threads bind their own. */
static _Thread_local int current_device;
static int bind_after = -1, bind_calls;
void fixture_configure_bind_after(int after) { bind_after = after; bind_calls = 0; }
static size_t memory_total[8], memory_used[8], graph_bytes;
static int immediate_copies;

void fixture_configure_immediate_copies(void) { immediate_copies = 1; }

void fixture_configure_memory(int device, size_t total, size_t used, size_t graph) {
    memory_total[device] = total;
    memory_used[device] = used;
    graph_bytes = graph;
}

struct device_info {
    int device_id, cuda_available, compute_capability_major, compute_capability_minor;
    int integrated, can_map_host_memory, unified_addressing;
    uint64_t total_memory_bytes;
    char name[128], driver_version[64], runtime_version[64];
};
int cuteafd_cuda_device_info(int device, struct device_info *info) {
    memset(info, 0, sizeof(*info));
    info->device_id = device;
    info->cuda_available = 1;
    info->compute_capability_major = 12;
    info->compute_capability_minor = 0;
    info->total_memory_bytes = memory_total[device];
    return 0;
}

int cuteafd_cuda_get_device(int *device) { *device = current_device; return 0; }
int cuteafd_cuda_memory_info(size_t *free_bytes, size_t *total) {
    *total = memory_total[current_device];
    *free_bytes = *total - memory_used[current_device];
    return 0;
}

static void event(char value) {
    if (quiet_events) return;
    FILE *file = fopen(event_path, "a");
    if (!file || fputc(value, file) == EOF || fclose(file)) abort();
}

void fixture_set_event_path(const char *path) {
    event_path = strdup(path);
    if (!event_path) abort();
}

__attribute__((destructor)) static void fixture_unloaded(void) {
    if (event_path) {
        event('U');
        free(event_path);
    }
}

int cuteafd_rdma_rc_endpoint_try_poll(void *handle, uint32_t sends,
                                    uint32_t recvs, void *out) {
    (void)handle; (void)sends; (void)recvs; (void)out;
    return 0;
}

int cuteafd_set_expert_hidden(uint32_t hidden) { (void)hidden; return 0; }

int cuteafd_alloc_host_buffer(size_t bytes, struct host_buffer *out) {
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->flags = 1;
    event('A');
    return 0;
}

int cuteafd_free_host_buffer(struct host_buffer *buffer) {
    event('F');
    free(buffer->ptr);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

struct device_buffer { void *ptr; size_t bytes; int device_id; uint64_t flags; };

void fixture_configure_pack(int pack, int drain) {
    pack_status = pack;
    drain_status = drain;
    drain_calls = 0;
    drain_after = 0;
}

void fixture_configure_drain_after(int after) { drain_after = after; }

int cuteafd_alloc_device_buffer(size_t bytes, struct device_buffer *out) {
    if (posix_memalign(&out->ptr, 256, bytes)) return 1;
    /* Zeroed so operand digests never read unspecified padding. */
    memset(out->ptr, 0, bytes);
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 1;
    memory_used[current_device] += bytes;
    event('D');
    return 0;
}

int cuteafd_free_device_buffer(struct device_buffer *buffer) {
    event('d');
    if (buffer->flags == 1) memory_used[buffer->device_id] -= buffer->bytes;
    free(buffer->ptr);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

int cuteafd_alloc_managed_device_buffer(size_t bytes, struct device_buffer *out) {
    // Host-resident managed pages deliberately do not reduce physical free.
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->device_id = current_device;
    out->flags = 2;
    event('D');
    return 0;
}

struct graph_exec { int device; size_t bytes; };
int cuteafd_cuda_graph_end_capture(void *stream, void **out) {
    (void)stream;
    struct graph_exec *graph = malloc(sizeof(*graph));
    if (!graph) return 1;
    graph->device = current_device;
    graph->bytes = graph_bytes;
    memory_used[current_device] += graph_bytes;
    *out = graph;
    event('G');
    return 0;
}
int cuteafd_cuda_graph_exec_destroy(void *handle) {
    struct graph_exec *graph = handle;
    memory_used[graph->device] -= graph->bytes;
    free(graph);
    event('g');
    return 0;
}

int cuteafd_fp8_w8a16_pack(const void *source, void *packed, void *scale,
                         int n, int k, int rule, void *stream) {
    (void)source; (void)packed; (void)scale; (void)n; (void)k;
    (void)rule; (void)stream;
    event('P');
    return pack_status;
}

struct upload { void *dst; const void *src; size_t bytes; };
struct upload_event { struct upload *copies; size_t count; };
static struct upload pending_uploads[65536];
static size_t pending_count;
static void finish_uploads(struct upload_event *upload) {
    for (size_t i = 0; i < upload->count; ++i) {
        struct upload *copy = &upload->copies[i];
        memcpy(copy->dst, copy->src, copy->bytes);
    }
    free(upload->copies);
    upload->copies = NULL;
    upload->count = 0;
}
int cuteafd_register_host_mapping(void *ptr, size_t bytes) {
    if (!ptr || !bytes) return 1;
    event('G'); return 0;
}
int cuteafd_unregister_host_mapping(void *ptr) {
    if (!ptr) return 1;
    event('g'); return 0;
}
int cuteafd_cuda_stream_create(void **out) {
    *out = (void *)1;
    event('V'); return 0;
}
static void *upload_events[128];
static size_t upload_event_count;

int cuteafd_cuda_event_create_ordering(void **out) {
    if (upload_event_count == 128) return 1;
    *out = calloc(1, sizeof(struct upload_event));
    if (!*out) return 1;
    upload_events[upload_event_count++] = *out;
    event('C');
    return 0;
}
int cuteafd_cuda_event_create(void **out) {
    return cuteafd_cuda_event_create_ordering(out);
}

int cuteafd_cuda_stream_wait_event(void *stream, void *handle) {
    (void)stream; (void)handle;
    event('W');
    return 0;
}

int cuteafd_cuda_zero_bytes_async(void *dst, size_t bytes, void *stream) {
    (void)stream;
    memset(dst, 0, bytes);
    return 0;
}

int cuteafd_copy_h2d_async(struct device_buffer dst, const void *source,
                         size_t bytes, void *stream);

int cuteafd_copy_d2d_async(struct device_buffer dst, struct device_buffer src,
                           size_t bytes, void *stream) {
    if (bytes > dst.bytes || bytes > src.bytes) return 1;
    if (!immediate_copies) return cuteafd_copy_h2d_async(dst, src.ptr, bytes, stream);
    if (dst.device_id != current_device) return 1;
    memcpy(dst.ptr, src.ptr, bytes);
    event('C');
    return pack_status;
}

int cuteafd_copy_d2h(void *dst, struct device_buffer src, size_t bytes) {
    if (bytes > src.bytes) return 1;
    memcpy(dst, src.ptr, bytes);
    return 0;
}

int cuteafd_copy_d2h_async(void *dst, struct device_buffer src, size_t bytes, void *stream) {
    (void)stream;
    if (!immediate_copies || bytes > src.bytes || src.device_id != current_device) return 1;
    memcpy(dst, src.ptr, bytes);
    event('J');
    return 0;
}

int cuteafd_copy_h2d_async(struct device_buffer dst, const void *source,
                         size_t bytes, void *stream) {
    (void)stream;
    if (bytes > dst.bytes) return 1;
    if (immediate_copies) {
        if (dst.device_id != current_device) return 1;
        memcpy(dst.ptr, source, bytes);
    } else {
        if (pending_count == 65536) return 1;
        pending_uploads[pending_count++] = (struct upload){dst.ptr, source, bytes};
    }
    event('Q');
    return 0;
}
int cuteafd_copy_h2d_batch_async(const struct device_buffer *dsts, const void *const *sources,
                                 const size_t *bytes, size_t count, void *stream) {
    for (size_t index = 0; index < count; ++index) {
        if (cuteafd_copy_h2d_async(dsts[index], sources[index], bytes[index], stream)) return 1;
    }
    return 0;
}
int cuteafd_cuda_event_record(void *handle, void *stream) {
    (void)stream;
    struct upload_event *upload = handle;
    if (upload->count) return 1;
    upload->copies = malloc(pending_count * sizeof(struct upload));
    if (pending_count && !upload->copies) return 1;
    if (pending_count) memcpy(upload->copies, pending_uploads, pending_count * sizeof(struct upload));
    upload->count = pending_count;
    pending_count = 0;
    event('R');
    return 0;
}
int cuteafd_cuda_event_synchronize(void *handle) {
    event('S');
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (status) return status;
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] == handle) {
            struct upload_event *upload = handle;
            finish_uploads(upload);
        }
    }
    return 0;
}

int cuteafd_cuda_event_destroy(void *handle) {
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i] == handle) { free(handle); upload_events[i] = NULL; }
    }
    event('E');
    return 0;
}

int cuteafd_cuda_stream_query(void *stream, int *ready) {
    (void)stream;
    *ready = immediate_copies;
    return 0;
}

int cuteafd_cuda_stream_synchronize(void *stream) {
    (void)stream;
    event('S');
    int status = ++drain_calls > drain_after ? drain_status : 0;
    if (status) return status;
    for (size_t i = 0; i < upload_event_count; ++i) {
        if (upload_events[i]) finish_uploads(upload_events[i]);
    }
    for (size_t i = 0; i < pending_count; ++i) {
        struct upload *copy = &pending_uploads[i];
        memcpy(copy->dst, copy->src, copy->bytes);
    }
    pending_count = 0;
    return 0;
}

int cuteafd_cuda_stream_destroy(void *stream) {
    (void)stream;
    event('T');
    return 0;
}

int cuteafd_cuda_set_device(int device) {
    if (bind_after >= 0 && bind_calls++ >= bind_after) return 1;
    current_device = device;
    return 0;
}

int cuteafd_copy_h2d(struct device_buffer dst, const void *source, size_t bytes) {
    if (bytes > dst.bytes) return 1;
    memcpy(dst.ptr, source, bytes);
    event('H');
    return 0;
}

int cuteafd_copy_d2d(struct device_buffer dst, struct device_buffer src, size_t bytes) {
    if (bytes > dst.bytes || bytes > src.bytes) return 1;
    memcpy(dst.ptr, src.ptr, bytes); return 0;
}
int cuteafd_cuda_zero_bytes(void *dst, size_t bytes) { memset(dst, 0, bytes); return 0; }
int cuteafd_zero_pitched_async(void *dst, uint64_t bytes, uint64_t pitch,
    uint64_t width, uint64_t rows, void *stream) {
    (void)stream;
    if (!dst || !width || !rows || pitch < width || width > bytes ||
        rows - 1 > (bytes - width) / pitch) return 1;
    for (uint64_t row = 0; row < rows; ++row)
        memset((uint8_t *)dst + row * pitch, 0, width);
    event('Z');
    return pack_status;
}
int cuteafd_pack_expert_plane_async(const uint8_t *source, uint64_t source_bytes,
    uint8_t *destination, uint64_t destination_bytes, uint32_t intermediate,
    uint32_t plane, void *stream) {
    int status = cuteafd_cuda_stream_synchronize(stream);
    if (status) return status;
    if (pack_status) return pack_status;
    uint32_t padded = (intermediate + 127) / 128 * 128;
    uint32_t n = plane < 2 ? 2 * padded : 5120;
    uint32_t k = plane < 2 ? 5120 : intermediate;
    int scales = plane % 2;
    uint32_t element = scales ? 1 : 4, divisor = scales ? 32 : 8;
    for (uint64_t index = 0; index < destination_bytes / element; ++index) {
        uint32_t row, col;
        cuteafd_expert_pack_coordinates(index, k, scales, &row, &col);
        uint64_t at = ((uint64_t)row * (k / divisor) + col) * element;
        if (row < n && col < k / divisor && at + element <= source_bytes)
            memcpy(destination + index * element, source + at, element);
        else memset(destination + index * element, 0, element);
    }
    event('P'); return 0;
}
int cuteafd_cuda_nvfp4_swizzle_scale_async(struct device_buffer src, struct device_buffer dst,
    size_t rows, size_t cols, void *stream) {
    int status = cuteafd_cuda_stream_synchronize(stream);
    if (status) return status;
    if (pack_status) return pack_status;
    size_t rp = (rows + 127) / 128 * 128, cp = (cols + 3) / 4 * 4;
    if (src.bytes < rows * cols || dst.bytes < rp * cp) return 1;
    for (size_t row = 0; row < rp; ++row) for (size_t col = 0; col < cp; ++col)
        ((uint8_t *)dst.ptr)[cuteafd_nvfp4_scale_offset(row, col, cp)] =
            row < rows && col < cols ? ((uint8_t *)src.ptr)[row * cols + col] : 0;
    event('P'); return 0;
}

int cuteafd_copy_d2d_2d_async(struct device_buffer dst, size_t dst_pitch,
                            struct device_buffer src, size_t src_pitch,
                            size_t width, size_t rows, void *stream) {
    (void)stream;
    for (size_t r = 0; r < rows; ++r)
        memcpy((char *)dst.ptr + r * dst_pitch,
               (const char *)src.ptr + r * src_pitch, width);
    event('C');
    return pack_status;
}

int cuteafd_fp8_grid_kmajor(const void *grid, void *kmajor, int64_t rows, int64_t kb, void *stream) {
    (void)stream;
    for (int64_t b = 0; b < kb; ++b)
        for (int64_t r = 0; r < rows; ++r)
            ((float *)kmajor)[b * rows + r] = ((const float *)grid)[(r / 128) * kb + b];
    event('K');
    return pack_status;
}

int cuteafd_fp8_block_dequant(const void *weight, const void *scale, void *out,
                            int rows, int cols, void *stream) {
    (void)weight; (void)scale; (void)out; (void)rows; (void)cols; (void)stream;
    event('B');
    return pack_status;
}

int cuteafd_fp8_row_quant_rule(const void *weight, void *out, void *scale,
                             int rows, int cols, int rule, void *stream) {
    (void)weight; (void)out; (void)scale; (void)rows; (void)cols;
    (void)rule; (void)stream;
    event('Q');
    return pack_status;
}

int cuteafd_last_error(char *out, size_t bytes) {
    snprintf(out, bytes, "injected CPU fixture failure");
    return 0;
}

/* EXL3 contexts model untracked driver residency and device-bound destruction. */
struct exl3_context { int device; size_t bytes; };
static int exl3_create(void **out, size_t bytes, char marker) {
    struct exl3_context *context = malloc(sizeof(*context));
    if (!context) return 1;
    context->device = current_device;
    context->bytes = bytes;
    memory_used[current_device] += bytes;
    *out = context;
    event(marker);
    return 0;
}
static void exl3_destroy(void *handle, char marker) {
    struct exl3_context *context = handle;
    if (context->device != current_device) abort();
    memory_used[current_device] -= context->bytes;
    event(marker);
    free(context);
}
int cuteafd_exl3_info(uint32_t *out, uint32_t count) {
    const uint32_t words[16] = {2,5120,1152,6,1,6,2,1,1,1,1,3,4,0,0,4};
    if (count != 16) return 1;
    memcpy(out, words, sizeof(words));
    return 0;
}
int cuteafd_exl3_create(void **out) { return exl3_create(out, 4096, 'K'); }
void cuteafd_exl3_destroy(void *handle) { exl3_destroy(handle, 'k'); }
int cuteafd_exl3_core(void *handle, void *const *ptrs, const int *scalars, void *stream) {
    (void)handle; (void)ptrs; (void)scalars; (void)stream; return 0;
}
int cuteafd_exl3_sum(void *handle, void *const *ptrs, const int *scalars, void *stream) {
    return cuteafd_exl3_core(handle, ptrs, scalars, stream);
}
int cuteafd_exl3_routes_create(void **out) { return exl3_create(out, 1024, 'R'); }
int cuteafd_exl3_routes_destroy(void *handle) { exl3_destroy(handle, 'r'); return 0; }
int cuteafd_exl3_routes_launch(void *handle, void *const *ptrs,
                             const uint64_t *bytes, int rows, void *stream) {
    (void)handle; (void)ptrs; (void)bytes; (void)rows; (void)stream; return 0;
}
int cuteafd_exl3_wire_initialize(void **out) { return exl3_create(out, 512, 'W'); }
void cuteafd_exl3_wire_destroy(void *handle) { exl3_destroy(handle, 'w'); }
int cuteafd_exl3_wire_decode(void *handle, const uint8_t *input, uint64_t in_bytes,
                           uint16_t *output, uint64_t out_bytes, uint32_t rows, void *stream) {
    (void)handle; (void)input; (void)in_bytes; (void)output;
    (void)out_bytes; (void)rows; (void)stream; return 0;
}
