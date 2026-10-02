// CPU-only fake native DSO. Its observable destructor proves actual dlclose;
// no CUDA, device, or transport implementation is linked into this fixture.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct host_buffer { void *ptr; size_t bytes; uint64_t flags; };
static char *event_path;
static int pack_status, drain_status;

static void event(char value) {
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
}

int cuteafd_alloc_device_buffer(size_t bytes, struct device_buffer *out) {
    out->ptr = malloc(bytes);
    if (!out->ptr) return 1;
    out->bytes = bytes;
    out->device_id = 0;
    out->flags = 1;
    event('D');
    return 0;
}

int cuteafd_free_device_buffer(struct device_buffer *buffer) {
    event('d');
    free(buffer->ptr);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
}

int cuteafd_fp8_w8a16_pack(const void *source, void *packed, void *scale,
                         int n, int k, int rule, void *stream) {
    (void)source; (void)packed; (void)scale; (void)n; (void)k;
    (void)rule; (void)stream;
    event('P');
    return pack_status;
}

int cuteafd_cuda_stream_synchronize(void *stream) {
    (void)stream;
    event('S');
    return drain_status;
}

int cuteafd_last_error(char *out, size_t bytes) {
    snprintf(out, bytes, "injected CPU fixture failure");
    return 0;
}
