#include "cuteafd_native.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void test_version() {
  char version[128] = {};
  assert(cuteafd_native_version(version, sizeof(version)) == CUTEAFD_STATUS_OK);
  assert(std::strstr(version, "cuteafd_native") != nullptr);
}

void test_cuda_device_info() {
  cuteafd_cuda_device_info_t info = {};
  assert(cuteafd_cuda_device_info(0, &info) == CUTEAFD_STATUS_OK);
  assert(info.device_id == 0);
  assert(info.name[0] != '\0');
}

void test_allocate_copy_free_roundtrip() {
  std::vector<unsigned char> input = {0, 1, 2, 3, 4, 5, 6, 7, 250, 251, 252, 253};
  std::vector<unsigned char> output(input.size(), 0);

  cuteafd_device_buffer_t buffer = {};
  assert(cuteafd_alloc_device_buffer(input.size(), &buffer) == CUTEAFD_STATUS_OK);
  assert(buffer.ptr != nullptr);
  assert(buffer.bytes == input.size());

  assert(cuteafd_copy_h2d(buffer, input.data(), input.size()) == CUTEAFD_STATUS_OK);
  assert(cuteafd_copy_d2h(output.data(), buffer, output.size()) == CUTEAFD_STATUS_OK);
  assert(output == input);

  assert(cuteafd_free_device_buffer(&buffer) == CUTEAFD_STATUS_OK);
  assert(buffer.ptr == nullptr);
  assert(buffer.bytes == 0);
}

void test_device_to_device_copy_roundtrip() {
  std::vector<unsigned char> input = {5, 8, 13, 21, 34, 55, 89, 144};
  std::vector<unsigned char> output(input.size(), 0);

  cuteafd_device_buffer_t src = {};
  cuteafd_device_buffer_t dst = {};
  assert(cuteafd_alloc_device_buffer(input.size(), &src) == CUTEAFD_STATUS_OK);
  assert(cuteafd_alloc_device_buffer(input.size(), &dst) == CUTEAFD_STATUS_OK);

  assert(cuteafd_copy_h2d(src, input.data(), input.size()) == CUTEAFD_STATUS_OK);
  assert(cuteafd_copy_d2d(dst, src, input.size()) == CUTEAFD_STATUS_OK);
  assert(cuteafd_copy_d2h(output.data(), dst, output.size()) == CUTEAFD_STATUS_OK);
  assert(output == input);

  assert(cuteafd_free_device_buffer(&src) == CUTEAFD_STATUS_OK);
  assert(cuteafd_free_device_buffer(&dst) == CUTEAFD_STATUS_OK);
}

void test_host_buffer_copy_roundtrip() {
  std::vector<unsigned char> input = {13, 21, 34, 55, 89, 144, 233, 255};
  std::vector<unsigned char> output(input.size(), 0);

  cuteafd_host_buffer_t host = {};
  assert(cuteafd_alloc_host_buffer(input.size(), &host) == CUTEAFD_STATUS_OK);
  assert(host.ptr != nullptr);
  assert(host.bytes == input.size());
  assert(host.flags != CUTEAFD_HOST_BUFFER_FLAG_NONE);
  std::memcpy(host.ptr, input.data(), input.size());

  cuteafd_device_buffer_t device = {};
  assert(cuteafd_alloc_device_buffer(input.size(), &device) == CUTEAFD_STATUS_OK);
  assert(cuteafd_copy_h2d(device, host.ptr, input.size()) == CUTEAFD_STATUS_OK);
  assert(cuteafd_copy_d2h(output.data(), device, output.size()) == CUTEAFD_STATUS_OK);
  assert(output == input);

  assert(cuteafd_free_device_buffer(&device) == CUTEAFD_STATUS_OK);
  assert(cuteafd_free_host_buffer(&host) == CUTEAFD_STATUS_OK);
  assert(host.ptr == nullptr);
  assert(host.bytes == 0);
  assert(host.flags == CUTEAFD_HOST_BUFFER_FLAG_NONE);
}

void test_error_propagation() {
  char tiny[4] = {};
  assert(cuteafd_native_version(tiny, sizeof(tiny)) == CUTEAFD_STATUS_BUFFER_TOO_SMALL);

  char error[128] = {};
  assert(cuteafd_last_error(error, sizeof(error)) == CUTEAFD_STATUS_OK);
  assert(std::strstr(error, "too small") != nullptr);

  assert(cuteafd_alloc_device_buffer(0, nullptr) == CUTEAFD_STATUS_INVALID_ARGUMENT);
  assert(cuteafd_last_error(error, sizeof(error)) == CUTEAFD_STATUS_OK);
  assert(std::strstr(error, "null") != nullptr);
}

void test_nccl_abi() {
  size_t unique_id_bytes = 0;
  const cuteafd_status_t size_status = cuteafd_nccl_unique_id_bytes(&unique_id_bytes);
  assert(size_status == CUTEAFD_STATUS_OK || size_status == CUTEAFD_STATUS_NCCL_UNAVAILABLE);
  if (size_status == CUTEAFD_STATUS_NCCL_UNAVAILABLE) {
    assert(unique_id_bytes == 0);
    return;
  }
  assert(unique_id_bytes > 0);
  std::vector<unsigned char> unique_id(unique_id_bytes, 0);
  assert(cuteafd_nccl_get_unique_id(unique_id.data(), unique_id.size()) == CUTEAFD_STATUS_OK);
}

void write_le16(std::vector<unsigned char>& bytes, size_t offset, uint16_t value) {
  bytes[offset] = static_cast<unsigned char>(value & 0xff);
  bytes[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xff);
}

void write_le32(std::vector<unsigned char>& bytes, size_t offset, uint32_t value) {
  for (size_t idx = 0; idx < 4; ++idx) {
    bytes[offset + idx] = static_cast<unsigned char>((value >> (idx * 8)) & 0xff);
  }
}

void write_le64(std::vector<unsigned char>& bytes, size_t offset, uint64_t value) {
  for (size_t idx = 0; idx < 8; ++idx) {
    bytes[offset + idx] = static_cast<unsigned char>((value >> (idx * 8)) & 0xff);
  }
}

std::vector<unsigned char> protocol_v2_frame(uint16_t kind, size_t payload_bytes) {
  constexpr size_t kHeaderBytes = 96;
  std::vector<unsigned char> frame(kHeaderBytes + payload_bytes, 0);
  const unsigned char magic[8] = {'D', 'S', '4', '1', 'R', 'T', 'E', '3'};
  std::memcpy(frame.data(), magic, sizeof(magic));
  write_le16(frame, 8, 3);
  write_le16(frame, 10, kind);
  write_le32(frame, 12, static_cast<uint32_t>(kHeaderBytes));
  write_le64(frame, kind == 1 ? 76 : 60, static_cast<uint64_t>(frame.size()));
  for (size_t idx = kHeaderBytes; idx < frame.size(); ++idx) {
    frame[idx] = static_cast<unsigned char>((idx * 17 + kind) & 0xff);
  }
  return frame;
}

void test_rdma_abi() {
  unsigned char endpoint_byte = 0;
  assert(cuteafd_rdma_rc_endpoint_post_recv_at(nullptr, 0, 1, 1) ==
         CUTEAFD_STATUS_INVALID_ARGUMENT);
  assert(cuteafd_rdma_rc_endpoint_send_at(nullptr, &endpoint_byte, 0, 1, 1) ==
         CUTEAFD_STATUS_INVALID_ARGUMENT);
  assert(cuteafd_rdma_rc_endpoint_send_parts_at(nullptr, &endpoint_byte, 1,
                                               &endpoint_byte, 1, 0, 1) ==
         CUTEAFD_STATUS_INVALID_ARGUMENT);
  assert(cuteafd_rdma_rc_endpoint_copy_recv_at(nullptr, &endpoint_byte, 1, 0, 1) ==
         CUTEAFD_STATUS_INVALID_ARGUMENT);

  cuteafd_rdma_device_info_t info = {};
  assert(cuteafd_rdma_device_info(&info) == CUTEAFD_STATUS_OK);
  assert(info.first_device_name[0] != '\0');
  assert(info.status[0] != '\0');

  std::vector<unsigned char> buffer(12288, 0);
  cuteafd_rdma_host_buffer_plan_t plan = {};
  assert(cuteafd_rdma_plan_host_buffer_registration(buffer.data(), buffer.size(), 4096, &plan) ==
         CUTEAFD_STATUS_OK);
  assert(plan.original_bytes == buffer.size());
  assert(plan.alignment == 4096);
  assert(plan.registered_span_bytes >= buffer.size());
  assert(plan.registered_span_bytes % 4096 == 0);
  assert(plan.span_aligned == 1);

  cuteafd_rdma_register_probe_t probe = {};
  const cuteafd_status_t status =
      cuteafd_rdma_register_host_buffer_probe(buffer.data(), buffer.size(), &probe);
  assert(status == CUTEAFD_STATUS_OK || status == CUTEAFD_STATUS_RDMA_UNAVAILABLE);
  if (status == CUTEAFD_STATUS_OK) {
    assert(probe.registered == 1);
    assert(probe.bytes == buffer.size());
    assert(probe.device_name[0] != '\0');
  }

  cuteafd_rdma_rc_qp_probe_t qp = {};
  const cuteafd_status_t qp_status = cuteafd_rdma_create_rc_qp_probe(1, 16, 16, 1, &qp);
  assert(qp_status == CUTEAFD_STATUS_OK || qp_status == CUTEAFD_STATUS_RDMA_UNAVAILABLE);
  assert(qp.port_num == 1);
  assert(qp.requested_send_wr == 16);
  assert(qp.requested_recv_wr == 16);
  assert(qp.requested_max_sge == 1);
  if (qp_status == CUTEAFD_STATUS_OK) {
    assert(qp.created == 1);
    assert(qp.qp_num != 0);
    assert(qp.actual_max_send_wr >= qp.requested_send_wr);
    assert(qp.actual_max_recv_wr >= qp.requested_recv_wr);
    assert(qp.actual_max_send_sge >= qp.requested_max_sge);
    assert(qp.actual_max_recv_sge >= qp.requested_max_sge);
    assert(qp.device_name[0] != '\0');
    assert(qp.status[0] != '\0');
  }

  cuteafd_rdma_rc_send_recv_probe_t loopback = {};
  const cuteafd_status_t loopback_status =
      cuteafd_rdma_rc_send_recv_loopback_probe(1, 12288, &loopback);
  assert(loopback_status == CUTEAFD_STATUS_OK ||
         loopback_status == CUTEAFD_STATUS_RDMA_UNAVAILABLE);
  assert(loopback.port_num == 1);
  assert(loopback.bytes == 12288);
  if (loopback_status == CUTEAFD_STATUS_OK) {
    assert(loopback.completed == 1);
    assert(loopback.payload_matches == 1);
    assert(loopback.sender_qp_num != 0);
    assert(loopback.receiver_qp_num != 0);
    assert(loopback.send_completions == 1);
    assert(loopback.recv_completions == 1);
    assert(loopback.poll_iterations > 0);
    assert(loopback.device_name[0] != '\0');
    assert(loopback.status[0] != '\0');
  }

  const std::vector<unsigned char> request_frame = protocol_v2_frame(1, 12288);
  const std::vector<unsigned char> response_frame = protocol_v2_frame(2, 12288);
  cuteafd_rdma_rc_protocol_v2_loopback_probe_t protocol_loopback = {};
  const cuteafd_status_t protocol_loopback_status =
      cuteafd_rdma_rc_protocol_v2_loopback_probe(1, request_frame.data(), request_frame.size(),
                                               response_frame.data(), response_frame.size(),
                                               &protocol_loopback);
  assert(protocol_loopback_status == CUTEAFD_STATUS_OK ||
         protocol_loopback_status == CUTEAFD_STATUS_RDMA_UNAVAILABLE);
  assert(protocol_loopback.port_num == 1);
  assert(protocol_loopback.request_bytes == request_frame.size());
  assert(protocol_loopback.response_bytes == response_frame.size());
  if (protocol_loopback_status == CUTEAFD_STATUS_OK) {
    assert(protocol_loopback.completed == 1);
    assert(protocol_loopback.request_payload_matches == 1);
    assert(protocol_loopback.response_payload_matches == 1);
    assert(protocol_loopback.client_qp_num != 0);
    assert(protocol_loopback.server_qp_num != 0);
    assert(protocol_loopback.send_completions == 2);
    assert(protocol_loopback.recv_completions == 2);
    assert(protocol_loopback.poll_iterations > 0);
    assert(protocol_loopback.device_name[0] != '\0');
    assert(protocol_loopback.status[0] != '\0');
  }
}

}  // namespace

int main() {
  test_version();
  test_cuda_device_info();
  test_allocate_copy_free_roundtrip();
  test_device_to_device_copy_roundtrip();
  test_host_buffer_copy_roundtrip();
  test_error_propagation();
  test_nccl_abi();
  test_rdma_abi();
  std::cout << "cuteafd_native_selftest passed\n";
  return 0;
}
