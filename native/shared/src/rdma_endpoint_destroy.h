#pragma once

namespace cuteafd {

// Registrations and their backing storage cannot retire while a QP can still
// write them. Keep this boundary testable without a live verbs device.
template <typename Endpoint, typename DestroyQp, typename Release>
bool destroy_rdma_endpoint(Endpoint* endpoint, DestroyQp destroy_qp, Release release) {
  if (endpoint == nullptr) return true;
  if (endpoint->qp != nullptr) {
    if (destroy_qp(endpoint->qp) != 0) return false;
    endpoint->qp = nullptr;
  }
  release(endpoint);
  return true;
}

}  // namespace cuteafd
