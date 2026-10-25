#pragma once
#include <cstdint>

namespace cuteafd {
// One startup byte, not a DSCP number. Reject malformed values rather than
// silently sending one side of a fabric connection into the lossy queue.
inline bool rdma_traffic_class(const char* text, uint8_t* result) {
  unsigned value = 0;
  if (text != nullptr && *text != '\0') {
    for (const char* p = text; *p; ++p) {
      if (*p < '0' || *p > '9') return false;
      value = value * 10 + static_cast<unsigned>(*p - '0');
      if (value > 255) return false;
    }
  }
  *result = static_cast<uint8_t>(value);
  return true;
}
}  // namespace cuteafd
