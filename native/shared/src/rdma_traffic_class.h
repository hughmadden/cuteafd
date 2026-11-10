#pragma once
#include <cstdint>

namespace cuteafd {
// GRH traffic class byte for every RC QP: DSCP 26 (RoCEv2), ECN ECT(0), so an
// ECN-marking switch signals congestion even with NIC congestion control off.
inline constexpr uint8_t kRdmaTrafficClass = 26 << 2 | 0b10;
}  // namespace cuteafd
