#pragma once
#include <cstdint>

namespace cuteafd {
// GRH traffic class byte for every RC QP: DSCP 26 (RoCEv2), ECN Not-ECT.
inline constexpr uint8_t kRdmaTrafficClass = 26 << 2;
}  // namespace cuteafd
