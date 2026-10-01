#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Two-GPU exchange over peer memory: SM-issued pushes with a release flag and
// a spinning acquire wait, no host involvement, graph-capturable (sequence
// numbers live in device memory, so replays advance them).
//
// A link is one direction A -> B: B owns `flag` (written by A's pushes) and
// `recv` (the payload); A owns `send_state` (u32 [2]: sequence, arrival
// counter); B owns `recv_state` (u32 [1]: last sequence waited for). All zeroed
// before first use. Peer access to B must be enabled on A.

// On A's stream: copy `bytes` (16-byte aligned, at most 2^40) from local
// `source` to peer `destination`, then publish the next sequence to the peer
// `flag` once every block's stores are visible system-wide. `blocks` 0 picks
// a default. The source must not be rewritten until the peer has waited.
int32_t cuteafd_peer_push_signal(void* destination, const void* source, uint64_t bytes,
    uint32_t* flag, uint32_t* send_state, uint32_t blocks, void* stream);
// On B's stream: one warp spins until `flag` reaches the next expected
// sequence (acquire), so later work on the stream sees the pushed bytes.
int32_t cuteafd_peer_wait(const uint32_t* flag, uint32_t* recv_state, void* stream);

// P2P probe for `cuteafd fabric --p2p`. Runs one measurement between devices
// `a` and `b` (peer access enabled both ways by the call) and writes the time
// per operation in microseconds (median of 5 repeats of `iterations` ops):
//   0 copy engine A->B one-way (cudaMemcpyAsync on B's stream)
//   1 SM pull A->B one-way (kernel on B reads A)
//   2 SM push A->B one-way (kernel on A writes B)
//   3 pinned host bounce A->host->B one-way (host-timed, includes a sync)
//   4 copy engine ping-pong with cross-device events (half round trip)
//   5 SM push ping-pong with cross-device events (half round trip)
//   6 SM push + flag ping-pong, eager launches (half round trip)
//   7 SM push + flag ping-pong, one CUDA graph per device (half round trip)
//   8 SM push + flag exchange (both push, both wait), eager (per round)
//   9 SM push + flag exchange, graphs (per round)
//  10 copy engine ping-pong with events, one multi-device graph (half round trip)
// `ingress` bit 0/1 keeps host->device DMA copies streaming into A/B during
// the measurement (a proxy for NIC->GPU ingress over the same PCIe links).
// `blocks` sizes the SM kernels (0: default). Returns a cudaError_t value.
int32_t cuteafd_p2p_probe(int32_t a, int32_t b, uint64_t bytes, int32_t test, uint32_t ingress,
    uint32_t iterations, uint32_t blocks, double* microseconds);
#ifdef __cplusplus
}
#endif
