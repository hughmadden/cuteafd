// kx7_decode.cuh: the reference device-side KX7 stream decode (format and packer in
// cuteafd_core::kx7; v1.1 = lane16, K split 4, 8-row ragged interleave, two-class code), shared
// by the KX7 kernels (kx7.cu's decode, and the GEMV and tile kernels that will fuse it). Kept
// line for line as the reference GEMV decodes: change both together.
//
// A lane owns one stream (lane16 layout, one K slice): 16 symbols per 64-wide K chunk. Its
// stream bytes are fetched 16 bytes at a time (ld.global.nc.L1::no_allocate; one quad in flight
// in registers, prefetch.global.L2 further ahead) and parked in a per-lane ring in shared
// memory laid out word-interleaved across the 32 lanes of a warp (word w of lane l at
// ring[(w % RING_WORDS) * 32 + l]) so every ring access is bank-conflict free; rows RING_WORDS
// and RING_WORDS+1 mirror rows 0 and 1 so a 3-word window never wraps. Decoding works on a
// 64-bit register window rebuilt from the ring once per 4 symbols (3 LDS + 2 funnel shifts);
// each symbol is: peek 16 bits, primary table (2^PBITS entries, shared memory), rare secondary
// table (codes longer than PBITS; variable-size sub-tables, also in shared memory), consume
// `len` bits. kx7_decode16_multi decodes G streams of one lane in lockstep so their dependent
// chains overlap. The 16 decoded symbols of a chunk are packed as 8 bf16x2 words in K order,
// exactly the words a BF16 kernel loads from the same 32 bytes of the weight row.
//
// Table entry (u32): leaf = (symbol << 16) | len (1..16); node = (index << 16) | (shr << 8) | 0xFF,
// where index is the sub-table's position in the table (primary entries included) and the
// sub-table has 2^(32-shr) entries indexed by the bits that follow the PBITS prefix:
// sub = (peek32 << PBITS) >> shr.
// Two-class variant (TWOCLASS = true; the tool's --code twoclass): every code is at most PBITS
// bits except the rare symbols, coded as ESC (Le bits) + R raw bits of rank; the primary table
// resolves everything in one lookup (ESC entry = (0xFF << 8) | (Le + R)) and the rank indexes a
// u16 table of 2^R patterns that follows the primary table in shared memory.
#pragma once
#include <cstdint>

#ifndef KX7_PBITS
#define KX7_PBITS 11
#endif
#ifndef KX7_L2PF
#define KX7_L2PF 32            // prefetch.global.L2 this many bytes past the quad being loaded (0 = off)
#endif
#define KX7_RING_WORDS 8
#define KX7_RING_ROWS  (KX7_RING_WORDS + 2)
#define KX7_RING_BYTES (KX7_RING_ROWS * 128)   // per (warp, stream slot): 10 rows x 32 lanes x 4 B
#define KX7_PRIMARY_WORDS (1u << KX7_PBITS)

__device__ __forceinline__ uint32_t kx7_bswap(uint32_t x) { return __byte_perm(x, 0u, 0x0123); }

__device__ __forceinline__ uint4 kx7_ld16(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void kx7_prefetch_l2(const uint8_t* p) {
#if KX7_L2PF > 0
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}

struct Kx7Stream {
    const uint8_t* gp;            // next quad to load (16-byte aligned)
    const uint8_t* tbase;         // ragged layout: where this stream's tail quads start (contiguous, 16 apart)
    uint32_t q0, q1, q2, q3;      // the quad in flight: loaded, not yet stored (raw little-endian words)
    uint32_t wq;                  // stream word index of q0 (= words stored so far)
    uint32_t pos;                 // bit position of the next unread bit, from the aligned base
    uint32_t hi, lo;              // the 64-bit window
    uint32_t nq;                  // quads left to load (loads past the stream are skipped)
    uint32_t n_int;               // of those, how many are still in the interleaved rounds (`stride` apart)
};

// Step gp to the quad after the one at gp: through the interleaved rounds (stride apart), then
// to the tail (tbase) and along it (16 apart). With n_int == nq (no ragged tail) the tail is never
// reached while loads remain.
__device__ __forceinline__ void kx7_advance(Kx7Stream& s, uint32_t stride) {
    if (s.n_int) { s.n_int--; s.gp = s.n_int ? s.gp + stride : s.tbase; }
    else s.gp += 16;
    s.nq--;
}

// ring_lane: this lane's word-0 slot (shared memory), row stride 32 words. The byte swap to
// MSB-first happens here, a chunk after the load, so nothing waits on the load.
__device__ __forceinline__ void kx7_store_quad(Kx7Stream& s, uint32_t* ring_lane) {
    uint32_t row = s.wq & (KX7_RING_WORDS - 1);
    uint32_t* r = ring_lane + row * 32;
    uint32_t a = kx7_bswap(s.q0), b = kx7_bswap(s.q1), c = kx7_bswap(s.q2), d = kx7_bswap(s.q3);
    r[0] = a; r[32] = b; r[64] = c; r[96] = d;
    if (row == 0) { ring_lane[KX7_RING_WORDS * 32] = a; ring_lane[(KX7_RING_WORDS + 1) * 32] = b; }
    s.wq += 4;
}
__device__ __forceinline__ void kx7_load_quad(Kx7Stream& s, uint32_t stride) {
    if (s.nq) {                                   // predicated: nothing is read past the stream
        uint4 v = kx7_ld16(s.gp);
        kx7_advance(s, stride);
        if (s.nq > 1) kx7_prefetch_l2(s.gp + (s.n_int > 1 ? stride : 16));   // two quads ahead
        s.q0 = v.x; s.q1 = v.y; s.q2 = v.z; s.q3 = v.w;
    }
}

// Start a stream: `off` is the byte offset of its first quad in `streams`, `len` its byte length,
// `stride` the distance between its consecutive quads (16: contiguous streams; 512: the 32
// streams of an 8-row block quad-interleaved, this lane being stream `lane` of the group) and
// `ragged` whether the interleave stops at the group's shortest stream, the tails following
// contiguously in lane order (then the 32 lanes of the warp must be the group's 32 streams and
// all call this together: the tail start comes from a warp min and a warp prefix sum of the
// quad counts). Two quads land in the ring (8 words of lead, the ring's capacity), the third
// stays in flight. Reads never go past the stream's quads; bits past its last symbol are never
// consumed.
__device__ __forceinline__ void kx7_init(Kx7Stream& s, const uint8_t* streams, uint32_t off, uint32_t len,
                                         uint32_t stride, bool ragged, uint32_t* ring_lane) {
    const uint8_t* base = streams + (off & ~15u);
    uint32_t nq = (len + (off & 15u) + 15u) >> 4;
    uint32_t n_int = nq;
    const uint8_t* tbase = base;
    if (ragged) {
        const int lane = threadIdx.x & 31;
        uint32_t qmin = __reduce_min_sync(0xffffffffu, nq);
        uint32_t excess = nq - qmin, prefix = excess;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) { uint32_t t = __shfl_up_sync(0xffffffffu, prefix, d); if (lane >= d) prefix += t; }
        prefix -= excess;                                           // exclusive: tail quads of lanes before me
        tbase = (base - lane * 16) + (size_t)qmin * stride + (size_t)prefix * 16;
        n_int = qmin;
    }
    s.gp = base; s.tbase = tbase; s.nq = nq; s.n_int = n_int;
    s.wq = 0;
    s.pos = (off & 15u) * 8u;
    uint4 a = make_uint4(0, 0, 0, 0), b = a, c = a;
    const uint8_t* p0 = s.gp; bool l0 = s.nq > 0; if (l0) kx7_advance(s, stride);
    const uint8_t* p1 = s.gp; bool l1 = s.nq > 0; if (l1) kx7_advance(s, stride);
    const uint8_t* p2 = s.gp; bool l2 = s.nq > 0; if (l2) kx7_advance(s, stride);
    if (l0) a = kx7_ld16(p0);
    if (l1) b = kx7_ld16(p1);
    if (l2) c = kx7_ld16(p2);
    if (s.nq > 0) kx7_prefetch_l2(s.gp);
    if (s.nq > 1) kx7_prefetch_l2(s.gp + (s.n_int > 1 ? stride : 16));
    s.q0 = a.x; s.q1 = a.y; s.q2 = a.z; s.q3 = a.w;
    kx7_store_quad(s, ring_lane);
    s.q0 = b.x; s.q1 = b.y; s.q2 = b.z; s.q3 = b.w;
    kx7_store_quad(s, ring_lane);
    s.q0 = c.x; s.q1 = c.y; s.q2 = c.z; s.q3 = c.w;
}

// Call before every 8 symbols (two windows): they consume at most 4 words and the second window
// reaches 2 words past its start word, so a lead of 5 words suffices, and one conditional store
// restores it: after a top-up the lead is >= 5, 8 symbols leave >= 1, and storing a quad when
// the lead is <= 4 brings it back to 5..8. The rows a store reuses held words wq-8..wq-5 <= W-4,
// so no word still needed is overwritten. (Predicated, no loop.)
__device__ __forceinline__ void kx7_topup(Kx7Stream& s, uint32_t* ring_lane, uint32_t stride) {
    uint32_t W = s.pos >> 5;
    if (s.wq - W <= KX7_RING_WORDS - 4) {
        kx7_store_quad(s, ring_lane);
        kx7_load_quad(s, stride);
    }
}

__device__ __forceinline__ void kx7_refresh(Kx7Stream& s, const uint32_t* ring_lane) {
    uint32_t W = s.pos >> 5;
    const uint32_t* r = ring_lane + (W & (KX7_RING_WORDS - 1)) * 32;
    uint32_t w0 = r[0], w1 = r[32], w2 = r[64];
    uint32_t sh = s.pos & 31;
    s.hi = __funnelshift_l(w1, w0, sh);
    s.lo = __funnelshift_l(w2, w1, sh);
}

// One symbol: returns the table entry (symbol in bits 16..31). Needs >= 16 valid window bits.
// Branch-free: every lane also reads the secondary table, a node lane at its sub-table entry
// and a leaf lane at word 0 (a broadcast, so the extra LDS costs about one wavefront), and a
// select picks the entry. Codes longer than PBITS carry 7 to 10% of the symbols, so a branch
// here would be taken by some lane of the warp at nearly every step, and it would split the
// decode into basic blocks that stop the G chains of kx7_decode16_multi from overlapping.
template <bool TWOCLASS>
__device__ __forceinline__ uint32_t kx7_sym(Kx7Stream& s, const uint32_t* __restrict__ lut, uint32_t esc_shift, uint32_t esc_mask) {
    uint32_t e = lut[s.hi >> (32 - KX7_PBITS)];
    uint32_t len = e & 0xFFu;
    if (TWOCLASS) {
        // rank bytes: (peek32 >> (31 - Le - R)) & (2^(R+1) - 2); zero (a broadcast read) for a
        // non-ESC lane. escmask = 0xFFFF for an ESC entry (its byte 1 is 0xFF).
        uint32_t escmask = __byte_perm(e, 0u, 0x4411);
        uint32_t rank2 = (s.hi >> esc_shift) & esc_mask & escmask;
        const uint16_t* rare = reinterpret_cast<const uint16_t*>(lut + KX7_PRIMARY_WORDS);
        uint32_t r = *reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(rare) + rank2);
        e = (e >> 16) | (r & escmask);                                    // the 16-bit symbol
    } else {
        uint32_t shr = __byte_perm(e, 0u, 0x4441);                        // byte 1: 32 - bits for a node
        uint32_t sub = (s.hi << KX7_PBITS) >> shr;                        // the bits after the prefix
        bool node = len > 16u;
        uint32_t idx2 = node ? (e >> 16) + sub : 0u;
        uint32_t e2 = lut[idx2];
        e = node ? e2 : e;
        len = e & 0xFFu;
    }
    s.hi = __funnelshift_l(s.lo, s.hi, len);
    s.lo <<= len;
    s.pos += len;
    return e;                                  // TWOCLASS: the symbol in bits 0..15; else the entry (symbol in 16..31)
}

// 16 symbols of one 64-wide K chunk for each of G streams of this lane, decoded in lockstep
// -> out[i][t] = bf16x2 word t (elements 2t, 2t+1) of stream i. Includes the top-ups.
template <int G, bool TWOCLASS>
__device__ __forceinline__ void kx7_decode16_multi(Kx7Stream (&s)[G], uint32_t* const (&ring_lane)[G],
                                                   const uint32_t* __restrict__ lut, uint32_t esc_shift, uint32_t esc_mask,
                                                   uint32_t stride, uint32_t (&out)[G][8]) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if ((w & 1) == 0) {
#pragma unroll
            for (int i = 0; i < G; ++i) kx7_topup(s[i], ring_lane[i], stride);
        }
#pragma unroll
        for (int i = 0; i < G; ++i) kx7_refresh(s[i], ring_lane[i]);
        uint32_t e[G][4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int i = 0; i < G; ++i) e[i][k] = kx7_sym<TWOCLASS>(s[i], lut, esc_shift, esc_mask);
#pragma unroll
        for (int i = 0; i < G; ++i) {
            if (TWOCLASS) {
                out[i][2 * w] = __byte_perm(e[i][0], e[i][1], 0x5410);    // low halves: sym0 | sym1 << 16
                out[i][2 * w + 1] = __byte_perm(e[i][2], e[i][3], 0x5410);
            } else {
                out[i][2 * w] = __byte_perm(e[i][0], e[i][1], 0x7632);    // high halves: sym0 | sym1 << 16
                out[i][2 * w + 1] = __byte_perm(e[i][2], e[i][3], 0x7632);
            }
        }
    }
}

// Copy the decode tables (primary + secondary, `words` u32) from global to shared memory.
__device__ __forceinline__ void kx7_load_tables(uint32_t* lut_s, const uint32_t* __restrict__ lut_g, int words, int tid, int threads) {
    for (int i = tid; i < words; i += threads) lut_s[i] = lut_g[i];
}
