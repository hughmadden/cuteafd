// MiniMax M3 host K/V placement: K/V pages at or above `resident` live in
// pinned host memory mapped into each plane's virtual range. The GPU L2 does
// not cache those reads, so every query row re-reading a selected host block
// pays PCIe again. Before a layer's sparse attention, `stage` copies the head
// slices of every host page the step's selections read into a staging page
// (once per page and head, whatever the number of rows selecting it), and
// `translate` rewrites the attention page table to point at the copies. The
// staging pages sit at ids [base, base + capacity) of every plane: one device
// allocation aliased by all layers. A page the staging area cannot hold keeps
// its host id, so attention reads it in place: the bytes read are the same.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kPageRows = 64;
constexpr int kPageBytes = kPageRows * 4 * 128;  // [64 rows][4 KV heads][128] E4M3
constexpr int kRowUnits = 4 * 128 / 16;          // uint4 per row
constexpr int kHeadUnits = 128 / 16;             // uint4 per head slice of a row
constexpr int kBlockTokens = 128;
constexpr int kTopk = 16;

// Slot map entries: 0 free, -1 being claimed, -2 overflow, s + 1 staged in slot s.
__device__ int claim(int32_t* map, int32_t page, int32_t* count, int32_t capacity) {
  int32_t value = atomicCAS(&map[page], 0, -1);
  if (value == 0) {
    const int32_t slot = atomicAdd(count, 1);
    value = slot < capacity ? slot + 1 : -2;
    atomicExch(&map[page], value);
  }
  while (value == -1) value = atomicAdd(&map[page], 0);
  return value > 0 ? value - 1 : -1;
}

__device__ void copy_slice(const uint4* source, uint4* destination, int lane) {
  for (int unit = lane; unit < kPageRows * kHeadUnits; unit += 32) {
    const int offset = unit / kHeadUnits * kRowUnits + unit % kHeadUnits;
    destination[offset] = source[offset];
  }
}

// One block per (row, KV head); each warp walks some of the row's selected
// blocks as the MSA attention kernels do: decode reads the first min(16,
// blocks visible) entries, the extend union every entry naming a visible
// block. A page is staged when it holds a token the row sees, so every
// unmasked read of a translated page finds its head slice; masked reads of
// other slices see stale bytes, as any reused page does.
__global__ void stage(const int32_t* q2k, uint32_t q2k_rows, uint32_t rows, const int32_t* table,
                      uint32_t stride, int32_t per_row, const int32_t* lengths, uint8_t* key, uint8_t* value,
                      int32_t resident, int32_t base, int32_t capacity, int32_t* map, int32_t* heads,
                      int32_t* count, unsigned long long* stats) {
  const int row = blockIdx.x, head = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  const int32_t visible = per_row ? lengths[row] : lengths[0] - int32_t(rows) + row + 1;
  if (visible <= 0) return;
  const int32_t* pages = table + (per_row ? int64_t(row) * stride : 0);
  int blocks = (visible + kBlockTokens - 1) / kBlockTokens;
  blocks = per_row && blocks < kTopk ? blocks : kTopk;
  for (int j = warp; j < blocks; j += warps) {
    const int32_t block = q2k[(int64_t(head) * q2k_rows + row) * kTopk + j];
    if (block < 0 || int64_t(block) * kBlockTokens >= visible) continue;
    for (int sub = 0; sub < 2; ++sub) {
      const int64_t entry = int64_t(block) * 2 + sub;
      if (entry * kPageRows >= visible || entry >= stride) break;
      const int32_t page = pages[entry];
      if (page < resident) continue;
      int32_t slot = -1;
      if (lane == 0) slot = claim(map, page, count, capacity);
      slot = __shfl_sync(0xffffffffu, slot, 0);
      if (slot < 0) {
        if (lane == 0) atomicAdd(&stats[1], 1ull);
        continue;
      }
      int32_t seen = 0;
      if (lane == 0) seen = atomicOr(&heads[slot], 1 << head);
      seen = __shfl_sync(0xffffffffu, seen, 0);
      if (seen & (1 << head)) continue;
      const int64_t from = int64_t(page) * kPageBytes + head * 128;
      const int64_t to = int64_t(base + slot) * kPageBytes + head * 128;
      copy_slice(reinterpret_cast<const uint4*>(key + from), reinterpret_cast<uint4*>(key + to), lane);
      copy_slice(reinterpret_cast<const uint4*>(value + from), reinterpret_cast<uint4*>(value + to), lane);
      if (lane == 0) atomicAdd(&stats[0], 1ull);
    }
  }
}

__global__ void translate(const int32_t* table, int32_t* out, uint64_t entries, const int32_t* map,
                          int32_t resident, int32_t base) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < entries; i += uint64_t(gridDim.x) * blockDim.x) {
    const int32_t page = table[i];
    const int32_t slot = page >= resident ? map[page] : 0;
    out[i] = slot > 0 ? base + slot - 1 : page;
  }
}

bool aligned(const void* pointer, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(pointer) % bytes == 0; }

}  // namespace

// Stage one layer's selected host pages. `q2k` is [4][q2k_rows][16] block ids;
// rows 0..rows-1 are live. Decode (`per_row` 1): row r reads table row r and
// sees `lengths[r]` tokens. Extend (`per_row` 0): every row reads table row 0
// and row r sees lengths[0] - rows + r + 1 tokens. `map` holds one entry per
// pool page, `heads` one per staging slot and `count` one; all three are zero
// before the call. `stats` accumulates [head slices copied, page reads left in
// host memory]. key/value are this layer's planes (16-byte aligned).
extern "C" int32_t cuteafd_minimax_kv_stage_async(const int32_t* q2k, uint32_t q2k_rows, uint32_t rows,
    const int32_t* table, uint32_t stride, int32_t per_row, const int32_t* lengths, void* key, void* value,
    int32_t resident, int32_t base, int32_t capacity, int32_t* map, int32_t* heads, int32_t* count,
    unsigned long long* stats, void* stream) {
  if (!q2k || !table || !lengths || !key || !value || !map || !heads || !count || !stats || rows == 0
      || rows > q2k_rows || stride == 0 || resident < 0 || base < resident || capacity <= 0
      || !aligned(key, 16) || !aligned(value, 16) || !aligned(stats, 8)) {
    return cudaErrorInvalidValue;
  }
  stage<<<dim3(rows, 4), 256, 0, static_cast<cudaStream_t>(stream)>>>(q2k, q2k_rows, rows, table, stride,
      per_row, lengths, static_cast<uint8_t*>(key), static_cast<uint8_t*>(value), resident, base, capacity,
      map, heads, count, stats);
  return cudaGetLastError();
}

// out[i] = the staging id of table[i] when `stage` copied it, else table[i].
extern "C" int32_t cuteafd_minimax_kv_translate_async(const int32_t* table, int32_t* out, uint64_t entries,
    const int32_t* map, int32_t resident, int32_t base, void* stream) {
  if (!table || !out || !map || entries == 0 || resident < 0 || base < resident) return cudaErrorInvalidValue;
  const uint64_t blocks = (entries + 255) / 256;
  translate<<<unsigned(blocks < 1024 ? blocks : 1024), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      table, out, entries, map, resident, base);
  return cudaGetLastError();
}
