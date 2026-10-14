# Issue #100: single-pass wide W8A8 GEMM in `native/shared/cuda/fp8_gemv.cu`

Worktree `~/Developer/cuteafd-ds-k100`, branch `duel/ds-100` (not pushed), based on `origin/work/p0`.
One file changed: `native/shared/cuda/fp8_gemv.cu`, +139/-4. One file added:
`native/tests/fp8_gemv_wide_selftest.cu` (the standalone harness).

## Verdict against the stop bar

The brief's stop bar: **the 1 FC + 5 KV total at M = 2048 must be at least 2x faster**, else report and
stop. Measured, one RTX PRO 6000, median of 51 CUDA-event timings:

| 1 FC + 5 KV, M = 2048 | before (split-K pass loop) | after (single pass) | delta |
| --- | --- | --- | --- |
| latency | 2278 us | **1359 us** | **1.68x** |
| kernel launches | 288 | **14** | **20.6x fewer** |
| delivered | 226 TFLOP/s | 379 TFLOP/s | +68% |

**1.68x < 2x: the bar is not met.** Per the brief I stop here rather than tune further: the change is
kept (strict win on every measured axis, correctness gate passes) and the remaining gap is diagnosed
below with the concrete lever that would close it.

## What changed

`run_wide`'s W8A8 route for more than `kMaxRows` (64) rows no longer loops over 128-row passes with
split-K. It runs whole-height single-split passes:

```cpp
if (w8a8 && rows > kMaxRows) {
  const int tile = w8a8_tile_rows(workspace_bytes, rows, k);
  if (tile >= 128) return run_wide_w8a8(x, packed, scale, out, out_f32, rows, k, n, tile, workspace, stream);
}
```

Three additions, all inside the file's existing anonymous namespace (no header, FFI, Rust, symbol or
ABI change of any kind):

1. `gemv_w8a8_wide_kernel<GROUPS>` -- `gemv_w8a8_kernel` with no K split. `blockIdx.y` is the 128-row
   activation tile (the grid's slowest dimension, so every live CTA reads the same weight tiles) and
   `blockIdx.x` covers 64 output columns (4 warps x 16). Each 128-wide K block's four `mma_e4m3`
   results are scaled by `(weight scale) * (activation scale)` and folded straight into the FP32 total
   in the same order the old kernel used, one 4-register group accumulator at a time (16 live
   accumulators would otherwise double the register demand at `GROUPS = 16`). BF16 or FP32 output is
   written directly, with `m >= rows` guarding the padding rows; there are no split partials and no
   reduce kernel.
2. `w8a8_tile_rows(workspace_bytes, rows, k)` -- the largest 128-multiple that fits the caller's
   *actual* workspace after `run_wide`'s own fragment base (`2 * kWideRows` FP32 scales/maxima plus
   `row_max_bytes(k, kWideRows)`), at `k` bytes per activation row for `xq` plus `k / 128` FP32 each
   for `sxa`, 256-byte aligned per region, stepping down a tile if the alignment estimate lands over.
   0 when not even one tile fits, and the old pass loop is kept then.
3. `run_wide_w8a8(...)` -- per pass: one `pack_x_e4m3_kernel` (whole-height activation tile) and one
   `gemv_w8a8_wide_kernel<16>` on a `(n / 16 / kWarps, stride / 128)` grid. `xq` and `sxa` live in the
   region the old route gave the split partials, so **no workspace size, contract or caller change is
   needed**; `wide_bytes` and its Rust mirror are untouched.

Derived tiles, against the daemon's actual scratch for GLM 5.3 Flash (30,532,608 B = the max over
shapes of `cuteafd_fp8_linear_workspace(2048, k, n, 2)`, itself exactly `wide_bytes(2048, 20480, 4096, true)`):

| projection | k, n | base | per row | tile | passes at M = 2048 | kernels |
| --- | --- | --- | --- | --- | --- | --- |
| fc | 20480, 4096 | 41,984 | 21,120 B | 1408 | 1408 + 640 | 4 |
| kv slice | 4096, 2048 (scratch 17,851,392 B) | 9,216 | 4,224 B | 2048 | 1 | 2 |

Properties the brief asked for:

- **No SM count is baked in.** The tile comes from the caller's `workspace_bytes`, `rows` and `k`; the
  grid is `(n / 16 / kWarps, stride / 128)`, so it scales with the row count. Nothing queries or
  assumes an SM count, L2 size or card. (The route it replaces was *more* device-dependent: its split
  count came from `split_plan` -> `sm_count()`.)
- **Every non-wide path is byte-for-byte unchanged.** The route is taken only for `w8a8 && rows > 64`,
  so `rows <= 64`, the W8A16 path and the `<= 8`-row f16 path still run exactly the old code. Proved
  twice: end to end (M in {1, 8, 16, 64} outputs are **bit-identical**, 0 differing bits, both shapes)
  and in machine code (below).
- **Machine code.** `nvcc -dc` of the pristine and the patched file, `cuobjdump -sass`, matched by
  kernel core name (the anonymous-namespace mangling embeds the file name and a content hash):
  **18 of 18 pre-existing device functions have byte-identical SASS bodies**; the patched build has
  exactly one device function the pristine build lacks, `gemv_w8a8_wide_kernel<16>`. Nothing else in
  the object moved.

## Correctness

Reference: an FP64 CPU reference of the *same* W8A8 math -- E4M3 weights and E4M3 activations (both
quantized in the kernels' own scheme: round-to-nearest-even saturating to E4M3, per row and 128-wide K
block), the products summed in double with the block scales applied per block. Rule 1 (pow2 weight
scales, the production rule: `mod.rs`'s default, and what `remote_dflash.rs` passes). One RTX PRO 6000
(188 SMs). `max_abs` and `max_rel` are over the whole output; `max_rel` is guarded to `|ref| >= 1% of
ref_max` so near-zero elements do not dominate; `rel@max` is the error at the largest reference element.

### 20480 -> 4096 (fc)

| M | ref_max | before max_abs | before max_rel | after max_abs | after max_rel | rel@max (after) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 27.61 | 7.413e-01 | 1.557e+00 | same (bit-identical) | | 5.439e-04 |
| 8 | 30.82 | 7.666e-01 | 1.678e+00 | same (bit-identical) | | 2.420e-03 |
| 16 | 31.18 | 6.247e-02 | 3.887e-03 | same (bit-identical) | | 1.892e-03 |
| 64 | 32.40 | 9.938e-02 | 3.887e-03 | same (bit-identical) | | 3.067e-03 |
| 65 | 33.05 | 6.250e-02 | 3.891e-03 | 6.250e-02 | 3.891e-03 | 1.375e-03 |
| 128 | 35.28 | 7.396e-02 | 3.891e-03 | 7.396e-02 | 3.891e-03 | 8.016e-04 |
| 129 | 33.68 | **6.694e-01** | **1.151e+00** | **8.463e-02** | 3.890e-03 | 1.936e-03 |
| 1024 | 39.17 | 1.245e-01 | 3.892e-03 | 1.245e-01 | 3.892e-03 | 2.122e-03 |
| 2048 | 41.25 | 1.182e-01 | 3.892e-03 | 1.182e-01 | 3.892e-03 | 9.322e-06 |

### 4096 -> 2048 (one KV slice)

| M | ref_max | before max_abs | before max_rel | after max_abs | after max_rel | rel@max (after) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11.15 | 2.572e-01 | 1.347e+00 | same (bit-identical) | | 1.448e-02 |
| 8 | 14.15 | 3.491e-01 | 1.725e+00 | same (bit-identical) | | 1.606e-03 |
| 16 | 15.05 | 3.124e-02 | 3.885e-03 | same (bit-identical) | | 5.743e-04 |
| 64 | 15.08 | 3.121e-02 | 3.886e-03 | same (bit-identical) | | 1.029e-03 |
| 65 | 16.90 | 3.125e-02 | 3.891e-03 | 3.125e-02 | 3.891e-03 | 1.564e-03 |
| 128 | 14.92 | 3.125e-02 | 3.889e-03 | 3.125e-02 | 3.889e-03 | 1.072e-03 |
| 129 | 15.10 | **2.746e-01** | **8.421e-01** | **3.122e-02** | 3.890e-03 | 1.896e-03 |
| 1024 | 16.57 | 5.260e-02 | 3.891e-03 | 5.260e-02 | 3.891e-03 | 3.174e-03 |
| 2048 | 17.65 | 5.474e-02 | 3.891e-03 | 5.474e-02 | 3.891e-03 | 1.616e-03 |

Reading of the table:

- Past the wide threshold the two routes agree to the last digit on every row count except 129: both
  sit at the E4M3 activation quantization floor (`max_rel` ~3.9e-3 against the FP64 reference, i.e.
  both are as accurate as W8A8 gets), so the change costs no accuracy and buys none, except at 129.
- M <= 64 is bit-identical, as required (no wide path is taken).
- **M = 129 is the one interesting row.** The old route's error there (6.694e-01 fc / 2.746e-01 kv) is
  ~10x its own normal band, and it is entirely one row. Dumping both routes' FP32 outputs and
  localizing: for k=20480 rows 0..127 differ only in isolated single elements <= 0.0625 (rounding
  order), while **row 128 differs in 3793 of 4096 elements**, worst old = -8.0625 vs new = -7.4062
  (col 3782); for k=4096, only rows 54 and 86 have one element each (0.0156) and **row 128 differs in
  1848 of 2048 elements**, worst old = -0.3691 vs new = -0.0947. That row is not garbage: over the
  row, `old = new` to a least-squares scale of 1.000196 with a residual of rms 0.187 on a signal of
  rms 7.20.
- Cause, in the old route's own code: it picks the activation format **per pass**, not per call
  (`const bool e4m3 = w8a8 && m > kW8a16Rows;`, `kW8a16Rows = 8`), and a pass of at most 8 rows is one
  chunk of `cuteafd_fp8_w8a16_linear` -- the f16 path, i.e. **BF16 activations, no activation
  quantization**, deliberately ("the same kernels and bits"). At 129 rows the trailing pass is exactly
  one row, so row 128 is computed without activation quantization while the reference (and the other
  128 rows) have it. A BF16-activation row differs from a W8A8 row by the activation quantization
  noise, predicted at `sigma_w * 0.036 * sqrt(k)` ~ 0.26 rms for fc -- the measured 0.187 rms is that,
  so the excess is the f16 trailing pass, not an error in the wide path. The same trailing pass
  appears at M = 136 (rows 128..135). It does not appear at M = 65: a 65-row call is a *single* pass
  of 65 rows, and 65 > 8 keeps the whole pass on E4M3.
- The new route keeps every row on E4M3, so it lands inside the normal band at M = 129
  (8.463e-02 / 3.122e-02) and its output is uniformly W8A8. The one behavioural trade-off: for a call
  with `rows > 128` and `rows % 128` in 1..8, the rows of that trailing pass now carry E4M3 activations
  instead of BF16 -- their bits change, and they move *closer* to the W8A8 reference. The brief allows
  the wide path's bits to differ from the old route's; calls with `rows <= 64`, and every W8A16 path,
  are untouched.

Contract checks (`--contract`, GPU against GPU, on operand sets whose arithmetic is exact -- the
properties the repo's own `native/tests/glm_draft_kernels_selftest.py` asserts for the wide path,
replicated over `rows` in {8, 64, 65, 96, 127, 128, 129, 136, 200, 1024, 2048} x both shapes, with the
pristine binary run as a control):

- mode 1 (wide W8A16) is bit-identical to mode 0 everywhere;
- mode 2 (W8A8) is bit-identical to mode 0 on ternary operands everywhere;
- mode 2's BF16 output is its FP32 output rounded, everywhere;
- mode 2 stays inside the accumulation-rounding bound (2e-3 of the operand set's own max) of mode 0 on
  E4M3-exact activations;
- plus a tail probe that reports `|mode2 - mode0|` per row region on generic operands, which measures
  the f16 trailing pass described above for both routes.

This is the one measurement of this task that the fair queue did not grant in time: `--contract` was
queued on gpu1 as ticket 1146 (`~/.cache/cuteafd/builds/ds-k100/contract_r1.log`) behind two
`cuteafd bench smoke` runs holding gpu1, and the run had not started when this report was written
(`./fp8_selftest_old_contract --rule 1 --contract; ./fp8_selftest_contract --rule 1 --contract`). Every
check that gates the brief's hard requirements is from a completed run: the FP64 reference table, the
bit-identical `M <= 64` dumps, the M = 129 localization, the SASS identity and the performance table.

## Performance

One RTX PRO 6000 (188 SMs), one harness process per binary, median of 51 iterations with CUDA events,
kernel counts from stream capture + `cudaGraphGetNodes` (exact, not estimated).

| | before | after | delta |
| --- | --- | --- | --- |
| fc 20480->4096, M = 2048 | 1106 us, 48 kernels | 914 us, 4 kernels | 1.21x, 12x fewer |
| KV 4096->2048, M = 2048 | 229 us, 48 kernels | 85 us, 2 kernels | 2.69x, 24x fewer |
| 1 FC + 5 KV, M = 2048 | 2278 us, 288 kernels | 1359 us, 14 kernels | 1.68x, 20.6x fewer |
| fc TFLOP/s | 311 | 376 | +21% |
| KV TFLOP/s | 150 | 404 | +169% |
| wave TFLOP/s | 226 | 379 | +68% |

Where the win comes from, per 128-row block of the row height (16 blocks at M = 2048):

- Both routes read the whole packed weight matrix once per 128-row block (fc: 83.9 MB x 16 = 1.34 GB;
  KV: 8.39 MB x 16 = 134 MB). Single-pass did not change that.
- The old route additionally round-trips split-K partials through memory: fc 12 splits x 128 rows x
  4096 cols x 4 B x 2 (write + read) x 16 blocks = **806 MB**; KV 16 x 128 x 2048 x 4 x 2 x 16 =
  **537 MB**, plus a third kernel per pass (48 launches for M = 2048).
- For KV the partials (537 MB) are 4x the weights (134 MB), which is why its win is 2.7x; for fc the
  partials (806 MB) are smaller than the weights (1.34 GB), which is why its win is only 1.2x.
- After the change the kernel is weight-traffic-bound at **~1.5 TB/s on both shapes** (fc 1.34 GB /
  914 us = 1.47 TB/s, KV 134 MB / 85 us = 1.58 TB/s), i.e. at the memory system, with the activation
  tiles (2.6 MB slabs) reused out of cache across the 64 column-CTAs.

## What I did not do

- **No tuning past the stop bar**, per the brief. The pass split (`1408 + 640` for fc) is the derived
  maximum tile; a balanced `1024 + 1024` moves the wave count by at most one CTA-duration per pass in
  either direction, so it is not the lever.
- **The lever that would clear 2x** is fewer weight re-reads per row block: the new kernel reads the
  full packed matrix once per 128-row block because a CTA covers only 128 activation rows. Making a
  CTA cover 256 rows (`GROUPS = 16` with 8 warps) or having two warps share an n-tile over K halves
  with a shared-memory reduction would halve that traffic, worth roughly the 1.2x fc gap. I did not
  implement it: it changes the kernel's register and occupancy shape and needs its own measurement
  round, which the stop bar says not to spend.
- **The harness is not wired into `native/CMakeLists.txt`.** It is a standalone probe like the repo's
  others (`glm_draft_kernels_selftest.py`, `v41_fp8_aot_qualification.py`), and the brief allows a
  standalone harness; I did not want to change a shared build file I cannot validate here.
- Rule 0 was not re-run: the route never touches quantization (`pack_x_e4m3_kernel` is called
  unchanged), so the scale rule cannot interact with it.

## Wall-time split

Derived from the agent transcript (13:32:57Z .. 14:03:43Z, 31.0 min), each gap between events charged
to the tool in flight (capped at 600 s per gap) and everything else to the action it precedes:

| | min | share |
| --- | --- | --- |
| GPU under the fair queue (hwlock waits, queue-wait poll loop, harness GPU sessions) | 10.0 | 32% |
| thinking/generation between tool calls (design, kernel and harness writing, reference, dump and SASS analysis) | 18.4 | 59% |
| model time immediately preceding file writes/edits | 1.3 | 4% |
| reading tools in flight (Read/Grep/Glob/cat/sed/diff) | 0.4 | 1% |
| `nvcc` compiles in the cpu container (5, incl. the `-dc` pair for the SASS diff) | 0.2 | 1% |
| uncategorized | 0.7 | 2% |

The GPU bucket is dominated by waiting: the queue wait plus the poll loop that watched the first
window are ~9.5 of its 10.0 min, and the harness's own GPU sessions are a small part of it (the FP64
CPU reference and the 51-iteration medians are seconds each). Compiles are cheap because only one
small translation unit is built in the cpu container rather than the whole native library.

## Reproducing

```bash
# harness, pristine vs patched source (the cpu container has nvcc, no GPU needed)
cd ~/.cache/cuteafd/builds/ds-k100/target/src
nvcc -std=c++17 -O3 -Xcompiler -fopenmp -arch=sm_120f -o ../fp8_selftest_old fp8_gemv_wide_selftest.cu fp8_gemv_pristine.cu -lcudart
nvcc -std=c++17 -O3 -Xcompiler -fopenmp -arch=sm_120f -o ../fp8_selftest_new fp8_gemv_wide_selftest.cu <worktree>/native/shared/cuda/fp8_gemv.cu -lcudart
# correctness (FP64 reference) + dumps + benches, then the contract checks, one GPU, fair queue
python3 scripts/hw/hwlock.py --locks gpu1 --wait 7200 -- scripts/build/cuteafd-dev.sh coordinator -- \
  bash -c 'cd /opt/cuteafd-target && ./fp8_selftest_new --rule 1 --dump dump_new && ./fp8_selftest_new --rule 1 --contract'
# SASS identity of every pre-existing kernel
nvcc -std=c++17 -O3 -arch=sm_120f -dc fp8_gemv_pristine.cu -o pristine_dc.o
nvcc -std=c++17 -O3 -arch=sm_120f -dc <worktree>/native/shared/cuda/fp8_gemv.cu -o new_dc.o
cuobjdump -sass pristine_dc.o > pristine.sass; cuobjdump -sass new_dc.o > new.sass
```

Logs: `~/.cache/cuteafd/builds/ds-k100/selftest_r1.log` (FP64 reference + benches, both binaries),
`~/.cache/cuteafd/builds/ds-k100/contract_r1.log` (contract + tail probe, both binaries).
