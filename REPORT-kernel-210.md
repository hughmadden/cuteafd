# #210 — one canonical FP8 decode reduction, row-invariant from 1 row to the program cap

Worker: DeepSeek Flash (kernel engineer), duel `duel/ds-210`.
Engine worktree `~/Developer/cuteafd-ds-k210` (branch `duel/ds-210`, based on
`origin/work/p0` at merge-base `e123753b`), fork `third_party/sparkinfer`
(branch `duel/ds-210`, not pushed).

## 1. The defect

The shared FP8 projection picked its kernel by the live row count: a
register-resident GEMV up to 16 rows (`MmaFp8Gemv`, one 16-row tile), a TMA GEMM
above. `TmaFp8Gemm` walks K in a different order than the GEMV, so a row's output
bits changed the moment the batch crossed the switch. On this hardware that was
103 of the 172 gated cases.

The threshold was not one number but three, and they disagreed:

* the Python route choice (`FP8_GEMV_ROWS`, `wide_rows=`) — 16 everywhere, 32 for
  half-head KDA, MiMo and the output-row wrappers;
* a baked `Int32(FP8_GEMV_ROWS)` / `Int32(32)` **scalar** passed into the program,
  which the kernel branches on at run time (so the Rust side could not move the
  boundary on its own);
* the Rust scalars `FP8_ROWS` / `FP8_GEMV_ROWS` in the per-family engines, plus
  the audited `DECODE_PROJECTION_THRESHOLDS` registry (`skinny_rows: 16`) that
  gates bucket sets.

A fourth, quieter site: `_glmf_output_rows.__call__` and
`GlmfOutputShard.__call__` overwrote the scalar with `Int32(32)` when
`total_rows <= 32` — a route that depended on a *different* program's row count
than the rows it was computing.

## 2. Design

**One canonical decode route per projection: thin GEMV tile up to
`FP8_GEMV_ROWS = 16`, a multi-tile GEMV of the *same* `(warps, groups)` up to the
program's capacity `FP8_DECODE_ROWS = 64`, TMA only past the capacity.**

The reason this works is structural, in `MmaFp8Gemv`
(`b12x/integration/cuteafd/_fp8_weights.py`):

* `m_tiles = ceil(max_rows/16)` scales only the accumulator set (`frags`) and the
  shared partial buffer. The `m_tiles` loop
  (`for mt in cutlass.range_constexpr(self.m_tiles)`, guarded by
  `live_tiles = ceil(rows/16)`) is the *innermost* loop of the tile.
* The K walk — `blocks = k/(128*warps)` per warp, the 128-K block loop, the 2x64
  chunk, the `groups` column slices — is outside it and does not read `m_tiles`.
  A row's K blocks, its per-warp K range (`k_begin = warp_id * k_per_warp`) and
  its scale addresses (`s_row` depends on `k_begin` and the lane's column group,
  never on `mt`) are the same at every width.
* The cross-warp fold is `for src in 0..warps`, a fixed order set by `warps`
  alone.

So a row's reduction is a function of `(warps, groups)` only. Two tiles built with
the same `(warps, groups)` are bit-identical over the rows they both cover;
`m_tiles` decides only which rows exist. That is the whole design: keep the tile
pair and the process split fixed, raise the row ceiling.

### Why not the alternatives

* **Freeze TMA at every decode width (`decode_recipe="w8a16"`).** It exists on the
  pin as an interim fallback and is bit-stable, but it is the *slow* route at the
  widths that matter: measured 1.8-2.6x the GEMV at M=1. Deleting it is part of
  this change.
* **Make the TMA GEMM's K order match the GEMV.** A tensor-core GEMM moves K
  through shared memory in a completely different shape; forcing the comparison
  order would cost the GEMM its throughput to fix 48 rows of decode.
* **Run the multi-tile GEMV with different `(warps, groups)` than the thin tile**
  (the obvious tuning instinct, and what the pin's `wide_warps=8, wide_groups=2`
  overrides did). Rejected: it re-introduces the defect at the 16-row boundary,
  which is inside the decode range. `Fp8Projection` now raises `ValueError` if a
  caller asks for it, so the mistake cannot be made quietly.
* **A separate 128-row GEMV tile for the `_m128` programs.** Deferred, not
  rejected — see section 7.

### Threshold as capacity, not as a policy switch

`FP8_DECODE_ROWS = 64` is not "the new threshold"; it is the row ceiling of the
decode programs (`_m64`) themselves. Below it there is no route change at all, so
there is nothing for a bucket to pad across — the audited registry now says
exactly that (`skinny_rows: FP8_DECODE_ROWS`) and the bucket assertions were
rewritten to the new invariant: no FP8 projection crosses an arithmetic route
inside the decode range (`check_bucket_thresholds(&[1,4,16,64], ...)`,
`&[1,4,8,16,64]`, `&[2,4,8,16,32,64,128]` all pass), while a set that would pad a
step past the capacity still fails (`[32, 72]` -> `kda.in.out.fp8`).

64 is inert with respect to the serving buckets rather than merely equal to them:
`check_bucket_thresholds` rejects `previous < t && t < bucket`, and a `t` equal to
a bucket value is never straddled. With the plain/spec bucket sets (`[1,4,16,64]`,
`[2,4,8,16,32,64,128]`) a threshold of 64 cannot be crossed by padding, so the
registry records the capacity without constraining the planner.

## 3. Diff summary

### Fork (`third_party/sparkinfer`, 12 files, +215/-161)

`b12x/integration/cuteafd/_fp8_weights.py` (+124/-...)

* adds `FP8_DECODE_ROWS = 64` next to `FP8_GEMV_ROWS = 16` (both exported) with the
  invariant written out in the module docstring;
* `RoutedFp8Projection(..., decode_rows=None)`: thin tile at `FP8_GEMV_ROWS`,
  `fp8_wide` at `decode_rows` **with the same `warps`**, `max_fp8_rows =
  decode_rows` (`fp8_wide is None` when equal);
* `Fp8Projection(..., wide_warps=None, wide_groups=None, decode_rows=None)`:
  `cap = max(decode_rows or FP8_DECODE_ROWS, wide_rows, gemv_rows)`, `gemv_wide`
  built at `cap` with the thin tile's `(warps, groups)`, and a `ValueError` if
  `(wide_warps, wide_groups) != (warps, groups)`. `wide_rows` is now only a floor
  for callers that pass it; `max_gemv_rows` is the cap, which is what every baked
  scalar reads;
* `projection(..., decode_rows=None)` forwards it; `key()` includes the wide
  tile's key.

`b12x/integration/cuteafd/glmf.py` (+90/-...)

* `_Fp8Switch(..., wide_rows=0, decode_rows=None)`: `cap = max(decode_rows or
  FP8_DECODE_ROWS, wide_rows)`, `max_fp8_rows = FP8_ROWS if fp8_wide is None else cap`;
* `_w8(...)` takes `decode_rows` and `warps`/`groups` overrides; the
  `wide_warps=8, wide_groups=2` overrides are **deleted** (the multi-tile GEMV
  inherits the thin tile's, which is the point of the change);
* `decode_recipe` is deleted everywhere (`_KdaW8`, `_compile_glmf_kda_w8`,
  `compile_glmf_kda_aot`), including its `raise ValueError` gate;
* `FP8_ROWS` (the `_Fp8Switch` thin tile, exported and used by `mimo_attention.py`
  and `qwen4.py` as their `gemv_rows`/`max_rows`) is now `FP8_GEMV_ROWS` from
  `_fp8_weights`, so the thin tile has **one** definition in Python instead of two
  copies of 16 that merely happened to agree;
* `_KdaW8.__init__` keeps the full-head in-projection tile for the head-split
  geometry: `in_tile = (8, 2) if (hidden, kda_heads, kda_head_dim) == (4096, 32, 128)
  else FP8_GEMV_CONFIG.get((p, hidden), (gemv_warps(hidden), 4))`. This is the
  pin's existing behaviour (it used to replace the projection outright with
  `warps=8, groups=2` after N slicing) preserved in the new shape — a half-head
  in-projection must keep the full-head program's K reduction or every row's bits
  move.

Consumers — every 16/32 route wart removed, every baked scalar moved to
`FP8_DECODE_ROWS`: `_glmf_output_rows.py`, `_glmf_output_shard.py`,
`_glmf_mla_output_rows.py` (the `total_rows <= 32 -> fp8_rows = 32` overwrite is
gone; the route scalar is unconditional), `glm_attention.py`, `glm_ffn.py`,
`qwen4_attention.py`, `qwen4_gdn.py`, `mimo_attention.py` (`MIMO_FP8_ROWS`
deleted), `mimo_ffn.py`. Docstrings that said "<= 16" now name the capacity.
`tests/integration/test_cuteafd_glmf_kda_w8_aot.py` drops the two
`decode_recipe="w8a16"` kwargs.

The gated test `tests/integration/test_fp8_projection_row_invariance.py` is
byte-identical to the pin (**not** modified — it is the judge's instrument).

### Engine (`~/Developer/cuteafd-ds-k210`, 7 files + submodule pin + lock, +61/-53)

* `rust/.../glm5_flash/engine.rs`: `FP8_ROWS: i32 = 16` -> `FP8_DECODE_ROWS: i32 = 64`
  (documented as the programs' capacity, with the `_m128` caveat in section 7);
  all seven FP8 `DECODE_PROJECTION_THRESHOLDS` entries `skinny_rows:
  FP8_DECODE_ROWS`; `kda.half.in.out.fp8.wide` renamed `kda.half.in.out.fp8` (one
  route now); the `kda_w8` and `fp8_scalars` baked scalars `FP8_DECODE_ROWS`; the
  audited skinny vector `[0,0,0,0,64,64,64,64,64,64,64]`; the threshold assertions
  and the `crossing` case rewritten around the capacity; the glmf.py FNV audit
  hash updated.
* `mimo_v2/engine.rs`: `FP8_DECODE_ROWS: i32 = 64` (was 32), and the now-dead
  `pub(crate) const FP8_ROWS: i32 = 16` is deleted rather than left as a second,
  wrong answer to the same question.
  `qwen4/engine.rs`: `FP8_PROJECTION_SKINNY_ROWS: usize = 64` (was 16).
  `glm5/dflash.rs`: the unused `pub(crate) const FP8_ROWS: usize = 128` and its
  "historical skinny-row limit" comment are deleted (the live
  `cuteafd-loader/.../mimo_v2/draft_config.rs:7` constant is a different, used
  one and stays).
* `python/tools/aot/export_b12x_dsv4_aot.py`: the KDA W8 job params no longer
  carry `decode_recipe`.
* `scripts/tests/test_glmf_wide_decode_export.py`, `test_qwen_vision_gate.py`:
  assert the exporter no longer passes `decode_recipe`, and that the fork's
  `FP8_DECODE_ROWS`, the Rust scalar and `defaults["gemv_rows"]` agree.
* `third_party/sparkinfer.lock.json`: pin bumped to the fix commit. Local-only —
  the fork branch is not pushed (section 7).

## 4. Evidence — the gate (Req 2)

`scripts/qualify/test_fp8_projection_row_invariance.py` is placed at its drop-in
location `third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py`
so that `from ..conftest import require_b12x` resolves as a package import instead
of the silent self-skip fallback (`def require_b12x(): pytest.skip(...)`), and so
`require_b12x()` finds a real SM120 GPU.

43 consumers x M in {16, 17, 32, 64}, row 0 compared bitwise against M=1.

**Before (pin `e087083e`), the same command in the `~/Developer/cuteafd-ds-k210b`
worktree pinned at that fork commit:**

```
103 failed, 69 passed, 8 warnings in 24.57s
```

**After (fork `b0c81048`, `~/Developer/cuteafd-ds-k210`):**

```
$ scripts/hw/hwlock.py --locks gpu0 --wait 7200 -- scripts/build/cuteafd-dev.sh coordinator -- \
    python -m pytest third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py
hwlock: ticket 1181 queued for gpu0 scripts/build/cuteafd-dev.sh coordinator
verified SparkInfer revision=b0c81048f66c256269079a29c1ad2105f9ddef78 \
  source_tree_sha256=34a8ecd6d503866475a8ffa8de399a795dc124f42b0255bea45df746317f7369
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /workspace/cuteafd/third_party/sparkinfer
configfile: pyproject.toml
collected 172 items

third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py . [  0%]
........................................................................ [ 42%]
........................................................................ [ 84%]
...........................                                              [100%]
====================== 172 passed, 71 warnings in 39.51s =======================
```

`rootdir: /workspace/cuteafd/third_party/sparkinfer` and `collected 172 items` are the two
lines that prove the test really ran rather than self-skipping: collected from its drop-in
location it resolves `from ..conftest import require_b12x` as a package import, and that
`require_b12x` skips unless it sees an SM120/SM121 device. Every one of the 172 cases executed
on GPU0.

M=1 bit-identity was measured the same way, on both trees, and is exact:

- `m1_bits.py` (untracked) drives the 43 gated consumers through the *test's own* helpers
  (`t._CASES`, `t._weights`, `t.W8`, `t._launch(program, kind, lone, w8, scale, bf16, one, 1, cap)`)
  with M=1 and dumps the int16 bit pattern of row 0 of each output plus the sentinel check.
- Tickets 1182 (before, revision `e087083e`) and 1185 (after, `b0c81048`); dumps
  `m1-before.pt` (891,396 B) and `m1-after.pt` (891,347 B).
- Comparison: `torch.equal(before[k], after[k])` for every consumer.

```
consumers before/after: 43 43
same key set: True
bit-identical rows: 43 of 43
int16 elements compared: 439232
```

All 43 consumers, 439,232 int16 values, bit-identical: 0 differing values, at any tolerance
since the comparison is exact equality. Sample (row 0, first four weights):

```
glm_dense_down before [-16646, -17408, -16629, -16721]  after [-16646, -17408, -16629, -16721]
glm_dense_gu   before [ 15987, -16837,  15972,  16189]  after [ 15987, -16837,  15972,  16189]
glm_index_wq   before [-17258, -17071,  15719, -16744]  after [-17258, -17071,  15719, -16744]
```

The M=1 route is unchanged by construction (a 1-row step takes the same thin tile in both
trees; the `route` classifier in `perf.py` records `gemv-thin` for M=1 in both dumps), so this
is the check that the *new* wide tile did not perturb the narrow one - e.g. by shrinking the
thin tile's `max_rows` or by moving a scalar.

## 5. Performance (Req 3)

### Method

`perf.py` (untracked, `.work-k210/`) builds each of the 43 consumers exactly as
`test_fp8_projection_row_invariance.py` builds them, then for each M in {1, 4, 16, 17, 32, 64}
runs three untimed warm launches and 64 launches timed with one CUDA event pair around the
projection call, one `torch.cuda.synchronize()` at the end, reporting median (and min/p90) in
microseconds. The warm launches are the AGENTS-mandated warm-up: per-token weight reads, first
CuTe DSL launch, first-use workspace. Both trees ran the same script on GPU0 through the fair
queue: before = ticket 1183 (revision `e087083e`), after = ticket 1184 (`b0c81048`); the two
runs finished five minutes apart on the same device, one launch per arm.

The `kind` column is the projection class under test (the test's own taxonomy):
`plain`/`w8`/`w8row`/`w8row32`/`half_in`/`rows`/`shard`/`mimo` are the production `Fp8Projection`
/ `RoutedFp8Projection`-free paths a real FP8 checkpoint uses; `routed` and `switch` are the
two classes that also own a BF16 copy of the weights, which the test exercises as "the
unmeasured equivalent, not a second copy of every BF16 projection".

### Results (median microseconds, after/before ratio in parentheses)

```
consumer                       tag    kind    N x K                           M=1                   M=4                  M=16                  M=17                  M=32                  M=64
                                                             before -> after (x)   before -> after (x)   before -> after (x)   before -> after (x)   before -> after (x)   before -> after (x)
glm_dense_down                 s16    plain   6144x12288      28.32->27.86 (0.98)   29.92->29.89 (1.00)   32.06->32.42 (1.01)   85.66->52.90 (0.62)   85.66->60.27 (0.70)   87.55->95.90 (1.10)
glm_dense_gu                   s16    plain   24576x6144      79.54->79.52 (1.00)   83.62->83.57 (1.00)   85.66->85.52 (1.00) 173.66->110.21 (0.63) 175.63->116.29 (0.66) 179.84->159.39 (0.89)
glm_index_wq                   s16    plain   4096x2048       18.74->18.62 (0.99)   18.56->18.11 (0.98)   18.62->18.61 (1.00)   18.62->18.62 (1.00)   18.62->18.54 (1.00)   18.62->18.62 (1.00)
glm_o                          s16    plain   6144x16384      46.54->46.69 (1.00)   46.75->46.75 (1.00)   48.40->48.26 (1.00)  113.68->69.28 (0.61)  114.16->73.38 (0.64) 114.21->118.43 (1.04)
glm_q_b                        s16    plain   16384x2048      18.66->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   32.02->21.70 (0.68)   27.87->21.70 (0.78)   29.89->33.98 (1.14)
glm_qkv_a                      s16    plain   2624x6144       18.62->18.62 (1.00)   18.62->18.53 (0.99)   18.62->18.37 (0.99)   25.76->19.60 (0.76)   25.76->19.65 (0.76)   25.82->29.86 (1.16)
glm_shared_down                s16    plain   6144x2048       18.62->18.58 (1.00)   18.62->18.59 (1.00)   18.62->18.62 (1.00)   21.70->18.59 (0.86)   18.62->18.62 (1.00)   18.62->18.59 (1.00)
glm_shared_gu                  s16    plain   4096x6144       19.65->18.62 (0.95)   19.52->18.62 (0.95)   18.66->18.59 (1.00)   42.14->19.65 (0.47)   42.11->19.65 (0.47)   42.50->29.89 (0.70)
glmf_dense_down                s16    w8      4096x12288      18.64->18.62 (1.00)   19.63->18.62 (0.95)   21.71->21.73 (1.00)   85.57->29.82 (0.35)   85.57->36.51 (0.43)   87.14->59.04 (0.68)
glmf_dense_down_half           s16    w8      4096x6144       18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   40.61->19.65 (0.48)   42.11->21.66 (0.51)   42.46->29.89 (0.70)
glmf_dense_gu                  s16    w8      24576x4096      44.67->42.66 (0.95)   46.75->46.75 (1.00)   50.85->50.34 (0.99)  101.82->69.06 (0.68)  104.10->71.33 (0.69) 112.29->105.41 (0.94)
glmf_dense_gu_half             s16    w8      12288x4096      18.62->18.62 (1.00)   18.62->18.59 (1.00)   18.62->18.62 (1.00)   54.94->34.10 (0.62)   56.48->35.94 (0.64)   58.51->56.99 (0.97)
glmf_kda_in                    s16    w8row   24896x4096      52.90->52.90 (1.00)   54.43->53.76 (0.99)   56.99->56.99 (1.00)  116.29->68.58 (0.59)  118.30->79.49 (0.67) 124.58->132.06 (1.06)
glmf_kda_in_half               s32    half_in 12576x4096      18.62->18.62 (1.00)   18.62->18.58 (1.00)   21.70->21.73 (1.00)   26.27->27.89 (1.06)   32.42->33.95 (1.05)   62.72->56.64 (0.90)
glmf_kda_o                     s16    w8row   4096x8192       18.62->19.49 (1.05)   19.57->19.65 (1.00)   18.62->18.62 (1.00)   68.85->19.65 (0.29)   68.91->21.70 (0.31)   69.15->35.94 (0.52)
glmf_kda_o_half                s32    w8row32 4096x4096       18.64->18.58 (1.00)   18.62->18.59 (1.00)   18.66->18.62 (1.00)   18.64->18.62 (1.00)   18.62->18.62 (1.00)   34.11->21.70 (0.64)
glmf_mla_o                     s16    w8      4096x16384      27.87->27.84 (1.00)   27.87->28.32 (1.02)   30.37->30.37 (1.00)  113.70->48.30 (0.42)  113.26->54.59 (0.48)  114.16->89.76 (0.79)
glmf_mla_o_half                s16    w8      4096x8192       19.46->18.62 (0.96)   18.64->18.62 (1.00)   18.62->18.62 (1.00)   58.91->19.65 (0.33)   58.91->21.70 (0.37)   58.91->34.46 (0.59)
glmf_mla_output_rows           s16    w8      4096x16384      27.84->27.82 (1.00)   28.32->29.78 (1.05)   30.37->30.37 (1.00)  112.29->48.46 (0.43)  113.46->54.16 (0.48)  114.18->91.17 (0.80)
glmf_mla_q_b                   s16    w8      16384x1536      19.62->18.62 (0.95)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   26.27->18.62 (0.71)   21.73->18.62 (0.86)   23.74->23.74 (1.00)
glmf_mla_q_b_half              s16    w8      8192x1536       18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   21.70->18.62 (0.86)   18.51->18.62 (1.01)   18.62->18.62 (1.00)
glmf_mla_qkv_a                 s16    w8      2048x4096       19.12->19.34 (1.01)   18.72->18.62 (0.99)   18.64->18.64 (1.00)   18.61->18.64 (1.00)   18.56->18.62 (1.00)   18.62->18.59 (1.00)
glmf_output_rows               s32    rows    4096x8192       18.62->18.62 (1.00)   18.62->18.59 (1.00)   18.62->18.62 (1.00)   18.99->19.65 (1.03)   21.70->21.76 (1.00)   69.15->34.46 (0.50)
glmf_output_shard              s32    shard   2048x8192       18.64->18.62 (1.00)   18.61->18.54 (1.00)   18.62->18.59 (1.00)   18.54->18.62 (1.00)   18.45->18.45 (1.00)   44.58->27.82 (0.62)
glmf_shared_down               s16    w8      4096x2048       18.62->18.59 (1.00)   18.56->18.58 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.59 (1.00)   18.62->21.70 (1.16)
glmf_shared_down_half          s16    w8      4096x1024       16.35->15.84 (0.97)   15.94->16.14 (1.01)   16.56->16.32 (0.99)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)
glmf_shared_gu                 s16    w8      4096x4096       18.43->18.62 (1.01)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   27.87->18.59 (0.67)   28.18->18.61 (0.66)   29.94->21.70 (0.72)
glmf_shared_gu_half            s16    w8      2048x4096       18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.59->18.62 (1.00)   18.59->18.62 (1.00)   18.59->18.59 (1.00)
mimo_dense_down                s32    mimo    4096x16384      27.84->30.91 (1.11)   27.89->27.84 (1.00)   31.07->30.37 (0.98)   48.46->50.27 (1.04)   56.48->56.99 (1.01)  118.05->93.82 (0.79)
mimo_dense_gu                  s32    mimo    32768x4096      73.38->73.07 (1.00)   74.59->73.06 (0.98)   77.44->75.17 (0.97)   85.66->87.71 (1.02) 102.02->103.41 (1.01) 122.53->163.20 (1.33)
mimo_o                         s32    mimo    4096x8192       19.46->18.62 (0.96)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   19.65->21.39 (1.09)   25.76->25.79 (1.00)   60.96->38.05 (0.62)
mimo_pro_dense_down            s32    mimo    6144x16384      46.75->48.77 (1.04)   46.75->48.77 (1.04)   50.82->54.37 (1.07)   73.34->77.26 (1.05)   82.99->87.65 (1.06) 118.34->138.11 (1.17)
mimo_pro_dense_gu              s32    mimo    32768x6144    120.48->119.76 (0.99) 120.48->120.48 (1.00) 123.60->125.87 (1.02) 138.88->156.48 (1.13) 148.94->169.01 (1.13) 189.90->241.28 (1.27)
mimo_pro_o                     s32    mimo    6144x16384      46.29->46.75 (1.01)   48.05->48.72 (1.01)   50.26->52.86 (1.05)   72.58->75.36 (1.04)   78.82->83.60 (1.06) 118.34->130.69 (1.10)
mimo_pro_qkv                   s32    mimo    27648x6144      97.58->97.95 (1.00)   99.89->99.90 (1.00) 100.05->102.05 (1.02) 114.34->126.59 (1.11) 120.45->134.78 (1.12) 188.00->200.32 (1.07)
mimo_qkv_full                  s32    mimo    13568x4096      18.98->18.66 (0.98)   18.62->18.62 (1.00)   19.65->19.65 (1.00)   26.27->27.84 (1.06)   29.82->30.37 (1.02)   63.01->48.80 (0.77)
mimo_qkv_swa                   s32    mimo    14848x4096      18.62->19.49 (1.05)   18.62->18.59 (1.00)   19.65->19.65 (1.00)   26.27->27.84 (1.06)   29.89->30.35 (1.02)   63.01->48.80 (0.77)
qwen_attn_in                   s16    plain   13952x2560      19.46->19.49 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   63.01->25.79 (0.41)   38.02->27.84 (0.73)   40.24->42.66 (1.06)
qwen_attn_o                    s16    plain   2560x6144       18.62->18.62 (1.00)   18.62->18.59 (1.00)   18.62->18.56 (1.00)   25.73->18.46 (0.72)   25.73->19.65 (0.76)   25.79->29.82 (1.16)
qwen_gdn_in                    s16    plain   16480x2560      18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   38.05->27.74 (0.73)   33.98->27.84 (0.82)   36.22->44.18 (1.22)
qwen_gdn_o                     s16    plain   2560x6144       18.62->18.62 (1.00)   18.62->18.62 (1.00)   18.62->18.62 (1.00)   25.73->19.49 (0.76)   25.76->19.62 (0.76)   25.79->29.86 (1.16)
routed_glm_q_b                 copy   routed  16384x2048      24.22->24.22 (1.00)   23.74->23.79 (1.00)   23.74->23.78 (1.00)   32.19->32.42 (1.01)   17.63->33.95 (1.93)   19.63->44.85 (2.28)
switch_glmf_kda_in             copy   switch  24896x4096      66.78->66.80 (1.00)   66.83->66.86 (1.00)   68.93->69.25 (1.00)  149.15->82.77 (0.55) 150.98->101.38 (0.67) 152.93->153.18 (1.00)

```

Summary over all 258 measured cells:

| band | cells | geomean after/before | median | best | worst |
| --- | --- | --- | --- | --- | --- |
| M = 1, 4, 16 | 129 | 0.9997 | 1.000 | 0.948 | 1.110 |
| M = 17, 32, 64 | 129 | 0.8174 | 0.974 | 0.285 | 2.284 |
| all | 258 | 0.9040 | 1.000 | 0.285 | 2.284 |

### Stop-bar verdict: not met, in either band

The brief's bar is M <= 16 within 3% and M = 17-64 within 15%. Both are breached:

- **M <= 16: FAIL.** 8 of 129 cells exceed 1.03, worst `mimo_dense_down` at M=1
  (27.84 -> 30.91 us, 1.110). The rest: 1.07 `mimo_pro_dense_down` M=16, 1.052 `mimo_pro_o`
  M=16, 1.051 `glmf_mla_output_rows` M=4, 1.046 `mimo_qkv_swa` M=1, 1.046 `glmf_kda_o` M=1,
  1.043/1.043 `mimo_pro_dense_down` M=1/M=4. Nothing in this band crosses 1.15.
- **M = 17-64: FAIL.** 10 of 129 cells exceed 1.15, worst `routed_glm_q_b` at M=64
  (19.63 -> 44.85 us, 2.284).

What those two bands actually contain:

1. **Every M >= 17 breacher is a shape whose *before* route was the W8A16/BF16 TMA at that
   width.** The classifier in `perf.py` records the route taken, so this is checkable from the
   dumps:

   | consumer | shape | kind | route before -> after | ratio |
   | --- | --- | --- | --- | --- |
   | `routed_glm_q_b` | 16384x2048 | routed | `bf16-tma` -> `gemv-wide` | 1.93 (M=32), 2.28 (M=64) |
   | `mimo_dense_gu` | 32768x4096 | mimo | `w8a16-tma` -> `gemv-wide` | 1.33 (M=64) |
   | `mimo_pro_dense_gu` | 32768x6144 | mimo | `w8a16-tma` -> `gemv-wide` | 1.27 (M=64) |
   | `qwen_gdn_in` | 16480x2560 | plain | `w8a16-tma` -> `gemv-wide` | 1.22 (M=64) |
   | `mimo_pro_dense_down` | 6144x16384 | mimo | `w8a16-tma` -> `gemv-wide` | 1.17 (M=64) |
   | `glmf_shared_down` | 4096x2048 | w8 | `w8a16-tma` -> `gemv-wide` | 1.16 (M=64) |
   | `qwen_attn_o` | 2560x6144 | plain | `w8a16-tma` -> `gemv-wide` | 1.16 (M=64) |
   | `glm_qkv_a` | 2624x6144 | plain | `w8a16-tma` -> `gemv-wide` | 1.16 (M=64) |
   | `qwen_gdn_o` | 2560x6144 | plain | `w8a16-tma` -> `gemv-wide` | 1.16 (M=64) |
   | `glm_q_b` | 16384x2048 | plain | `w8a16-tma` -> `gemv-wide` | 1.14 (M=64) |
   | `glm_dense_down` | 6144x12288 | plain | `w8a16-tma` -> `gemv-wide` | 1.10 (M=64) |

   The TMA GEMM is nearly flat in M - `glm_dense_down` costs 85.66 us at M=17 and 87.55 us at
   M=64 - because it is weight-bandwidth bound and its tensor-core work for 17-64 extra rows is
   nearly free. The FP8 GEMV reads half the bytes per weight (FP8 vs the BF16 copy) but pays for
   extra rows in serial MMA work per K block, so its per-row advantage shrinks as M approaches
   its 64-row capacity and the two cross over somewhere around M=32-64, per shape. **Routing
   those rows back to the TMA is the defect this task exists to remove**, so this cost cannot be
   bought back on those shapes; the choice is row-invariance at 1.1-1.3x on 10 of 43 shapes at
   M=64, or the current 103/172 bit failures.

2. **The M <= 16 band is flat in aggregate** (geomean 0.9997, median 1.000, 121 of 129 cells
   within 3%). Two sub-classes are worth naming rather than waving at:
   * The three `mimo_pro_*` cells (1.04, 1.04, 1.07 on `mimo_pro_dense_down`; 1.02-1.06 on
     `mimo_pro_o`/`mimo_pro_qkv`) are a near-constant additive offset at every width, including
     M=1 where both trees take the same `gemv-thin` route with the same tile geometry
     (`max_rows=16`, same warps/groups) and the same grid. A constant offset on the same
     kernel is more plausibly session-level (separate containers, five minutes apart,
     DRAM-resident shapes of 100-160 MB) than code, but these two arms were not interleaved,
     so I report it as unexplained rather than explained.
   * `mimo_dense_down` M=1 (1.11) is the largest single number in the band and is also the
     first width measured for that consumer, on the largest DRAM footprint in the set
     (4096x16384: 134 MB FP8 + 268 MB BF16 copy). Per the AGENTS rule on warm-up I did not
     re-run it; the honest reading is that it is either first-touch cost or session noise, and
     a second session is what would separate them.

### What the change buys

The widths that serving actually pads to are the buckets the Rust registries admit - 1, 4, 16,
64 on the plain path and 2, 4, 8, 16, 32, 64, 128 on the speculative path - so a 17-32 row
batch is served as a 32-row step and 33-64 as 64 rows. In that band the fix is a large win on
the shapes it was opened for:

| consumer | shape | M=17 | M=32 | M=64 |
| --- | --- | --- | --- | --- |
| `glmf_kda_o` (KDA output projection) | 4096x8192 | 0.29 | 0.31 | 0.52 |
| `glmf_mla_o_half` | 4096x8192 | 0.33 | 0.37 | 0.59 |
| `glmf_dense_down` | 4096x12288 | 0.35 | 0.43 | 0.68 |
| `glmf_mla_o` / `glmf_mla_output_rows` | 4096x16384 | 0.42/0.43 | 0.48 | 0.79/0.80 |
| `glm_shared_gu` | 4096x6144 | 0.47 | 0.47 | 0.70 |
| `switch_glmf_kda_in` (glmf decode switch) | 24896x4096 | 0.55 | 0.67 | 1.00 |
| `qwen_attn_in` | 13952x2560 | 0.41 | 0.73 | 1.06 |
| `glm_dense_gu` | 24576x6144 | 0.63 | 0.66 | 0.89 |

Geomean over the M >= 17 band is 0.8174: the fix makes a padded decode step roughly 18% faster
on average while making it bit-exact, with the wins concentrated in the 17-32 band (up to 3.5x)
and the losses concentrated at M=64 on shapes where the TMA was already better. At M = 1-16
(the C1 decode case) the projection costs what it cost before.

## 6. SM export comparison (Req 4)

### Result

`scripts/build/compare-sm-exports.py --sms 188,170` over all eleven groups
(`glm, glm2, glmf, glmf2, mimo, mimo2, mimop, mimop2, mimof, mimof2, qwen4`), CPU-only, no GPU
lock (`cuteafd-dev.sh cpu`):

```
Compared 296 programs: 296 identical objects, 0 different.
Report: /workspace/cuteafd/.work-k210/sm-exports/comparison.json
```

Byte-identical objects (and identical SASS, `sass=True` per line) for SM188 and SM170, so no
program touched by this change bakes in a card's SM count.

### Why that is structural, not luck

* The touched fork files contain no reference to an SM count at all:
  `for f in $(git diff --name-only e087083e HEAD); do grep -Hn "sm_count\|SmCount\|SM_COUNT" $f; done`
  over the 12 changed fork files prints nothing.
* `MmaFp8Gemv` derives its grid from the shape alone (`n // cols` column tiles, one block per
  column tile) and its block size from `warps`; `FP8_DECODE_ROWS`/`FP8_GEMV_ROWS` are row
  counts. The wide tile is the same kernel with a larger `m_tiles` (accumulator set) and the
  same `(warps, groups)`; nothing about it is sized by L2 or SM count.
* The three programs that *do* take an SM count are the sparse-MLA/plan-metadata programs,
  which this change does not touch; the verified comparison above covers them anyway.
* Engine side: the planner still derives its verify bucket from the runtime SM count
  (`cudaDevAttr`), unchanged; the only new scalar is the threshold itself.

## 7. Residuals and limits

* **`_m128` programs are canonical to 64, not 128.** With
  `--glmf-wide-decode-rows 128` a step of 65-128 rows runs the `_m128` programs,
  whose FP8 GEMV capacity is `FP8_DECODE_ROWS`; rows 65-128 of such a step still
  take the W8A16 TMA. Every program's *default* decode range (<= 64 rows) is
  row-invariant and the 172-case gate covers exactly that range. A 128-row GEMV
  tile is the follow-up if a 128-row decode step becomes a served shape. This is
  recorded in the `FP8_DECODE_ROWS` doc comment in `engine.rs`, not hidden.
* **The fork pin is local-only.** Kernel changes are supposed to go to
  `../sparkinfer-glmrt` master first, and this duel forbids pushing. The fix commit
  (`b0c81048`) exists only in this worktree; `third_party/sparkinfer.lock.json` points at it,
  so a clone of `duel/ds-210` cannot verify the pin (`b0c81048`, tree
  `34a8ecd6...`) until the fork branch is
  pushed and re-pinned. Everything else in the engine diff is self-contained.
* **Measurements are one launch per arm.** Both performance arms and the M=1 dumps ran once,
  per the AGENTS frugality rule. The 1.11 at `mimo_dense_down` M=1 and the flat ~4%
  `mimo_pro_*` offsets are the two numbers a second, interleaved session would settle; neither
  is load-bearing for the verdict, which fails on numbers 2-3x larger than those.

## 8. Wall time (own timestamps)

All timestamps are my own (log mtimes, commit dates, transcript), +0800.

| phase | window | wall |
| --- | --- | --- |
| read brief, `_fp8_weights.py`, the three threshold sites; derive the invariant | 00:16-00:31 | 15 min |
| fork edits (12 files) + engine edits (7 files + lock); design frozen at commit `b0c81048` | 00:31-01:12 | 41 min |
| first gate round: smoke + baseline invariance (tickets 1172, 1175), first export pass, first cargo pass | 00:30-01:08 | 38 min, mostly overlapping the coding |
| second/third gate rounds: `gemv_warps` NameError fix found by the export gate, two dead Rust constants deleted, export re-run (01:23, 296/296), container `scripts/tests` | 01:04-01:23 | 19 min |
| GPU queue wait + five final runs (invariance after, m1 before/after, perf before/after) | 01:12-01:29 | 17 min |
| crash recovery: orchestrator died mid-flight, re-queued the killed GPU tickets detached | ~01:12-01:16 | ~4 min |
| analysis (M=1 diff, perf table, stop-bar verdict) + this report | 01:16-02:05 | 49 min |

Total elapsed session: 00:16 -> 02:05 = **109 min**, which overlaps: the CPU gates ran while
GPU tickets were queued and while I edited. The last 30 minutes are three `cargo
test --workspace` passes that had to be re-run for the reasons above and one that hung. The single largest cost is **queue wait**: seven
tickets for this task on GPU0 (1172, 1175, 1181, 1182, 1183, 1184, 1185), each 1-4 minutes of
actual GPU work, and a transcript audit at 01:04 found 24.4 of the first 47.8 minutes inside
idle gaps longer than 45 s, nearly all of it waiting for the queue to reach my ticket.

Build mechanics was the most expensive *engineering* phase, not the design or the code: four
`cargo test --workspace` launches (the first three were invalidated by later edits and one hung
and had to be killed), three export passes (the first died on `NameError: gemv_warps`, the
second on a stale non-empty output root, the third clean), and two `scripts/tests` runs - the
container run reported 8 failures that the host run of the same file does not reproduce
(below), so the gate was run again where AGENTS puts script tests, on the host.

### Gate results (host, this worktree)

| gate | result |
| --- | --- |
| `pytest test_fp8_projection_row_invariance.py` (GPU0, ticket 1181) | **172 passed** (baseline `e087083e`: 103 failed, 69 passed) |
| `compare-sm-exports.py --sms 188,170` | **296 programs: 296 identical, 0 different** |
| `cargo test --workspace -j 16 --locked` (host) | **58 binaries, 3053 passed, 0 failed, 228 ignored** (`--no-fail-fast`, two tests skipped: see below) |
| `.venv/bin/python -m pytest -q scripts/tests` (host) | **2965 passed, 18 skipped, 305 subtests passed, 0 failed** in 542.75s |
| M=1 row-0 bit identity, 43 consumers | **43/43 identical, 439,232 int16 values** |

Both remaining gates were re-launched at 01:34 after the last source edit (the two dead
`FP8_ROWS` deletions at 01:15) - the earlier runs predate it. The cargo gate stalls on this
host in `families::deepseek_v4::placement_tests::planner_equals_runtime_deepseek_v4` and its
`_missing_tp2_package` sibling: two full `cargo test --workspace` passes hung there with the
process alive and the log silent for 15+ minutes, and a second agent's run in another worktree
was passing `--skip` for exactly those two ids at the same time. I did not touch those tests
(this change is FP8 projections), so the gate above is the workspace run with those two ids
skipped and `--no-fail-fast`; nothing else is skipped and 0 tests fail. Note on `scripts/tests`: the
container run of 01:18 reported `8 failed, 2957 passed`, all eight in
`test_fp8_bf16_artifact_staging.py`, every one of them a version mismatch of the form
`SparkInfer revision 3fe4c930... differs from b0c81048...` on *fixture* packages the test
writes itself from the lock file it reads. `3fe4c930` is not this worktree's pin, not the
baseline's and not any lock file on the host, so that container saw a source tree other than
the one under test for the duration of that run. The same file run again on this worktree now
passes 103/103 in the container and 103/103 on the host, so the eight failures were an
environment transient (this worktree shares the repo's `.git`/submodule object store with the
other agent's worktree, which was re-pinning its own tree at the time), not this change. The
host is where AGENTS puts script tests and the host run below is the gate.

## 9. Reproduce

The five commands that produced the evidence. Fork revision `b0c81048` (branch `duel/ds-210`,
unpushed) pinned by `third_party/sparkinfer.lock.json`; the engine changes are commit
`bf87212` on branch `duel/ds-210`.

```bash
# the hard gate (needs a GPU; through the fair queue, tickets 1181/1175)
cd ~/Developer/cuteafd-ds-k210
scripts/hw/hwlock.py --locks gpu0 --wait 7200 -- scripts/build/cuteafd-dev.sh coordinator -- \
  python -m pytest third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py -q
# -> 172 passed, 71 warnings in 39.51s     (rootdir .../third_party/sparkinfer, collected 172 items)

# M=1 bit-identity and the performance table (untracked measurement scripts, same driver as the test)
scripts/hw/hwlock.py --locks gpu0 --wait 7200 -- scripts/build/cuteafd-dev.sh coordinator -- \
  python .work-k210/m1_bits.py .work-k210/m1-after.pt
scripts/hw/hwlock.py --locks gpu0 --wait 7200 -- scripts/build/cuteafd-dev.sh coordinator -- \
  python .work-k210/perf.py .work-k210/perf-after.json
python3 -c "
import torch
b=torch.load('.work-k210/m1-before.pt'); a=torch.load('.work-k210/m1-after.pt')
print(sum(torch.equal(v, a[k]) for k, v in b.items()), 'of', len(b), 'identical')"

# SM export gate (CPU-only, no GPU lock)
scripts/build/cuteafd-dev.sh cpu -- python scripts/build/compare-sm-exports.py \
  --sms 188,170 --output-dir .work-k210/sm-exports \
  --groups glm,glm2,glmf,glmf2,mimo,mimo2,mimop,mimop2,mimof,mimof2,qwen4
# -> Compared 296 programs: 296 identical objects, 0 different.

# host gates
cd rust && CARGO_TARGET_DIR=~/.cache/cuteafd/builds/ds-k210/target \
  cargo test --workspace -j 16 --locked --no-fail-fast -- \
  --skip families::deepseek_v4::placement_tests::planner_equals_runtime_deepseek_v4 \
  --skip families::deepseek_v4::placement_tests::planner_equals_runtime_deepseek_v4_missing_tp2_package
/home/tj/Developer/cuteafd/.venv/bin/python -m pytest -q scripts/tests
```

The before-state arms (`perf-before.json`, `m1-before.pt`) were produced by the same two scripts
in `~/Developer/cuteafd-ds-k210b` (a second worktree of the same engine branch point, fork at
`e087083e`) - see section 8 for their tickets. `analyze_perf.py` (scratchpad) turns the two perf
JSONs into the table in section 5 and the stop-bar verdict.
