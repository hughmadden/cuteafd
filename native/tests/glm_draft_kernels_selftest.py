"""GLM 5.3 Flash drafter kernels: the vocabulary head's tensor-op launch and the FP8 GEMM modes.

Usage: python native/tests/glm_draft_kernels_selftest.py /path/to/libcuteafd_native.so
           [--head SHARD.safetensors] [--head-shape VOCAB WIDTH] [--head-only]
           [--head-rows 8,64] [--profile-head] [--quick]
           [--mimo-forward SNAPSHOT/dflash --fp8-rows 8,64 --profile-mimo --fp8-baseline LIB]

Run with Python containing CUDA PyTorch, on one GPU.

1. Vocabulary head (`cuteafd_vocabulary_head_launch_tensor_op`, `--draft-head tensor`): BF16
   tensor cores with FP32 accumulation against an FP64 reference and against the pedantic FP32
   path (`cuteafd_vocabulary_head_launch_width`) and the few-row kernel (`cuteafd_vocab_head_rows`).
   Reports the drafter's top-16 agreement (BF16-rounded logits, larger first, lower index on
   ties, as `glm_dflash_topk` ranks them) and times each path by row count. Rank changes qualify
   only within adjacent FP64 near-ties (1e-5 relative per row).
   `--head-input` qualifies each file at its serving row count, not as one concatenated GEMM.
   `--head` reads `lm_head.weight` from a checkpoint shard; otherwise the weight is synthetic.
2. FP8 GEMMs (`cuteafd_fp8_linear`, `--draft-linear`): mode 1 (passes of 128 rows) must equal
   mode 0 (passes of 64) bit for bit at every row count; mode 2 (W8A8 past 8 rows) must equal
   mode 0 bit for bit up to 8 rows; past them it must match mode 0 to accumulation rounding on
   activations that E4M3 holds exactly (every row and 128-wide K block scaled to an amax of 448
   times a power of two), and bit for bit on ternary operands, whose every partial sum is exact
   in any accumulator (a fragment-layout slip misses by O(1) either way). The drafters take BF16
   outputs: in every mode and at every row count the BF16 output must be the FP32 output rounded
   to nearest even, bit for bit (the plain W8A16 entry point too). Times each mode on the
   drafter's GEMM shapes.
"""
import argparse
import ctypes as C
from contextlib import nullcontext
import json
from pathlib import Path
import struct
import sys

import torch

P = C.c_void_p
DEFAULT_HEAD_SHAPE = (154880, 4096)
# The DFlash2 drafter's GEMMs (k, n): attention and MLP convolutions, q|k|v, o, gate|up, down,
# the selector projection, and the context update's fc and k|v rows.
SHAPES = [(4096, 1024), (4096, 6144), (4096, 4096), (4096, 24576), (12288, 4096), (4096, 256), (20480, 4096),
          (4096, 2048)]


def bind(path):
    lib = C.CDLL(path)
    sig = {
        "cuteafd_vocab_head_rows": ([P, P, P, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_create_vocab": ([P, C.c_uint64, C.c_int32, C.c_int32, C.c_int32, C.POINTER(P)],
                                                 C.c_int32),
        "cuteafd_vocabulary_head_launch_width": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_launch_tensor_op": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_v41_markov_destroy": ([P], C.c_int32),
        "cuteafd_glm_dflash_topk_workspace": ([C.c_int32], C.c_uint64),
        "cuteafd_glm_dflash_topk": ([P, P, P, P, C.c_int32, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_fp8_w8a16_pack": ([P, P, P, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_fp8_w8a16_workspace": ([C.c_int32, C.c_int32, C.c_int32], C.c_size_t),
        "cuteafd_fp8_w8a16_linear": ([P, P, P, P, C.c_int32, C.c_int32, C.c_int32, C.c_int32, P, C.c_size_t, P],
                                     C.c_int32),
        "cuteafd_fp8_linear_workspace": ([C.c_int32, C.c_int32, C.c_int32, C.c_int32], C.c_size_t),
        "cuteafd_fp8_linear": ([P, P, P, P, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32, P, C.c_size_t, P],
                               C.c_int32),
    }
    for name, (args, ret) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, ret
    return lib


def read_head(path, vocab, width):
    """Load just the requested BF16 lm_head.weight from a safetensors shard."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        meta = header["lm_head.weight"]
        assert meta["dtype"] == "BF16" and meta["shape"] == [vocab, width], meta
        start, end = meta["data_offsets"]
        assert end - start == vocab * width * 2, meta
        f.seek(8 + size + start)
        data = bytearray(f.read(end - start))
    return torch.frombuffer(data, dtype=torch.bfloat16).view(vocab, width).cuda()


def timed(fn, runs):
    stream = torch.cuda.current_stream()
    fn()
    times = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(stream)
        fn()
        b.record(stream)
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def top16(logits):
    """The drafter's top 16 of each row: BF16-rounded logits, larger first, lower index on ties."""
    rounded = logits.to(torch.bfloat16).float()
    index = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float64)
    # Exact for |logit| < 2^20 at this vocabulary: the BF16 value, then the lower index.
    key = rounded.double() * 2.0 ** 20 - index / 2.0 ** 18
    return torch.topk(key, 16, dim=1).indices


def head_rank_mismatches(expected, candidate, oracle, epsilon):
    """Allow only changes inside groups of adjacent FP64 near-ties."""
    union = sorted(set(expected) | set(candidate), key=lambda i: (-oracle[i], i))
    groups = {}
    group = 0
    previous = None
    for token in union:
        if previous is not None and abs(oracle[token] - oracle[previous]) > epsilon:
            group += 1
        groups[token] = group
        previous = token
    bad_order = any(groups[a] != groups[b] for a, b in zip(expected, candidate))
    bad_set = sorted(groups[i] for i in expected) != sorted(groups[i] for i in candidate)
    first_differs = expected[0] != candidate[0]
    top_gap = abs(oracle[expected[0]] - oracle[expected[1]])
    bad_first = first_differs and (groups[expected[0]] != groups[candidate[0]] or top_gap > epsilon)
    return {
        "top1": int(first_differs),
        "set": int(set(expected) != set(candidate)),
        "order": int(expected != candidate),
        "bad_top1": int(bad_first),
        "bad_set": int(bad_set),
        "bad_order": int(bad_order),
    }


def head_rank_selftest():
    def check(expected, candidate, scores):
        return head_rank_mismatches(expected, candidate, dict(enumerate(scores)), 1e-5)

    assert not any(check([0, 1, 2], [0, 1, 2], [3., 2., 1.]).values())
    assert check([0, 1, 2], [1, 0, 2], [3., 3. - 1e-6, 1.])["bad_order"] == 0
    assert check([0, 1, 2], [1, 0, 2], [3., 2., 1.])["bad_top1"] == 1
    assert check([0, 1, 2], [1, 0, 2], [3., 2.999, 1.])["bad_top1"] == 1
    assert check([0, 1, 2], [0, 1, 3], [3., 2., 1., 1. - 1e-6])["bad_set"] == 0
    assert check([0, 1, 2], [0, 1, 3], [3., 2., 1., .5])["bad_set"] == 1
    assert check([0, 1, 2], [0, 2, 1], [3., 2., 1.])["bad_order"] == 1


def head_check(lib, args):
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(53)
    vocab, width = args.head_shape
    rows_all = max(24, max(args.head_rows))
    head_rank_selftest()
    batches = []
    if args.head_input:
        inputs = []
        offset = 0
        for path in args.head_input:
            data = bytearray(Path(path).read_bytes())
            assert len(data) > 0 and len(data) % (width * 2) == 0, (path, len(data), width)
            tensor = torch.frombuffer(data, dtype=torch.bfloat16).view(-1, width)
            rows = tensor.shape[0]
            assert rows in args.head_rows and rows % 8 == 0, (path, rows, args.head_rows)
            inputs.append(tensor)
            batches.append((Path(path).name, offset, rows))
            offset += rows
        x = torch.cat(inputs).cuda()
        rows_all = x.shape[0]
        assert 8 <= rows_all <= 1024 and rows_all >= max(args.head_rows), rows_all
    else:
        x = (torch.randn(rows_all, width, device="cuda") * torch.linspace(
            0.5, 2.0, rows_all, device="cuda")[:, None]).bfloat16()
        batches.append(("synthetic", 0, rows_all))
    weight = read_head(args.head, vocab, width) if args.head else (
        torch.randn(vocab, width, device="cuda") * 0.02).bfloat16()
    weight_bytes = weight.numel() * weight.element_size()
    workspace = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
    handle = P()
    assert lib.cuteafd_vocabulary_head_create_vocab(workspace.data_ptr(), workspace.numel(), width, rows_all, vocab,
                                                    C.byref(handle)) == 0
    out = {name: torch.empty(rows_all, vocab, device="cuda") for name in ("pedantic", "tensor", "few")}

    def launch(name, rows, offset=0):
        input_ptr = x[offset:].data_ptr()
        output_ptr = out[name][offset:].data_ptr()
        if name == "pedantic":
            status = lib.cuteafd_vocabulary_head_launch_width(handle, input_ptr, weight.data_ptr(),
                                                              output_ptr, rows, stream)
        elif name == "tensor":
            status = lib.cuteafd_vocabulary_head_launch_tensor_op(handle, input_ptr, weight.data_ptr(),
                                                                  output_ptr, rows, stream)
        else:
            status = lib.cuteafd_vocab_head_rows(input_ptr, weight.data_ptr(), output_ptr, rows, width,
                                                 vocab, stream)
        assert status == 0, (name, rows, status)

    try:
        for _, offset, rows in batches:
            for name in out:
                launch(name, rows, offset)
        torch.cuda.synchronize()
        # Real inputs qualify every row; synthetic smoke retains its eight-row FP64 gate.
        reference_rows = rows_all if args.head_input else 8
        ref = torch.cat([x[:reference_rows].double() @ weight[v:v + 16384].double().T
                         for v in range(0, vocab, 16384)], dim=1)
        scale = ref.abs().amax().item()
        reference_top = top16(ref)
        expected_rows = reference_top.cpu().tolist()
        epsilons = (ref.abs().amax(dim=1) * 1e-5).cpu().tolist()
        failures = []
        rank_counts = {}
        for name in out:
            err = (out[name][:reference_rows].double() - ref).abs().amax().item() / scale
            print(f"head: {name} max |error| / max |logit| over {reference_rows} rows: {err:.2e}")
            if err >= (1e-3 if name == "tensor" else 1e-5):
                failures.append((name, "logit error", err))
            candidate = top16(out[name][:reference_rows])
            candidate_rows = candidate.cpu().tolist()
            indices = torch.cat((reference_top, candidate), dim=1)
            oracle_rows = ref.gather(1, indices).cpu().tolist()
            counts = {key: 0 for key in ("top1", "set", "order", "bad_top1", "bad_set", "bad_order")}
            details = []
            for row, (expected, actual, scores) in enumerate(zip(
                    expected_rows, candidate_rows, oracle_rows)):
                ids = expected + actual
                oracle = dict(zip(ids, scores))
                result = head_rank_mismatches(expected, actual, oracle, epsilons[row])
                for key, value in result.items():
                    counts[key] += value
                if result["order"] and len(details) < 8:
                    source, offset, _ = next(batch for batch in batches if batch[1] <= row < batch[1] + batch[2])
                    changed = [(a, b, abs(oracle[a] - oracle[b]) / epsilons[row])
                               for a, b in zip(expected, actual) if a != b]
                    details.append((source, row - offset, changed, result))
            rank_counts[name] = counts
            print(f"head: {name} FP64 rank mismatches over {reference_rows} rows: {counts}; "
                  "epsilon=1e-5 * row max |FP64 logit|")
            for detail in details:
                print(f"head: {name} mismatch source,row,(expected,actual,gap/epsilon),counts: {detail}")
        for name, counts in rank_counts.items():
            for key in ("bad_top1", "bad_set", "bad_order"):
                if counts[key] > rank_counts["pedantic"][key] or rank_counts["pedantic"][key]:
                    failures.append((name, key, counts[key], "pedantic", rank_counts["pedantic"][key]))
        baseline = top16(out["pedantic"])
        for name in ("tensor", "few"):
            candidate = top16(out[name])
            same_first = (baseline[:, 0] == candidate[:, 0]).float().mean().item()
            same_top = (baseline == candidate).all(dim=1).float().mean().item()
            same_set = sum(set(baseline[r].tolist()) == set(candidate[r].tolist())
                           for r in range(rows_all)) / rows_all
            print(f"head: {name} against pedantic over {rows_all} rows: top-1 {100 * same_first:.2f}%, "
                  f"top-16 set {100 * same_set:.2f}%, top-16 in order {100 * same_top:.2f}%")
            if not args.head_input and name == "tensor":
                assert same_first >= 0.95

        # Exercise the serving selector on 8-row blocks, excluding each anchor.
        drafted = rows_all // 8 * 7
        unary = torch.empty(drafted, 16, device="cuda")
        candidates = torch.empty(drafted, 16, device="cuda", dtype=torch.int32)
        top_workspace = torch.empty(lib.cuteafd_glm_dflash_topk_workspace(drafted), device="cuda", dtype=torch.uint8)

        def select(name, rows):
            assert lib.cuteafd_glm_dflash_topk(out[name].data_ptr(), unary.data_ptr(), candidates.data_ptr(),
                top_workspace.data_ptr(), rows // 8, 8, 7, vocab, stream) == 0

        for name in out:
            select(name, rows_all // 8 * 8)
            torch.cuda.synchronize()
            expected = top16(out[name][:rows_all // 8 * 8].view(-1, 8, vocab)[:, 1:].reshape(-1, vocab))
            assert torch.equal(candidates.long(), expected), ("native selector", name)

        runs = 5 if args.quick else 20
        print(f"head [{vocab},{width}]: {weight_bytes} BF16 bytes; bandwidth assumes one useful weight read "
              "per launch, not hardware-measured traffic")
        for rows in args.head_rows:
            for name in out:
                elapsed = timed(lambda: launch(name, rows), runs)
                print(f"head rows={rows} route={name} median_ms={elapsed:.6f} "
                      f"useful_weight_GB_s={weight_bytes / elapsed / 1e6:.1f}")
                if rows % 8 == 0:
                    elapsed = timed(lambda: select(name, rows), runs)
                    print(f"topk rows={rows} route={name} median_ms={elapsed:.6f}")

        if args.profile_head:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                for rows in args.head_rows:
                    for name in out:
                        with torch.cuda.nvtx.range(f"head/{name}/rows{rows}"):
                            launch(name, rows)
                        if rows % 8 == 0:
                            with torch.cuda.nvtx.range(f"topk/{name}/rows{rows}"):
                                select(name, rows)
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
        assert not failures, ("head qualification", failures)
    finally:
        torch.cuda.synchronize()
        assert lib.cuteafd_v41_markov_destroy(handle) == 0


def e4m3_exact_rows(rows, k, generator):
    """BF16 activations that E4M3 holds exactly at the W8A8 scale: E4M3 values times 2^j per row and
    128-wide block, each block with one entry of magnitude 448 * 2^j (so amax / 448 = 2^j)."""
    codes = torch.randint(0, 256, (rows, k), generator=generator, dtype=torch.int32)
    codes = codes[(codes & 0x7F) != 0x7F]  # no NaN codes
    codes = torch.cat([codes, torch.zeros(rows * k, dtype=torch.int32)])[:rows * k].view(rows, k)
    values = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    values[:, ::128] = 448.0 * torch.where(torch.rand(rows, k // 128, generator=generator) < 0.5, -1.0, 1.0)
    exponents = torch.randint(-12, -2, (rows, k // 128), generator=generator).float()
    values = values * torch.repeat_interleave(torch.exp2(exponents), 128, dim=1)
    assert torch.equal(values.bfloat16().float(), values)
    return values.bfloat16().cuda()


def ternary(rows, k, scale, generator):
    """{-1, 0, 1} * scale per row and 128-wide block (2^j for activations), the first entry of
    each block 1, so every block's amax is its scale."""
    values = torch.randint(-1, 2, (rows, k), generator=generator).float()
    values[:, ::128] = 1.0
    return (values * scale).bfloat16().cuda()


def fp8_check(lib, args):
    stream = torch.cuda.current_stream().cuda_stream
    generator = torch.Generator().manual_seed(8)
    shapes = args.fp8_shapes or (SHAPES[:3] if args.quick else SHAPES)
    profile_buffers = []
    baseline = bind(args.fp8_baseline) if args.fp8_baseline else None
    # 136 and 200 exercise a full pass followed by the skinny or wide remainder.
    row_counts = sorted(set((1, 8, 9, 16, 33, 64, 65, 96, 127, 128, 136, 200)) | set(args.fp8_rows))
    for k, n in shapes:
        weight = (torch.randn(n, k, generator=generator) * 0.02).bfloat16().cuda()
        packed = torch.empty(n * k, device="cuda", dtype=torch.uint8)
        scale = torch.empty(n * k // 128, device="cuda", dtype=torch.float32)
        assert lib.cuteafd_fp8_w8a16_pack(weight.data_ptr(), packed.data_ptr(), scale.data_ptr(), n, k, 1, stream) == 0
        rows_max = max(row_counts)
        sizes = [lib.cuteafd_fp8_linear_workspace(rows_max, k, n, mode) for mode in (0, 1, 2)]
        assert sizes[0] == lib.cuteafd_fp8_w8a16_workspace(rows_max, k, n), sizes
        assert sizes[0] <= sizes[1] <= sizes[2], sizes
        workspace = torch.empty(sizes[2], device="cuda", dtype=torch.uint8)

        def run(mode, x, rows, out):
            # FP32 output for an FP32 `out`, BF16 for a BF16 one.
            out_f32 = int(out.dtype == torch.float32)
            status = lib.cuteafd_fp8_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), out.data_ptr(), out_f32,
                                            rows, k, n, mode, workspace.data_ptr(), workspace.numel(), stream)
            assert status == 0, (k, n, mode, rows, out_f32, status)

        if baseline:
            baseline_bytes = max(baseline.cuteafd_fp8_linear_workspace(rows_max, k, n, mode) for mode in (0, 1, 2))
            baseline_workspace = torch.empty(baseline_bytes, device="cuda", dtype=torch.uint8)

            def run_baseline(mode, x, rows, out):
                status = baseline.cuteafd_fp8_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), out.data_ptr(),
                    int(out.dtype == torch.float32), rows, k, n, mode, baseline_workspace.data_ptr(),
                    baseline_workspace.numel(), stream)
                assert status == 0, ("baseline", k, n, mode, rows, status)

            # Include power-of-two boundaries, padding rows, and wide-pass tails.
            edge = torch.empty(rows_max, k, dtype=torch.bfloat16)
            for row in range(rows_max):
                if row % 7 == 0:
                    edge[row].zero_()
                else:
                    exponent = (row % 41) - 20
                    value = 2.0 ** exponent
                    edge[row].fill_(value * (1.0 + (row % 3) / 128.0))
                    edge[row, 1::2].neg_()
            for label, source in (("random", None), ("pow2", edge.cuda())):
                if source is None:
                    source = (torch.randn(rows_max, k, generator=generator) * 3.0).bfloat16().cuda()
                for rows in row_counts:
                    for mode in (0, 1, 2):
                        for dtype in (torch.float32, torch.bfloat16):
                            candidate = torch.empty(rows, n, device="cuda", dtype=dtype)
                            reference = torch.empty_like(candidate)
                            run(mode, source, rows, candidate)
                            run_baseline(mode, source, rows, reference)
                            torch.cuda.synchronize()
                            assert torch.equal(candidate, reference), ("baseline bits", label, k, n, rows, mode, dtype)
            print(f"fp8 [{k}, {n}]: baseline bit equality passed at {row_counts}")

        x = (torch.randn(rows_max, k, generator=generator) * torch.linspace(0.1, 3.0, rows_max)[:, None]).bfloat16()
        x = x.cuda()
        exact = e4m3_exact_rows(rows_max, k, generator)
        # Ternary operands: weights {-1, 0, 1} / 64 (E4M3 256 at the pow2 scale 2^-14), activations
        # {-1, 0, 1} * 448 * 2^j per block; every product and partial sum is exact.
        weight3 = ternary(n, k, 2.0 ** -6, generator)
        packed3 = torch.empty(n * k, device="cuda", dtype=torch.uint8)
        scale3 = torch.empty(n * k // 128, device="cuda", dtype=torch.float32)
        assert lib.cuteafd_fp8_w8a16_pack(weight3.data_ptr(), packed3.data_ptr(), scale3.data_ptr(), n, k, 1,
                                          stream) == 0
        x3 = ternary(rows_max, k, torch.repeat_interleave(
            448.0 * torch.exp2(torch.randint(-12, -2, (rows_max, k // 128), generator=generator).float()), 128, dim=1),
            generator)
        for rows in row_counts:
            outs = {}
            for mode in (0, 1, 2):
                outs[mode] = torch.full((rows, n), float("nan"), device="cuda")
                run(mode, x, rows, outs[mode])
            # The plain entry point is mode 0.
            legacy = torch.empty(rows, n, device="cuda")
            assert lib.cuteafd_fp8_w8a16_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), legacy.data_ptr(), 1,
                                                rows, k, n, workspace.data_ptr(), workspace.numel(), stream) == 0
            # BF16 outputs (the drafters' dtype): the FP32 ones rounded, in every mode.
            outs16 = {}
            for mode in (0, 1, 2):
                outs16[mode] = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
                run(mode, x, rows, outs16[mode])
            legacy16 = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
            assert lib.cuteafd_fp8_w8a16_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), legacy16.data_ptr(),
                                                0, rows, k, n, workspace.data_ptr(), workspace.numel(), stream) == 0
            torch.cuda.synchronize()
            assert torch.equal(legacy, outs[0]), (k, n, rows)
            assert torch.equal(legacy16, outs[0].bfloat16()), ("BF16 output of the W8A16 entry point", k, n, rows)
            for mode in (0, 1, 2):
                assert torch.equal(outs16[mode], outs[mode].bfloat16()), ("BF16 output is not the FP32 one rounded",
                                                                          k, n, mode, rows)
            assert torch.equal(outs[1], outs[0]), ("wide is not bit-identical", k, n, rows)
            if rows <= 8:
                assert torch.equal(outs[2], outs[0]), ("w8a8 changed one draft block", k, n, rows)
            else:
                rms = outs[0].pow(2).mean().sqrt().item()
                drift = (outs[2] - outs[0]).pow(2).mean().sqrt().item() / rms
                exact0, exact2 = torch.empty(rows, n, device="cuda"), torch.empty(rows, n, device="cuda")
                run(0, exact, rows, exact0)
                run(2, exact, rows, exact2)
                torch.cuda.synchronize()
                # Both sum exact products in FP32; FP8 tensor cores may hold fewer bits while
                # they sum within an MMA. A wrong fragment layout misses by O(1).
                bound = exact0.abs().amax().item() * 2e-3
                gap = (exact2 - exact0).abs().amax().item()
                assert gap <= bound, ("w8a8 differs on exact activations", k, n, rows, gap, bound)
                tern = [torch.empty(rows, n, device="cuda") for _ in range(2)]
                for mode, out in zip((0, 2), tern):
                    assert lib.cuteafd_fp8_linear(x3.data_ptr(), packed3.data_ptr(), scale3.data_ptr(), out.data_ptr(),
                                                  1, rows, k, n, mode, workspace.data_ptr(), workspace.numel(),
                                                  stream) == 0
                torch.cuda.synchronize()
                assert torch.equal(tern[0], tern[1]), ("w8a8 differs on ternary operands", k, n, rows)
                if rows in (16, 128):
                    print(f"fp8 [{k}, {n}] {rows} rows: w8a8 relative RMS drift {drift:.2e} (E4M3 activations); "
                          f"on exact activations max |gap| {gap:.2e} (bound {bound:.2e})")
        runs = 5 if args.quick else 20
        weight_bytes = packed.numel() + scale.numel() * scale.element_size()
        for rows in args.fp8_rows:
            out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            for mode in (0, 1, 2):
                elapsed = timed(lambda: run(mode, x, rows, out), runs)
                print(f"fp8 k={k} n={n} rows={rows} mode={mode} median_ms={elapsed:.6f} "
                      f"useful_weight_GB_s={weight_bytes / elapsed / 1e6:.1f}")
                if baseline:
                    base_ms = timed(lambda: run_baseline(mode, x, rows, out), runs)
                    print(f"fp8 baseline k={k} n={n} rows={rows} mode={mode} median_ms={base_ms:.6f} "
                          f"speedup={base_ms / elapsed:.4f}")
        if args.profile_fp8:
            profile_buffers.append((k, n, x, packed, scale, workspace,
                                    torch.empty(max(args.fp8_rows), n, device="cuda", dtype=torch.bfloat16)))
    if args.profile_fp8:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            for k, n, x, packed, scale, workspace, out in profile_buffers:
                for rows in args.fp8_rows:
                    for mode in (0, 1, 2):
                        with torch.cuda.nvtx.range(f"fp8/k{k}/n{n}/rows{rows}/mode{mode}"):
                            assert lib.cuteafd_fp8_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(),
                                out.data_ptr(), 0, rows, k, n, mode, workspace.data_ptr(), workspace.numel(), stream) == 0
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()


def mimo_forward_check(lib, args):
    """Checkpoint-weight forward census; fixed synthetic embeddings/KV, not a serving acceptance test."""
    from safetensors import safe_open

    sig = {
        "cuteafd_glm_dflash_rmsnorm": ([P, P, P, C.c_int32, C.c_int32, C.c_float, P], C.c_int32),
        "cuteafd_glm_dflash_silu_mul": ([P, P, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_mimo_dflash_qk_rope": ([P] * 8 + [C.c_int32] * 4 + [C.c_float] * 3 + [P], C.c_int32),
        "cuteafd_mimo_dflash_attention_workspace": ([C.c_int32] * 5, C.c_uint64),
        "cuteafd_mimo_dflash_attention": ([P] * 11 + [C.c_int32] * 7 + [C.c_float, P], C.c_int32),
        "cuteafd_mimo_dflash_add_norm": ([P] * 4 + [C.c_int32] * 2 + [C.c_float, P], C.c_int32),
    }
    baseline = bind(args.fp8_baseline) if args.fp8_baseline else None
    for native in (lib, baseline):
        if native is not None:
            for name, (types, ret) in sig.items():
                getattr(native, name).argtypes, getattr(native, name).restype = types, ret
    directory = Path(args.mimo_forward)
    cfg = json.loads((directory / "config.json").read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kv_heads, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    block, ring = cfg["block_size"], cfg["sliding_window"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    rope_dim = int(dim * cfg.get("partial_rotary_factor", 1.0))
    v_scale = cfg["dflash_config"].get("attention_value_scale", 1.0)
    assert dim == 128 and ring == 1024 and not cfg.get("is_causal", False), cfg
    assert all(rows % block == 0 for rows in args.fp8_rows), (args.fp8_rows, block)
    assert 0 <= args.mimo_context <= ring, args.mimo_context
    max_rows = max(args.fp8_rows)
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(26)
    shapes = [(h, (heads + 2 * kv_heads) * dim), (heads * dim, h), (h, 2 * inter), (inter, h)]

    def pack(weight, k, n):
        assert weight.dtype == torch.bfloat16 and list(weight.shape) == [n, k], weight.shape
        packed = torch.empty(n * k, device="cuda", dtype=torch.uint8)
        scale = torch.empty(n * k // 128, device="cuda")
        assert lib.cuteafd_fp8_w8a16_pack(weight.data_ptr(), packed.data_ptr(), scale.data_ptr(),
                                        n, k, args.mimo_scale_rule, stream) == 0
        torch.cuda.synchronize()
        return packed, scale

    layers = []
    with safe_open(directory / "dflash_draft_model.safetensors", framework="pt", device="cpu") as checkpoint:
        def tensor(name):
            result = checkpoint.get_tensor(name)
            assert result.dtype == torch.bfloat16, (name, result.dtype)
            return result.cuda()

        final_norm = tensor("norm.weight")
        for index in range(cfg["num_hidden_layers"]):
            p = f"layers.{index}"
            a = f"{p}.self_attn"
            matrices = [torch.cat([tensor(f"{a}.{name}_proj.weight") for name in ("q", "k", "v")]),
                        tensor(f"{a}.o_proj.weight"),
                        torch.cat([tensor(f"{p}.mlp.{name}_proj.weight") for name in ("gate", "up")]),
                        tensor(f"{p}.mlp.down_proj.weight")]
            weights = [pack(weight, k, n) for weight, (k, n) in zip(matrices, shapes)]
            del matrices
            layers.append(dict(weights=weights, input_norm=tensor(f"{p}.input_layernorm.weight"),
                post_norm=tensor(f"{p}.post_attention_layernorm.weight"), q_norm=tensor(f"{a}.q_norm.weight"),
                k_norm=tensor(f"{a}.k_norm.weight"),
                sinks=tensor(f"{a}.attention_sink_bias") if cfg["dflash_config"].get("attention_sink_bias") else None,
                k_ring=(torch.randn(max_rows // block, ring, kv_heads, dim, device="cuda") * 0.1).bfloat16(),
                v_ring=(torch.randn(max_rows // block, ring, kv_heads, dim, device="cuda") * 0.1).bfloat16()))
    seed = (torch.randn(max_rows, h, device="cuda") * 0.2).bfloat16()

    def bank(native):
        buffers = {name: torch.empty(max_rows, width, device="cuda", dtype=torch.bfloat16)
                   for name, width in (("h", h), ("n", h), ("qkv", shapes[0][1]), ("q", heads * dim),
                                       ("k", kv_heads * dim), ("v", kv_heads * dim), ("attn", heads * dim),
                                       ("delta", h), ("gate_up", 2 * inter), ("act", inter))}
        buffers["scratch"] = torch.empty(max(native.cuteafd_fp8_linear_workspace(max_rows, k, n, 0)
                                               for k, n in shapes), device="cuda", dtype=torch.uint8)
        buffers["attention"] = torch.empty(native.cuteafd_mimo_dflash_attention_workspace(
            max_rows // block, heads, kv_heads, block, ring + block), device="cuda", dtype=torch.uint8)
        return buffers

    candidate = bank(lib)
    reference = bank(baseline) if baseline else None

    tables = {rows: {
        "slots": torch.arange(rows // block, device="cuda", dtype=torch.int32),
        "lengths": torch.full((rows // block,), args.mimo_context, device="cuda", dtype=torch.int32),
        "ends": torch.full((rows // block,), args.mimo_context, device="cuda", dtype=torch.int32),
        "positions": torch.arange(args.mimo_context, args.mimo_context + block,
                                  device="cuda", dtype=torch.int64).repeat(rows // block),
    } for rows in args.fp8_rows}

    def run(native, buffers, rows, labels=False):
        table = tables[rows]
        ptr = lambda name: buffers[name].data_ptr()
        buffers["h"][:rows].copy_(seed[:rows])
        assert native.cuteafd_glm_dflash_rmsnorm(ptr("h"), layers[0]["input_norm"].data_ptr(), ptr("n"),
                                                rows, h, eps, stream) == 0
        for index, layer in enumerate(layers):
            with torch.cuda.nvtx.range(f"mimo/layer{index}/rows{rows}") if labels else nullcontext():
                def linear(which, source, target):
                    k, n = shapes[which]
                    packed, scale = layer["weights"][which]
                    assert native.cuteafd_fp8_linear(ptr(source), packed.data_ptr(), scale.data_ptr(), ptr(target),
                        0, rows, k, n, 0, ptr("scratch"), buffers["scratch"].numel(), stream) == 0

                linear(0, "n", "qkv")
                assert native.cuteafd_mimo_dflash_qk_rope(ptr("qkv"), layer["q_norm"].data_ptr(),
                    layer["k_norm"].data_ptr(), table["positions"].data_ptr(), None, ptr("q"), ptr("k"), ptr("v"),
                    rows, heads, kv_heads, rope_dim, theta, eps, v_scale, stream) == 0
                assert native.cuteafd_mimo_dflash_attention(ptr("q"), ptr("k"), ptr("v"),
                    layer["k_ring"].data_ptr(), layer["v_ring"].data_ptr(), table["slots"].data_ptr(),
                    table["lengths"].data_ptr(), table["ends"].data_ptr(),
                    layer["sinks"].data_ptr() if layer["sinks"] is not None else None, ptr("attn"), ptr("attention"),
                    rows // block, block, heads, kv_heads, ring, ring + block, ring, dim ** -0.5, stream) == 0
                linear(1, "attn", "delta")
                assert native.cuteafd_mimo_dflash_add_norm(ptr("h"), ptr("delta"), layer["post_norm"].data_ptr(),
                                                          ptr("n"), rows, h, eps, stream) == 0
                linear(2, "n", "gate_up")
                assert native.cuteafd_glm_dflash_silu_mul(ptr("gate_up"), ptr("act"), rows, inter, stream) == 0
                linear(3, "act", "delta")
                next_norm = layers[index + 1]["input_norm"] if index + 1 < len(layers) else final_norm
                assert native.cuteafd_mimo_dflash_add_norm(ptr("h"), ptr("delta"), next_norm.data_ptr(), ptr("n"),
                                                          rows, h, eps, stream) == 0

    head_weight = read_head(args.head, cfg["vocab_size"], h) if args.head else None
    if head_weight is not None:
        head_workspace = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
        head_handle = P()
        assert lib.cuteafd_vocabulary_head_create_vocab(head_workspace.data_ptr(), head_workspace.numel(),
            h, max_rows, cfg["vocab_size"], C.byref(head_handle)) == 0
        head_logits = torch.empty(max_rows, cfg["vocab_size"], device="cuda")
        head_unary = torch.empty(max_rows // block * (block - 1), 16, device="cuda")
        head_candidates = torch.empty_like(head_unary, dtype=torch.int32)
        top_workspace = torch.empty(lib.cuteafd_glm_dflash_topk_workspace(head_unary.shape[0]),
                                    device="cuda", dtype=torch.uint8)

        def tokens(buffers, rows):
            assert lib.cuteafd_vocabulary_head_launch_width(head_handle, buffers["n"].data_ptr(),
                head_weight.data_ptr(), head_logits.data_ptr(), rows, stream) == 0
            assert lib.cuteafd_glm_dflash_topk(head_logits.data_ptr(), head_unary.data_ptr(),
                head_candidates.data_ptr(), top_workspace.data_ptr(), rows // block, block, block - 1,
                cfg["vocab_size"], stream) == 0
            return head_candidates[:rows // block * (block - 1)].clone()

    try:
        for rows in args.fp8_rows:
            run(lib, candidate, rows)
            if baseline:
                run(baseline, reference, rows)
                torch.cuda.synchronize()
                for name in ("h", "n", "qkv", "attn", "gate_up", "act"):
                    assert torch.equal(candidate[name][:rows], reference[name][:rows]), ("forward bits", rows, name)
                print(f"mimo forward rows={rows}: checkpoint-weight fixed synthetic-input baseline bits passed")
                if head_weight is not None:
                    actual, expected = tokens(candidate, rows), tokens(reference, rows)
                    torch.cuda.synchronize()
                    assert torch.equal(actual, expected), ("fixed native draft top16", rows)
                    print(f"mimo forward rows={rows}: fixed synthetic-input draft tokens/top16 bit-identical")
            elapsed = timed(lambda: run(lib, candidate, rows), 5 if args.quick else 20)
            print(f"mimo forward rows={rows} context={args.mimo_context} median_ms={elapsed:.6f}")
            if baseline:
                base_ms = timed(lambda: run(baseline, reference, rows), 5 if args.quick else 20)
                print(f"mimo forward baseline rows={rows} median_ms={base_ms:.6f} speedup={base_ms / elapsed:.4f}")
        if args.profile_mimo:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                for rows in args.fp8_rows:
                    with torch.cuda.nvtx.range(f"mimo/forward/rows{rows}"):
                        run(lib, candidate, rows, labels=True)
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()

    finally:
        torch.cuda.synchronize()
        if head_weight is not None:
            assert lib.cuteafd_v41_markov_destroy(head_handle) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--head", help="checkpoint shard holding lm_head.weight (default: a synthetic head)")
    parser.add_argument("--head-input", action="append", help="actual BF16 final-norm row dump; may repeat")
    parser.add_argument("--head-shape", nargs=2, type=int, metavar=("VOCAB", "WIDTH"),
                        default=DEFAULT_HEAD_SHAPE, help="expected BF16 head dimensions")
    parser.add_argument("--head-rows", default="8,16,24,32,64,96,128", help="comma-separated row counts, at most 128")
    parser.add_argument("--profile-head", action="store_true",
                        help="one NVTX-labelled head/top-k shot inside cudaProfilerStart/Stop after checks")
    parser.add_argument("--fp8-shapes", help="comma-separated KxN shapes; overrides --quick's shape subset")
    parser.add_argument("--fp8-baseline", help="sealed baseline library for exact FP8 output comparison")
    parser.add_argument("--fp8-rows", default="8,16,32,64,128", help="comma-separated profiling row counts")
    parser.add_argument("--profile-fp8", action="store_true",
                        help="one NVTX-labelled projection shot after checks; use with --skip-head")
    parser.add_argument("--mimo-forward", help="MiMo dflash directory; run its forward instead of separate head/FP8 checks")
    parser.add_argument("--mimo-context", type=int, default=1024, help="fixed synthetic ring context length")
    parser.add_argument("--mimo-scale-rule", type=int, choices=(0, 1, 2), default=0,
                        help="FP8 weight scale rule: 0 amax (MiMo default), 1 pow2, 2 best")
    parser.add_argument("--profile-mimo", action="store_true", help="NVTX-labelled native forward after checks")
    parser.add_argument("--quick", action="store_true", help="three GEMM shapes, fewer timing runs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-head", action="store_true")
    group.add_argument("--head-only", action="store_true", help="skip unrelated FP8 checks")
    args = parser.parse_args()
    try:
        args.head_rows = [int(value) for value in args.head_rows.split(",")]
    except ValueError:
        parser.error("--head-rows requires comma-separated integers")
    if not args.head_rows or any(rows < 1 or rows > 128 for rows in args.head_rows):
        parser.error("--head-rows must be in [1,128]")
    vocab, width = args.head_shape
    if vocab < 1 or vocab > 64 * 12 * 256 or width < 256 or width > 6336 or width % 256:
        parser.error("head shape requires a supported vocabulary and width: multiple of 256, at most 6336")
    if args.profile_head and args.skip_head:
        parser.error("--profile-head requires head checks")
    if args.profile_fp8 and (args.head_only or args.profile_head or not args.skip_head):
        parser.error("--profile-fp8 requires --skip-head and one capture range")
    try:
        args.fp8_rows = [int(value) for value in args.fp8_rows.split(",")]
        args.fp8_shapes = [tuple(int(value) for value in shape.lower().split("x"))
                           for shape in args.fp8_shapes.split(",")] if args.fp8_shapes else None
    except ValueError:
        parser.error("invalid FP8 row counts or KxN shapes")
    if not args.fp8_rows or any(rows < 1 or rows > 2048 for rows in args.fp8_rows):
        parser.error("--fp8-rows must be in [1,2048]")
    if args.fp8_shapes and any(len(shape) != 2 or shape[0] < 128 or shape[0] % 128 or
                              shape[1] < 16 or shape[1] % 16 for shape in args.fp8_shapes):
        parser.error("--fp8-shapes requires K multiples of 128 and N multiples of 16")
    if args.profile_mimo and not args.mimo_forward:
        parser.error("--profile-mimo requires --mimo-forward")
    if args.mimo_forward and (args.profile_fp8 or args.profile_head or args.head_only):
        parser.error("--mimo-forward has its own profile range and cannot run separate head/FP8 checks")
    lib = bind(args.library)
    print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")
    if args.mimo_forward:
        mimo_forward_check(lib, args)
    else:
        if not args.skip_head:
            head_check(lib, args)
        if not args.head_only:
            fp8_check(lib, args)
    print("glm draft kernels: all checks passed")


if __name__ == "__main__":
    sys.exit(main())
