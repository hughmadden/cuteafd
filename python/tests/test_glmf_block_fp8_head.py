"""Engine block-FP8 head wrapper against FP64; opt in in the CUDA image."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

if os.environ.get("CUTEAFD_GLMF_BLOCK_HEAD_GPU") != "1":
    pytest.skip("requires explicit GLM Flash block-FP8 head CUDA gate", allow_module_level=True)

torch = pytest.importorskip("torch")


@torch.no_grad()
def test_engine_block_fp8_head_fp64_and_write_bounds():
    root = Path(os.environ.get("CUTEAFD_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    spec = importlib.util.spec_from_file_location("glmf_head_exporter", root / "python/tools/aot/export_b12x_dsv4_aot.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    import cutlass
    from b12x.integration.cuteafd import GLM53_FLASH
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection

    vocab, hidden, max_rows = 154880, GLM53_FLASH.hidden, 257
    assert hidden == 4096
    program = exporter.compile_glmf_head_block_fp8(GLM53_FLASH)
    bf16_projection = TmaBf16Projection(vocab, hidden, out_dtype=cutlass.Float32, tiles_by_rows=())
    bf16_program = compile_program(bf16_projection, name="glmf_head_bf16_oracle", operands=(
        Operand("x", torch.bfloat16, f"[rows,{hidden}]"),
        Operand("w", torch.bfloat16, f"[{vocab},{hidden}]"),
        Operand("logits", torch.float32, f"[rows,{vocab}]", "out")),
        scalars=(Scalar("rows"),), key=(bf16_projection.key(), "f32"))
    generator = torch.Generator(device="cuda").manual_seed(1795)
    inputs = torch.randn((max_rows, hidden), generator=generator, device="cuda").bfloat16()
    weights = torch.empty((vocab, hidden), device="cuda", dtype=torch.float8_e4m3fn)
    bf16_weights = torch.empty((vocab, hidden), device="cuda", dtype=torch.bfloat16)
    scales = torch.empty((vocab // 128, hidden // 128), device="cuda", dtype=torch.float32)
    oracle = torch.empty((max_rows, vocab), device="cuda", dtype=torch.float64)
    magnitude = torch.empty_like(oracle)
    raw_oracle = torch.empty_like(oracle)
    x64 = inputs.double()
    for first in range(0, vocab, 4096):
        count = min(4096, vocab - first)
        blocks = (torch.randn((count // 128, 128, hidden // 128, 128),
            generator=generator, device="cuda") * 0.02).bfloat16().float()
        scale = (blocks.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-12)
        quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
        weights[first:first + count] = quantized.reshape(count, hidden)
        scales[first // 128:(first + count) // 128] = scale
        raw = (quantized.float() * scale[:, None, :, None]).reshape(count, hidden)
        # Both routes widen scaled E4M3 weights through BF16 before MMA.
        bf16_weights[first:first + count] = raw.bfloat16()
        dequantized = bf16_weights[first:first + count].double()
        oracle[:, first:first + count] = x64 @ dequantized.T
        magnitude[:, first:first + count] = x64.abs() @ dequantized.abs().T
        raw_oracle[:, first:first + count] = x64 @ raw.double().T
    sentinel = -123456.0
    for rows in (1, 4, 8, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257):
        guarded = torch.full((rows + 2, vocab), sentinel, device="cuda", dtype=torch.float32)
        out = guarded[1:rows + 1]
        program.launch(inputs[:rows], weights, scales, out, scalars=(rows,))
        torch.cuda.synchronize()
        assert bool((guarded[[0, -1]] == sentinel).all())
        assert bool(torch.isfinite(out).all())
        error = (out.double() - oracle[:rows]).abs()
        bound = 1e-4 * magnitude[:rows] + 1e-6
        assert bool((error <= bound).all()), (rows, float(error.max()))
        relative = ((out.double() - raw_oracle[:rows]).square().sum(-1) /
            raw_oracle[:rows].square().sum(-1)).sqrt()
        assert bool((relative < 0.006).all()), (rows, relative.tolist())
        first_result = out.clone()
        for row in range(rows):
            program.launch(inputs[row:row + 1], weights, scales, out[row:row + 1], scalars=(1,))
        torch.cuda.synchronize()
        row_exact = torch.equal(out, first_result)
        if rows <= 16:
            assert row_exact, f"rows={rows}: GEMV span differs from row launch"
        # GEMM may reorder sums, but cannot change the widened BF16 operands.
        assert bool(((out.double() - oracle[:rows]).abs() <= bound).all())
        bf16_program.launch(inputs[:rows], bf16_weights, out, scalars=(rows,))
        torch.cuda.synchronize()
        assert bool(((out.double() - oracle[:rows]).abs() <= bound).all())
        bf16_exact = torch.equal(out, first_result)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(inputs[:rows], weights, scales, out, scalars=(rows,))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first_result), f"rows={rows}: graph replay differs"
        assert bool((guarded[[0, -1]] == sentinel).all())
        print(json.dumps(dict(rows=rows, shape=[vocab, hidden],
            route="gemv" if rows <= 16 else "w8a16",
            max_abs_error=float(error.max()), max_relative_raw_fp64=float(relative.max()),
            canaries=True, row_exact=row_exact, bf16_exact=bf16_exact, graph_exact=True)))
        if os.environ.get("CUTEAFD_GLMF_BLOCK_HEAD_TIMING") == "1":
            bf16_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(bf16_graph):
                bf16_program.launch(inputs[:rows], bf16_weights, out, scalars=(rows,))
            times = {}
            for name, candidate in (("block_fp8", graph), ("bf16", bf16_graph)):
                for _ in range(3):
                    candidate.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    candidate.replay()
                end.record()
                end.synchronize()
                times[name] = start.elapsed_time(end) / 20
            print(json.dumps(dict(rows=rows, milliseconds=times,
                microseconds_per_row={name: ms * 1000 / rows for name, ms in times.items()},
                speedup=times["bf16"] / times["block_fp8"])))
