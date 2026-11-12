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
    from b12x.integration.cuteafd import GLM53_FLASH

    vocab, hidden, max_rows = 154880, GLM53_FLASH.hidden, 65
    assert hidden == 4096
    program = exporter.compile_glmf_head_block_fp8(GLM53_FLASH)
    generator = torch.Generator(device="cuda").manual_seed(1795)
    inputs = torch.randn((max_rows, hidden), generator=generator, device="cuda").bfloat16()
    weights = torch.empty((vocab, hidden), device="cuda", dtype=torch.float8_e4m3fn)
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
        # Canonical MmaFp8Gemv widens scaled weights through BF16 before MMA.
        dequantized = raw.bfloat16().double()
        oracle[:, first:first + count] = x64 @ dequantized.T
        magnitude[:, first:first + count] = x64.abs() @ dequantized.abs().T
        raw_oracle[:, first:first + count] = x64 @ raw.double().T
    sentinel = -123456.0
    for rows in (1, 4, 8, 15, 16, 17, 31, 64, 65):
        guarded = torch.full((rows + 2, vocab), sentinel, device="cuda", dtype=torch.float32)
        out = guarded[1:rows + 1]
        for first in range(0, rows, 16):
            count = min(16, rows - first)
            program.launch(inputs[first:first + count], weights, scales,
                out[first:first + count], scalars=(count,))
        torch.cuda.synchronize()
        assert torch.equal(guarded[0], torch.full_like(guarded[0], sentinel))
        assert torch.equal(guarded[-1], torch.full_like(guarded[-1], sentinel))
        assert bool(torch.isfinite(out).all())
        error = (out.double() - oracle[:rows]).abs()
        assert bool((error <= 1e-4 * magnitude[:rows] + 1e-6).all()), (rows, float(error.max()))
        relative = ((out.double() - raw_oracle[:rows]).square().sum(-1) /
            raw_oracle[:rows].square().sum(-1)).sqrt()
        assert bool((relative < 0.006).all()), (rows, relative.tolist())
        first_result = out.clone()
        for row in range(rows):
            program.launch(inputs[row:row + 1], weights, scales, out[row:row + 1], scalars=(1,))
        torch.cuda.synchronize()
        assert torch.equal(out, first_result), f"rows={rows}: span launch differs from row launch"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for first in range(0, rows, 16):
                count = min(16, rows - first)
                program.launch(inputs[first:first + count], weights, scales,
                    out[first:first + count], scalars=(count,))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first_result), f"rows={rows}: graph replay differs"
        assert bool((guarded[[0, -1]] == sentinel).all())
        print(json.dumps(dict(rows=rows, shape=[vocab, hidden],
            max_abs_error=float(error.max()), max_relative_raw_fp64=float(relative.max()),
            canaries=True, row_exact=True, graph_exact=True)))
        if os.environ.get("CUTEAFD_GLMF_BLOCK_HEAD_TIMING") == "1":
            for _ in range(3):
                graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end) / 20
            print(json.dumps(dict(rows=rows, milliseconds=milliseconds,
                microseconds_per_row=milliseconds * 1000 / rows)))
