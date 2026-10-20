"""Opt-in SM120 row-FP8 vocabulary gates on checksum-verified checkpoint blocks."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"),
                                reason="requires native EXLR8 checkpoint and SM120")


def real_rows(stem, poison):
    spec = importlib.util.spec_from_file_location(
        "vocabulary_oracle", Path(__file__).parents[1] / "reference/exlr8.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    tensors = ref.Tensors(Path(os.environ["CUTEAFD_EXLR8_SNAPSHOT"]) /
                          "dense/exlr8-vocab.safetensors")
    record = tuple(map(int, tensors.array(stem + ".record")))
    _, n, k, _, side, _, rotation, _ = record
    assert record[0] == 5 and side == 1 and rotation == 0
    checksums = tensors.array(stem + ".checksum")
    payload = torch.full((n, k), poison, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(n, device="cuda")
    rows, decoded = [], []
    for block in (0, n // 256, n // 128 - 1):
        first = block * 128
        raw = tensors.raw(stem + ".fp8", first*k, 128*k)
        scale = tensors.raw(stem + ".scale", first*4, 128*4)
        assert ref.xxhash64(raw + scale) == int(checksums[block])
        q = torch.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), device="cuda").reshape(128, k)
        s = torch.tensor(np.frombuffer(scale, dtype="<f4").copy(), device="cuda")
        payload[first:first+128].copy_(q)
        scales[first:first+128].copy_(s)
        rows.extend(range(first, first+128))
        # Decode E4M3 exactly, then multiply in F64, independently of the kernel.
        decoded.append(q.view(torch.float8_e4m3fn).double() * s.double()[:, None])
    return payload, scales, torch.tensor(rows, device="cuda"), torch.cat(decoded)


def test_native_embedding_real_checkpoint_rows():
    from b12x.integration.cuteafd.fp8_row_embedding import compile_fp8_row_embedding_aot
    payload, scales, selected, weight = real_rows("embed_tokens", 0x7f)
    n, k = payload.shape
    # First/last, repeated and out-of-order vocabulary IDs, plus invalid IDs.
    positions = torch.tensor([0, 383, 129, 0, 382, 128, 1], device="cuda")
    ids = torch.cat((selected[positions], torch.tensor([-1, n], device="cuda"))).int()
    capacity = len(ids) + 3
    output = torch.empty((capacity, k), dtype=torch.bfloat16, device="cuda")
    program = compile_fp8_row_embedding_aot(hidden=k, vocabulary=n, max_rows=capacity)
    ctas = torch.cuda.get_device_properties(0).multi_processor_count * 4
    expected = torch.cat((weight[positions].bfloat16(),
                          torch.zeros((2, k), dtype=torch.bfloat16, device="cuda")))
    immutable = [ids.clone(), scales.clone(), payload[selected].clone()]
    for rows in (1, 4, len(ids)):
        for limit in (1, ctas):
            output.fill_(float("nan"))
            program.launch(ids, payload, scales, output, scalars=(rows, limit))
            assert torch.equal(output[:rows], expected[:rows])
            assert torch.isnan(output[rows:]).all()
    assert torch.equal(ids, immutable[0]) and torch.equal(scales, immutable[1])
    assert torch.equal(payload[selected], immutable[2])


def test_native_head_real_checkpoint_rows():
    from b12x.integration.cuteafd.scaled_linear import compile_scaled_linear_aot
    payload, scales, selected, weight = real_rows("lm_head", 0)
    n, k = payload.shape
    torch.manual_seed(408)
    rows = 129
    x = (torch.randn((rows, k), device="cuda") * .05).bfloat16()
    output = torch.empty((rows, n), device="cuda")
    partial = torch.empty(1, device="cuda")
    program = compile_scaled_linear_aot(input_features=k, output_features=n,
        encoding="fp8_row", sharded_side=1, max_rows=rows)
    ctas = torch.cuda.get_device_properties(0).multi_processor_count * 4
    program.launch(x.flatten(), payload.flatten(), scales, partial, output.flatten(),
                   scalars=(rows, ctas))
    expected = output[:, selected].clone()
    oracle = x.double() @ weight.T
    bf16 = (x @ weight.bfloat16().T).double()
    error = torch.linalg.vector_norm(expected.double() - oracle)
    bound = torch.linalg.vector_norm(bf16 - oracle)
    assert error <= 1.01 * bound, (error.item(), bound.item())
    for live in (1, 4, 16, 64, 65, 128, 129):
        output.fill_(float("nan"))
        program.launch(x.flatten(), payload.flatten(), scales, partial, output.flatten(),
                       scalars=(live, 1 if live % 2 else ctas))
        assert torch.equal(output[:live, selected].view(torch.int32),
                           expected[:live].view(torch.int32))
        assert torch.isnan(output[live:]).all()
