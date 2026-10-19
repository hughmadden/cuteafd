"""W8 same-input official/FP64 gates; opt in inside the matching CUDA image.

The official operators are extracted unchanged from the pinned checkpoint,
not reimplemented as the baseline. No full model or tensor shard is loaded.
"""
from __future__ import annotations

import ast
import __future__
import json
from functools import lru_cache
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("CUTEAFD_KIMI_W8_GPU") != "1":
    pytest.skip("requires explicit Kimi W8 CUDA gate", allow_module_level=True)

torch = pytest.importorskip("torch")
from torch import nn


@lru_cache(maxsize=1)
def official():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = Path(os.environ.get("CUTEAFD_KIMI_SNAPSHOT", hub / "models--moonshotai--Kimi-K3" /
        "snapshots/f831ab66814297da540d832a5235f8e904f29d06"))
    source = snapshot / "modeling_kimi_linear.py"
    if not source.is_file():
        pytest.fail("official Kimi modeling source is required for this gate")
    names = {"SituAndMul", "KimiRMSNorm", "_apply_attn_res", "_get_situ_activation_params", "KimiMLP"}
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in tree.body} == names
    ns = dict(torch=torch, nn=nn)
    exec(compile(tree, str(source), "exec", flags=__future__.annotations.compiler_flag), ns)
    return ns


def assert_accuracy(actual, baseline, oracle):
    """FP64 global and EVERY-row error, with no absolute tolerance floor."""
    actual, baseline = actual.cpu().double(), baseline.cpu().double()
    oracle = oracle.cpu().double()
    assert actual.shape == baseline.shape == oracle.shape
    assert torch.isfinite(actual).all() and torch.isfinite(baseline).all() and torch.isfinite(oracle).all()
    candidate = (actual - oracle).square().sum(-1)
    reference = (baseline - oracle).square().sum(-1)
    assert bool((candidate <= reference * 1.01 ** 2).all()), (
        "per-row FP64 error exceeds 1.01x official", candidate.tolist(), reference.tolist())
    assert candidate.sum() <= reference.sum() * 1.01 ** 2, "global FP64 error exceeds 1.01x official"


def residual_oracle(values, score_weight, norm_weight, eps):
    """Independent FP64 reductions; preserve the two semantic BF16 boundaries.

    Leave the final product unrounded so the measurement includes the final
    storage error, just as the unrounded FP64 Situ oracle does.
    """
    v = values.cpu().double()
    scores = ((v / (v.square().mean(-1, keepdim=True) + eps).sqrt()) *
        score_weight.cpu().double()).sum(-1)
    exp = (scores - scores.max(-1, keepdim=True).values).exp()
    probs = exp / exp.sum(-1, keepdim=True)
    mixed = (v * probs[..., None]).sum(1).to(torch.bfloat16).double()
    normalized = (mixed / (mixed.square().mean(-1, keepdim=True) + eps).sqrt()).to(torch.bfloat16).double()
    return normalized * norm_weight.cpu().double()


def inputs(rows, candidates, case):
    g = torch.Generator().manual_seed(273 + candidates)
    # Padding rows ensure the kernel uses the fixed arena stride, not live M.
    arena = torch.randn(9, rows + 3, 7168, generator=g).bfloat16()
    delta = torch.randn(rows, 7168, generator=g).bfloat16()
    if case == "zero":
        arena.zero_()
        delta.zero_()
    elif case == "small":
        arena *= 2.0 ** -12
        delta *= 2.0 ** -12
    elif case == "cancel":
        delta = -arena[8, :rows].clone()
        delta[:, ::7] += 2.0 ** -8
    norm = (1 + 0.1 * torch.randn(7168, generator=g)).bfloat16()
    proj = (0.01 * torch.randn(7168, generator=g)).bfloat16()
    if case == "saturated":
        proj *= 1000
    following = (1 + 0.2 * torch.randn(7168, generator=g)).bfloat16()
    return arena.cuda(), delta.cuda(), norm.cuda(), proj.cuda(), following.cuda()


@pytest.mark.parametrize("intermediate", [3072, 6144, 16896, 33792])
@pytest.mark.parametrize("compensated", [False, True], ids=["direct", "compensated"])
@torch.no_grad()
def test_ffn_chain_official_fp64(intermediate, compensated):
    from b12x.integration.cuteafd import kimi_residual, pointwise
    # The full-width projections use the production hidden size; M16 bounds
    # FP64 work while exercising a genuine batched dense/shared FFN chain.
    rows, hidden = 16, 7168
    g = torch.Generator().manual_seed(273)
    x = torch.randn(rows, hidden, generator=g).bfloat16().cuda()
    config = SimpleNamespace(hidden_size=hidden, intermediate_size=intermediate,
        hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0)
    mlp = official()["KimiMLP"](config).cuda().bfloat16()
    for parameter in mlp.parameters():
        parameter.copy_((torch.randn(parameter.shape, generator=g) * 0.01).bfloat16().cuda())
    baseline = mlp(x)
    packed = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), 0)
    projected = torch.empty(rows, 2 * intermediate, dtype=torch.bfloat16, device="cuda")
    activated = torch.empty(rows, intermediate, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    projection_options = {"compensated": True} if compensated else {}
    gate_up = pointwise.compile_bf16_projection_aot(width=hidden, outputs=2 * intermediate, **projection_options)
    activation = kimi_residual.compile_kimi_situ_aot(intermediate)
    down = pointwise.compile_bf16_projection_aot(width=intermediate, outputs=hidden, **projection_options)
    gate_up.launch(x, packed, projected, scalars=(rows,))
    activation.launch(projected, activated, scalars=(rows,))
    down.launch(activated, mlp.down_proj.weight, out, scalars=(rows,))
    torch.cuda.synchronize()
    # Preserve the BF16 projection and activation boundaries; the final FP64
    # product stays unrounded so output-storage error is measured as well.
    xx = x.cpu().double()
    gate = (xx @ mlp.gate_proj.weight.cpu().double().T).bfloat16().double()
    up = (xx @ mlp.up_proj.weight.cpu().double().T).bfloat16().double()
    oracle_activation = ((4 * (gate / 4).tanh() / (1 + (-gate).exp())) *
                         (25 * (up / 25).tanh())).bfloat16().double()
    oracle = oracle_activation @ mlp.down_proj.weight.cpu().double().T
    assert_accuracy(out, baseline, oracle)
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gate_up.launch(x, packed, projected, scalars=(rows,))
        activation.launch(projected, activated, scalars=(rows,))
        down.launch(activated, mlp.down_proj.weight, out, scalars=(rows,))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first), "FFN graph replay is not byte-exact"


# All candidate counts and site transitions; wide prefill is bounded separately.
SITES = [(1, "keep", True, False)] + [
    (c, mode, append, True) for c in range(2, 10)
    for mode, append in (("add", False), ("replace", False), ("keep", False))] + [
    (c, "add", True, True) for c in range(2, 9)]


@pytest.mark.parametrize("candidates,mode,append,mix", SITES)
@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("case", ["normal", "zero", "small", "cancel", "saturated"])
@torch.no_grad()
def test_residual_official_fp64(candidates, mode, append, mix, rows, case):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_attn_res_rmsnorm_aot
    arena, delta, norm, proj, following = inputs(rows, candidates, case)
    original = arena.clone()
    prefix = arena[8, :rows].clone()
    if mode == "add":
        prefix = prefix + delta
    elif mode == "replace":
        prefix = delta.clone()
    blocks = arena[:candidates - 1, :rows].transpose(0, 1).contiguous()
    definitions = official()
    res_norm = definitions["KimiRMSNorm"](7168, eps=1e-5).cuda().bfloat16()
    out_norm = definitions["KimiRMSNorm"](7168, eps=1e-5).cuda().bfloat16()
    projection = nn.Linear(7168, 1, bias=False).cuda().bfloat16()
    res_norm.weight.copy_(norm)
    out_norm.weight.copy_(following)
    projection.weight.copy_(proj[None])
    score_weight = norm.float() * proj.float()
    baseline = out_norm(definitions["_apply_attn_res"](prefix, blocks, projection, res_norm) if mix else prefix)
    values = torch.cat((blocks, prefix[:, None]), 1)
    # Include the official F32 score-weight product's error in the comparison.
    oracle_weight = norm.cpu().double() * proj.cpu().double()
    oracle = residual_oracle(values, oracle_weight, following, 1e-5)
    program = compile_kimi_attn_res_rmsnorm_aot(7168, candidates, mode, append, mix)
    out = torch.empty_like(prefix)
    program.launch(arena, delta, score_weight, following, out, scalars=(rows, arena.shape[1], 1e-5))
    torch.cuda.synchronize()
    assert_accuracy(out, baseline, oracle)
    assert torch.equal(arena[8, :rows], prefix)
    if append:
        assert torch.equal(arena[candidates - 1, :rows], prefix)
    changed = {8} if mode != "keep" else set()
    if append:
        changed.add(candidates - 1)
    for c in range(9):
        assert torch.equal(arena[c, rows:], original[c, rows:]), "padding row was overwritten"
        if c not in changed:
            assert torch.equal(arena[c], original[c]), "immutable block was overwritten"
    first = out.clone()
    arena.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(arena, delta, score_weight, following, out, scalars=(rows, arena.shape[1], 1e-5))
    arena.copy_(original)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first), "graph replay is not byte-exact"


@pytest.mark.parametrize("candidates", [2, 9])
@torch.no_grad()
def test_residual_prefill_official_fp64(candidates):
    # Reuse the identical gate, including fixed-stride padding and graph replay.
    test_residual_official_fp64(candidates, "add", False, True, 4096, "normal")


@pytest.mark.parametrize("candidates,mode,append,mix", [
    (1, "keep", True, False), (2, "add", True, True),
    (9, "add", False, True), (9, "replace", False, True), (9, "keep", False, True)])
@torch.no_grad()
def test_residual_same_row_width_and_position(candidates, mode, append, mix):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_attn_res_rmsnorm_aot
    seed, delta, norm, proj, following = inputs(1, candidates, "normal")
    score_weight = norm.float() * proj.float()
    program = compile_kimi_attn_res_rmsnorm_aot(7168, candidates, mode, append, mix)
    expected = None
    # The same compiled callable, inputs and weights must not depend on M or row position.
    for rows in (1, 16, 4096):
        arena = seed[:, :1].repeat(1, rows + 3, 1)
        change = delta.repeat(rows, 1)
        out = torch.empty(rows, 7168, device="cuda", dtype=torch.bfloat16)
        program.launch(arena, change, score_weight, following, out,
                       scalars=(rows, rows + 3, 1e-5))
        torch.cuda.synchronize()
        if expected is None:
            expected = out[0].clone()
        for position in {0, min(1, rows - 1), rows - 1}:
            assert torch.equal(out[position], expected), "residual depends on width/row position"
        del arena, change, out


@pytest.mark.parametrize("intermediate", [3072, 6144, 16896, 33792])
@torch.no_grad()
def test_situ_same_row_width_and_position(intermediate):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_situ_aot
    g = torch.Generator().manual_seed(273)
    row = (torch.randn(1, 2 * intermediate, generator=g) * 8).bfloat16().cuda()
    program = compile_kimi_situ_aot(intermediate)
    expected = None
    for rows in (1, 16, 4096):
        x = row.repeat(rows, 1)
        out = torch.empty(rows, intermediate, device="cuda", dtype=torch.bfloat16)
        program.launch(x, out, scalars=(rows,))
        torch.cuda.synchronize()
        if expected is None:
            expected = out[0].clone()
        for position in {0, min(1, rows - 1), rows - 1}:
            assert torch.equal(out[position], expected), "Situ depends on width/row position"
        del x, out


@pytest.mark.parametrize("intermediate", [3072, 6144, 16896, 33792])
@pytest.mark.parametrize("rows", [1, 16, 4096])
@torch.no_grad()
def test_situ_official_fp64(intermediate, rows):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_situ_aot
    g = torch.Generator().manual_seed(273)
    x = (torch.randn(rows, 2 * intermediate, generator=g) * 8).bfloat16().cuda()
    # Include threshold/saturation values alongside the same random inputs.
    special = torch.tensor([-100, -25, -4, -1e-4, 0, 1e-4, 4, 25, 100], device="cuda").bfloat16()
    x[:, :special.numel()] = special
    x[:, intermediate:intermediate + special.numel()] = special.flip(0)
    baseline = official()["SituAndMul"](4.0, 25.0)(x)
    out = torch.empty(rows, intermediate, dtype=torch.bfloat16, device="cuda")
    program = compile_kimi_situ_aot(intermediate)
    program.launch(x, out, scalars=(rows,))
    torch.cuda.synchronize()
    # Chunk CPU FP64 work, avoiding a second full-width double prefill allocation.
    for start in range(0, rows, 16):
        gate, up = x[start:start + 16].cpu().double().chunk(2, dim=-1)
        oracle = (4 * (gate / 4).tanh() / (1 + (-gate).exp())) * (25 * (up / 25).tanh())
        assert_accuracy(out[start:start + 16], baseline[start:start + 16], oracle)
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(x, out, scalars=(rows,))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first), "graph replay is not byte-exact"


@pytest.mark.parametrize("hidden", [3584, 7168])
@pytest.mark.parametrize("rows", [1, 16, 4096])
@pytest.mark.parametrize("case", ["normal", "zero", "small", "large"])
@torch.no_grad()
def test_rmsnorm_official_fp64(hidden, rows, case):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_rmsnorm_aot
    g = torch.Generator().manual_seed(273)
    x = torch.randn(rows, hidden, generator=g).bfloat16().cuda()
    if case == "zero":
        x.zero_()
    elif case == "small":
        x *= 2.0 ** -12
    elif case == "large":
        x *= 2.0 ** 12
    norm = official()["KimiRMSNorm"](hidden, eps=1e-5).cuda().bfloat16()
    norm.weight.copy_((1 + 0.2 * torch.randn(hidden, generator=g)).bfloat16().cuda())
    baseline = norm(x)
    out = torch.empty_like(x)
    program = compile_kimi_rmsnorm_aot(hidden=hidden)
    program.launch(x, norm.weight, out, scalars=(rows, 1e-5))
    torch.cuda.synchronize()
    xx = x.cpu().double()
    normalized = (xx / (xx.square().mean(-1, keepdim=True) + 1e-5).sqrt()).bfloat16().double()
    assert_accuracy(out, baseline, normalized * norm.weight.cpu().double())
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(x, norm.weight, out, scalars=(rows, 1e-5))
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first, out), "RMSNorm graph replay is not byte-exact"


@pytest.mark.parametrize("hidden", [3584, 7168])
@torch.no_grad()
def test_rmsnorm_same_row_width_and_position(hidden):
    from b12x.integration.cuteafd.kimi_residual import compile_kimi_rmsnorm_aot
    g = torch.Generator().manual_seed(273)
    row = torch.randn(1, hidden, generator=g).bfloat16().cuda()
    weight = (1 + 0.2 * torch.randn(hidden, generator=g)).bfloat16().cuda()
    program = compile_kimi_rmsnorm_aot(hidden=hidden)
    expected = None
    for rows in (1, 16, 4096):
        x = row.repeat(rows, 1)
        out = torch.empty_like(x)
        program.launch(x, weight, out, scalars=(rows, 1e-5))
        torch.cuda.synchronize()
        if expected is None:
            expected = out[0].clone()
        for position in {0, min(1, rows - 1), rows - 1}:
            assert torch.equal(out[position], expected), "RMSNorm depends on width/row position"


@pytest.mark.skipif(not os.environ.get("CUTEAFD_KIMI_W8_REPORT"), reason="optional resource/timing report")
@pytest.mark.parametrize("kind,width", [("residual", c) for c in (1, 2, 4, 5, 9)] +
                         [("norm", w) for w in (3584, 7168)] +
                         [("situ", w) for w in (3072, 6144, 16896, 33792)])
@torch.no_grad()
def test_w8_resource_and_timing_report(kind, width):
    from b12x.integration.cuteafd import kimi_residual
    from b12x.integration.cuteafd._common import exportable_compilation
    directory = Path(os.environ["CUTEAFD_KIMI_W8_REPORT"]) / f"{kind}-{width}"
    directory.mkdir(parents=True, exist_ok=True)
    with exportable_compilation():
        if kind == "residual":
            program = kimi_residual.compile_kimi_attn_res_rmsnorm_aot(
                7168, width, "keep", width == 1, width != 1)
        elif kind == "norm":
            program = kimi_residual.compile_kimi_rmsnorm_aot(hidden=width)
        else:
            program = kimi_residual.compile_kimi_situ_aot(width)
    program.export_to_c(str(directory), "kernel", "kimi_w8")
    records = []
    for rows in (1, 16, 4096):
        if kind == "residual":
            arena, delta, norm, proj, following = inputs(rows, width, "normal")
            out = torch.empty_like(delta)
            args = (arena, delta, norm.float() * proj.float(), following, out)
            scalars = (rows, rows + 3, 1e-5)
        elif kind == "norm":
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            weight = torch.ones(width, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(x)
            args, scalars = (x, weight, out), (rows, 1e-5)
        else:
            x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
            args, scalars = (x, out), (rows,)
        # Warm once, then measure device work in a graph, not Python dispatch.
        program.launch(*args, scalars=scalars)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                program.launch(*args, scalars=scalars)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        records.append(dict(rows=rows, device_us=start.elapsed_time(end) * 1000 / 100))
        del graph, args, out
    (directory / "timing.json").write_text(json.dumps(dict(
        geometry=dict(program.geometry), timing=records, warm_batches=1,
        repetitions_per_graph=100), indent=2) + "\n")
