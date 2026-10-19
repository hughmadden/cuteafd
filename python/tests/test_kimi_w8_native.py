"""W8 same-input official/FP64 gates; opt in inside the matching CUDA image.

The official operators are extracted unchanged from the pinned checkpoint,
not reimplemented as the baseline. No full model or tensor shard is loaded.
"""
from __future__ import annotations

import ast
import __future__
from functools import lru_cache
import os
from pathlib import Path

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
    names = {"SituAndMul", "KimiRMSNorm", "_apply_attn_res"}
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
    oracle = residual_oracle(values, score_weight, following, 1e-5)
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
