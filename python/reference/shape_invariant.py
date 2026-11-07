"""Fixed row geometry for reference arithmetic, never for the serving engine."""
from __future__ import annotations

import torch
import torch.nn.functional as F

ROWS = 128
_linear = F.linear
_einsum = torch.einsum


def linear(x, weight, bias=None):
    """Keep GEMM M fixed even when a window or an expert's token count changes."""
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return _linear(x, weight, bias)
    result = []
    for start in range(0, flat.shape[0], ROWS):
        chunk = flat[start:start + ROWS]
        count = chunk.shape[0]
        if count != ROWS:
            padded = chunk.new_zeros((ROWS, chunk.shape[-1]))
            padded[:count].copy_(chunk)
            chunk = padded
        result.append(_linear(chunk, weight, bias)[:count])
    return torch.cat(result, dim=0).reshape(*x.shape[:-1], weight.shape[0])


def einsum(equation, *operands):
    """Fix query-row batch geometry for V4.1 grouped projection/attention."""
    axes = {"bsgd,grd->bsgr": (0,), "bmhd,bmtd->bmht": (0, 1),
            "bmht,bmtd->bmhd": (0, 1)}.get(equation)
    if axes is None:
        return _einsum(equation, *operands)
    count = operands[0].shape[1]
    if count == 0:
        return _einsum(equation, *operands)
    result = []
    for start in range(0, count, ROWS):
        width = min(ROWS, count - start)
        args = list(operands)
        for axis in axes:
            chunk = args[axis][:, start:start + ROWS]
            if width != ROWS:
                shape = list(chunk.shape)
                shape[1] = ROWS
                padded = chunk.new_zeros(shape)
                padded[:, :width].copy_(chunk)
                chunk = padded
            args[axis] = chunk
        result.append(_einsum(equation, *args)[:, :width])
    return torch.cat(result, dim=1)


class ForcedRouteQualification:
    """Compare arithmetic on eager expert IDs, observing natural cutoff flips.

    Callbacks supply checkpoint-native choice scores and combine weights; only
    IDs are teacher-forced. Published scoring must never install these hooks.
    """
    def __init__(self, *, relative_margin_limit=1e-5):
        self.relative_margin_limit = relative_margin_limit
        self.eager = None
        self.report = None

    def observe(self, gate, *, choices, weights=None):
        from contextlib import contextmanager

        @contextmanager
        def hooked():
            original = F.linear
            logits = None
            active = False

            def before(module, args):
                nonlocal active
                active = True

            def linear_call(*args, **kwargs):
                nonlocal logits
                result = original(*args, **kwargs)
                if active:
                    logits = result
                return result

            def after(module, args, output):
                nonlocal active
                active = False
                if logits is None or not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError("unsupported qualification router output")
                ids, native_weights = output
                scores = choices(logits).detach()
                if ids.ndim != 2 or scores.ndim != 2 or ids.shape[0] != scores.shape[0]:
                    raise ValueError("qualification router geometry differs")
                if not torch.isfinite(scores).all():
                    raise ValueError("non-finite qualification choice scores")
                if weights is None:
                    self.eager = (ids.detach().clone(), scores.clone())
                    return output
                if self.eager is None:
                    raise ValueError("forced routing requires an eager observation")
                eager_ids, eager_scores = self.eager
                if eager_ids.shape != ids.shape or eager_scores.shape != scores.shape:
                    raise ValueError("eager/adapted router geometry differs")
                flips = []
                changed = (eager_ids.sort(-1).values != ids.sort(-1).values).any(-1)
                for token in torch.nonzero(changed).flatten().tolist():
                    eager_set, natural_set = set(eager_ids[token].tolist()), set(ids[token].tolist())
                    removed, added = sorted(eager_set - natural_set), sorted(natural_set - eager_set)
                    margins = []
                    for arm in (eager_scores, scores):
                        left, right = arm[token, removed], arm[token, added]
                        delta = (left[:, None] - right[None, :]).abs()
                        scale = torch.maximum(left[:, None].abs(), right[None, :].abs()).clamp_min(1e-30)
                        margins.append({"absolute": float(delta.max()), "relative": float((delta / scale).max()),
                                        "eager_only_scores": left.tolist(), "natural_only_scores": right.tolist()})
                    flips.append({"token": token, "eager_only_experts": removed,
                                  "natural_only_experts": added, "margins": margins})
                self.report = {"selection_flip_tokens": len(flips), "flips": flips,
                               "relative_margin_limit": self.relative_margin_limit,
                               "trajectory": "natural choices on teacher-forced upstream state",
                               "passed": all(margin["relative"] <= self.relative_margin_limit
                                             for flip in flips for margin in flip["margins"])}
                forced_weights = weights(logits, eager_ids)
                if forced_weights.shape != native_weights.shape or forced_weights.dtype != native_weights.dtype:
                    raise ValueError("forced route weight geometry/dtype differs")
                return eager_ids, forced_weights

            handles = [gate.register_forward_pre_hook(before), gate.register_forward_hook(after)]
            F.linear = linear_call
            try:
                yield
            finally:
                F.linear = original
                for handle in handles:
                    handle.remove()
        return hooked()

    def require_valid(self):
        if self.report is None or not self.report["passed"]:
            raise ValueError("natural routing flip exceeds relative F32 margin limit")


def qualify(a, manifest, execute, *, route_forcing=False):
    """Nested layer-major probes must not change the outer family's dtype."""
    if getattr(a, "_prefix_probe", False):
        return None
    from fidelity_windows import qualify_prefix
    dtype = torch.get_default_dtype()
    previous = getattr(a, "_route_forced_qualification", False)
    a._route_forced_qualification = route_forcing
    try:
        proof = qualify_prefix(a, manifest, execute)
        from fidelity_windows import canonical
        (a.out / "prefix-qualification.json").write_bytes(canonical(proof) + b"\n")
        return proof
    finally:
        a._route_forced_qualification = previous
        torch.set_default_dtype(dtype)


def bounded_eager(function, rows=1024):
    """Call unchanged eager arithmetic per query block, retaining all keys.

    Golden runners discard attention weights; returning them would reconstruct
    the quadratic allocation that this wrapper is intended to avoid.
    """
    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.training:
            raise ValueError("bounded reference attention requires inference without dropout")
        outputs = []
        for start in range(0, query.shape[-2], rows):
            end = min(start + rows, query.shape[-2])
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                if mask.shape[-2] != query.shape[-2]:
                    raise ValueError("attention mask query coverage differs")
                mask = mask[..., start:end, :]
            sliced = dict(kwargs)
            for name in ("selected_kv", "selected_valid"):
                if sliced.get(name) is not None:
                    sliced[name] = sliced[name][:, start:end]
            output, weights = function(module, query[..., start:end, :], key, value, mask,
                                       scaling=scaling, dropout=dropout, **sliced)
            outputs.append(output)
            del weights
        return torch.cat(outputs, dim=1), None
    return forward


def bounded_sparse(function, rows=128):
    """Bound official-checkpoint sparse hooks before their KV gather allocation."""
    def forward(q, kv, attn_sink, topk_idxs, softmax_scale):
        outputs = []
        for start in range(0, q.shape[1], rows):
            outputs.append(function(q[:, start:start + rows], kv, attn_sink,
                                    topk_idxs[:, start:start + rows], softmax_scale))
        return torch.cat(outputs, dim=1)
    return forward


def fixed_index_topk(scores, slots):
    """Official score selection, fixed slots and ascending block-id reduction order.

    Stable score sorting resolves cutoff ties toward lower block ids. Padding is
    -inf and its ids are outside the causal extent, so the official mask turns
    those entries into -1 before sparse attention gathers them.
    """
    from collections import namedtuple
    if scores.shape[-1] < slots:
        scores = F.pad(scores, (0, slots - scores.shape[-1]), value=-float("inf"))
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :slots]
    indices = indices.sort(dim=-1).values
    result = namedtuple("FixedTopk", "values indices")
    return result(scores.gather(-1, indices), indices)


def install_index_topk(module):
    """Adapt only the checkpoint Indexer.forward selection, not its score math."""
    import ast
    import inspect
    import textwrap
    function = module.Indexer.forward
    if getattr(function, "_fixed_index_topk", False):
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    replacements = 0

    class Selection(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal replacements
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "topk"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "index_score"):
                replacements += 1
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="_reference_fixed_index_topk", ctx=ast.Load()),
                    args=[node.func.value, ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="index_topk", ctx=ast.Load())], keywords=[]), node)
            return node

    tree = Selection().visit(tree)
    if replacements != 1:
        raise ValueError("unsupported official Indexer.forward top-k site")
    module.__dict__["_reference_fixed_index_topk"] = fixed_index_topk
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), "exec"),
         module.__dict__, scope)
    scope["forward"]._fixed_index_topk = True
    module.Indexer.forward = scope["forward"]


def install_eager(module):
    """Bound the family's eager fallback without replacing its math or mask."""
    if hasattr(module, "eager_attention_forward"):
        module.eager_attention_forward = bounded_eager(module.eager_attention_forward)


def install():
    """Also cover checkpoint modules and HC code that call F.linear directly."""
    F.linear = linear
    torch.einsum = einsum
