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


def qualify(a, manifest, execute):
    """Nested layer-major probes must not change the outer family's dtype."""
    if getattr(a, "_prefix_probe", False):
        return None
    from fidelity_windows import qualify_prefix
    dtype = torch.get_default_dtype()
    try:
        proof = qualify_prefix(a, manifest, execute)
        from fidelity_windows import canonical
        (a.out / "prefix-qualification.json").write_bytes(canonical(proof) + b"\n")
        return proof
    finally:
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
