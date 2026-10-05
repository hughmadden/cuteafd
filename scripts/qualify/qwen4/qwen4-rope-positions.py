#!/usr/bin/env python3
"""Regenerate host M-RoPE fixtures by executing the pinned get_rope_index methods.

CPU torch is sufficient; no checkpoint, CUDA or transformers import is needed.
Only the two reference methods are extracted, unmodified, from the source AST.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch


def reference(source: Path):
    tree = ast.parse(source.read_text())
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpModel":
            methods = [m for m in node.body if isinstance(m, ast.FunctionDef)
                       and m.name in ("get_rope_index", "get_vision_position_ids")]
    if len(methods) != 2:
        raise ValueError("pinned Qwen4ExpModel positional methods not found")
    namespace = {"torch": torch, "itertools": itertools}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    model.get_vision_position_ids = lambda *a, **kw: namespace["get_vision_position_ids"](model, *a, **kw)
    return lambda *a, **kw: namespace["get_rope_index"](model, *a, **kw)


def fixtures(source: Path):
    get_rope_index = reference(source)
    rng = random.Random(6308)
    cases = []
    for index in range(50):
        ids, types, images = [], [], []
        image_count = 0 if index < 5 else 1 + index % 4
        for image in range(image_count):
            text = [10 + rng.randrange(200) for _ in range(rng.randrange(1, 11))] + [248053]
            ids.extend(text)
            types.extend([0] * len(text))
            h, w = 1 + rng.randrange(6), 1 + rng.randrange(7)
            images.append({"start": len(ids), "grid": [1, 2 * h, 2 * w]})
            ids.extend([248056] * (h * w))
            types.extend([1] * (h * w))
            ids.append(248054)
            types.append(0)
        text = [20 + rng.randrange(200) for _ in range(1 + rng.randrange(20))]
        ids.extend(text)
        types.extend([0] * len(text))
        positions, delta = get_rope_index(torch.tensor([ids]), torch.tensor([types]),
                                         image_grid_thw=torch.tensor([i["grid"] for i in images]) if images else None)
        cases.append({"name": f"layout-{index:02d}", "ids": ids, "images": images,
                      "positions": positions[:, 0, :].T.tolist(), "delta": int(delta.item())})
    return {"reference": "Qwen4ExpModel.get_rope_index/get_vision_position_ids",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = fixtures(args.transformers_source)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {len(data['cases'])} reference layouts to {args.output}")


if __name__ == "__main__":
    main()
