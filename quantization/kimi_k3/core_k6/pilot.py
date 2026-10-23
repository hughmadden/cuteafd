#!/usr/bin/env python3
"""Subset codebook decisions using CoreEngine's unchanged causal replay phases.

Replay always uses published Core-K6 predecessors, never an arm's candidates.
Calibration and held-out prompts are independently selected from original splits.
This mode writes candidates/results only; it cannot export or publish a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ARMS = ("mcg", "mcg-post-e4m3", "e4m3")
SCHEMA = "cuteafd-core-codebook-pilot-v1"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-snapshot", type=Path, required=True)
    result.add_argument("--replay-snapshot", type=Path, required=True)
    result.add_argument("--tokens", type=Path, required=True)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--layers", default="1,3")
    result.add_argument("--arms", default=",".join(ARMS))
    result.add_argument("--calibration-tokens", type=int, default=32768)
    result.add_argument("--heldout-tokens", type=int, default=4096)
    result.add_argument("--preflight", action="store_true")
    return result


def configuration(argv=None):
    args = parser().parse_args(argv)
    args.layers = tuple(int(value) for value in args.layers.split(","))
    args.arms = tuple(args.arms.split(","))
    if (not args.layers or tuple(sorted(set(args.layers))) != args.layers
            or args.layers[0] < 1 or args.layers[-1] > 3):
        raise ValueError("bounded pilot requires sorted unique layers in 1..3")
    if args.arms != ARMS:
        raise ValueError("pilot requires the three ordered codebook arms")
    if not 32768 <= args.calibration_tokens <= 65536 or not 1 <= args.heldout_tokens <= 16384:
        raise ValueError("pilot token budget outside admission")
    return args


def subset(records, budget):
    """Deterministic original-order prefix of complete, unmodified prompts."""
    selected, count = [], 0
    for row in records:
        selected.append(row)
        count += len(row["input_ids"])
        if count >= budget:
            return selected
    raise ValueError("corpus split is smaller than the requested pilot")


def validate_result(row):
    if set(row) != {"schema", "tensor", "arm", "nmse", "bf16_output_proxy_nmse",
                    "calibration_tokens", "heldout_tokens", "weight_nmse"}:
        raise ValueError("invalid pilot result fields")
    if row["schema"] != SCHEMA or row["arm"] not in ARMS or not row["tensor"]:
        raise ValueError("invalid pilot result identity")
    for key in ("nmse", "bf16_output_proxy_nmse", "weight_nmse"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("nonfinite or negative pilot metric")
    for key in ("calibration_tokens", "heldout_tokens"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0:
            raise ValueError("invalid pilot observation count")
    return row


def markdown(rows):
    by_tensor = {}
    for row in rows:
        validate_result(row)
        by_tensor.setdefault(row["tensor"], {})[row["arm"]] = row
    lines = ["| Tensor | mcg K6 NMSE | post-E4M3 NMSE | e4m3 K6 NMSE | e4m3/mcg |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for tensor, arms in sorted(by_tensor.items()):
        if set(arms) != set(ARMS):
            raise ValueError("incomplete pilot arm inventory")
        a, b, c = (arms[arm]["nmse"] for arm in ARMS)
        lines.append(f"| {tensor} | {a:.8g} | {b:.8g} | {c:.8g} | {c/a:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = configuration(argv)
    # Keep argument/schema CPU tests independent of torch and model imports.
    from . import run
    from .recipe import reviewed_modules
    from .storage import StageStore
    import torch
    from safetensors.torch import load_file
    from gptqmodel.utils import kimi_k3_core_format as codec
    from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as quantizer

    args.root.mkdir(parents=True, exist_ok=True)
    calibration = subset(run.load_token_records(args.tokens, "calibration"), args.calibration_tokens)
    heldout = subset(run.load_token_records(args.tokens, "heldout"), args.heldout_tokens)
    if {row["id"] for row in calibration} & {row["id"] for row in heldout}:
        raise ValueError("pilot calibration and held-out prompt IDs overlap")
    token_hash = lambda rows: hashlib.sha256(run.canonical(rows).encode()).hexdigest()
    identity = dict(schema=SCHEMA, layers=args.layers, arms=args.arms,
                    calibration_tokens=sum(len(row["input_ids"]) for row in calibration),
                    heldout_tokens=sum(len(row["input_ids"]) for row in heldout),
                    calibration_sha256=token_hash(calibration), heldout_sha256=token_hash(heldout),
                    calibration="deterministic-original-prompt-subset-rebuild",
                    replay="published-Core-K6-predecessors", head=False,
                    source_index_sha256=run.sha256_file(args.source_snapshot / "model.safetensors.index.json"),
                    replay_index_sha256=run.sha256_file(args.replay_snapshot / "model.safetensors.index.json"))
    run.atomic_json(args.root / "pilot-plan.json", identity)
    if args.preflight:
        for snapshot in (args.source_snapshot, args.replay_snapshot):
            run.header_source(snapshot)
        print(run.canonical(identity), flush=True)
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86 * run.GIB / torch.cuda.get_device_properties(0).total_memory)
    source = codec.KimiK3EXL3Source(args.source_snapshot)
    replay = codec.KimiK3EXL3Source(args.replay_snapshot)
    selection = reviewed_modules()
    candidates = args.root / "candidates"
    candidates.mkdir(exist_ok=True)
    rows = []

    class PublishedCheckpoints:
        def load(self, module):
            # The production branch uses this to skip unnecessary MLA stage B.
            # Force that phase to collect the actual q_b input Hessian.
            return None

    class PilotEngine(run.CoreEngine):
        def __init__(self, split, records):
            root = args.root / split
            root.mkdir(exist_ok=True)
            super().__init__(replay, records, root, root / "unused-export", identity,
                             selection, run.Progress(root, host_limit=50))
            self.split = split

        def input_sample(self, block, store, phase, name):
            # Proxy compares actual BF16 linear outputs on the first held-out
            # batch, while the main NMSE uses every held-out observation.
            _, _, state = next(self._read_batches(store, "a" if phase == "b" else phase))
            if phase == "a":
                value = state["hidden"]
            elif phase == "b":
                value = block.self_attn.q_a_layernorm(block.self_attn.q_a_proj(state["hidden"]))
            elif phase == "c":
                value = state["o_input"]
            elif phase == "d":
                value = state["hidden"]
            elif name == "routed-up":
                value = state["routed_input"]
            else:
                _, _, prior = next(self._read_batches(store, "d"))
                value = self.math.shared_down_input(block.block_sparse_moe.shared_experts, prior["hidden"])
            return value.reshape(-1, value.shape[-1])[:128].contiguous()

        def _quant_group(self, block, checkpoints, modules, store, phase, hessian):
            if block.layer_idx not in args.layers:
                return
            raw = self._raw(store, phase, hessian)
            sample = self.input_sample(block, store, phase, hessian) if self.split == "heldout" else None
            for module in modules:
                weight = source.tensor(module + ".weight", self.device)
                logical_shape = tuple(weight.shape)
                if self.split == "calibration":
                    for arm in ("mcg", "e4m3"):
                        packed, metrics, evidence = codec.quantize_core_projection(weight, raw, codebook=arm)
                        run.commit_safetensors(candidates / (module + "." + arm + ".safetensors"), packed)
                        run.atomic_json(candidates / (module + "." + arm + ".json"),
                                        dict(metrics=metrics, evidence=evidence))
                        self.progress.emit("pilot_candidate", module=module, arm=arm)
                else:
                    denominator = float(((weight.float() @ raw["H"]) * weight.float()).sum(dtype=torch.float64))
                    reference = sample @ weight.T
                    for arm in args.arms:
                        packed = load_file(str(candidates / (module + "." + ("mcg" if arm == "mcg-post-e4m3" else arm) + ".safetensors")))
                        if arm == "mcg-post-e4m3":
                            packed["e4m3"] = packed.pop("mcg")
                        # Compare the serialized native reconstruction, not the
                        # quantizer's pre-serialization error estimate.
                        dense = codec.reconstruct_core_projection(packed, logical_shape, device=self.device, dtype=torch.float32)
                        error = dense - weight.float()
                        numerator = float(((error @ raw["H"]) * error).sum(dtype=torch.float64))
                        proxy = sample @ dense.to(torch.bfloat16).T
                        result = validate_result(dict(schema=SCHEMA, tensor=module, arm=arm,
                            nmse=max(0., numerator) / denominator,
                            bf16_output_proxy_nmse=float((proxy.float() - reference.float()).square().sum(dtype=torch.float64) / reference.float().square().sum(dtype=torch.float64)),
                            weight_nmse=float(error.square().sum(dtype=torch.float64) / weight.float().square().sum(dtype=torch.float64)),
                            calibration_tokens=identity["calibration_tokens"], heldout_tokens=identity["heldout_tokens"]))
                        rows.append(result)
                        run.atomic_json(args.root / "results.json", dict(identity=identity, results=rows))
                        run.atomic_json(args.root / (module + "." + arm + ".json"), result)
                        self.progress.emit("pilot_result", **result)
                        del dense, error, proxy, packed
                del weight
                run.compact_memory()
            del raw, sample
            run.compact_memory()

        @torch.inference_mode()
        def pilot_layer(self, layer, current):
            store = StageStore(self.root / f"layers/{layer:03d}/stages", identity)
            block = replay.load_block(layer, self.device, include_experts=False)
            self._execute_layer(block, current, store, PublishedCheckpoints(),
                                {name: None for name in ("a", "b", "c", "d", "e", "f")})
            del block
            run.compact_memory()
            manifest = store.load("f")
            batches = run.hardlink_frontier(self.boundary, layer, store.directory("f"), manifest)
            current = self.boundary.commit(layer, batches)
            store.clear()
            self.boundary.retire_old()
            return current

    with torch.inference_mode(), run.exclusive_run(args.root):
        for split, records in (("calibration", calibration), ("heldout", heldout)):
            engine = PilotEngine(split, records)
            current = engine.initial()
            for layer in range(max(args.layers) + 1):
                current = engine.pilot_layer(layer, current)
            # A pilot does not retain multi-GiB replay activations after use.
            import shutil
            shutil.rmtree(engine.root / "frontier")
        table = markdown(rows)
        (args.root / "pilot-table.md").write_text(table)
        ratios = {row["tensor"]: row["nmse"] for row in rows if row["arm"] == "mcg"}
        passed = all(row["nmse"] <= 1.1 * ratios[row["tensor"]] for row in rows if row["arm"] == "e4m3")
        run.atomic_json(args.root / "summary.json", dict(identity=identity, passed=passed,
                                                        stop="pilot-only-no-full-quant", results=rows))
        print(table, flush=True)
        print(f"Within-10-percent bar: {passed}; stopping after pilot.", flush=True)
    source.close()
    replay.close()


if __name__ == "__main__":
    main()
