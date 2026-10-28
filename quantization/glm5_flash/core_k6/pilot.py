#!/usr/bin/env python3
"""Short Flash format pilot and native/staged replay qualification.

Candidates are scored against a fixed BF16 predecessor stream. This bounded
mode cannot publish or launch production. The production recipe decision is
automatic after its measured replay and projection checks pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("GLMF_GPTQMODEL_SOURCE", str(REPO / "third_party/gptqmodel")))
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "glm5_flash.core_k6"

from glm5.core_k6.pilot import tp2_projection_bytes as full_tp2_bytes, weighted_nmse
from kimi_k3.core_k6.pilot import fp8_quantize, fp8_reconstruct, payload_bytes, subset
from kimi_k3.uniform_k2.storage import (
    RollingBoundary, atomic_json, canonical, commit_safetensors, sha256_file, verify_receipt,
)

ARMS = ("mcg-k6", "fp8-block128", "bf16")
LAYERS = (1, 4, 7)
SCHEMA = "cuteafd-glm53-flash-short-pilot-v1"
PREFIX = "model.language_model."
CHECKS = ("kda_boundary_byte_exact", "mla_boundary_byte_exact", "mhc_boundary_byte_exact",
          "mtp_boundary_byte_exact", "serialized_projection_replay")
GIB = 1024**3


def modules(layer):
    """One dense KDA, one MoE KDA and one pooled-indexer MLA representative."""
    if layer not in LAYERS:
        raise ValueError("layer is outside the shortened pilot")
    prefix = f"{PREFIX}layers.{layer}."
    attention = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if layer == 7 else (
        "q_proj", "k_proj", "v_proj", "o_proj")
    mlp = "mlp." if layer < 3 else "mlp.shared_experts."
    return [prefix + "self_attn." + part for part in attention] + [
        prefix + mlp + part for part in ("gate_proj", "up_proj", "down_proj")]


def tp2_projection_bytes(name, shape, arm):
    return full_tp2_bytes(name, shape, arm)


def select_records(calibration, heldout, calibration_tokens=4096, heldout_tokens=1024):
    """Keep whole prompts and original ordering; add one natural sparse witness."""
    cal = subset(calibration, calibration_tokens)
    witness = next((r for r in calibration if len(r["input_ids"]) > 2051), None)
    if witness is not None and witness["id"] not in {r["id"] for r in cal}:
        cal.append(witness)
    held = subset(heldout, heldout_tokens)
    if {r["id"] for r in cal} & {r["id"] for r in held}:
        raise ValueError("pilot splits overlap")
    return cal, held, witness


def exact_state(actual, expected):
    import torch
    if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in actual):
        raise ValueError("native/staged Flash boundary differs")


def native_staged_boundary(numerical, block, state, capture=None):
    """Also compare the indexer's state when its decoder hides that return."""
    import torch
    observed = []
    indexer = getattr(block.self_attn, "indexer", None)
    handle = None if indexer is None else indexer.register_forward_hook(
        lambda module,args,value: observed.append(value.detach().clone()))
    try:
        actual = numerical.native_forward(block, state)
        if capture is not None:
            capture.close()
        expected = numerical.staged_forward(block, state)
        exact_state(actual, expected)
        stats = None
        if indexer is not None:
            if len(observed) != 2 or not torch.equal(observed[0], observed[1]):
                raise ValueError("native/staged Flash pooled-indexer state differs")
            tokens = state["hidden"].shape[1]
            valid_last = int(observed[0][0,-1].ge(0).sum())
            stats = dict(byte_exact=True, tokens=tokens, valid_last_row=valid_last,
                sparse_selection=valid_last<tokens, raw_topk=indexer.index_topk,
                pool_size=indexer.index_kpool, pool_budget=indexer.index_topk//indexer.index_kpool)
        return actual, stats
    finally:
        if handle is not None:
            handle.remove()
        if capture is not None:
            capture.attach()


def progress(root, event, **fields):
    import psutil
    import torch
    state = dict(event=event, time=datetime.now(timezone.utc).isoformat(),
        rss_gib=psutil.Process().memory_info().rss/GIB,
        cuda_gib=torch.cuda.memory_allocated()/GIB,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
        free_gib=shutil.disk_usage(root).free/GIB, **fields)
    with (root/"events.jsonl").open("a") as f:
        f.write(canonical(state)+"\n")
    atomic_json(root/"status.json", state)
    print(canonical(state), flush=True)
    if state["rss_gib"] > 58 or state["cuda_gib"] > 86 or state["free_gib"] < 64:
        raise MemoryError("Flash pilot exceeded host/GPU/disk admission")


def compact():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Capture:
    def __init__(self, numerical, block, layer):
        self.math, self.block, self.layer = numerical, block, layer
        self.raw, self.handles = {}, []
        self.attach()

    def attach(self):
        prefix = f"{PREFIX}layers.{self.layer}."
        for name in modules(self.layer):
            module = self.math.module(self.block, name.removeprefix(prefix))
            self.handles.append(module.register_forward_pre_hook(self.hook(name)))

    def hook(self, name):
        def accumulate(module, args):
            import torch
            x = args[0].detach().reshape(-1, module.in_features).float()
            if name not in self.raw:
                self.raw[name] = dict(H=torch.zeros((x.shape[1], x.shape[1]), device=x.device), count=0)
            self.raw[name]["H"].addmm_(x.T, x)
            self.raw[name]["count"] += len(x)
        return accumulate

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def commit_batch(path, state):
    return dict(filename=Path(path).name, **commit_safetensors(path, state))


def _to_device(state, device="cuda:0"):
    return {k: v.to(device) for k, v in state.items()}


def capture_split(source, numerical, records, root, identity):
    import torch
    from safetensors.torch import load_file
    boundary = RollingBoundary(root/"frontier", identity)
    current = boundary.current(checksum=True)
    if current is None:
        partial = boundary.directory(-1)
        if partial.exists():
            if partial.is_symlink():
                raise ValueError("refuse symlink pilot frontier")
            shutil.rmtree(partial)
        entries = []
        embedding = source.tensor(PREFIX+"embed_tokens.weight", "cuda:0")
        for index, row in enumerate(records):
            ids = torch.tensor(row["input_ids"], device="cuda:0")
            hidden = torch.nn.functional.embedding(ids, embedding).unsqueeze(0)
            hidden = hidden.unsqueeze(2).expand(-1, -1, source.config["hc_mult"], -1).contiguous()
            entries.append(commit_batch(partial/f"{index:05d}.safetensors", dict(hidden=hidden)))
        current = boundary.commit(-1, entries)
        del embedding, hidden, ids
        source.close()
        compact()
    for layer in range(current["layer"]+1, max(LAYERS)+1):
        partial = boundary.directory(layer)
        if partial.is_symlink():
            raise ValueError("refuse symlink pilot successor")
        if partial.exists():
            shutil.rmtree(partial)
        progress(root, "loading_layer", layer=layer)
        block = source.load_block(layer, "cuda:0", include_experts=True)
        capture = Capture(numerical, block, layer) if layer in LAYERS else None
        entries, proofs = [], []
        try:
            for index, entry in enumerate(current["manifest"]["batches"]):
                state = _to_device(load_file(str(boundary.root/current["directory"]/entry["filename"])))
                if index == 0 or records[index]["id"] == identity["long_context_witness"]:
                    successor, indexer_stats = native_staged_boundary(numerical, block, state, capture)
                    if layer in LAYERS:
                        path = root/"proof-inputs"/f"layer-{layer:03d}-{index:05d}.safetensors"
                        proofs.append(dict(layer=layer, row_id=records[index]["id"], tokens=len(records[index]["input_ids"]),
                            file=str(path.relative_to(root)), receipt=commit_safetensors(path, state)))
                    progress(root, "native_boundary_exact", layer=layer, row_id=records[index]["id"],
                             tokens=len(records[index]["input_ids"]), indexer=indexer_stats)
                else:
                    successor = numerical.native_forward(block, state)
                entries.append(commit_batch(partial/f"{index:05d}.safetensors", successor))
                del state, successor
            if capture is not None:
                receipts = {}
                if set(capture.raw) != set(modules(layer)):
                    raise ValueError("pilot hook inventory differs")
                for name, raw in capture.raw.items():
                    if raw["count"] != sum(len(r["input_ids"]) for r in records):
                        raise ValueError("pilot must capture only actual observations")
                    path = root/"hessians"/(name+".safetensors")
                    receipts[name] = commit_safetensors(path, dict(H=raw["H"], count=torch.tensor(raw["count"])))
                atomic_json(root/"hessians"/f"layer-{layer}.json", dict(identity=identity, files=receipts))
                atomic_json(root/"proof-inputs"/f"layer-{layer}.json", dict(identity=identity, entries=proofs))
                del raw, receipts
            current = boundary.commit(layer, entries)
        finally:
            if capture is not None:
                capture.close()
            del capture, block
            compact()
        progress(root, "layer_committed", layer=layer)
    return current


def load_hessian(root, split, layer, name, identity):
    from safetensors.torch import load_file
    directory = root/split/"hessians"
    manifest = json.loads((directory/f"layer-{layer}.json").read_text())
    if manifest["identity"] != identity or set(manifest["files"]) != set(modules(layer)):
        raise ValueError("pilot Hessian identity/inventory differs")
    path = directory/(name+".safetensors")
    verify_receipt(path, manifest["files"][name], checksum=True)
    raw = load_file(str(path), device="cuda:0")
    if int(raw["count"]) != identity[split+"_tokens"]:
        raise ValueError("pilot Hessian count differs")
    return dict(H=raw["H"], count=int(raw["count"]), finalized=False)


def score(source, numerical, root, identity):
    from safetensors.torch import load_file
    rows = []
    for layer in LAYERS:
        for name in modules(layer):
            raw = load_hessian(root, "calibration", layer, name, identity)
            weight = source.tensor(name+".weight", "cuda:0")
            source.close()
            shape = list(weight.shape)
            for arm in ARMS:
                if arm == "bf16":
                    dense = weight.float()
                    size = weight.numel()*weight.element_size()
                else:
                    path = root/"candidates"/(name+"."+arm+".safetensors")
                    receipt = path.with_suffix(".json")
                    if receipt.exists():
                        prior = json.loads(receipt.read_text())
                        if prior["identity"] != identity:
                            raise ValueError("candidate identity differs")
                        verify_receipt(path, prior["receipt"], checksum=True)
                        packed = load_file(str(path))
                    else:
                        path.unlink(missing_ok=True)
                        packed = (numerical.quantize_projection(weight, raw, bits=6)[0] if arm == "mcg-k6"
                                  else fp8_quantize(weight, arm))
                        rec = commit_safetensors(path, packed)
                        atomic_json(receipt, dict(identity=identity, receipt=rec))
                    size = sum(v.numel()*v.element_size() for v in packed.values())
                    if size != payload_bytes(arm, shape):
                        raise ValueError("pilot candidate byte accounting differs")
                    dense = (numerical.reconstruct_projection(packed, shape, device="cuda:0", dtype=weight.dtype).float()
                             if arm == "mcg-k6" else fp8_reconstruct(packed, arm, "cuda:0").to(weight.dtype).float())
                    del packed
                heldout = load_hessian(root, "heldout", layer, name, identity)
                row = dict(tensor=name, arm=arm, nmse=weighted_nmse(weight, dense, heldout["H"]),
                    payload_bytes=size, bpw=size*8/weight.numel(), tp2_bytes=tp2_projection_bytes(name, shape, arm),
                    calibration_tokens=identity["calibration_tokens"], heldout_tokens=identity["heldout_tokens"])
                rows.append(row)
                atomic_json(root/"results.json", dict(identity=identity, results=rows))
                progress(root, "pilot_result", **row)
                del dense, heldout
            del raw, weight
            compact()
    return rows


def qualify_serialized_replay(source, numerical, root, identity):
    import torch
    from safetensors.torch import load_file
    checks = dict.fromkeys(CHECKS, False)
    for layer in LAYERS:
        block = source.load_block(layer, "cuda:0", include_experts=True)
        prefix = f"{PREFIX}layers.{layer}."
        for name in modules(layer):
            path = root/"candidates"/(name+".mcg-k6.safetensors")
            receipt = json.loads(path.with_suffix(".json").read_text())
            if receipt["identity"] != identity:
                raise ValueError("serialized replay candidate identity differs")
            verify_receipt(path, receipt["receipt"], checksum=True)
            packed = load_file(str(path))
            module = numerical.module(block, name.removeprefix(prefix))
            restored = numerical.reconstruct_projection(packed, list(module.weight.shape), device="cuda:0", dtype=module.weight.dtype)
            if not bool(torch.isfinite(restored).all()):
                raise ValueError("serialized projection replay is nonfinite")
            module.weight.copy_(restored)
            del packed, restored
        manifest = json.loads((root/"calibration/proof-inputs"/f"layer-{layer}.json").read_text())
        if manifest["identity"] != identity or not manifest["entries"]:
            raise ValueError("serialized replay proof inputs differ")
        for entry in manifest["entries"]:
            path = root/"calibration"/entry["file"]
            verify_receipt(path, entry["receipt"], checksum=True)
            state = _to_device(load_file(str(path)))
            native, indexer_stats = native_staged_boundary(numerical, block, state)
            if not bool(torch.isfinite(native["hidden"]).all()):
                raise ValueError("quantized native replay is nonfinite")
            progress(root, "serialized_boundary_exact", layer=layer, row_id=entry["row_id"], tokens=entry["tokens"], indexer=indexer_stats)
            del state, native
        checks["mla_boundary_byte_exact" if layer == 7 else "kda_boundary_byte_exact"] = True
        del block
        compact()
    checks["mhc_boundary_byte_exact"] = True
    checks["serialized_projection_replay"] = True
    return checks


def qualify_mtp(source, numerical, root, identity, row):
    """Compare the Flash adapter with its independently written native MTP body.

    Use a real eight-layer BF16 prefix as a bounded input witness. This tests
    alignment and decoder arithmetic, not final-layer calibration quality.
    """
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm
    from gptqmodel.models.definitions.glm5_next import Glm5NextMTPDecoderLayer
    from gptqmodel.utils import glm5_flash_mtp as mtp
    boundary = RollingBoundary(root/"calibration/frontier", identity)
    current = boundary.current(checksum=True)
    if current is None or current["layer"] != max(LAYERS):
        raise ValueError("MTP bounded target-prefix witness is absent")
    entry = current["manifest"]["batches"][0]
    frontier = load_file(str(boundary.root/current["directory"]/entry["filename"]), device="cuda:0")["hidden"]
    ids = torch.tensor([row["input_ids"]], dtype=torch.int64, device="cuda:0")
    embedding = source.tensor(PREFIX+"embed_tokens.weight", "cuda:0")
    norm = Glm5NextTextRMSNorm(source.config["hidden_size"], source.config["rms_norm_eps"]).to("cuda:0")
    norm.weight = nn.Parameter(source.tensor(PREFIX+"norm.weight", "cuda:0"), requires_grad=False)
    source.close()
    aligned = mtp.aligned_inputs(frontier, ids, embedding, norm)
    expected_positions = torch.arange(1, ids.shape[1], device=ids.device).unsqueeze(0)
    if (not torch.equal(aligned["input_ids"], ids[:, 1:])
            or not torch.equal(aligned["position_ids"], expected_positions)
            or aligned["target_hidden"].shape[1] != ids.shape[1]-1):
        raise ValueError("Flash MTP shifted-token alignment differs")
    adapter = mtp.load_mtp(source, "cuda:0", include_experts=True)
    language_model = nn.Module()
    language_model.norm = norm
    language_model.hc_head = Glm5NextTextHyperHead()
    language_model.embed_tokens = nn.Embedding.from_pretrained(embedding, freeze=True)
    with torch.device("meta"):
        oracle = Glm5NextMTPDecoderLayer(source.native_config(), language_model=language_model)
    for name in ("enorm", "hnorm", "eh_proj"):
        setattr(oracle, name, getattr(adapter, name))
    for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
        setattr(oracle, name, getattr(adapter.decoder, name))
    oracle.shared_head.norm = adapter.shared_head_norm
    oracle.eval()
    def compare():
        actual = adapter.forward_aligned(aligned)
        expected, topk = oracle(frontier, input_ids=aligned["input_ids"],
            position_ids=aligned["position_ids"], attention_mask=aligned["attention_mask"], use_cache=False)
        if not torch.equal(actual["normalized_hidden"], expected):
            raise ValueError("native Flash MTP adapter differs from checkpoint MTP body")
        if topk is not None and not torch.equal(actual.get("prev_topk_indices"), topk):
            raise ValueError("native Flash MTP index state differs")
        state = dict(hidden=adapter.eh_proj(adapter.envelope(aligned)), position_ids=aligned["position_ids"])
        native_staged_boundary(numerical, adapter.decoder, state)
    compare()
    # Exercise MTP's actual K4 serialized projection path without quantizing
    # every draft expert in a format pilot.
    envelope = adapter.envelope(aligned).reshape(-1, 2*source.config["hidden_size"]).float()
    raw = dict(H=envelope.T@envelope, count=envelope.shape[0], finalized=False)
    packed, metrics, evidence = numerical.quantize_projection(adapter.eh_proj.weight, raw, bits=4)
    path = root/"candidates/mtp-eh-k4.safetensors"
    receipt = commit_safetensors(path, packed)
    reloaded = load_file(str(path))
    adapter.eh_proj.weight.copy_(numerical.reconstruct_projection(reloaded, list(adapter.eh_proj.weight.shape),
        device="cuda:0", dtype=adapter.eh_proj.weight.dtype))
    compare()
    proof = dict(byte_exact=True, serialized_k4_byte_exact=True, prefix_layers=max(LAYERS)+1,
        final_target_frontier=False, tokens=ids.shape[1], shifted_pairs=ids.shape[1]-1,
        source_row=row["id"], contract=mtp.MTP_CONTRACT, projection_evidence=evidence,
        projection_metrics=metrics, projection_receipt=receipt)
    atomic_json(root/"mtp-proof.json", proof)
    progress(root, "mtp_boundary_exact", tokens=ids.shape[1], shifted_pairs=ids.shape[1]-1)
    del adapter, oracle, language_model, embedding, norm, frontier, aligned, envelope, raw, packed, reloaded
    compact()
    return proof


def validate_qualification(report):
    if report.get("status") != "passed" or set(report.get("native_checks", {})) != set(CHECKS):
        raise ValueError("pilot qualification is incomplete")
    if not all(report["native_checks"][name] is True for name in CHECKS):
        raise ValueError("native Flash replay qualification failed")
    expected = {(name, arm) for layer in LAYERS for name in modules(layer) for arm in ARMS}
    rows = report.get("results", [])
    if len(rows) != len(expected) or {(r["tensor"], r["arm"]) for r in rows} != expected:
        raise ValueError("pilot result inventory differs")
    import math
    if any(not isinstance(r["nmse"], (int,float)) or not math.isfinite(r["nmse"]) or r["nmse"] < 0 for r in rows):
        raise ValueError("pilot metric is invalid")
    if any(r["nmse"] != 0 for r in rows if r["arm"] == "bf16"):
        raise ValueError("BF16 reference differs")
    for name in {r["tensor"] for r in rows}:
        arms = {r["arm"]:r for r in rows if r["tensor"] == name}
        if arms["mcg-k6"]["nmse"] > arms["fp8-block128"]["nmse"]:
            raise ValueError("K6 heldout error exceeds the FP8 control: " + name)
    return report


def make_identity(source, tokens, calibration, heldout, witness, reviewed):
    files = [Path(__file__), Path(__file__).with_name("recipe.py"), Path(__file__).with_name("run.py"),
        REPO/"quantization/glm5_flash/prepare_corpus.py", REPO/"quantization/kimi_k3/prepare_corpus.py",
        REPO/"quantization/glm5/core_k6/pilot.py", REPO/"quantization/kimi_k3/core_k6/pilot.py",
        REPO/"third_party/gptqmodel/gptqmodel/utils/glm5_flash_core.py",
        REPO/"third_party/gptqmodel/gptqmodel/utils/glm5_flash_mtp.py"]
    return dict(schema=SCHEMA, source=str(source.snapshot), arms=list(ARMS), layers=list(LAYERS),
        source_inventory=reviewed, source_config_sha256=sha256_file(source.snapshot/"config.json"),
        source_index_sha256=sha256_file(source.snapshot/"model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(tokens/"manifest.json"),
        calibration_sha256=hashlib.sha256(canonical(calibration).encode()).hexdigest(),
        heldout_sha256=hashlib.sha256(canonical(heldout).encode()).hexdigest(),
        calibration_tokens=sum(len(r["input_ids"]) for r in calibration),
        heldout_tokens=sum(len(r["input_ids"]) for r in heldout),
        replay="fixed-native-bf16-reference-no-candidate-affects-replay",
        selection="whole-prompts-original-order-prefix-plus-first-natural-over2051",
        long_context_witness=None if witness is None else witness["id"],
        head="unmeasured-final-normalized-input-requires-full-45-layer-replay",
        code={str(p.relative_to(REPO)):sha256_file(p) for p in files},
        gptqmodel_commit=subprocess.check_output(["git", "-C", str(REPO/"third_party/gptqmodel"), "rev-parse", "HEAD"], text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot", "tokens", "root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--heldout-tokens", type=int, default=1024)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.calibration_tokens <= 8192 or not 256 <= args.heldout_tokens <= 2048:
        parser.error("short pilot token budget outside admission")
    from glm5_flash.prepare_corpus import load_token_records
    from .recipe import BASE_REVISION, inventory
    from gptqmodel.utils import glm5_flash_core as numerical
    source = numerical.GlmFlashBF16Source(args.source_snapshot)
    if source.snapshot.name != BASE_REVISION:
        raise ValueError("pilot source is not the pinned Flash BF16 revision")
    reviewed = inventory(source)
    calibration, heldout, witness = select_records(load_token_records(args.tokens, "calibration"),
        load_token_records(args.tokens, "heldout"), args.calibration_tokens, args.heldout_tokens)
    identity = make_identity(source, args.tokens, calibration, heldout, witness, reviewed)
    root = args.root.resolve()
    output = args.output or root/"qualification.json"
    root.mkdir(parents=True, exist_ok=True)
    if (root/"plan.json").exists() and json.loads((root/"plan.json").read_text()) != identity:
        raise ValueError("pilot identity differs; choose a fresh root")
    if args.preflight:
        print(canonical(dict(admission="passed", identity=identity)), flush=True)
        return
    if output.exists():
        prior = validate_qualification(json.loads(output.read_text()))
        if prior["identity"] != identity:
            raise ValueError("completed pilot identity differs")
        print("Pilot already complete; no GPU work.")
        return
    import torch
    from kimi_k3.core_k6.run import exclusive_run
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86*GIB/torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    with torch.inference_mode(), exclusive_run(root):
        atomic_json(root/"plan.json", identity)
        for split, records in (("calibration", calibration), ("heldout", heldout)):
            (root/split).mkdir(exist_ok=True)
            capture_split(source, numerical, records, root/split, identity)
        rows = score(source, numerical, root, identity)
        checks = qualify_serialized_replay(source, numerical, root, identity)
        # The MTP adapter supplies an independent native/aligned test against
        # actual Flash checkpoint weights; the witness prefix is explicitly
        # identified and does not claim full-model quality calibration.
        mtp_proof = qualify_mtp(source, numerical, root, identity, calibration[0])
        if mtp_proof.get("byte_exact") is not True:
            raise ValueError("native Flash MTP boundary qualification failed")
        checks["mtp_boundary_byte_exact"] = True
        observed_indexer = [json.loads(line) for line in (root/"calibration/events.jsonl").read_text().splitlines()
            if json.loads(line).get("event") == "native_boundary_exact" and json.loads(line).get("indexer") is not None]
        if witness is not None and not any(row["row_id"] == witness["id"] and row["indexer"]["sparse_selection"]
                                           for row in observed_indexer):
            raise ValueError("natural long prompt did not exercise sparse indexer selection")
        report = dict(status="passed", identity=identity, native_checks=checks, layers=list(LAYERS),
            results=rows, mtp=mtp_proof, native_indexer_evidence=observed_indexer, elapsed_seconds=time.monotonic()-started,
            recipe_decision="retain-K6-base-core-K4-routed-and-MTP-conservative-native-keep-list",
            head_measured=False, full_model_quality_measured=False)
        validate_qualification(report)
        atomic_json(output, report)
        atomic_json(root/"summary.json", report)
        # Receipts and metric evidence remain, but no activation banks or
        # standalone candidate weights survive a fully qualified short pilot.
        for split in ("calibration", "heldout"):
            for name in ("frontier", "proof-inputs"):
                path = root/split/name
                if path.is_symlink():
                    raise ValueError("refuse symlink pilot cleanup")
                if path.exists():
                    shutil.rmtree(path)
        for path in (root/"candidates").glob("*.safetensors"):
            path.unlink()
        progress(root, "complete", results=len(rows), elapsed_seconds=report["elapsed_seconds"])
    source.close()


if __name__ == "__main__":
    main()
