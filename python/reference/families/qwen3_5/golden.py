#!/usr/bin/env python3
"""Official dense Qwen BF16 text goldens; MTP off, eager attention, native F32 GDN."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import finish_golden, load_set, verify_snapshot, write_scored_logits


def qualification_reference():
    path = ROOT / "scripts/qualify/qwen3_5/reference.py"
    spec = importlib.util.spec_from_file_location("qwen35_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_windows(args, model, identity, provenance):
    import numpy as np
    import torch
    from shape_invariant import qualify

    manifest = load_set(args.windows, "qwen3_5")
    verify_snapshot(manifest, args.snapshot)
    args.out.mkdir(parents=True, exist_ok=True)
    proof = qualify(args, manifest, lambda probe: run_windows(probe, model, identity, provenance))
    started, rows, loss, count = time.monotonic(), [], 0.0, 0
    with torch.inference_mode():
        for window in manifest["windows"]:
            ids = torch.tensor([window["tokens"]], dtype=torch.long, device="cuda:0")
            start = window["score_from"]
            positions = torch.arange(start - 1, len(window["tokens"]) - 1, device="cuda:0")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float()
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite golden logits")
            # Bound the FP64 scoring workspace, independently of reference inference.
            total = sum(torch.nn.functional.cross_entropy(
                logits[i:i + 32].double(), ids[0, start + i:start + i + 32],
                reduction="sum").item() for i in range(0, len(logits), 32))
            entry = write_scored_logits(args.out, window, logits.cpu().numpy().astype(np.float32))
            entry.update(nll=total / len(logits), scored_tokens=len(logits))
            folder = args.out / entry["path"]
            (folder / "meta.json").write_text(json.dumps(entry, sort_keys=True) + "\n")
            rows.append(entry)
            loss += total
            count += len(logits)
            print(f"{window['id']}: {len(ids[0])} input, {len(logits)} scored, NLL {entry['nll']:.9f}", flush=True)
            del logits, ids, positions
    finish_golden(args.out, manifest, rows, snapshot=str(args.snapshot), snapshot_identity=identity,
                  prefix_qualification=proof, seconds=time.monotonic() - started,
                  nll=loss / count, scored_tokens=count, media="text-only", **provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as reference
    from shape_invariant import install, install_eager

    manifest = load_set(args.windows, "qwen3_5")
    identity = verify_snapshot(manifest, args.snapshot)
    qualification = qualification_reference()
    _, _, _, source = qualification.reference_identity(args.snapshot)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Use the same bounded/fixed-row reference adapters as the other family goldens.
    model, fp32 = qualification.load_reference(args.snapshot)
    install()
    install_eager(reference)
    import platform
    provenance = dict(platform={"machine": platform.machine(), "gpu": torch.cuda.get_device_name(0),
                                "capability": list(torch.cuda.get_device_capability(0))},
                      reference="transformers.Qwen3_5ForConditionalGeneration",
                      transformers=transformers.__version__, torch=torch.__version__,
                      reference_sha256=hashlib.sha256(source).hexdigest(),
                      qualification_sha256=hashlib.sha256(Path(qualification.__file__).read_bytes()).hexdigest(),
                      golden_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      native_fp32_parameters=fp32, dtype="bf16", mtp=False, use_cache=False,
                      attention="eager", gates={"gdn": "silu", "full_attention": "sigmoid"})
    run_windows(args, model, identity, provenance)


if __name__ == "__main__":
    main()
