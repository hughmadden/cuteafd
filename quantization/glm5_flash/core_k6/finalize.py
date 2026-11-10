#!/usr/bin/env python3
"""Finalize completed Flash GPU evidence under an explicit reporting-only policy.

This command never replays, quantizes, initializes CUDA or deletes evidence.
It retains the original capture identity and audits every historical receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/"quantization"))
if __package__ in (None, ""):
    __package__ = "glm5_flash.core_k6"

from .pilot import (ARMS, CHECKS, LAYERS, PREFIX, decision_policy, modules,
    select_records, validate_policy_change, validate_qualification, validate_report_code)
from kimi_k3.uniform_k2.storage import atomic_json, canonical, sha256_file, verify_receipt

ORIGINAL_PILOT = "104b8d557"
PILOT_PATH = "quantization/glm5_flash/core_k6/pilot.py"


def _event(rows, kind, **fields):
    matches = [row for row in rows if row.get("event") == kind
               and all(row.get(k) == value for k, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one measured {kind} event for {fields}, found {len(matches)}")
    return matches[0]


def native_checks_from_events(identity, calibration, heldout, root_events, cal_events, held_events, mtp):
    """Derive checks only from the historical code's successful proof events."""
    witness = identity["long_context_witness"]
    cal_witnesses = [calibration[0]] + [r for r in calibration if r["id"] == witness and r != calibration[0]]
    observed = []
    for rows, records in ((cal_events, cal_witnesses), (held_events, [heldout[0]])):
        for layer in range(max(LAYERS)+1):
            _event(rows, "layer_committed", layer=layer)
            for record in records:
                event = _event(rows, "native_boundary_exact", layer=layer, row_id=record["id"],
                               tokens=len(record["input_ids"]))
                indexer = event.get("indexer")
                if layer in (3,7):
                    if (not isinstance(indexer, dict) or indexer.get("byte_exact") is not True
                            or indexer.get("raw_topk") != 2048 or indexer.get("pool_size") != 4
                            or indexer.get("pool_budget") != 512
                            or indexer.get("tokens") != len(record["input_ids"])):
                        raise ValueError("native pooled-indexer proof differs")
                    sparse = indexer["valid_last_row"] < indexer["tokens"]
                    if indexer.get("sparse_selection") is not sparse:
                        raise ValueError("reported sparse-selection event contradicts observation")
                    if record["id"] == witness and not sparse:
                        raise ValueError("natural long witness did not exercise sparse selection")
                    observed.append(event)
                elif indexer is not None:
                    raise ValueError("KDA boundary unexpectedly declares a DSA indexer")
    for layer in LAYERS:
        for record in cal_witnesses:
            event = _event(root_events, "serialized_boundary_exact", layer=layer, row_id=record["id"],
                           tokens=len(record["input_ids"]))
            if layer == 7:
                indexer = event.get("indexer", {})
                if (indexer.get("byte_exact") is not True
                        or indexer.get("tokens") != len(record["input_ids"])
                        or indexer.get("pool_budget") != 512
                        or (record["id"] == witness and indexer.get("sparse_selection") is not True)):
                    raise ValueError("serialized pooled-indexer proof differs")
    first = calibration[0]
    _event(root_events, "mtp_boundary_exact", tokens=len(first["input_ids"]), shifted_pairs=len(first["input_ids"])-1)
    evidence = mtp.get("projection_evidence", {})
    metrics = mtp.get("projection_metrics", {})
    if (mtp.get("byte_exact") is not True or mtp.get("serialized_k4_byte_exact") is not True
            or mtp.get("prefix_layers") != max(LAYERS)+1 or mtp.get("final_target_frontier") is not False
            or mtp.get("source_row") != first["id"] or mtp.get("tokens") != len(first["input_ids"])
            or mtp.get("shifted_pairs") != len(first["input_ids"])-1
            or evidence.get("bits") != 4 or evidence.get("identity_rows") != 0
            or evidence.get("hessian_rows") != len(first["input_ids"])-1
            or metrics.get("hessian_sample_count") != len(first["input_ids"])-1
            or metrics.get("quantizer_path") != "hessian_ldlq"
            or metrics.get("hessian_metric_status") != "ok"
            or mtp.get("contract", {}).get("positions") != "shifted-token-position[t+1]"):
        raise ValueError("independent native/serialized Flash MTP proof differs")
    # Each true value is backed by the complete event inventories above; no
    # inference is made from the fact that a process happened to exit.
    return dict.fromkeys(CHECKS, True), observed


def audit_evidence(root, source, tokens, repo=REPO):
    from safetensors import safe_open
    from glm5_flash.prepare_corpus import load_token_records
    root, source, tokens, repo = (Path(p).resolve() for p in (root,source,tokens,repo))
    recorded = {}
    def read(relative):
        path = root/relative
        recorded[relative] = sha256_file(path)
        return json.loads(path.read_text())
    def receipt(relative, expected):
        path = root/relative
        verify_receipt(path, expected, checksum=True)
        recorded[relative] = expected["sha256"]
    identity = read("plan.json")
    if source != Path(identity["source"]).resolve():
        raise ValueError("finalization source snapshot differs from capture")
    for filename,key in (("config.json","source_config_sha256"),
                         ("model.safetensors.index.json","source_index_sha256")):
        if sha256_file(source/filename) != identity[key]:
            raise ValueError("source metadata changed after capture")
    if sha256_file(tokens/"manifest.json") != identity["token_manifest_sha256"]:
        raise ValueError("prepared corpus manifest changed after capture")
    calibration, heldout, witness = select_records(load_token_records(tokens,"calibration"),
                                                  load_token_records(tokens,"heldout"))
    for split,records in (("calibration",calibration),("heldout",heldout)):
        if (hashlib.sha256(canonical(records).encode()).hexdigest() != identity[split+"_sha256"]
                or sum(len(r["input_ids"]) for r in records) != identity[split+"_tokens"]):
            raise ValueError("pilot selected corpus rows changed")
    if (None if witness is None else witness["id"]) != identity["long_context_witness"]:
        raise ValueError("pilot natural long witness changed")
    results = read("results.json")
    if results["identity"] != identity:
        raise ValueError("metric result capture identity differs")
    rows = results["results"]
    expected = {(name,arm) for layer in LAYERS for name in modules(layer) for arm in ARMS}
    if len(rows) != len(expected) or {(r["tensor"],r["arm"]) for r in rows} != expected:
        raise ValueError("historical metric inventory is incomplete")
    events = {}
    for name in ("events.jsonl", "calibration/events.jsonl", "heldout/events.jsonl"):
        path = root/name
        recorded[name] = sha256_file(path)
        events[name] = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        event = _event(events["events.jsonl"], "pilot_result", tensor=row["tensor"], arm=row["arm"])
        if any(event.get(k) != value for k,value in row.items()):
            raise ValueError("historical metric and measured event differ")
        if (row["calibration_tokens"] != identity["calibration_tokens"]
                or row["heldout_tokens"] != identity["heldout_tokens"]):
            raise ValueError("historical metric row counts differ")
        if row["arm"] != "bf16":
            relative = "candidates/"+row["tensor"]+"."+row["arm"]+".safetensors"
            item = read(str(Path(relative).with_suffix(".json")))
            if item["identity"] != identity:
                raise ValueError("historical candidate identity differs")
            receipt(relative,item["receipt"])
            with safe_open(str(root/relative), framework="pt", device="cpu") as reader:
                size = 0
                for key in reader.keys():
                    tensor = reader.get_slice(key)
                    import math
                    size += math.prod(tensor.get_shape())*{"I16":2,"F16":2,"F32":4,"I32":4,"U8":1}[tensor.get_dtype()]
                if size != row["payload_bytes"]:
                    raise ValueError("historical candidate payload accounting differs")
    for split,records in (("calibration",calibration),("heldout",heldout)):
        for layer in LAYERS:
            manifest = read(f"{split}/hessians/layer-{layer}.json")
            if manifest["identity"] != identity or set(manifest["files"]) != set(modules(layer)):
                raise ValueError("historical Hessian identity/inventory differs")
            for name,expected_receipt in manifest["files"].items():
                relative = f"{split}/hessians/{name}.safetensors"
                receipt(relative,expected_receipt)
                with safe_open(str(root/relative), framework="pt",device="cpu") as reader:
                    if (set(reader.keys()) != {"H","count"}
                            or int(reader.get_tensor("count")) != identity[split+"_tokens"]
                            or reader.get_slice("H").get_dtype() != "F32"):
                        raise ValueError("historical Hessian observation contract differs")
            proofs = read(f"{split}/proof-inputs/layer-{layer}.json")
            if proofs["identity"] != identity:
                raise ValueError("historical replay input identity differs")
            expected_rows = {records[0]["id"]:records[0]}
            expected_rows.update({r["id"]:r for r in records if r["id"] == witness["id"]} if witness else {})
            if {entry["row_id"] for entry in proofs["entries"]} != set(expected_rows):
                raise ValueError("historical replay input row inventory differs")
            for entry in proofs["entries"]:
                if entry["layer"] != layer or entry["tokens"] != len(expected_rows[entry["row_id"]]["input_ids"]):
                    raise ValueError("historical replay input geometry differs")
                receipt(split+"/"+entry["file"],entry["receipt"])
    mtp = read("mtp-proof.json")
    receipt("candidates/mtp-eh-k4.safetensors",mtp["projection_receipt"])
    checks, observed = native_checks_from_events(identity,calibration,heldout,
        events["events.jsonl"],events["calibration/events.jsonl"],events["heldout/events.jsonl"],mtp)
    policy_hash = hashlib.sha256(canonical(decision_policy()).encode()).hexdigest()
    report = dict(status="passed",identity=identity,native_checks=checks,layers=list(LAYERS),results=rows,
        mtp=mtp,native_indexer_evidence=observed,decision_policy=decision_policy(),
        recipe_decision="retain-K6-base-core-K4-routed-and-MTP-conservative-native-keep-list",
        head_measured=False,full_model_quality_measured=False,evidence_root=str(root))
    validate_qualification(report)
    timestamps = [datetime.fromisoformat(row["time"]) for values in events.values() for row in values]
    report["elapsed_seconds"] = (max(timestamps)-min(timestamps)).total_seconds()
    revision = subprocess.check_output(["git","-C",str(repo),"rev-parse",ORIGINAL_PILOT],text=True).strip()
    original = subprocess.check_output(["git","-C",str(repo),"show",revision+":"+PILOT_PATH],text=True)
    allowed = validate_policy_change(original,(repo/PILOT_PATH).read_text(),identity["code"][PILOT_PATH])
    audit = dict(status="passed", identity=identity,native_checks=checks,verified_files=recorded,
        verified_candidate_payloads=43,metric_rows=len(rows),native_event_count=len(observed),
        decision_policy_sha256=policy_hash,original_pilot_git_revision=revision,
        finalization="CPU-only-existing-completed-GPU-evidence-no-numerical-rerun-no-evidence-deletion",
        time=datetime.now(timezone.utc).isoformat())
    atomic_json(root/"finalization-audit.json",audit)
    report["policy_migration"] = dict(schema="cuteafd-flash-pilot-report-only-migration-v1",
        original_pilot_git_revision=revision,original_pilot_sha256=identity["code"][PILOT_PATH],
        current_pilot_sha256=sha256_file(repo/PILOT_PATH),allowed_reporting_nodes=allowed,
        original_plan_sha256=recorded["plan.json"],decision_policy_sha256=policy_hash,
        audit_file="finalization-audit.json",audit_sha256=sha256_file(root/"finalization-audit.json"),
        finalizer_code_sha256=sha256_file(repo/"quantization/glm5_flash/core_k6/finalize.py"),
        reason="FP8-control-is-nearly-lossless-on-some-source-projections; admit-bounded-K6-compression-by-absolute-error")
    validate_report_code(report,repo)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.root/"qualification.json"
    if output.exists():
        raise ValueError("refuse to overwrite an existing qualification; retain its immutable audit")
    report = audit_evidence(args.root,args.source_snapshot,args.tokens)
    atomic_json(output,report)
    atomic_json(args.root/"summary.json",report)
    print(canonical(dict(status="passed",report=str(output),metric_rows=len(report["results"]),
        maximum_k6_nmse=max(r["nmse"] for r in report["results"] if r["arm"] == "mcg-k6"),
        native_checks=report["native_checks"],capture_identity_retained=True)),flush=True)


if __name__ == "__main__":
    main()
