#!/usr/bin/env python3
"""Tabulate `cuteafd::memory` ledger reports from coordinator or expertd logs.

Usage: memory-audit.py LOG [LOG ...] [--last | --sequence N] [--json]

Each log line written by `shared::memory_report` carries one JSON report:
per device the runtime's used bytes, the ledger's tracked bytes by category,
the untracked rest (CUDA context, modules, cuBLAS, graph executables),
weights by tensor stem and resident format, and dual-format tensors.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GIB = float(1 << 30)

# Tensor stem -> weight group (coarse, family-agnostic).
GROUPS = [
    (r"embed_tokens|\bembed\b|wte", "embedding"),
    (r"lm_head|\bhead\b", "lm_head"),
    (r"indexer|\.index|wk\b|weights_proj|w_ik|w_iq", "indexer"),
    (r"shared_expert|shared_experts", "shared_expert"),
    (r"\.mlp\.experts|\.experts\.", "routed_expert"),
    (r"\.mlp\.gate\.|router|e_score_correction|tid2eid", "router"),
    (r"mtp|nextn|eh_proj|enorm|hnorm|dspark", "speculator"),
    (r"norm", "norm"),
    (r"hc_|\.hc\.", "hyper_connection"),
    (r"self_attn|\.attn\.|attention|kv_b|q_a|q_b|kv_a|o_proj|qkv|wo_|wq|wkv|linear_attn|kda|compressor", "attention"),
    (r"\.mlp\.|ffn|gate_up|down_proj|up_proj|gate_proj", "dense_ffn"),
]


def category(scope):
    """Planner category of a ledger scope (mirrors cuteafd_core::memory_layout::Category::of_scope)."""
    root = scope.split("/")[0]
    simple = {"weights": "weights", "embedding": "embedding", "experts": "experts", "local-experts": "experts",
              "drafter": "drafter", "kv": "kv", "prefix": "prefix", "workspace": "workspace", "sampler": "workspace",
              "spark-intake": "workspace", "probe": "workspace", "transport": "transport", "peer-split": "transport",
              "staging": "staging", "mapped-table": "tables", "ple": "tables"}
    if root in simple:
        return simple[root]
    if root == "v41":
        stage = scope[4:]
        low = stage.lower()
        if "draft" in low:
            return "drafter"
        if "engram" in low:
            return "tables"
        if "weights" in low:
            return "weights"
        if low.startswith("kv") or "kv cache" in low:
            return "kv"
        if "prefix" in low or "snapshot" in low:
            return "prefix"
        if "expert" in low:
            return "experts"
        if "transport" in low or "tp2" in low:
            return "transport"
        if "workspace" in low or "lane" in low or "vision" in low:
            return "workspace"
    return "runtime"


def group(stem):
    for pattern, name in GROUPS:
        if stem and re.search(pattern, stem):
            return name
    return "other"


def reports(path):
    with open(path, errors="replace") as handle:
        for line in handle:
            if "memory ledger" not in line:
                continue
            line = ANSI.sub("", line)
            start = line.find("report=")
            if start < 0:
                continue
            text = line[start + len("report="):]
            try:
                value, _ = json.JSONDecoder().raw_decode(text)
            except json.JSONDecodeError:
                continue
            seq = re.search(r"sequence=(\d+)", line)
            value["_sequence"] = int(seq.group(1)) if seq else None
            yield value


def gib(n):
    return f"{n / GIB:8.2f}"


def summarize(report):
    out = {"stage": report.get("stage"), "devices": [], "pinned": report.get("pinned", {})}
    weights = defaultdict(int)
    for space, device, scope, tensor, fmt, nbytes, _count in report.get("weights", []):
        weights[(space, device, group(tensor), fmt or "native")] += nbytes
    for dev in report.get("devices", []):
        entry = dict(dev)
        cats = defaultdict(int)
        for scope, nbytes in dev.get("scopes", {}).items():
            cats[category(scope)] += nbytes
        cats["runtime"] += max(dev.get("untracked", 0), 0)
        entry["categories"] = dict(cats)
        entry["weights"] = {f"{g}/{f}": b for (s, d, g, f), b in sorted(weights.items())
                            if s != "pinned" and d == dev["device"]}
        out["devices"].append(entry)
    out["dual_formats"] = report.get("dual_formats", [])
    out["host"] = report.get("host", {})
    return out


def render(path, summary):
    print(f"== {path} (stage {summary['stage']})")
    for dev in summary["devices"]:
        print(f"device {dev['device']}: used {gib(dev.get('used', 0))} GiB of {gib(dev.get('total', 0))}, "
              f"tracked {gib(dev['tracked'])}, untracked {gib(dev.get('untracked', 0))}, peak tracked {gib(dev['peak'])}")
        print("   by category: " + ", ".join(f"{c} {n / GIB:.2f}" for c, n in
                                             sorted(dev["categories"].items(), key=lambda kv: -kv[1])))
        for scope, nbytes in sorted(dev["scopes"].items(), key=lambda kv: -kv[1]):
            print(f"   {scope:<44} {gib(nbytes)}")
        if dev["weights"]:
            print("   weights by group/format:")
            for key, nbytes in sorted(dev["weights"].items(), key=lambda kv: -kv[1]):
                print(f"     {key:<42} {gib(nbytes)}")
    pinned = summary["pinned"]
    if pinned.get("tracked"):
        print(f"pinned host: {gib(pinned['tracked'])} GiB (peak {gib(pinned.get('peak', 0))})")
        for scope, nbytes in sorted(pinned.get("scopes", {}).items(), key=lambda kv: -kv[1]):
            print(f"   {scope:<44} {gib(nbytes)}")
    host = summary.get("host") or {}
    if host:
        print("host: " + ", ".join(f"{k} {gib(v).strip()}" for k, v in host.items()))
    for dual in summary["dual_formats"]:
        print(f"DUAL device {dual['device']}: {dual['tensor']} resident as {', '.join(dual['formats'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare", help="`cuteafd plan --layout --json` output to compare against")
    parser.add_argument("--device-map", default="rtx0=0,rtx1=1",
                        help="planner device=ledger device pairs for the logs given (e.g. spark0=0)")
    args = parser.parse_args()
    results = {}
    for path in args.logs:
        found = list(reports(path))
        if args.sequence is not None:
            found = [r for r in found if r["_sequence"] == args.sequence]
        if not found:
            print(f"{path}: no memory ledger reports", file=sys.stderr)
            continue
        results[path] = summarize(found[-1])
    if args.compare:
        plan = json.load(open(args.compare))["memory_layout"]
        names = {}
        for device in plan["devices"]:
            prefix = "rtx" if device["kind"] == "rtx" else "spark"
            names[f"{prefix}{device['index']}"] = device
        mapping = dict(pair.split("=") for pair in args.device_map.split(","))
        for path, summary in results.items():
            ledger = {str(d["device"]): d for d in summary["devices"]}
            for planned, ledger_id in mapping.items():
                if planned not in names or ledger_id not in ledger:
                    continue
                predicted = defaultdict(int)
                for item in names[planned]["items"]:
                    predicted[item["category"]] += item["bytes"]
                measured = ledger[ledger_id]["categories"]
                print(f"== {path} {planned} (ledger device {ledger_id}): predicted vs measured GiB")
                total_p = total_m = 0
                for cat in sorted(set(predicted) | set(measured)):
                    p_, m_ = predicted.get(cat, 0), measured.get(cat, 0)
                    total_p += p_; total_m += m_
                    err = (p_ - m_) / m_ * 100 if m_ else float("nan")
                    print(f"   {cat:<12} {p_ / GIB:8.2f} {m_ / GIB:8.2f} {err:+7.1f}%")
                err = (total_p - total_m) / total_m * 100 if total_m else float("nan")
                print(f"   {'total':<12} {total_p / GIB:8.2f} {total_m / GIB:8.2f} {err:+7.1f}%")
        return
    if args.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    else:
        for path, summary in results.items():
            render(path, summary)


if __name__ == "__main__":
    main()
