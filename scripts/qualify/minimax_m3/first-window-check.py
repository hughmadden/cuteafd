#!/usr/bin/env python3
"""File preconditions and result gates for first-window.sh; preflight is local only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
MODEL = "MiniMaxAI/MiniMax-M3.1-Flash"


def config(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            raise ValueError("configuration must contain literal KEY=VALUE entries")
        value = value.strip().strip("\"'")
        if any(char in value for char in "`$\n\r"):
            raise ValueError("configuration interpolation is not permitted")
        values[key.strip()] = value
    return values


def hosts(values):
    count = int(values["SPARK_COUNT"])
    result = [values.get(f"SPARK_{rank}_HOST", "") for rank in range(count)]
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host) for host in result):
        raise ValueError("every rank needs an explicit SPARK_<rank>_HOST")
    if len(set(result)) != count:
        raise ValueError("Spark ranks must use distinct hosts")
    if values.get("SPARK_HOSTS", ",".join(result)) != ",".join(result):
        raise ValueError("SPARK_HOSTS must match the explicit rank order")
    for rank in range(count):
        if not re.fullmatch(r"[0-9A-Fa-f.:]+", values.get(f"SPARK_{rank}_LANE_A", "")):
            raise ValueError("every rank needs SPARK_<rank>_LANE_A")
    return result


def checkpoint(snapshot):
    if snapshot.name != REVISION:
        raise ValueError("snapshot directory must name the pinned revision")
    cfg = json.loads((snapshot / "config.json").read_text())
    if cfg.get("model_type") != "minimax_m3_vl":
        raise ValueError("snapshot is not the MiniMax M3 release")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError("checkpoint has no tensor shards")
    for name in shards:
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError("checkpoint shard escapes snapshot")
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if not 2 <= size <= 128 * 1024 * 1024:
                raise ValueError("invalid safetensors header length")
            header = json.loads(handle.read(size))
            tensors = [entry for key, entry in header.items() if key != "__metadata__"]
            extent = max(entry["data_offsets"][1] for entry in tensors)
            if not extent or path.stat().st_size != 8 + size + extent:
                raise ValueError("full weight payloads required; header-only or incomplete shard")
            # Sparse header snapshots preserve apparent lengths; reject their holes too.
            if getattr(path.stat(), "st_blocks", extent // 512) * 512 < extent:
                raise ValueError("full weight payloads required; sparse header snapshot")
            nonzero = False
            for offset in (0, extent // 2, max(0, extent - 4096)):
                handle.seek(8 + size + offset)
                nonzero |= any(handle.read(min(4096, extent - offset)))
            if not nonzero:
                raise ValueError("full weight payloads required; zero payload samples")


def reference(path):
    meta = json.loads((path / "meta.json").read_text())
    windows = json.loads((path / "windows.json").read_text())
    if (meta.get("schema") != "cuteafd.fidelity.golden/2"
            or meta.get("family") != "minimax_m3" or meta.get("checkpoint") != MODEL
            or meta.get("activation") != "served-cache" or meta.get("diagnostic_only") is True
            or meta.get("set_sha256") != windows.get("set_sha256")):
        raise ValueError("a complete served-cache MiniMax fidelity reference is required")
    proof = meta.get("prefix_qualification", {})
    if (proof.get("schema") != "cuteafd.fidelity.prefix/1" or proof.get("passed") is not True
            or proof.get("finite") is not True or proof.get("different_rows") != []
            or proof.get("set_sha256") != meta.get("set_sha256")
            or proof.get("snapshot_identity") != meta.get("snapshot_identity")
            or meta.get("snapshot_identity", {}).get("snapshot_revision") != REVISION):
        raise ValueError("reference lacks a passing byte-exact prefix proof")
    if not meta.get("windows"):
        raise ValueError("reference has no windows")
    for window in meta["windows"]:
        folder = (path / window["path"]).resolve()
        if not folder.is_relative_to(path.resolve()):
            raise ValueError("reference window escapes its root")
        for name in ("tokens.bin", "logits.bin", "layer03.bin", "layer05.bin"):
            if not (folder / name).is_file() or not (folder / name).stat().st_size:
                raise ValueError(f"reference window lacks {name}; export --layers 3 5")
        if (folder / "tokens.bin").stat().st_size < 66 * 4:
            raise ValueError("prefix checks at 63/64/65 require every window to have at least 66 tokens")
    diagnostic_window(path)


def diagnostic_window(path):
    meta = json.loads((path / "meta.json").read_text())
    eligible = [(len_window, window["id"]) for window in meta["windows"]
                for len_window in [(path / window["path"] / "tokens.bin").stat().st_size // 4]
                if len_window >= 2177]
    if not eligible:
        raise ValueError("MSA selection qualification needs a window with >=2177 tokens to exercise pruning")
    return min(eligible)[1]


def preflight(card):
    required = ("MINIMAX_SNAPSHOT", "MINIMAX_GOLDEN", "MINIMAX_CONFIG_DIR", "MINIMAX_PLAN_BIN", "MINIMAX_RUN_ID")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError("set " + ", ".join(missing))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,19}", os.environ["MINIMAX_RUN_ID"]):
        raise ValueError("MINIMAX_RUN_ID must be a fresh identifier, at most 20 characters")
    planner = Path(os.environ["MINIMAX_PLAN_BIN"])
    if not planner.is_file() or not os.access(planner, os.X_OK):
        raise ValueError("MINIMAX_PLAN_BIN must be an existing executable")
    checkpoint(Path(os.environ["MINIMAX_SNAPSHOT"]))
    reference(Path(os.environ["MINIMAX_GOLDEN"]))
    cards = ("pro6000", "5090") if card == "all" else (card,)
    all_hosts = None
    for current in cards:
        expected_hosts = None
        gpu = os.environ.get(f"MINIMAX_GPU_{current.upper()}", "")
        if gpu not in ("0", "1"):
            raise ValueError(f"set MINIMAX_GPU_{current.upper()} to an explicit local GPU index (0 or 1)")
        for tp in (4, 2, 3):
            values = config(Path(os.environ["MINIMAX_CONFIG_DIR"]) / f"minimax-m31-flash-nvfp4-{current}-tp{tp}.config")
            required_values = {"SPARK_COUNT": str(tp), "RTX_GPUS": "1", "KV_CACHE": "fp8",
                               "SPECULATOR": "off", "COORDINATOR_GPU": gpu,
                               "COORDINATOR_GPU_BUDGET_GIB": "31.8" if current == "5090" else "95.5"}
            for key, wanted in required_values.items():
                if values.get(key) != wanted:
                    raise ValueError(f"{current} TP{tp}: {key} must be {wanted}")
            current_hosts = hosts(values)
            if expected_hosts is None:
                expected_hosts = current_hosts
                if all_hosts is None:
                    all_hosts = current_hosts
                elif all_hosts != current_hosts:
                    raise ValueError("--card all requires a shared rank pool; otherwise run each --card separately")
            elif current_hosts != expected_hosts[:tp]:
                raise ValueError("TP2/TP3 must use prefixes of the TP4 rank list")
            if values.get("MODEL_ID") not in (MODEL, os.environ["MINIMAX_SNAPSHOT"]):
                raise ValueError("configuration model differs from the pinned checkpoint")
            if values.get("MODEL_REVISION") != REVISION:
                raise ValueError("configuration revision differs from the pinned checkpoint")
            if not 1024 <= int(values.get("EXPERT_PORT", "0")) <= 65535:
                raise ValueError("set an explicit unprivileged EXPERT_PORT in each configuration")
    print("Local preconditions pass. Device availability, artifacts and numerics remain unverified.")


def load_harness():
    spec = importlib.util.spec_from_file_location("minimax_golden_run", ROOT / "scripts/bench/golden-run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api(url):
    import urllib.request
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Say hello."}],
                         "max_tokens": 16, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    choices = body.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    if body.get("error") or not (message.get("content") or message.get("reasoning_content")):
        raise ValueError("API returned an error or empty completion")


def result_gate(path, kind):
    result = json.loads(path.read_text())
    partial = kind != "full"
    prefix = kind != "diagnostic"
    if (result.get("schema") != "cuteafd.minimax.golden/1" or result.get("passed") is not True
            or result.get("partial_target") is not partial or result.get("prefix_byte_exact") is not prefix
            or not result.get("windows")):
        raise ValueError("golden summary does not prove the requested target and prefix gates")
    def number(row, key):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"missing finite golden metric {key}")
        return value
    for window in result["windows"]:
        if window.get("partial_target") is not partial or window.get("prefix_byte_exact") is not prefix:
            raise ValueError("window target/prefix proof differs from the summary")
        if partial and (number(window, "cosine") < 0.99 or number(window, "relative_l2") > 0.05):
            raise ValueError("partial golden failed cosine >=0.99 or relative L2 <=0.05")
    if not partial:
        if (number(result, "scored_tokens") <= 0 or result.get("gate_scope") != "generated"
                or number(result, "top1") < 0.90 or number(result, "mean_kl") > 0.06):
            raise ValueError("full golden failed top-1 >=90% or KL <=0.06 nat")


def runtime_inputs():
    from urllib.parse import urlparse
    url = urlparse(os.environ["MINIMAX_URL"])
    address = os.environ["MINIMAX_ADDR"]
    if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.username or url.password or url.query or url.fragment or url.path not in ("", "/")
            or not url.port or str(url.port) != address.rsplit(":", 1)[-1]
            or any(char.isspace() for char in address)):
        raise ValueError("MINIMAX_URL must be the local HTTP endpoint on MINIMAX_ADDR's port")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "hosts", "locks", "plan", "api", "expert-port", "window", "result", "runtime", "budgets"))
    parser.add_argument("value")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--kind", choices=("full", "partial", "diagnostic"), default="full")
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            preflight(args.value)
        elif args.mode == "hosts":
            print(",".join(hosts(config(args.value))))
        elif args.mode == "locks":
            print(",".join(load_harness().locks_for([args.gpu], hosts(config(args.value)))))
        elif args.mode == "api":
            api(args.value)
        elif args.mode == "window":
            print(diagnostic_window(Path(args.value)))
        elif args.mode == "result":
            result_gate(Path(args.value), args.kind)
        elif args.mode == "runtime":
            runtime_inputs()
        elif args.mode == "budgets":
            values = config(args.value)
            worker = int(values.get("SPARK_DEVICE_BUDGET_BYTES", str(108 << 30)))
            if worker <= 0:
                raise ValueError("Spark memory budget must be positive")
            print(values["COORDINATOR_GPU_BUDGET_GIB"], worker)
        elif args.mode == "expert-port":
            port = int(config(args.value).get("EXPERT_PORT", "0"))
            if not 1024 <= port <= 65535:
                raise ValueError("set an explicit unprivileged EXPERT_PORT in each configuration")
            print(port)
        else:
            deployment = json.loads(Path(args.value).read_text())
            report = deployment.get("report", {})
            if deployment.get("wip_launchable") is not True:
                raise ValueError("deployment is not WIP launchable")
            if report.get("fits") is not True or report.get("placement_supported") is not True:
                raise ValueError("plan must support placement and fit all budgets")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"first-window: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
