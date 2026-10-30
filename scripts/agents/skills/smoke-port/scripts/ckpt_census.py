#!/usr/bin/env python3
"""Byte census of a safetensors checkpoint from its headers alone.

Reads the config (`config.json`, or Mistral-native `params.json`), the shard
index and every shard's JSON header -- from a local snapshot or from a Hugging
Face repo at a pinned revision with HTTP range reads -- and reports, without
reading a single weight byte:

* every tensor group (layer and expert indices collapsed), with dtype, count,
  bytes, and every distinct shape with the layers that carry it (per-layer
  variation such as NoPE layers or KV-sharing layers shows up here);
* bytes by component (routed experts / coordinator / drafter / vision /
  embedding+head), using overridable regexes;
* the Spark fit: widest per-rank slice of the routed experts for each TP world
  (intermediate split in whole 128-row blocks), against a budget.

Standard library only. Run it isolated: `python3 -I ckpt_census.py ...`.

    ckpt_census.py --local /path/to/snapshot --out census
    ckpt_census.py --hf org/repo --rev <sha> --out census   # HF_TOKEN if gated

`--stub-dir DIR` also writes a header-only snapshot: every non-weight file
(configs, index, tokenizer, template) copied, and every shard as a sparse file
of its true size holding only its header. `cuteafd plan DIR` and other
header-only tools then run on a checkpoint that was never downloaded; any code
that reads weight bytes reads zeros, so never serve from a stub.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

GIB = 1 << 30
DEFAULT_CATEGORIES = [
    ("routed_experts", r"(^|\.)experts\.\d+\.|\.experts\.(w[123]|gate_up|down)"),
    ("drafter", r"^(mtp|mtp_dspark|dspark|dflash|draft|model\.mtp|model\.layers\.mtp)"),
    ("vision", r"(^|\.)(vision|visual|vit|patch_merger|pre_mm_projector|vision_language_adapter|mm_projector|multi_modal_projector|audio)"),
    ("embed_head", r"^(tok_embeddings|output|lm_head|model\.embed_tokens|embed_tokens)\.|^(model\.)?(embed|head)\b"),
]


def fetch(req: urllib.request.Request, tries: int = 6) -> bytes:
    """GET with bounded retries on 429/5xx, honoring Retry-After."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = e.headers.get("Retry-After")
            time.sleep(min(60.0, float(wait) if wait and wait.isdigit() else 2.0 ** attempt))
    raise AssertionError("unreachable")


class Source:
    def __init__(self, local: str | None, hf: str | None, rev: str | None):
        self.local = Path(local) if local else None
        self.hf, self.rev = hf, rev or "main"
        self.token = os.environ.get("HF_TOKEN")

    def _url(self, name: str) -> str:
        return f"https://huggingface.co/{self.hf}/resolve/{self.rev}/{name}"

    def _get(self, name: str, rng: tuple[int, int] | None = None) -> bytes:
        if self.local:
            with open(self.local / name, "rb") as f:
                if rng is None:
                    return f.read()
                f.seek(rng[0])
                return f.read(rng[1] - rng[0] + 1)
        req = urllib.request.Request(self._url(name))
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if rng is not None:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        return fetch(req)

    def exists(self, name: str) -> bool:
        try:
            if self.local:
                return (self.local / name).is_file()
            self._get(name, (0, 0))
            return True
        except Exception:
            return False

    def json(self, name: str) -> dict:
        return json.loads(self._get(name))

    def header_bytes(self, shard: str) -> bytes:
        """The shard's 8-byte length prefix and JSON header."""
        n = struct.unpack("<Q", self._get(shard, (0, 7)))[0]
        return self._get(shard, (0, 7 + n))

    def header(self, shard: str) -> dict:
        return json.loads(self.header_bytes(shard)[8:])

    def files(self) -> list[tuple[str, int]]:
        """(path, size) of every file in the snapshot."""
        if self.local:
            return [(str(p.relative_to(self.local)), p.stat().st_size)
                    for p in sorted(self.local.rglob("*")) if p.is_file() and ".cache" not in p.parts]
        url = f"https://huggingface.co/api/models/{self.hf}/tree/{self.rev}?recursive=true"
        req = urllib.request.Request(url)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        return [(e["path"], e.get("size", 0)) for e in json.loads(fetch(req)) if e.get("type") == "file"]


def write_stub(s: Source, shards: list[str], stub: Path, max_copy: int) -> tuple[int, int]:
    """Header-only snapshot: small files copied, shards sparse at true size."""
    stub.mkdir(parents=True, exist_ok=True)
    copied = sparse = 0
    for path, size in s.files():
        dst = stub / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path in shards:
            with open(dst, "wb") as f:
                f.write(s.header_bytes(path))
                f.truncate(size)  # sparse: apparent size only, no data blocks
            sparse += 1
        elif not path.endswith((".safetensors", ".bin", ".pt", ".gguf")) and size <= max_copy:
            dst.write_bytes(s._get(path))
            copied += 1
    return copied, sparse


def group_name(tensor: str) -> tuple[str, tuple[int, ...]]:
    """Collapse every numeric path segment; return the group and the indices."""
    idx = tuple(int(m) for m in re.findall(r"(?<=\.)\d+(?=\.|$)", tensor))
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "N", tensor), idx


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local", help="local snapshot directory")
    src.add_argument("--hf", help="Hugging Face repo id (org/name)")
    ap.add_argument("--rev", help="revision (commit SHA preferred) for --hf")
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.json and <out>.md")
    ap.add_argument("--spark-budget-gib", type=float, default=100.0, help="per-Spark expert budget (planner default 100)")
    ap.add_argument("--expert-intermediate", type=int, help="routed expert intermediate size (read from config if omitted)")
    ap.add_argument("--category", action="append", default=[], metavar="NAME=REGEX",
                    help="extra/override category regex, tried before the defaults")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stub-dir", help="also write a header-only sparse snapshot here (see the module doc)")
    ap.add_argument("--stub-max-copy-mib", type=int, default=64, help="largest non-weight file the stub copies")
    a = ap.parse_args()

    s = Source(a.local, a.hf, a.rev)
    cfg_name = next((n for n in ("config.json", "params.json") if s.exists(n)), None)
    cfg = s.json(cfg_name) if cfg_name else {}
    idx_name = next((n for n in ("model.safetensors.index.json", "consolidated.safetensors.index.json") if s.exists(n)), None)
    if idx_name:
        shards = sorted(set(s.json(idx_name)["weight_map"].values()))
    else:  # single-file checkpoints
        shards = [n for n in ("model.safetensors", "consolidated.safetensors") if s.exists(n)][:1]
    if not shards:
        print("no safetensors shards found", file=sys.stderr)
        return 1

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        headers = dict(zip(shards, ex.map(s.header, shards)))

    cats = [(c.split("=", 1)[0], re.compile(c.split("=", 1)[1])) for c in a.category]
    cats += [(n, re.compile(r)) for n, r in DEFAULT_CATEGORIES]
    groups: dict[str, dict] = {}
    by_cat: dict[str, int] = defaultdict(int)
    total = 0
    for shard, h in headers.items():
        h.pop("__metadata__", None)
        for name, t in h.items():
            nbytes = t["data_offsets"][1] - t["data_offsets"][0]
            total += nbytes
            g, ix = group_name(name)
            cat = next((n for n, r in cats if r.search(name)), "coordinator")
            by_cat[cat] += nbytes
            e = groups.setdefault(g, {"dtype": set(), "count": 0, "bytes": 0, "shapes": defaultdict(list), "category": cat})
            e["dtype"].add(t["dtype"])
            e["count"] += 1
            e["bytes"] += nbytes
            e["shapes"][json.dumps(t["shape"])].append(ix[0] if ix else -1)

    # Spark fit: the planner splits each expert's intermediate rows in whole
    # 128-row blocks, so the widest rank holds ceil(blocks / tp) blocks.
    inter = a.expert_intermediate or (cfg.get("moe") or {}).get("expert_hidden_dim") or cfg.get("moe_intermediate_size")
    routed = by_cat.get("routed_experts", 0)
    fit = []
    if inter and routed:
        blocks = inter // 128
        for tp in range(1, 9):
            if blocks < tp:
                break
            share = -(-blocks // tp) / blocks
            fit.append({"tp": tp, "share": share, "gib_per_rank": routed * share / GIB,
                        "fits": routed * share <= a.spark_budget_gib * GIB})

    out = {
        "source": a.local or f"{a.hf}@{s.rev}", "config_file": cfg_name, "index_file": idx_name,
        "shards": len(shards), "tensor_bytes": total,
        "bytes_by_category": dict(by_cat),
        "groups": {g: {"category": e["category"], "dtype": sorted(e["dtype"]), "count": e["count"], "bytes": e["bytes"],
                       "shapes": {k: compress(v) for k, v in e["shapes"].items()}} for g, e in groups.items()},
        "spark_fit": {"expert_intermediate": inter, "budget_gib": a.spark_budget_gib, "worlds": fit},
    }
    Path(f"{a.out}.json").write_text(json.dumps(out, indent=1))

    md = [f"# Checkpoint census: `{out['source']}`", "",
          f"Config `{cfg_name}`, index `{idx_name}`, {len(shards)} shards, {total:,} tensor bytes "
          f"({total / GIB:.2f} GiB). Headers only; no weight bytes read.", "",
          "| Component | GiB |", "| --- | ---: |"]
    md += [f"| {c} | {b / GIB:.2f} |" for c, b in sorted(by_cat.items(), key=lambda x: -x[1])]
    if fit:
        md += ["", f"Spark fit (routed experts, intermediate {inter} in 128-row blocks, budget {a.spark_budget_gib:g} GiB):", "",
               "| TP | widest share | GiB/rank | fits |", "| ---: | ---: | ---: | --- |"]
        md += [f"| {f['tp']} | {f['share']:.4f} | {f['gib_per_rank']:.1f} | {'yes' if f['fits'] else 'no'} |" for f in fit]
    md += ["", "| Group | Component | dtype | count | GiB | shapes (layers) |", "| --- | --- | --- | ---: | ---: | --- |"]
    for g, e in sorted(out["groups"].items(), key=lambda x: -x[1]["bytes"]):
        shp = "; ".join(f"{k} ({v})" if v != "-1" else k for k, v in e["shapes"].items())
        md.append(f"| `{g}` | {e['category']} | {','.join(e['dtype'])} | {e['count']} | {e['bytes'] / GIB:.3f} | {shp} |")
    Path(f"{a.out}.md").write_text("\n".join(md) + "\n")
    print(f"{a.out}.md: {len(groups)} groups, {total / GIB:.2f} GiB; " +
          ", ".join(f"{c} {b / GIB:.2f}" for c, b in sorted(by_cat.items(), key=lambda x: -x[1])))
    if a.stub_dir:
        copied, sparse = write_stub(s, shards, Path(a.stub_dir), a.stub_max_copy_mib << 20)
        print(f"stub {a.stub_dir}: {copied} files copied, {sparse} sparse shards (headers only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
