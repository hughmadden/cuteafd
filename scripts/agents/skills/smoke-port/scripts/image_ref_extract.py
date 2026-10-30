#!/usr/bin/env python3
"""Pull a model's reference implementation out of a serving image, no Docker.

Day-0 model support usually ships first in a vendor-tagged serving image
(`vllm/vllm-openai:<model>`, `lmsysorg/sglang:<model>`) before it lands in the
engine's main branch. This reads the image straight from its registry: it
resolves the manifest for one platform, picks the layer that installed the
engine's Python package (from the image history, or `--layer`), streams that
one layer and extracts only the files matching `--include` (default: `*.py`
and dist-info metadata). Nothing is executed; extracted files are data.

    image_ref_extract.py vllm/vllm-openai:mistral-large-4 --out ref/
    image_ref_extract.py lmsysorg/sglang:mistral-large-4 --package sglang --out ref/

Then search the tree for the architecture name from the checkpoint config,
e.g. `grep -rn MistralLarge4 ref/` (vLLM's registry is
`vllm/model_executor/models/registry.py`; newer model packages live under
`vllm/models/<name>/`). Writes `<out>/IMAGE.json` with the digests read.

Standard library only; honors HTTPS_PROXY. Run it isolated: `python3 -I`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def parse_ref(ref: str) -> tuple[str, str, str]:
    """`[registry/]repo[:tag|@digest]` -> (registry, repo, reference)."""
    registry = "registry-1.docker.io"
    first = ref.split("/", 1)[0]
    if "/" in ref and ("." in first or ":" in first) and first != "docker.io":
        registry, ref = ref.split("/", 1)
    elif first == "docker.io":
        ref = ref.split("/", 1)[1]
    if "@" in ref:
        repo, tag = ref.split("@", 1)
    elif ":" in ref.rsplit("/", 1)[-1]:
        repo, tag = ref.rsplit(":", 1)
    else:
        repo, tag = ref, "latest"
    if registry == "registry-1.docker.io" and "/" not in repo:
        repo = f"library/{repo}"
    return registry, repo, tag


class Registry:
    def __init__(self, registry: str, repo: str):
        self.base, self.repo, self.token = f"https://{registry}/v2/{repo}", repo, None
        if registry == "registry-1.docker.io":
            url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
            with urllib.request.urlopen(url, timeout=60) as r:
                self.token = json.load(r)["token"]

    def open(self, path: str, accept: str | None = None):
        req = urllib.request.Request(f"{self.base}/{path}")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if accept:
            req.add_header("Accept", accept)
        return urllib.request.urlopen(req, timeout=600)

    def json(self, path: str, accept: str | None = None) -> dict:
        with self.open(path, accept) as r:
            return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="e.g. vllm/vllm-openai:mistral-large-4")
    ap.add_argument("--out", required=True, help="directory to extract into (created; must be empty or new)")
    ap.add_argument("--arch", default="amd64", help="platform architecture (amd64, arm64)")
    ap.add_argument("--package", default="vllm", help="Python package whose install layer to extract")
    ap.add_argument("--layer", type=int, action="append", help="layer index(es) to extract instead of the history pick")
    ap.add_argument("--tries", type=int, default=3, help="candidate layers to try before giving up")
    ap.add_argument("--max-layer-mib", type=int, default=2048, help="skip candidate layers larger than this")
    ap.add_argument("--include", action="append", default=None,
                    help="glob(s) of member paths to extract (default: *.py and *.dist-info/METADATA)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    includes = a.include or ["*.py", "*.dist-info/METADATA", "*.dist-info/direct_url.json"]

    registry, repo, tag = parse_ref(a.image)
    reg = Registry(registry, repo)
    top = reg.json(f"manifests/{tag}", ACCEPT)
    manifest_digest = None
    if "manifests" in top:  # index / manifest list: pick the platform
        pick = next(m for m in top["manifests"]
                    if m.get("platform", {}).get("architecture") == a.arch and m["platform"].get("os") == "linux")
        manifest_digest = pick["digest"]
        man = reg.json(f"manifests/{manifest_digest}", ACCEPT)
    else:
        man = top
    cfg = reg.json(f"blobs/{man['config']['digest']}")
    layers = man["layers"]

    # Map non-empty history entries onto layers, then rank the likely install
    # layers: a wheel install that names the package, then any wheel install,
    # then a pip install naming the package; smallest first within a rank (a
    # requirements layer can be several GB). Try them until one holds
    # `<package>/__init__.py`.
    history = [h for h in cfg.get("history", []) if not h.get("empty_layer")]

    def rank(i: int) -> int | None:
        cb = history[i].get("created_by", "")
        named, whl, pip = a.package.lower() in cb.lower(), ".whl" in cb, "pip install" in cb
        return 0 if named and whl else 1 if whl and pip else 2 if named and pip else None

    if a.layer:
        chosen, explicit = a.layer, True
    else:
        ranked = [i for i in range(min(len(history), len(layers))) if rank(i) is not None
                  and layers[i]["size"] <= a.max_layer_mib << 20]
        chosen, explicit = sorted(ranked, key=lambda i: (rank(i), layers[i]["size"]))[:a.tries], False
    if not chosen:
        print("no install layer found in the history; pass --layer N. Layers:", file=sys.stderr)
        for i, (lay, h) in enumerate(zip(layers, history)):
            print(f"  {i:3d} {lay['size'] >> 20:6d} MiB {h.get('created_by', '')[-120:]}", file=sys.stderr)
        return 1

    extracted, used = 0, []
    for i in chosen:
        lay = layers[i]
        print(f"layer {i}: {lay['size'] >> 20} MiB {lay['digest']}", file=sys.stderr)
        with reg.open(f"blobs/{lay['digest']}") as r, tarfile.open(fileobj=r, mode="r|*") as tar:
            for m in tar:
                if m.isfile() and any(fnmatch.fnmatch(m.name, g) for g in includes):
                    tar.extract(m, out, filter="data")
                    extracted += 1
        used.append(i)
        if not explicit and any(out.rglob(f"{a.package}/__init__.py")):
            break
    chosen = used
    (out / "IMAGE.json").write_text(json.dumps({
        "image": a.image, "registry": registry, "repo": repo, "tag": tag, "arch": a.arch,
        "manifest_digest": manifest_digest, "config_digest": man["config"]["digest"], "created": cfg.get("created"),
        "layers": [{"index": i, "digest": layers[i]["digest"], "size": layers[i]["size"]} for i in chosen],
        "files": extracted,
    }, indent=1))
    pkgs = sorted({str(p.parent) for p in out.rglob(f"{a.package}/__init__.py")})
    print(f"{extracted} files -> {out}; package roots: {pkgs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
