#!/usr/bin/env python3
"""Read-only local admission before joining and after winning the GPU queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.core_k6.container import IMAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if socket.gethostname().lower() != "oliver":
        raise RuntimeError("this recipe's production placement is Oliver only")
    for relative in (".venv/bin/python",):
        if not (REPO / relative).is_file():
            raise FileNotFoundError(relative)
    for directory, paths in ((REPO, ["quantization/kimi_k3/core_k6", "quantization/kimi_k3/prepare_corpus.py",
                                    "quantization/kimi_k3/uniform_k2/storage.py"]),
                             (REPO / "third_party/gptqmodel", [])):
        dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--", *paths], text=True)
        if dirty:
            raise RuntimeError(f"uncommitted production code in {directory}")
    lock = json.loads((REPO / "third_party/gptqmodel.lock.json").read_text())
    actual = subprocess.check_output(["git", "-C", str(REPO / "third_party/gptqmodel"), "rev-parse", "HEAD"], text=True).strip()
    helper = runpy.run_path(str(REPO / "scripts/build/verify-sparkinfer-source.py"))
    if actual != lock["revision"] or helper["source_tree_sha256"](REPO / "third_party/gptqmodel") != lock["source_tree_sha256"]:
        raise RuntimeError("GPTQModel differs from its committed source lock")
    qualification = json.loads((Path.home() / ".cache/cuteafd/builds/kimi-core-k6/qualification.json").read_text())
    if (qualification["gptqmodel_commit"] != actual
            or qualification["source_tree_sha256"] != lock["source_tree_sha256"]
            or qualification["native_report"]["status"] != "passed"):
        raise RuntimeError("this GPTQModel source has not passed target GPU qualification")
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "scripts/build/assert-build-filesystem.py"),
        str(Path.home() / ".cache/cuteafd/builds/kimi-k3/exllamav3")], check=True)
    subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(devices) != 1 or "RTX PRO 6000" not in devices[0] or int(devices[0].rsplit(",",1)[1].strip()) < 96000:
        raise RuntimeError("expected one local 96GB RTX PRO 6000")
    subprocess.run([str(REPO / ".venv/bin/python"), str(Path(__file__).with_name("run.py")),
                    *arguments, "--preflight"], check=True, timeout=240)
    print(json.dumps(dict(admission="passed", host="oliver", image=IMAGE)), flush=True)


if __name__ == "__main__":
    main()
