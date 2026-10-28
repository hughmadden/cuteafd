"""Read-only admission for the bounded Flash pilot, quantizer and publisher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
from kimi_k3.uniform_k2.storage import sha256_file
from kimi_k3.core_k6.container import IMAGE

CHECKS = ("kda_boundary_byte_exact", "mla_boundary_byte_exact", "mhc_boundary_byte_exact",
          "mtp_boundary_byte_exact", "serialized_projection_replay")


def require_local_source():
    if socket.gethostname().lower() != "maxq":
        raise ValueError("Flash quantization is authorized only on maxq")
    result = subprocess.check_output(["nest", "where", "hf:zai-org/GLM-5.3-Flash-BF16"],
                                     text=True, timeout=60)
    if not any(line.split() == ["maxq", "ready"] for line in result.splitlines()):
        raise ValueError("the entire BF16 source must be local in Sparknest on maxq")


def verify_fork():
    fork = REPO / "third_party/gptqmodel"
    for directory, paths in ((REPO, ["quantization/glm5_flash"]), (fork, [])):
        if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--", *paths],
                                   text=True).strip():
            raise ValueError("uncommitted Flash production source")
    actual = subprocess.check_output(["git", "-C", str(fork), "rev-parse", "HEAD"], text=True).strip()
    lock = json.loads((REPO / "third_party/gptqmodel.lock.json").read_text())
    digest = runpy.run_path(str(REPO / "scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"](fork)
    if actual != lock["revision"] or digest != lock["source_tree_sha256"]:
        raise ValueError("GPTQModel pin or source-tree lock differs")
    return actual


def qualified(report_path, fork_commit):
    report = json.loads(Path(report_path).read_text())
    identity = report["identity"]
    if (report.get("status") != "passed" or identity.get("gptqmodel_commit") != fork_commit
            or any(report.get("native_checks", {}).get(name) is not True for name in CHECKS)
            or report.get("layers") != [1, 4, 7]):
        raise ValueError("Flash native/reconstructed pilot qualification is incomplete")
    code = identity.get("code", {})
    if not code:
        raise ValueError("pilot has no immutable numerical source identity")
    for name, digest in code.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"producer changed after Flash qualification: {name}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--publication-only", action="store_true")
    parser.add_argument("--qualification-report", type=Path, default=Path.home() /
        ".cache/cuteafd/quantization/glm53-flash-k4-core-k6/pilot/qualification.json")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.publication_only and args.qualification_only:
        parser.error("publication and pilot are separate modes")
    if socket.gethostname().lower() != "maxq":
        raise ValueError("Flash production placement is maxq only")
    commit = verify_fork()
    if not args.publication_only:
        require_local_source()
        if not args.qualification_only:
            qualified(args.qualification_report, commit)
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        devices = subprocess.check_output([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                          text=True, timeout=30).strip().splitlines()
        if (len(devices) != 1 or "RTX PRO 6000" not in devices[0]
                or int(devices[0].rsplit(",", 1)[1].strip()) < 96000):
            raise ValueError("expected exactly one local 96GB RTX PRO 6000")
        subprocess.run(["docker", "image", "inspect", IMAGE], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
    runner = Path(__file__).with_name("pilot.py" if args.qualification_only else "run.py")
    subprocess.run([str(REPO / ".venv/bin/python"), str(runner), *arguments, "--preflight"],
                   check=True, timeout=540)
    print(json.dumps(dict(admission="passed", host="maxq", fork_commit=commit,
                          pilot=args.qualification_only, publication=args.publication_only)), flush=True)


if __name__ == "__main__":
    main()
