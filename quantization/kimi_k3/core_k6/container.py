#!/usr/bin/env python3
"""Run the pinned uv environment inside the local SM120 development image."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
IMAGE = "cuteafd-coordinator-dev:tc-aa74e55ca2130070"


def command(arguments, *, name, fork=None):
    home = Path.home()
    environment = dict(HOME=str(home), CUDA_VISIBLE_DEVICES="0",
        CUDA_HOME="/usr/local/cuda-13.4", TORCH_CUDA_ARCH_LIST="12.0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", CC="/usr/bin/gcc", CXX="/usr/bin/g++",
        MAX_JOBS="4", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        FLA_DISABLE_TENSOR_CACHE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
        GPTQMODEL_EXLLAMAV3_BUILD_ROOT=str(home/".cache/cuteafd/builds/kimi-k3/exllamav3"),
        LD_LIBRARY_PATH=str(REPO/".venv/lib/python3.12/site-packages/torch/lib")+":/usr/local/cuda-13.4/lib64")
    if fork:
        environment.update(KIMI_GPTQMODEL_SOURCE=str(Path(fork).resolve()), PYTHONPATH=str(Path(fork).resolve()))
    result = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--network", "none",
        "--ulimit", "nofile=65536:65536",
        "--user", f"{os.getuid()}:{os.getgid()}", "--stop-timeout", "30", "--workdir", str(REPO),
        "--mount", f"type=bind,src={home},dst={home}",
        "--mount", "type=bind,src=/usr/local/cuda-13.4,dst=/usr/local/cuda-13.4,readonly",
        "--entrypoint", str(REPO/".venv/bin/python")]
    for key, value in environment.items():
        result += ["-e", f"{key}={value}"]
    return result + [IMAGE, *map(str, arguments)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fork", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        parser.error("a Python script or -m invocation is required")
    sys.exit(subprocess.call(command(arguments, name=args.name, fork=args.fork)))


if __name__ == "__main__":
    main()
