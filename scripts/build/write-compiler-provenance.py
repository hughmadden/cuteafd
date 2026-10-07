#!/usr/bin/env python3
"""Seal actual compiler/cache settings alongside an opt-in build's binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def output(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def seal(source, binary):
    identity = {}
    try:
        identity = json.loads((source / "BUILD_IDENTITY.json").read_text())
    except (OSError, ValueError):
        pass
    if not identity.get("commit"):
        commit = output("git", "rev-parse", "HEAD", cwd=source)
        if commit:
            identity = {"commit": commit, "dirty": bool(output("git", "status", "--porcelain",
                                                              "--untracked-files=no", cwd=source))}
        else:
            try:
                stamp = (source / ".cuteafd-source-revision").read_text().split()
            except OSError:
                stamp = []
            identity = {"commit": stamp[0] if stamp else "unknown", "dirty": "dirty" in stamp[1:]}
    active = os.environ.get("CUTEAFD_KACHE_ACTIVE")
    fallback = Path(os.environ.get("CUTEAFD_KACHE_WARNING_DIR", "/nonexistent")).exists()
    with binary.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"schema": 1, "source": identity,
            "rustc": output("rustc", "-vV"), "cargo": output("cargo", "-V"),
            "kache_mode": "fallback" if fallback else os.environ.get("CUTEAFD_KACHE_MODE", "disabled"),
            "kache": output(active, "--version") if active else None,
            "rustflags": os.environ.get("RUSTFLAGS", ""),
            "cargo_encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
            "path_remapping": "kache injected" if active else "none added by cuteafd",
            "rustc_wrapper": os.environ.get("RUSTC_WRAPPER", ""),
            "binary_sha256": binary_sha256}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.write_text(json.dumps(seal(args.source, args.binary), indent=2) + "\n")
