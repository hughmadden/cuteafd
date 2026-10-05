#!/usr/bin/env python3
"""WP-4 matching-SM121 byte and TCP overhead gate (reference in Spark container)."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import struct
import time

import numpy as np


def harness():
    spec = importlib.util.spec_from_file_location("tower_gate", Path(__file__).with_name("qualify-vision.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoder_identity(snapshot):
    headers = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with shard.open("rb") as source:
            size = int.from_bytes(source.read(8), "little")
            entries = json.loads(source.read(size))
        for name, entry in entries.items():
            if name.startswith("visual."):
                if entry["dtype"] != "BF16":
                    raise ValueError(f"unsupported reference tower dtype: {name}: {entry['dtype']}")
                start, end = entry["data_offsets"]
                headers[name] = {"dtype": "Bf16", "shape": entry["shape"], "byte_offset": 8 + size + start, "byte_length": end - start}
    digest = hashlib.sha256(b"cuteafd-encoder-v1")
    for value in (b"mimo_v2", snapshot.name.encode()):
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    # Rust sorts the outer BTreeMap; serde_json preserves inner field insertion order.
    headers = {name: headers[name] for name in sorted(headers)}
    digest.update(hashlib.sha256(json.dumps(headers, separators=(",", ":")).encode()).digest())
    digest.update(struct.pack("<II", 1, 121))
    return digest.hexdigest()


def reference(args):
    gate = harness()
    cfg = json.loads((args.snapshot / "config.json").read_text())
    spec, blob = gate.pack_spec(cfg, gate.read_visual(args.snapshot))
    args.reference.with_suffix(".id").write_text(encoder_identity(args.snapshot))
    lib = C.CDLL(str(args.library))
    lib.cuteafd_vision_required.argtypes = [C.POINTER(gate.Spec), C.POINTER(gate.Ledger)]
    lib.cuteafd_vision_create.argtypes = [C.POINTER(gate.Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    lib.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    lib.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_uint64, gate.OBSERVER, C.c_void_p]
    lib.cuteafd_vision_destroy.argtypes = [C.c_void_p]

    def check(code):
        if code:
            raise RuntimeError(f"native vision status {code}")

    ledger = gate.Ledger()
    owner = C.c_void_p()
    check(lib.cuteafd_vision_required(C.byref(spec), C.byref(ledger)))
    check(lib.cuteafd_vision_create(C.byref(spec), 0, ledger.weights + ledger.scratch + ledger.blas_workspace, C.byref(owner)))
    try:
        check(lib.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob
        lut = gate.normalization_lut()
        outputs = {}
        for tokens in (256, 1024, 4096):
            h, w, rgb = gate.fixture(tokens)
            out = np.empty((tokens, spec.output_width), np.uint16)
            for _ in range(2):
                check(lib.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, lut.ctypes.data, h, w, out.ctypes.data, out.nbytes, gate.OBSERVER(), None))
            outputs[str(tokens)] = out
        np.savez(args.reference, **outputs)
    finally:
        check(lib.cuteafd_vision_destroy(owner))


def receive(stream, length):
    data = bytearray()
    while len(data) < length:
        chunk = stream.recv(min(length - len(data), 1 << 20))
        if not chunk:
            raise RuntimeError("encoder connection closed")
        data.extend(chunk)
    return bytes(data)


def remote(args):
    gate = harness()
    expected = np.load(args.reference)
    host, port = args.address.rsplit(":", 1)
    results = {"address": args.address, "sm": 121, "samples": {}, "byte_exact": True}
    with socket.create_connection((host, int(port)), timeout=60) as stream:
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        header = receive(stream, 88)
        magic, identity, plan, patches, width, patch, merge = struct.unpack("<8s32s32s4I", header)
        assert magic == b"CAFDVI01" and plan.hex() == args.plan_hash
        assert identity.hex() == args.reference.with_suffix(".id").read_text().strip(), "remote/local EncoderId mismatch"
        assert (patches, patch, merge) == (16384, 16, 2)
        results["encoder_id"] = identity.hex()
        stream.sendall(header)
        assert receive(stream, 4) == bytes(4)
        for repeat in range(4):
            for tokens in ((4096, 1024, 256) if repeat % 2 else (256, 1024, 4096)):
                h, w, rgb = gate.fixture(tokens)
                key = hashlib.sha256(rgb.tobytes()).digest()
                request = struct.pack("<I32s3IQQ", 1, key, 1, h, w, tokens, rgb.nbytes)
                start = time.perf_counter()
                stream.sendall(request)
                stream.sendall(rgb.tobytes())
                status, reply_key, elapsed_ns, length = struct.unpack("<I32sQQ", receive(stream, 52))
                assert status == 0 and reply_key == key and length == tokens * width * 2
                output = receive(stream, length)
                wall_ms = (time.perf_counter() - start) * 1000
                exact = output == expected[str(tokens)].tobytes()
                results["byte_exact"] &= exact
                assert exact, f"remote/local SM121 bytes differ at {tokens} tokens"
                if repeat:
                    encode_ms = elapsed_ns / 1e6
                    results["samples"].setdefault(str(tokens), []).append({
                        "wall_ms": wall_ms, "encode_ms": encode_ms,
                        "tcp_overhead_ms": max(0, wall_ms - encode_ms),
                        "tcp_fraction": max(0, wall_ms - encode_ms) / encode_ms,
                    })
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--address")
    parser.add_argument("--plan-hash", default="ab" * 32)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if len(arguments.plan_hash) != 64 or any(c not in "0123456789abcdef" for c in arguments.plan_hash):
        parser.error("--plan-hash must be 64 lowercase hexadecimal characters")
    if arguments.address:
        if arguments.output is None:
            parser.error("remote mode requires --output")
        try:
            host, port = arguments.address.rsplit(":", 1)
            if not host or not 1 <= int(port) <= 65535:
                raise ValueError()
        except ValueError:
            parser.error("--address must be HOST:PORT with port 1..65535")
    elif arguments.snapshot is None or arguments.library is None:
        parser.error("reference mode requires --snapshot and --library")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.address:
        remote(arguments)
    else:
        reference(arguments)
