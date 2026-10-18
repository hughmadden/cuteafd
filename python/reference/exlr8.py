"""CPU EXLR8 v2 oracle, implemented from kindlingai/exlr8 FORMAT.md.

No implementation from that unlicensed repository is used. Matrices use the
format's (input, output) orientation, not Hugging Face's transposed storage.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        return json.loads(stream.read(size)), 8 + size


class Tensors:
    def __init__(self, path):
        self.path = Path(path)
        self.meta, self.start = header(path)

    def raw(self, name, offset=0, length=None):
        desc = self.meta[name]
        begin, end = desc["data_offsets"]
        length = end - begin - offset if length is None else length
        if offset < 0 or length < 0 or offset + length > end - begin:
            raise ValueError(f"range outside tensor {name}")
        with self.path.open("rb") as stream:
            stream.seek(self.start + begin + offset)
            data = stream.read(length)
        if len(data) != length:
            raise ValueError(f"truncated tensor {name}")
        return data

    def array(self, name):
        desc = self.meta[name]
        dtype = {"F16": "<f2", "F32": "<f4", "I8": "i1", "U8": "u1",
                 "I16": "<i2", "I32": "<i4", "I64": "<i8", "U64": "<u8"}[desc["dtype"]]
        return np.frombuffer(self.raw(name), dtype=dtype).reshape(desc["shape"])


def xxhash64(data, seed=0):
    """XXH64 canonical little-endian algorithm (seed zero in EXLR8)."""
    mask = (1 << 64) - 1
    p1, p2, p3, p4, p5 = (11400714785074694791, 14029467366897019727,
                          1609587929392839161, 9650029242287828579, 2870177450012600261)
    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & mask
    def lane(a, value):
        return (rol((a + value * p2) & mask, 31) * p1) & mask
    at = 0
    if len(data) >= 32:
        lanes = [(seed + p1 + p2) & mask, (seed + p2) & mask, seed & mask, (seed - p1) & mask]
        while at + 32 <= len(data):
            lanes = [lane(a, v) for a, v in zip(lanes, struct.unpack_from("<4Q", data, at))]
            at += 32
        h = sum(rol(a, n) for a, n in zip(lanes, (1, 7, 12, 18))) & mask
        for a in lanes:
            h = ((h ^ lane(0, a)) * p1 + p4) & mask
    else:
        h = (seed + p5) & mask
    h = (h + len(data)) & mask
    while at + 8 <= len(data):
        h = (rol(h ^ lane(0, struct.unpack_from("<Q", data, at)[0]), 27) * p1 + p4) & mask
        at += 8
    if at + 4 <= len(data):
        h = (rol(h ^ (struct.unpack_from("<I", data, at)[0] * p1 & mask), 23) * p2 + p3) & mask
        at += 4
    for value in data[at:]:
        h = rol(h ^ (value * p5 & mask), 11) * p1 & mask
    h ^= h >> 33
    h = h * p2 & mask
    h ^= h >> 29
    h = h * p3 & mask
    return h ^ (h >> 32)


def e4m3_round(values):
    """Finite E4M3, nearest-even with saturation; no Torch/CUDA dependency."""
    x = np.asarray(values, dtype=np.float32)
    magnitude = np.minimum(np.abs(x), 448.0)
    exponent = np.floor(np.log2(np.maximum(magnitude, 2.0 ** -6)))
    step = np.exp2(exponent - 3)
    return np.copysign(np.minimum(np.rint(magnitude / step) * step, 448.0), x)


def decode_tiles(data, k):
    if k not in (2, 3, 4) or len(data) % (32 * k):
        raise ValueError("trellis data must contain complete K2/K3/K4 tiles")
    words = np.frombuffer(data, dtype="<u2").reshape(-1, 16 * k)
    swapped = words.reshape(-1, 8 * k, 2)[:, :, ::-1].reshape(-1, 16 * k)
    bits = ((swapped[:, :, None] >> np.arange(15, -1, -1)) & 1).reshape(-1, 256 * k)
    ends = np.arange(1, 257) * k - 1
    states = np.zeros((len(words), 256), dtype=np.uint32)
    for shift in range(16):
        states |= bits[:, (ends - shift) % (256 * k)].astype(np.uint32) << shift
    x = (states * np.uint32(0xCBAC1FED) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    low = (x & 0xFFFF).astype("<u2").view("<f2")
    high = (x >> 16).astype("<u2").view("<f2")
    values = e4m3_round((low + high).astype(np.float32))
    out = np.empty((len(words), 16, 16), dtype=np.float32)
    for t in range(32):
        r, c = 2 * (t % 4), t // 4
        for i, (row, col) in enumerate(((r, c), (r+1, c), (r+8, c), (r+9, c),
                                       (r, c+8), (r+1, c+8), (r+8, c+8), (r+9, c+8))):
            out[:, row, col] = values[:, 8*t+i]
    return out


def hadamard128(values, axis=-1):
    x = np.moveaxis(np.asarray(values, dtype=np.float32), axis, -1)
    if x.shape[-1] % 128:
        raise ValueError("Hadamard dimension must be a multiple of 128")
    shape = x.shape
    x = x.copy().reshape(-1, 128)
    width = 1
    while width < 128:
        view = x.reshape(-1, 128 // (2 * width), 2, width)
        a, b = view[:, :, 0].copy(), view[:, :, 1].copy()
        view[:, :, 0], view[:, :, 1] = a + b, a - b
        width *= 2
    return np.moveaxis((x / np.sqrt(np.float32(128))).reshape(shape), -1, axis)


def decode_unit(data, k, projection, hidden):
    tiles = decode_tiles(data, k)
    if len(tiles) != hidden // 16 * 8:
        raise ValueError("incorrect unit length")
    if projection in ("gate", "up"):
        return tiles.reshape(hidden // 16, 8, 16, 16).transpose(0, 2, 1, 3).reshape(hidden, 128)
    if projection == "down":
        return tiles.reshape(hidden // 128, 8, 8, 16, 16).transpose(1, 3, 0, 2, 4).reshape(128, hidden)
    raise ValueError(f"unknown projection {projection}")


def expert_unit(snapshot, layer, k, expert, unit, projection, shared=False):
    index = Tensors(Path(snapshot) / "experts" / f"exlr8-layer-{layer:03}.index.safetensors")
    prefix = f"{'shared.' if shared else ''}k{k}."
    e = 0 if shared else expert
    p = ("gate", "up", "down").index(projection)
    offset, length = index.array(prefix + "index")[e, p, unit]
    with (Path(snapshot) / "experts" / f"exlr8-layer-{layer:03}.k{k}.safetensors").open("rb") as stream:
        stream.seek(int(offset))
        data = stream.read(int(length))
    if len(data) != length:
        raise ValueError("truncated unit")
    channel_name, block_name = ("svh", "su_block") if projection != "down" else ("suh", "sv_block")
    channels = index.array(prefix + projection + "." + channel_name)[e, unit*128:(unit+1)*128]
    blocks = index.array(prefix + projection + "." + block_name)[e]
    expected = int(index.array(prefix + "checksum")[e, p, unit])
    actual = xxhash64(data + channels.tobytes() + blocks.tobytes())
    if actual != expected:
        raise ValueError(f"unit checksum mismatch: {actual:016x} != {expected:016x}")
    hidden = len(index.array("signs.in"))
    trellis = decode_unit(data, k, projection, hidden)
    hidden_side = index.array("signs.out" if projection == "down" else "signs.in") * np.repeat(blocks.astype(np.float32), 128)
    suh, svh = (channels, hidden_side) if projection == "down" else (hidden_side, channels)
    weight = hadamard128(hadamard128(trellis, 0), 1) * suh[:, None] * svh[None, :]
    return weight, trellis, f"{actual:016x}"


def mxfp8(values):
    x = np.asarray(values, dtype=np.float32)
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    exponent = np.ceil(np.log2(np.maximum(maximum, np.finfo(np.float32).tiny) / 448))
    scale = np.exp2(np.clip(exponent, -127, 127))
    return (e4m3_round(blocks / scale) * scale).reshape(x.shape)


def activation_errors(weight, seed=408, rows=32):
    rng = np.random.default_rng(seed)
    result = {}
    for name in ("gaussian", "outliers"):
        x = rng.standard_normal((rows, weight.shape[0]), dtype=np.float32)
        if name == "outliers":
            x *= np.where(rng.random(x.shape) < .01, 30, 1)
        reference = x @ weight
        denominator = np.linalg.norm(reference)
        result[name] = {
            "a16_relative": float(np.linalg.norm(x.astype(np.float16).astype(np.float32) @ weight - reference) / denominator),
            "a8_relative": float(np.linalg.norm(mxfp8(x) @ weight - reference) / denominator),
        }
    return result


def bf16_unit(snapshot, layer, expert, unit, projection):
    root = Path(snapshot)
    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensor = Tensors(root / index["weight_map"][name])
    rows, cols = tensor.meta[name]["shape"]
    if projection != "down":
        raw = tensor.raw(name, unit * 128 * cols * 2, 128 * cols * 2)
        shape = (128, cols)
    else:
        raw = b"".join(tensor.raw(name, (r * cols + unit * 128) * 2, 256) for r in range(rows))
        shape = (rows, 128)
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--bf16-snapshot", type=Path)
    parser.add_argument("--layers", default="3,22,40,77")
    parser.add_argument("--expert", type=int, default=7)
    parser.add_argument("--unit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for layer in map(int, args.layers.split(",")):
        for projection in ("gate", "up", "down"):
            reference = bf16_unit(args.bf16_snapshot, layer, args.expert, args.unit, projection) if args.bf16_snapshot else None
            for k in (2, 3, 4):
                weight, trellis, checksum = expert_unit(args.snapshot, layer, k, args.expert, args.unit, projection)
                row = dict(layer=layer, expert=args.expert, unit=args.unit, projection=projection,
                           k=k, checksum=checksum, e4m3_exact=bool(np.array_equal(trellis, e4m3_round(trellis))),
                           activation=activation_errors(weight))
                if reference is not None:
                    row["weight_relative_error"] = float(np.linalg.norm(weight - reference) / np.linalg.norm(reference))
                results.append(row)
    args.output.write_text(json.dumps({"source": "FORMAT.md independent CPU oracle", "units": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
