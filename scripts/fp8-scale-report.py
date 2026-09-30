#!/usr/bin/env python3
"""fp8-scale-report.py SNAPSHOT --pattern REGEX [--layout row128|block128] [--rows-per-chunk N]

For each BF16 2-D tensor of SNAPSHOT whose name matches REGEX, quantizes it to E4M3 (round to nearest even,
saturating) with FP32 scales per 128-wide K block and output row (row128) or per 128 x 128 block (block128),
under two scale rules:
  amax: s = amax / 448 (what the engine's FP8 copies used)
  pow2: s = the smallest power of two >= amax / 448 (Hugh Madden's glm53f-afd pow2_scale: a BF16 value with
        at most E4M3's 3 mantissa bits inside the block's range then quantizes exactly)
  best: per block, whichever of the two leaves the smaller squared error (never worse than amax)
and reports the share of weights the copy reproduces exactly and the relative RMS error ||q*s - w|| / ||w||.
Host only (numpy)."""
import argparse, json, os, re, struct, sys
import numpy as np


def tensors(snapshot):
    index = os.path.join(snapshot, "model.safetensors.index.json")
    files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
    for f in files:
        path = os.path.join(snapshot, f)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                yield name, meta, path, 8 + n


def bf16(path, base, meta, first, count):
    rows, cols = meta["shape"]
    start = base + meta["data_offsets"][0] + first * cols * 2
    raw = np.fromfile(path, dtype=np.uint16, count=count * cols, offset=start)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(count, cols)


def e4m3(v):
    a = np.abs(v)
    e = np.clip(np.floor(np.log2(np.maximum(a, np.float32(2.0 ** -9)))), -6, 8)
    step = np.exp2(e - 3).astype(np.float32)
    q = np.minimum(np.round(a / step) * step, np.float32(448.0))
    return np.copysign(q, v).astype(np.float32)


def pow2_scale(amax):
    b = amax.astype(np.float32).view(np.uint32).astype(np.int64)
    x = np.clip((b >> 23) - 135 + ((b & 0x7FFFFF) > 0x600000), -126, 127)
    s = ((x + 127).astype(np.uint32) << 23).view(np.float32)
    return np.where(amax > 0, s, np.float32(1.0))


def amax_scale(amax):
    return np.where(amax > 0, (amax / np.float32(448.0)).astype(np.float32), np.float32(1.0))


def measure(w, layout):
    rows, cols = w.shape
    blocks = w.reshape(rows, cols // 128, 128)
    amax = np.abs(blocks).max(axis=2)  # [rows, kb]
    if layout == "block128":
        r = rows // 128
        amax = np.repeat(amax[: r * 128].reshape(r, 128, -1).max(axis=1), 128, axis=0)
    out, errs = {}, {}
    for rule, scale in (("amax", amax_scale), ("pow2", pow2_scale)):
        s = scale(amax)[:, :, None]
        deq = e4m3(blocks / s) * s
        errs[rule] = deq - blocks
    if layout == "block128":
        r = rows // 128
        blockwise = lambda e: np.repeat((e.astype(np.float64) ** 2).sum(axis=2).reshape(r, 128, -1).sum(axis=1), 128, axis=0)
    else:
        blockwise = lambda e: (e.astype(np.float64) ** 2).sum(axis=2)
    pick = (blockwise(errs["pow2"]) < blockwise(errs["amax"]))[:, :, None]
    errs["best"] = np.where(pick, errs["pow2"], errs["amax"])
    for rule, err in errs.items():
        out[rule] = (int((err == 0).sum()), float((err.astype(np.float64) ** 2).sum()))
    return out, float((w.astype(np.float64) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--layout", choices=["row128", "block128"], default="row128")
    ap.add_argument("--rows-per-chunk", type=int, default=4096)
    ap.add_argument("--verbose", action="store_true", help="one line per tensor")
    args = ap.parse_args()
    pattern = re.compile(args.pattern)
    totals = {}
    print(f"{'tensor':64s} {'shape':>14s} {'exact amax':>10s} {'exact pow2':>10s} {'exact best':>10s} "
          f"{'relRMS amax':>12s} {'relRMS pow2':>12s} {'relRMS best':>12s}")
    for name, meta, path, base in sorted(tensors(args.snapshot), key=lambda t: t[0]):
        if not pattern.search(name) or meta["dtype"] != "BF16" or len(meta["shape"]) != 2 or meta["shape"][1] % 128:
            continue
        rows, cols = meta["shape"]
        if args.layout == "block128" and rows % 128:
            continue
        exact = {"amax": 0, "pow2": 0, "best": 0}
        sq = {"amax": 0.0, "pow2": 0.0, "best": 0.0}
        norm = 0.0
        chunk = max(128, args.rows_per_chunk // 128 * 128)
        for first in range(0, rows, chunk):
            count = min(chunk, rows - first)
            got, n = measure(bf16(path, base, meta, first, count), args.layout)
            norm += n
            for rule in exact:
                exact[rule] += got[rule][0]
                sq[rule] += got[rule][1]
        n = rows * cols
        rel = {r: (sq[r] / norm) ** 0.5 if norm else 0.0 for r in sq}
        if args.verbose:
            print(f"{name:64s} {str(rows)+'x'+str(cols):>14s} {exact['amax']/n:10.4f} {exact['pow2']/n:10.4f} "
                  f"{exact['best']/n:10.4f} {rel['amax']:12.4e} {rel['pow2']:12.4e} {rel['best']:12.4e}", flush=True)
        kind = re.sub(r"\.\d+\.", ".N.", name)
        t = totals.setdefault(kind, [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0])
        t[0] += n; t[1] += exact["amax"]; t[2] += exact["pow2"]; t[3] += exact["best"]
        t[4] += sq["amax"]; t[5] += sq["pow2"]; t[6] += sq["best"]; t[7] += norm; t[8] += 1
    print(f"\nsummary by tensor kind ({args.layout}; tensors, weights, exact share and relative RMS per rule)")
    for kind, t in sorted(totals.items()):
        print(f"{kind:64s} {t[8]:>4d} {t[0]:>12d} {t[1]/t[0]:10.4f} {t[2]/t[0]:10.4f} {t[3]/t[0]:10.4f} "
              f"{(t[4]/t[7])**0.5:12.4e} {(t[5]/t[7])**0.5:12.4e} {(t[6]/t[7])**0.5:12.4e}")


if __name__ == "__main__":
    sys.exit(main())
