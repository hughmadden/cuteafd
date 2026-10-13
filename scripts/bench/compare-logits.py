#!/usr/bin/env python3
"""Compare two F32 logits dumps (rows x vocab), as written by a golden's --save-logits.

  scripts/bench/compare-logits.py A B VOCAB [--tokens tokens.bin --offset K] [--json OUT]

Reports each file's sha256 and byte equality first, then over the common rows:
max |a - b|, mean and max KL(a||b) (float64), top-1 agreement and, with the
token file and the row offset of A's first row in it, each side's mean NLL of
the next token. Rows are streamed in blocks, so full-prompt dumps (~1 GB) stay
within a few hundred MB of memory. Exit 0 when both files were compared (the
verdict is in the output), 2 on unusable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BLOCK_ROWS = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def log_softmax(x: np.ndarray) -> np.ndarray:
    top = x.max(1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(1, keepdims=True))


def compare(a_path: Path, b_path: Path, vocab: int, tokens: Path | None = None, offset: int = 0) -> dict:
    if vocab <= 0:
        raise ValueError('vocab must be positive')
    sizes = {name: path.stat().st_size for name, path in (('a', a_path), ('b', b_path))}
    for name, size in sizes.items():
        if size % (4 * vocab):
            raise ValueError(f'{name}: {size} bytes is not a whole number of {vocab}-wide F32 rows')
    result = dict(a=str(a_path), b=str(b_path), vocab=vocab, sha256_a=sha256(a_path), sha256_b=sha256(b_path),
                  rows_a=sizes['a'] // (4 * vocab), rows_b=sizes['b'] // (4 * vocab))
    result['byte_identical'] = result['sha256_a'] == result['sha256_b'] and sizes['a'] == sizes['b']
    rows = min(result['rows_a'], result['rows_b'])
    result['rows'] = rows
    if rows == 0:
        return result
    a = np.memmap(a_path, dtype='<f4', mode='r', shape=(result['rows_a'], vocab))
    b = np.memmap(b_path, dtype='<f4', mode='r', shape=(result['rows_b'], vocab))
    ids = None
    if tokens is not None:
        ids = np.fromfile(tokens, dtype='<u4')
    max_diff, kl_sum, kl_max, agree, nll_a, nll_b, scored = 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0
    nonfinite = 0
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        xa, xb = np.asarray(a[start:stop], dtype=np.float64), np.asarray(b[start:stop], dtype=np.float64)
        finite = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        nonfinite += int((~finite).sum())
        xa, xb = xa[finite], xb[finite]
        if not len(xa):
            continue
        max_diff = max(max_diff, float(np.abs(xa - xb).max()))
        la, lb = log_softmax(xa), log_softmax(xb)
        kl = (np.exp(la) * (la - lb)).sum(1)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
        agree += int((xa.argmax(1) == xb.argmax(1)).sum())
        if ids is not None:
            index = np.arange(start, stop)[finite] + offset + 1
            keep = index < len(ids)
            nxt = ids[index[keep]]
            nll_a -= float(la[keep][np.arange(len(nxt)), nxt].sum())
            nll_b -= float(lb[keep][np.arange(len(nxt)), nxt].sum())
            scored += len(nxt)
    compared = rows - nonfinite
    result.update(nonfinite_rows=nonfinite, max_abs_diff=max_diff, mean_kl=kl_sum / max(compared, 1),
                  max_kl=kl_max, top1_agreement=agree / max(compared, 1))
    if ids is not None:
        result.update(nll_a=nll_a / max(scored, 1), nll_b=nll_b / max(scored, 1), nll_rows=scored)
    return result


def describe(result: dict) -> str:
    lines = [f"byte-identical: {result['byte_identical']}  rows {result['rows_a']} vs {result['rows_b']}",
             f"sha256 a {result['sha256_a']}", f"sha256 b {result['sha256_b']}"]
    if result['rows']:
        lines.append(f"rows {result['rows']}: max |diff| {result['max_abs_diff']:.4e} | mean KL(a||b) {result['mean_kl']:.6f} "
                     f"max {result['max_kl']:.6f} | top-1 agreement {100 * result['top1_agreement']:.2f}%"
                     + (f" | {result['nonfinite_rows']} nonfinite rows skipped" if result['nonfinite_rows'] else ''))
    if 'nll_a' in result:
        lines.append(f"mean NLL a {result['nll_a']:.4f} b {result['nll_b']:.4f} over {result['nll_rows']} rows")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('vocab', type=int)
    parser.add_argument('--tokens', type=Path, help="token file (u32) the rows score against")
    parser.add_argument('--offset', type=int, default=0, help="index in --tokens of A's first row")
    parser.add_argument('--json', type=Path, help='also write the result as JSON')
    args = parser.parse_args()
    try:
        result = compare(args.a, args.b, args.vocab, args.tokens, args.offset)
    except (OSError, ValueError) as error:
        print(f'compare-logits: {error}', file=sys.stderr)
        return 2
    print(describe(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
