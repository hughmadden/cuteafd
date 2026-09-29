#!/usr/bin/env python3
"""Compare two golden directories (golden.py outputs): per-layer cosine for
layers both saved, logits top-1 agreement and mean KL(a || b).

  compare.py A_DIR B_DIR
"""
import sys
from pathlib import Path

import numpy as np


def bf16(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def main() -> None:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    for layer in sorted(p.name for p in b.glob("layer*.bin")):
        x, y = bf16(a / layer), bf16(b / layer)
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x))
        print(f"{layer}: cosine {cos:.6f} rel_l2 {rel:.3e}")
    tokens = np.frombuffer((a / "tokens.bin").read_bytes(), dtype=np.int32)
    la = np.frombuffer((a / "logits.bin").read_bytes(), dtype=np.float32).reshape(len(tokens), -1)
    lb = np.frombuffer((b / "logits.bin").read_bytes(), dtype=np.float32).reshape(len(tokens), -1)
    agree = float((la.argmax(1) == lb.argmax(1)).mean())
    def logsoftmax(l):
        l = l - l.max(1, keepdims=True)
        return l - np.log(np.exp(l).sum(1, keepdims=True))
    pa, pb = logsoftmax(la.astype(np.float64)), logsoftmax(lb.astype(np.float64))
    kl = float((np.exp(pa) * (pa - pb)).sum(1).mean())
    next_a = float((la.argmax(1)[:-1] == tokens[1:]).mean())
    next_b = float((lb.argmax(1)[:-1] == tokens[1:]).mean())
    print(f"logits: top-1 agreement {agree:.4f} | mean KL {kl:.5f} nats | next-token accuracy {next_a:.4f} vs {next_b:.4f}")


if __name__ == "__main__":
    main()
