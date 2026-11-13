"""Byte-exact oracle for the ML4 Markov head's row-split joint projection (#1285).

Usage:
  nvcc -shared -std=c++17 -O3 -arch=sm_120f -Xcompiler -fPIC \
      -DCUTEAFD_ML4_DSPSPARK_LEGACY_ORACLE \
      -o /tmp/libml4_oracle.so native/families/mistral_large4/cuda/ml4_dspark.cu
  python native/tests/ml4_dspark_oracle.py /tmp/libml4_oracle.so \
      [--snapshot /mnt/sparknest/hf-home/hub/models--mistralai--Mistral-Large-4-1T-A52B/snapshots/<rev>] \
      [--bench]

The oracle build exports both the current `cuteafd_ml4_dspark_markov` (projection
rows split over sequences x row-blocks sized from the device's SM count) and
`cuteafd_ml4_dspark_markov_oracle` (the pre-#1285 single-CTA joint kernel). Both
run the same inputs at sequences 1, 4, 7, 16 and 32; the greedy drafts, the BF16
state and embedding, and the BF16-rounded joint projection must be byte-identical.

Inputs are random synthetic weights and, with `--snapshot`, the official
checkpoint's own Markov weights and a real drafter forward pass through the
reference (`python/reference/families/mistral_large4/dspark.py`) for `hidden` and
the soft-capped base logits. `--bench` reports one timing per arm per call.

The library can contain only ml4_dspark.cu, built with the macro above (the
production build does not compile the legacy kernel).
"""
import argparse
import ctypes as C
import statistics
import sys
from pathlib import Path

import torch

P, I = C.c_void_p, C.c_int32
ARMS = {"new": "cuteafd_ml4_dspark_markov", "old": "cuteafd_ml4_dspark_markov_oracle"}
RANK, ROWS, BLOCKS = 512, 3 * 512, 296
SEQUENCES = (1, 4, 7, 16, 32)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))


def bind(path):
    lib = C.CDLL(path)
    lib.cuteafd_ml4_dspark_markov_workspace.argtypes = [I, I]
    lib.cuteafd_ml4_dspark_markov_workspace.restype = C.c_uint64
    lib.cuteafd_ml4_dspark_markov_norms.argtypes = [P, P, I, I, P]
    lib.cuteafd_ml4_dspark_markov_norms.restype = I
    for name in ARMS.values():
        getattr(lib, name).argtypes = [P] * 10 + [I] * 5 + [P]
        getattr(lib, name).restype = I
    return lib


def regions(workspace, sequences, block):
    """(state, embedding, joint) byte views of the documented workspace layout."""
    values = sequences * BLOCKS * 8
    top = sequences * block * 4
    state = values + top
    embed = state + sequences * RANK * 2
    joint = embed + sequences * RANK * 2
    joint_bytes = 4 * sequences * ROWS
    return (workspace[state:embed], workspace[embed:joint], workspace[joint:joint + joint_bytes])


def real_inputs(snapshot, sequences, block, hidden_width, vocab, seed):
    """hidden [sequences * block, hidden] BF16, base [sequences * block, vocab]
    FP32 and anchors [sequences] from the checkpoint's own drafter weights."""
    from families.mistral_large4 import dspark, ops

    get = dspark.checkpoint_getter(Path(snapshot), "cuda")
    drafter = dspark.Drafter(get)
    weights = {name: get(name) for name in ("autoregressive_head.embedding.weight",
                                            "autoregressive_head.joint_proj.weight",
                                            "autoregressive_head.joint_proj.bias",
                                            "autoregressive_head.output.weight")}
    generator = torch.Generator().manual_seed(seed)
    start = 16
    rows = sequences * block
    taps = (torch.randn(start + rows + block, 5 * hidden_width, generator=generator) * 0.5).bfloat16().cuda()
    anchors = torch.randint(vocab, (sequences,), generator=generator, dtype=torch.int32).cuda()
    positions = torch.arange(taps.shape[0], device="cuda")
    hidden = []
    with torch.no_grad():
        context = drafter.context(taps, positions)
        for s in range(sequences):
            position = start + s * block
            hidden.append(drafter.body(int(anchors[s]), position, drafter.window_of(context, position)))
        hidden = torch.cat(hidden).to(torch.bfloat16).contiguous()
        base = ops.softcap(torch.nn.functional.linear(hidden, get("head")).float(), drafter.softcap)
    return weights, hidden, base.contiguous(), anchors


def synthetic_inputs(sequences, block, hidden_width, vocab, seed):
    generator = torch.Generator().manual_seed(seed)
    weights = {"autoregressive_head.embedding.weight": (torch.randn(vocab, RANK, generator=generator) * 0.05)
               .bfloat16().cuda(),
               "autoregressive_head.output.weight": (torch.randn(vocab, RANK, generator=generator) * 0.05)
               .bfloat16().cuda(),
               "autoregressive_head.joint_proj.weight":
                   (torch.randn(ROWS, 2 * RANK + hidden_width, generator=generator) * 0.02).bfloat16().cuda(),
               "autoregressive_head.joint_proj.bias": (torch.randn(ROWS, generator=generator) * 0.1)
               .bfloat16().cuda()}
    hidden = (torch.randn(sequences * block, hidden_width, generator=generator) * 0.5).bfloat16().cuda()
    # Soft-capped base logits with a few strong candidates per row (near-ties included).
    base = torch.tanh(torch.randn(sequences * block, vocab, generator=generator) * 0.3) * 30
    for row in range(sequences * block):
        hot = torch.randint(vocab, (4,), generator=generator)
        base[row, hot] = 29.0 + torch.rand(4, generator=generator) * 0.05
    anchors = torch.randint(vocab, (sequences,), generator=generator, dtype=torch.int32).cuda()
    return weights, hidden.cuda(), base.cuda(), anchors


def compare(lib, label, sequences, weights, hidden, base, anchors, vocab, hidden_width, block, workspace_bytes,
            stream, norms):
    drafts, workspaces = {}, {}
    for arm in ("old", "new"):
        drafts[arm] = torch.zeros(sequences, block, dtype=torch.int32, device="cuda")
        workspaces[arm] = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        rc = getattr(lib, ARMS[arm])(
            base.data_ptr(), hidden.data_ptr(), weights["autoregressive_head.embedding.weight"].data_ptr(),
            weights["autoregressive_head.joint_proj.weight"].data_ptr(),
            weights["autoregressive_head.joint_proj.bias"].data_ptr(),
            weights["autoregressive_head.output.weight"].data_ptr(), norms.data_ptr(), anchors.data_ptr(),
            drafts[arm].data_ptr(), workspaces[arm].data_ptr(), sequences, block, vocab, RANK, hidden_width, stream)
        assert rc == 0, (arm, rc)
    torch.cuda.synchronize()
    old, new = workspaces["old"], workspaces["new"]
    (old_state, old_embed, old_joint), (new_state, new_embed, new_joint) = (regions(w, sequences, block)
                                                                           for w in (old, new))
    checks = [("drafts", drafts["old"].view(torch.uint8), drafts["new"].view(torch.uint8)),
              ("state", old_state, new_state),
              ("embedding", old_embed, new_embed),
              # The projection is stored widened from BF16; compare its BF16 bits.
              ("joint (bf16)", old_joint.view(torch.float32).bfloat16().view(torch.uint8),
               new_joint.view(torch.float32).bfloat16().view(torch.uint8))]
    bad = [name for name, before, after in checks if not torch.equal(before, after)]
    print(f"{label} sequences {sequences}: " + ("byte-identical (drafts, state, embedding, joint)"
                                                if not bad else f"DIFFERS in {', '.join(bad)}"))
    return len(bad)


def run_case(lib, label, sequences, weights, hidden, base, anchors, vocab, hidden_width, block, workspace_bytes,
             stream):
    norms = torch.empty(vocab, dtype=torch.float32, device="cuda")
    assert lib.cuteafd_ml4_dspark_markov_norms(
        weights["autoregressive_head.output.weight"].data_ptr(), norms.data_ptr(), vocab, RANK, stream) == 0
    return compare(lib, label, sequences, weights, hidden, base, anchors, vocab, hidden_width, block,
                   workspace_bytes, stream, norms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("library")
    p.add_argument("--vocab", type=int, default=204800)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot", type=Path, help="official checkpoint snapshot for the real-input arm")
    p.add_argument("--bench", action="store_true", help="time one run of each arm per call")
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    lib = bind(args.library)
    vocab, hidden_width, block = args.vocab, args.hidden, args.block
    stream = torch.cuda.current_stream().cuda_stream
    workspace_bytes = lib.cuteafd_ml4_dspark_markov_workspace(32, block)
    failures = 0
    for sequences in SEQUENCES:
        weights, hidden, base, anchors = synthetic_inputs(sequences, block, hidden_width, vocab, args.seed)
        failures += run_case(lib, "synthetic", sequences, weights, hidden, base, anchors, vocab, hidden_width, block,
                             workspace_bytes, stream)
        if args.snapshot:
            weights, hidden, base, anchors = real_inputs(args.snapshot, sequences, block, hidden_width, vocab,
                                                         args.seed)
            failures += run_case(lib, "checkpoint", sequences, weights, hidden, base, anchors, vocab, hidden_width,
                                 block, workspace_bytes, stream)
    assert failures == 0, f"{failures} region(s) differ between the split and single-CTA joint kernels"
    if args.bench:
        bench(lib, vocab, hidden_width, block, workspace_bytes, stream, args.iterations, args.seed)
    print("ml4 dspark markov oracle: pass")


def bench(lib, vocab, hidden_width, block, workspace_bytes, stream, iterations, seed):
    """One warm run per arm: milliseconds per whole `cuteafd_ml4_dspark_markov` call."""
    for sequences in SEQUENCES:
        weights, hidden, base, anchors = synthetic_inputs(sequences, block, hidden_width, vocab, seed)
        norms = torch.empty(vocab, dtype=torch.float32, device="cuda")
        assert lib.cuteafd_ml4_dspark_markov_norms(
            weights["autoregressive_head.output.weight"].data_ptr(), norms.data_ptr(), vocab, RANK, stream) == 0
        drafts = torch.zeros(sequences, block, dtype=torch.int32, device="cuda")
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        for arm in ("old", "new"):
            fn = getattr(lib, ARMS[arm])
            args = (base.data_ptr(), hidden.data_ptr(), weights["autoregressive_head.embedding.weight"].data_ptr(),
                    weights["autoregressive_head.joint_proj.weight"].data_ptr(),
                    weights["autoregressive_head.joint_proj.bias"].data_ptr(),
                    weights["autoregressive_head.output.weight"].data_ptr(), norms.data_ptr(), anchors.data_ptr(),
                    drafts.data_ptr(), workspace.data_ptr(), sequences, block, vocab, RANK, hidden_width, stream)
            for _ in range(3):
                assert fn(*args) == 0
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(iterations):
                start.record()
                assert fn(*args) == 0
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            print(f"sequences {sequences} {arm}: {statistics.median(times):.3f} ms/call "
                  f"(min {min(times):.3f}, max {max(times):.3f})")


if __name__ == "__main__":
    main()
