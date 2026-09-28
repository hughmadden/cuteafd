# Working on cuteafd

Read `PLAN.md` first. This file is the standing rules. Code is king: keep
external documentation to this file, `PLAN.md`, and one README; measurements
go in commit messages as short before → after tables with conditions.

## Hosts

- `raptor`: coordinator, x86-64, 2× RTX PRO 6000 Blackwell 96 GB (SM120),
  power-capped at 325 W while TJ is away. Fabric 10.55.0.22 / 10.55.1.22.
- Sparks (GB10, SM121, ARM64, 121 GiB unified; `nvidia-smi` memory reads
  N/A, measure with CUDA): ostrich, dodo, emu, kiwi, rhea, moa at
  10.55.0.1–6. Ranks 0–5 in that order. All six are the pool. TP4 on the
  first four is the qualified V4.1 default until a six-rank layout wins.
- Passwordless SSH by hostname. Fan-out: `scripts/run-on-hosts.sh`.
- Root: `agent-sudo --agent-context "why" CMD` (remote human approval).

## Storage

- Every host mounts sparknest at `/mnt/sparknest`; `HF_HOME` is
  `/mnt/sparknest/hf-home`. Sealed local copies read at NVMe speed; files
  without a local copy stream over RoCE at ~5 GB/s. Reads never replicate.
- `nest where hf:ORG/MODEL` shows copies; `nest replicate SEL --hosts
  @sparks --wait` places them; `nest evict` removes (never the last copy);
  `nest plan --free` when space is tight. Replicate a model to every rank
  while working on it, then shrink to one copy or 1/N. Manage space.
- `/mnt/scratch` and `/mnt/models` are slow archive stores (150 MB/s
  write, 500 MB/s read). Never build on `/mnt/scratch` (NTFS kernel bug).

## Build and run

- Build only under `~/.cache/cuteafd/builds/<task>` on root NVMe; run
  `scripts/assert-build-filesystem.py` on every path first. Never reuse a
  Cargo cache that has seen filesystem errors.
- `./wip.sh --slot S --role both` for iteration, `scripts/run-wip.sh
  --wip-slot S --restart` to launch; `./build.sh` and `./run.sh` for
  release images. Slots isolate artifacts, not GPUs or ports: serialize
  builds and performance runs, one model served at a time.
- Submodules are pinned with tree locks (`third_party/*.lock.json`).
  Kernel changes go to `../sparkinfer-glmrt` master, quantizer changes to
  `../GPTQModel` main; push there first, then bump pin and lock here.
- Kernels: CuTe-DSL/Triton AOT exports from the b12x fork are the default;
  hand CUDA only where measured to pay. SM120 and SM121 are both targets.
- Run CUDA/PyTorch checks inside the matching architecture's container.

## Engineering rules

- Never slower than the engine being replaced: V4.1 Flash parity (C1 code
  decode on 1 and 2 RTX, 8K prefill, tool eval) is checked at every phase.
- Correctness first, then warm-up, then identical-config A/B, interleaved,
  three runs for a final number. Judge speculation by emitted tok/s, not
  acceptance. Profiling perturbs timing.
- Unsupported is a result, not a crash: `cuteafd plan` names the tensors,
  formats, shapes and the exporter or kernel to add.
- Load speed is a feature; do not regress readiness time.
- Preserve graph pointer/shape/workspace lifetimes; drain queued work before
  publishing or releasing storage; weight and workspace admission precede
  allocation; zero steady-state graph captures per request.
- Rust: typed errors inside crates, `anyhow` at edges, `tracing`, no
  `unsafe` outside FFI and verbs layers, each block with a safety comment.
- Keep weights, build artifacts, benchmark runs and local config out of Git.

## Git

- Work on `main`. Small commits, push after every green step. Tag phase
  boundaries `p0`, `p1`, … Imperative subjects; performance commits carry
  the measurement table. No attribution trailers.
