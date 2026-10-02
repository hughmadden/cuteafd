# Working on cuteafd

Read `PLAN.md` first. This file is the standing rules. Code is king: keep
external documentation to this file, `PLAN.md`, one README and the
`benchmarks/` index; measurements go in commit messages as short before →
after tables with conditions.

## Hosts

- `raptor`: coordinator, x86-64, 2× RTX PRO 6000 Blackwell 96 GB (SM120),
  power-capped at 325 W while TJ is away. Fabric 10.55.0.22 / 10.55.1.22.
- Sparks (GB10, SM121, ARM64, 121 GiB unified; `nvidia-smi` memory reads
  N/A, measure with CUDA): ostrich, dodo, emu, kiwi, rhea, moa at
  10.55.0.1–6. Ranks 0–5 in that order. All six are the pool. TP4 on the
  first four is the qualified V4.1 default until a six-rank layout wins.
- Fabric: Sparks have two RoCE ports at 200 Gb/s each on separate
  subnets (rail A 10.55.0.x, rail B 10.55.1.x); raptor has one 400 Gb/s
  port carrying both rail subnets. The switch is being raised from 100G to
  200G. Read rates and states at startup (`rdma link`, sysfs `rate`); do
  not assume them. Dual rail at 100G caused head-of-line blocking against
  200+ Gb/s PCIe ingress, so rail use is a measured decision.
- Passwordless SSH by hostname. Fan-out: `scripts/launch/run-on-hosts.sh`.
- Root: `agent-sudo --agent-context "why" CMD` (remote human approval).

## Storage

- Every host mounts sparknest at `/mnt/sparknest`; `HF_HOME` is
  `/mnt/sparknest/hf-home`. Sealed local copies read at NVMe speed; files
  without a local copy stream over RoCE at ~5 GB/s. Reads never replicate.
- `nest where hf:ORG/MODEL` shows copies; `nest replicate SEL --hosts
  @sparks --wait` places them; `nest evict` removes (never the last copy);
  `nest plan --free` when space is tight. Replicate a model to every rank
  while working on it, then shrink to one copy or 1/N. Manage space.
- Each host's `~/.cache/huggingface/hub` is a symlink into `/mnt/sparknest`;
  containers must mount the resolved hub (run.sh does) or `/mnt/sparknest`.
- `/mnt/scratch` and `/mnt/models` are slow archive stores (150 MB/s
  write, 500 MB/s read). Never build on `/mnt/scratch` (NTFS kernel bug).

## Build and run

- Build only under `~/.cache/cuteafd/builds/<task>` on root NVMe; run
  `scripts/build/assert-build-filesystem.py` on every path first. Never reuse a
  Cargo cache that has seen filesystem errors.
- `./wip.sh --slot S --role both` for iteration, `./run.sh --wip S
  --restart` to launch; `./build.sh` and `./run.sh` for
  release images. Slots isolate artifacts, not GPUs or ports: serialize
  builds and performance runs, one model served at a time.
- Submodules are pinned with tree locks (`third_party/*.lock.json`).
  Kernel changes go to `../sparkinfer-glmrt` master, quantizer changes to
  `../GPTQModel` main; push there first, then bump pin and lock here.
  A SparkInfer bump also needs `scripts/build/build-dev-images.sh` (shared
  dev images on raptor and every Spark, then `./wip.sh --recreate`);
  `wip.sh` and `run.sh --wip` refuse a stale dev image and say so.
- Kernels: CuTe-DSL/Triton AOT exports from the b12x fork are the default;
  hand CUDA only where measured to pay. SM120 and SM121 are both targets.
- Run CUDA/PyTorch checks inside the matching architecture's container.
- Host checks: `cargo check/test --workspace` from `rust/` with
  `CARGO_TARGET_DIR=~/.cache/cuteafd/builds/<task>/target` (no Python
  needed). Script tests: `.venv/bin/python -m pytest -q scripts/tests`
  (`uv venv --python 3.12 .venv` + pytest numpy tokenizers jsonschema pyyaml);
  43 inherited failures remain (work/p0 96edf07), add none.
- `./build.sh` (release pair, ~15 min coordinator + Spark leg): set
  `CUTEAFD_RELEASE_BUILD_ROOT` and `CUTEAFD_RELEASE_REMOTE_BUILD_DIR` under
  `~/.cache/cuteafd/builds/`, and `CUTEAFD_RELEASE_SPARK_TP_ROLES=` for a
  TP4-only pair. It reads the live checkout while assembling images: edit in
  a git worktree until it finishes. Crates download from crates.io each build
  and can crawl while the WAN is busy; it is slow, not stuck.
- Iterate with `./wip.sh --slot S` then `./run.sh --wip S --restart`; A/B two
  checkouts with `scripts/bench/bench-ab.py`. `cuteafd plan MODEL` (any HF id or
  snapshot dir) says what a checkpoint needs before any kernel work;
  `cuteafd fabric` shows ports, link/PCIe rates, subnets and the rail plan
  (services log the same line at startup).

## Engineering rules

- Never slower than the engine being replaced: V4.1 Flash parity (C1 code
  decode on 1 and 2 RTX, 8K prefill, tool eval) is checked at every phase.
- Correctness first, then warm-up, then identical-config A/B, interleaved,
  three runs for a final number. Judge speculation by emitted tok/s, not
  acceptance. Profiling perturbs timing.
- Benchmark each model on two reference configs only: the natural minimum
  (1× RTX + the fewest Sparks it fits) and the maximum (2× RTX + 4 or 6
  Sparks, whichever divides the model sensibly). Other layouts need
  correctness gates, not perf tables; the planner's estimates cover them.
- Tiered gates. Merges and features: cargo/script tests (failing ids, not
  counts), golden NLL/byte-exactness on one GPU or loopback, and the
  feature's own measurement. Changes to shared hot paths (transport, expert
  exchange, native lib, sampler) add a quick V4.1 parity: one launch of the
  candidate (WIP images) vs a baseline measured the same day, C1 + C16 code
  decode only (~10 min); escalate to 3 interleaved sessions per arm only if a
  metric is below 0.98 after warm-up (benches run one untimed batch per
  concurrency level; DeepSeek engram tables and first-use workspaces make the
  first wide batch after a launch ~10% slow — explain, don't re-run). Full V4.1 parity (3 sessions per arm, all metrics)
  runs at release cuts only. Release images are built for release cuts, not
  to verify branches; agentic benches gate with 1–2 short sessions, the full
  bench runs at release.
- Published results. The root README holds the only exhaustive table: the
  basic benchmark profile for every family on its natural-minimum and
  maximum hardware. Re-run a family's rows after changes that target that
  family's code (or a shared hot path that plausibly moves it); skip
  irrelevant changes, staleness is fine. Other profiles run only when TJ
  asks: their exports (`report.svg` + `report.json` from `cuteafd bench`,
  the same runner as the dashboard) go to
  `benchmarks/<family>/<date>-<profile>-<hardware>/` and get a line in
  `benchmarks/README.md` (per family, newest first: date, profile, hardware,
  build). Commit them straight on top of `main` or the working branch; no
  release or branch needed.
- Unsupported is a result, not a crash: `cuteafd plan` names the tensors,
  formats, shapes and the exporter or kernel to add.
- Load speed is a feature; do not regress readiness time.
- Preserve graph pointer/shape/workspace lifetimes; drain queued work before
  publishing or releasing storage; weight and workspace admission precede
  allocation; zero steady-state graph captures per request.
- Rust: typed errors inside crates, `anyhow` at edges, `tracing`, no
  `unsafe` outside FFI and verbs layers, each block with a safety comment.
- Keep weights, build artifacts, benchmark runs and local config out of Git
  (published reports under `benchmarks/` are the exception).

## Git

- Work on `main`. Small commits, push after every green step. Tag phase
  boundaries `p0`, `p1`, … Imperative subjects; performance commits carry
  the measurement table. No attribution trailers.
