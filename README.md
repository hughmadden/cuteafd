<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/cuteafd-logo-color-dark.svg">
    <img src="assets/brand/cuteafd-logo-color.svg" alt="cuteafd" width="480">
  </picture>
</p>

CuteAFD brings frontier-scale open-weights models into the home lab at
data-center speed. It disaggregates attention from the routed experts: one or
two consumer-Blackwell RTX PRO 6000 cards run attention, the dense
backbone, routing and sampling, while a pool of DGX Sparks (GB10, SM121) holds
the routed experts and answers over RoCE. The same engine loads a model's
standard Hugging Face checkpoint directly — no side files, no repacking — and
runs it on the formats it actually ships in: official FP8 and MXFP4,
NVIDIA ModelOpt NVFP4, and EXL3, with native kernels that honor each
checkpoint's own numerics instead of converting everything to one internal
format.

- Attention/FFN disaggregation (AFD): RTX cards own the backbone, Sparks own
  the experts, exchanging activations over RoCE with GPU-direct landing.
- Robust quant support: official FP8/MXFP4, NVIDIA ModelOpt NVFP4, and EXL3,
  loaded from the checkpoint's own `config.json` and tensor headers.
- Exact prefix caching for agentic work: the deepest cached snapshot that
  prefixes a request is restored byte-identical, not approximated.
- Own your intelligence: your weights, your hardware, your rate limits (none),
  agentic coding at full speed on a machine you control, not a shared tenant.

## Models

Basic benchmark profile per family on its natural-minimum (1× RTX + fewest
Sparks) and maximum (2× RTX + 4 or 6 Sparks) hardware. Other reports:
[`benchmarks/`](benchmarks/README.md).

<!-- results:begin -->
_Pending the first Release smoke run across the full family × quant × hardware matrix._
<!-- results:end -->

Each family's page ([`docs/models/`](docs/models/)) has its supported
checkpoints and quants, engineering summary and known limits:
[DeepSeek V4.1 Flash](docs/models/deepseek_v41.md),
[DeepSeek V4 Flash/Pro](docs/models/deepseek_v4.md),
[GLM 5.3](docs/models/glm5.md), [GLM 5.3 Flash](docs/models/glm5_flash.md),
[MiMo V2 Flash / V2.6 Pro](docs/models/mimo_v2.md),
[Qwen 3.8 Flash Next](docs/models/qwen4.md).

## Storage

The checkpoints and quants behind these numbers were served from
[SparkNest](https://github.com/tpurtell/sparknest), a distributed model store
across the cluster: hosts that already hold a sealed local copy of a shard
read it at NVMe speed, hosts without one stream it over RoCE at roughly
5 GB/s. SparkNest is what made the quoted load times possible; it is a
separate project, not required to run CuteAFD, and any standard Hugging Face
cache layout works.

## Quick start

1. `cuteafd plan MODEL` (any Hugging Face model id or local snapshot
   directory) reports what the checkpoint needs — tensors, formats, shapes,
   and which kernels are missing — before you touch a GPU.
2. Pick or adapt a config under [`examples/configs/`](examples/configs/) or
   edit `cuteafd.config` for your own topology (coordinator GPUs, Spark
   ranks, TP/EP layout).
3. `./run.sh` launches the release images named in the config,
   `ghcr.io/tpurtell/cuteafd-coordinator:v0.1.0` on the RTX host and
   `ghcr.io/tpurtell/cuteafd-spark-expert:v0.1.0` on each Spark; `docker pull`
   them on those hosts first (`./run.sh` does not pull). `./wip.sh --slot S --role both`
   plus `./run.sh --wip S --restart` is the faster loop while iterating.

## Working on it

[`AGENTS.md`](AGENTS.md) is the standing guide for agents and collaborators
working on CuteAFD. [`PLAN.md`](PLAN.md) is the roadmap.
