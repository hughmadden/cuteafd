#!/usr/bin/env python3
"""DeepSeek V4 Flash prefill built from the b12x (SparkInfer fork) kernels.

This is the executable specification the Rust engine follows: one decoder layer
(and a layer-by-layer driver) for ONE sequence prefilled from position 0,
T tokens, every op a b12x entry point unless marked ``# TORCH FALLBACK:``.
The numerical truth is the official ``inference/model.py`` (V4-Flash-0731);
``golden.py`` records it and ``compare.py`` diffs this module against it.

Per layer (``DeepseekV4Layer.forward``), stream = [T, hc=4, dim] BF16:

  mHC pre (attn)       norm.mhc.run_pre           Sinkhorn mix, collapse, attn_norm fused
  q / kv / KV page     attention.dsv4_producer     wq_a|wkv GEMM, q_norm, kv_norm, RoPE,
                                                   584-B FP8 page record, wq_b, per-head
                                                   RMS + RoPE on q
  compressor (C4/C128) attention.dsv4_compressor   gated pooling (+overlap/ape), norm,
                                                   RoPE, FP8 compressed page (+C4 index K)
  index query (C4)     dsv4_producer.run_indexer   index wq_b, RoPE, Hadamard, FP4 QAT
  index top-k (C4)     attention.dsa_indexer       relu-weighted score, causal top-512,
                                                   physical slots
  attention            attention.compressed_sparse_mla  window(128) + compressed, sink
  inv RoPE + wo_a/wo_b gemm.wo_projection (inv_rope)
  mHC post + pre (ffn) norm.mhc.run_post_pre       ffn_norm fused
  router               torch gate GEMM + moe.fused_moe.route_topk (hash: torch)
  routed experts       moe.fused_moe (w4a8 MXFP4)
  shared expert        gemm.block_fp8_linear x2 (+ torch SwiGLU)
  mHC post             norm.mhc.run_post
Model: torch embedding gather, lanes = repeat(4); head = norm.mhc.run_head
(sigmoid hc_head collapse + final RMSNorm fused) then an FP32 torch GEMM.

Caches written by one layer's prefill (fp8 cache_format, page = 256 source
tokens, identity page table here: page p holds tokens [256p, 256p+256)):
  window/main KV  [ceil(T/256), 149760] u8: 256 x (448 FP8 + 64 BF16 RoPE) payload,
                  then 256 x 8 scale bytes (7 UE8M0 per 64-dim group + pad), pad.
  C4 compressed   [ceil(T/256), 37440] u8: same record, 64 rows/page (1 row = 4 tokens).
  C4 index K      [ceil(T/256), 8448] u8: 64 x 128 FP8 then 64 x FP32 per-row scale.
  C128 compressed [ceil(T/256), 1728] u8: same record, 2 rows/page.
  compressor state (for continuation/decode, fp32): C4 kv/score [S,16,1024] and
  index [S,16,256]; C128 [S,256,512].
``describe_prefill_buffers(T)`` prints every buffer and scratch size.

Run on a GPU inside the SM120/SM121 container; see compare.py for the driver.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = Path(
    "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
)

# Fixed DSV4 geometry baked into the b12x ops (they validate these).
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
SOURCE_PAGE_TOKENS = 256          # window/main page, and source tokens per compressed page
MAIN_PAGE_BYTES = 149_760         # 256 * 584 rounded up to a 576-byte multiple
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_PAGE_ROWS = 64
INDEX_PAGE_BYTES = 8_448          # 64 * 128 FP8 + 64 * FP32 scale
WINDOW = 128                      # also the SWA width fed to sparse MLA
PREFILL_SELECTION_TILE = 64       # SM120 MG prefill consumes indexed slots in 64-wide tiles


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def compressed_page_rows(ratio: int) -> int:
    return SOURCE_PAGE_TOKENS // ratio


def compressed_page_bytes(ratio: int) -> int:
    """fp8 compressed-MLA page: rows x 584 B rounded up to a 576-B multiple."""
    return _align(compressed_page_rows(ratio) * 584, 576)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    """inference/config.json plus the ModelArgs defaults it relies on."""

    vocab_size: int = 129_280
    dim: int = 4_096
    moe_inter_dim: int = 2_048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_heads: int = 64
    n_routed_experts: int = 256
    n_activated_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1_024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1_024
    window_size: int = 128
    original_seq_len: int = 65_536
    rope_theta: float = 10_000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6  # ModelArgs default; not in config.json
    compress_rope_theta: float = 160_000.0
    compress_ratios: tuple[int, ...] = ()
    score_func: str = "sqrtsoftplus"

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "Config":
        raw = json.loads((Path(snapshot) / "inference" / "config.json").read_text())
        names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in raw.items() if k in names}
        values["compress_ratios"] = tuple(raw["compress_ratios"])
        cfg = cls(**values)
        assert cfg.score_func == "sqrtsoftplus" and cfg.head_dim == HEAD_DIM
        return cfg

    def ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer])

    def is_hash_layer(self, layer: int) -> bool:
        return layer < self.n_hash_layers


# --------------------------------------------------------------------------- checkpoint


class Checkpoint:
    """Per-tensor safetensors access (never reads a whole shard)."""

    def __init__(self, snapshot: Path):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.snapshot = Path(snapshot)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self._files: dict[str, Any] = {}

    def _file(self, name: str):
        shard = self.index[name]
        if shard not in self._files:
            self._files[shard] = self._safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self._files[shard]

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str, device: torch.device | None = None) -> torch.Tensor:
        tensor = self._file(name).get_tensor(name)
        return tensor if device is None else tensor.to(device, non_blocking=False)

    def rows(self, name: str, row_ids: list[int], device: torch.device) -> torch.Tensor:
        """Gather rows of a 2-D tensor through slices (embedding, tid2eid)."""
        sl = self._file(name).get_slice(name)
        return torch.cat([sl[r : r + 1] for r in row_ids], dim=0).to(device)


# --------------------------------------------------------------------------- b12x plumbing


_COMPAT_INSTALLED = False


def install_b12x_compat() -> None:
    """COMPAT SHIM (not a torch fallback): make the pinned dsv4_producer importable.

    At sparkinfer 7fcc094e, ``b12x/attention/dsv4_producer/_impl.py`` still
    imports ``plan_block_fp8_linear_scratch`` / ``BlockFP8LinearScratchPlan`` from
    ``b12x.gemm._shared.block_fp8``; commit 77351c13 (prepared-plan migration)
    removed both, so ``import b12x.attention.dsv4_producer._impl`` raises
    ImportError. The producer only needs "a scratch plan with scratch_specs() and
    bind(scratch, source, packed_weight, output, expected_m, activation_block_size)
    whose binding block_fp8_linear_mxfp8 accepts" -- exactly a prepared
    ``gemm.block_fp8_linear`` plan declared with activation_block_size=128 (the
    checkpoint's per-128 activation quantization). Fix upstream by porting the
    producer to ``block_fp8_linear.plan``; delete this shim then.
    """
    global _COMPAT_INSTALLED
    if _COMPAT_INSTALLED:
        return
    import b12x.gemm._shared.block_fp8 as block_fp8

    if not hasattr(block_fp8, "plan_block_fp8_linear_scratch"):
        from b12x.gemm import block_fp8_linear

        class PreparedBlockFP8LinearScratchPlan:
            def __init__(self, caps):
                self.caps = dataclasses.replace(caps, activation_block_size=128)
                self.plan = prepare(block_fp8_linear.plan(self.caps), "compat.block_fp8_linear")

            def scratch_specs(self):
                return self.plan.scratch_specs()

            def bind(self, *, scratch, source, packed_weight, output, expected_m=None,
                     activation_block_size=128, bias=None):
                return block_fp8_linear.bind(
                    self.plan, scratch=scratch, source=source, packed_weight=packed_weight,
                    output=output, bias=bias, expected_m=expected_m,
                    activation_block_size=activation_block_size,
                )

        block_fp8.BlockFP8LinearScratchPlan = PreparedBlockFP8LinearScratchPlan
        block_fp8.plan_block_fp8_linear_scratch = PreparedBlockFP8LinearScratchPlan
    _COMPAT_INSTALLED = True


def prepare(plan, name: str):
    """Materialize a declarative b12x Plan with its default configuration.

    Serving prepares every plan in one PreparationSession before graph capture
    (autotuning optional); ``prepare_default`` is the same lifecycle for one
    request with a no-op priming call and no timing race.
    """
    from b12x.preparation import PreparedCall, prepare_default

    if getattr(plan, "prepared", None) is None:
        prepare_default(plan.request(name=name, prepare_call=lambda state: PreparedCall(run=lambda: None)))
    return plan


class ScratchLedger:
    """Allocates caller-owned scratch from ScratchBufferSpecs and records sizes."""

    def __init__(self):
        self.entries: dict[str, int] = {}
        self.buffers: dict[str, int] = {}

    def scratch(self, name: str, specs):
        specs = tuple(specs)
        tensors = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in specs)
        self.entries[name] = sum(int(s.nbytes) for s in specs)
        return tensors[0] if len(tensors) == 1 else tensors

    def buffer(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        self.buffers[name] = tensor.numel() * tensor.element_size()
        return tensor

    def report(self) -> str:
        lines = ["scratch (caller-owned, per op):"]
        lines += [f"  {k:40s} {v:>14,d} B" for k, v in self.entries.items()]
        lines.append("buffers (outputs / caches):")
        lines += [f"  {k:40s} {v:>14,d} B" for k, v in self.buffers.items()]
        return "\n".join(lines)


# --------------------------------------------------------------------------- RoPE


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
                         device=None) -> torch.Tensor:
    """Copy of model.py precompute_freqs_cis (YaRN when original_seq_len > 0)."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, n):
        if lo == hi:
            hi += 0.001
        return torch.clamp((torch.arange(n, dtype=torch.float32, device=device) - lo) / (hi - lo), 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_cos_sin(cfg: Config, ratio: int, positions: int, device) -> torch.Tensor:
    """b12x RoPE table: FP32 [positions, 64] = [cos(32) | sin(32)], interleaved pairs.

    Window-only layers (ratio 0) use theta 1e4 without YaRN; compressed layers use
    compress_rope_theta with YaRN (model.py Attention.__init__). The compressor
    and C4 indexer of a layer share its table.
    """
    if ratio:
        orig, theta = cfg.original_seq_len, cfg.compress_rope_theta
    else:
        orig, theta = 0, cfg.rope_theta
    fc = precompute_freqs_cis(cfg.rope_head_dim, positions, orig, theta, cfg.rope_factor,
                              cfg.beta_fast, cfg.beta_slow, device=device)
    return torch.cat((fc.real, fc.imag), dim=-1).float().contiguous()


def apply_rope_torch(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor,
                     inverse: bool = False) -> torch.Tensor:
    """model.py apply_rotary_emb on the trailing 64 dims (fp32 math, input dtype out)."""
    half = cos_sin.shape[-1] // 2
    cos, sin = cos_sin[positions, :half], cos_sin[positions, half:]
    shape = [x.shape[0]] + [1] * (x.ndim - 2) + [half]
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    pairs = x[..., -2 * half:].float().unflatten(-1, (half, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    out = x.clone()
    out[..., -2 * half:] = rotated.to(x.dtype)
    return out


# --------------------------------------------------------------------------- prefill metadata


@dataclass
class PrefillMetadata:
    """Index/slot metadata for one sequence prefilled from position 0.

    All slots are physical cache slots; with the identity page table used here
    physical == logical. An engine maps logical positions through its page table.
    """

    tokens: int
    positions: torch.Tensor          # int64 [T]
    positions_i32: torch.Tensor      # int32 [T]
    main_pages: int
    main_slots: torch.Tensor         # int64 [T]  token t -> window/main page slot
    swa_indices: torch.Tensor        # int32 [T, 128]  window slots, -1 padded
    swa_lengths: torch.Tensor        # int32 [T]  min(t+1, 128)
    compress: dict[int, dict[str, Any]] = field(default_factory=dict)


def build_prefill_metadata(T: int, device, ratios=(4, 128), index_topk: int = 512) -> PrefillMetadata:
    i64 = dict(dtype=torch.int64, device=device)
    i32 = dict(dtype=torch.int32, device=device)
    pos = torch.arange(T, **i64)
    # model.py get_window_topk_idxs(start_pos=0): row t sees [max(0, t-127) .. t].
    start = (pos - (WINDOW - 1)).clamp(min=0)[:, None]
    idx = start + torch.arange(WINDOW, **i64)[None, :]
    swa = torch.where(idx <= pos[:, None], idx, torch.full_like(idx, -1)).to(torch.int32).contiguous()
    swa_len = torch.minimum(pos + 1, torch.full_like(pos, WINDOW)).to(torch.int32)
    meta = PrefillMetadata(
        tokens=T, positions=pos, positions_i32=pos.to(torch.int32), main_pages=math.ceil(T / SOURCE_PAGE_TOKENS),
        main_slots=pos.clone(), swa_indices=swa, swa_lengths=swa_len,
    )
    for ratio in ratios:
        n = T // ratio
        rows = compressed_page_rows(ratio)
        pages = max(1, math.ceil(n / rows))
        visible = ((pos + 1) // ratio).to(torch.int32)  # completed groups visible to row t
        g = torch.arange(n, **i32) * ratio
        entry = dict(
            groups=n, pages=pages,
            # dsv4_compressor prefill metadata (one sequence, rows of hidden_states)
            active_groups=torch.tensor([n], **i32),
            group_source_starts=g.clone(),                   # first source row of group j
            group_rope_positions=g.clone(),                  # RoPE position j*ratio (model.py freqs_cis[:cutoff:ratio])
            compressed_slots=torch.arange(n, **i32),         # group j -> compressed slot j
            active_sequences=torch.tensor([1], **i32),
            sequence_offsets=torch.tensor([0, T], **i32),
            state_sequence_ids=torch.tensor([0], **i32),
            visible=visible,
        )
        if ratio == 4:
            # dsa_indexer: shared page table (all rows see the same pages), causal
            # per-row lengths, physical-slot output; attention length = min(topk, visible).
            entry["index_page_table"] = torch.arange(pages, **i32)[None, :].expand(T, pages)
            entry["index_cache_lengths"] = visible.contiguous()
            entry["index_active_width"] = torch.tensor([pages * INDEX_PAGE_ROWS], **i32)
            entry["indexed_lengths"] = torch.clamp(visible, max=index_topk).contiguous()
            entry["indexed_width"] = index_topk
        else:
            # model.py get_compress_topk_idxs(start_pos=0): dense causal list of completed groups.
            width = max(PREFILL_SELECTION_TILE, _align(max(n, 1), PREFILL_SELECTION_TILE))
            cols = torch.arange(width, **i32)[None, :]
            entry["indexed_indices"] = torch.where(cols < visible[:, None], cols, torch.full_like(cols, -1)).contiguous()
            entry["indexed_lengths"] = visible.contiguous()
            entry["indexed_width"] = width
        meta.compress[ratio] = entry
    return meta


# --------------------------------------------------------------------------- weights


def _u8(t: torch.Tensor) -> torch.Tensor:
    """E8M0 scale / packed-FP4 I8 bytes as uint8 (the b12x byte ABI)."""
    return t.contiguous().view(torch.uint8)


@dataclass
class LayerWeights:
    layer: int
    ratio: int
    hash_routing: bool
    # mHC
    hc_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # fn [24,4d] f32, scale [3], base [24]
    hc_ffn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    attn_norm: torch.Tensor                                    # bf16 [d] (fused into mhc pre)
    ffn_norm: torch.Tensor
    # attention
    producer: Any                                              # dsv4_producer.Weights
    attn_sink: torch.Tensor                                    # f32 [64]
    compressor: Any = None                                     # dsv4_compressor.Weights
    indexer: Any = None                                        # dsv4_producer.IndexerWeights
    wo: Any = None                                             # wo_projection.Weights (MXFP8)
    wo_a_bf16: torch.Tensor | None = None                      # [8, 1024, 4096] (wo_mode=reference)
    wo_b_fp8: Any = None                                       # block_fp8_linear.Weight (wo_mode=reference)
    # ffn
    gate_weight: torch.Tensor | None = None                    # bf16 [256, d]
    gate_bias: torch.Tensor | None = None                      # f32 [256]    (score layers)
    tid2eid: torch.Tensor | None = None                        # i32 [V, 6]   (hash layers)
    experts: Any = None                                        # fused_moe.PreparedExperts
    shared_w13: Any = None                                     # block_fp8_linear.Weight [gate;up] 4096 x 4096
    shared_w2: Any = None                                      # block_fp8_linear.Weight 4096 x 2048
    moe_activation: str = "silu_v41"


def load_layer_weights(ckpt: Checkpoint, cfg: Config, layer: int, device: torch.device,
                       *, wo_mode: str = "b12x", moe_activation: str = "silu_v41") -> LayerWeights:
    """Load one layer and pack it for the b12x ops (one-time load work)."""
    install_b12x_compat()
    from b12x.attention import dsv4_compressor, dsv4_producer
    from b12x.gemm import block_fp8_linear, wo_projection

    p = f"layers.{layer}."
    g = lambda name: ckpt.get(p + name, device)  # noqa: E731
    ratio = cfg.ratio(layer)

    # -- attention producer: joint [wq_a; wkv] GEMM + wq_b, FP8 with 128x128 UE8M0 blocks.
    producer = dsv4_producer.pack_weights(
        g("attn.wq_a.weight"), _u8(g("attn.wq_a.scale")),
        g("attn.wq_b.weight"), _u8(g("attn.wq_b.scale")),
        g("attn.wkv.weight"), _u8(g("attn.wkv.scale")),
        g("attn.q_norm.weight").contiguous(), g("attn.kv_norm.weight").contiguous(),
    )
    lw = LayerWeights(
        layer=layer, ratio=ratio, hash_routing=cfg.is_hash_layer(layer),
        hc_attn=(g("hc_attn_fn").float().contiguous(), g("hc_attn_scale").float().contiguous(),
                 g("hc_attn_base").float().contiguous()),
        hc_ffn=(g("hc_ffn_fn").float().contiguous(), g("hc_ffn_scale").float().contiguous(),
                g("hc_ffn_base").float().contiguous()),
        attn_norm=g("attn_norm.weight").contiguous(), ffn_norm=g("ffn_norm.weight").contiguous(),
        producer=producer, attn_sink=g("attn.attn_sink").float().contiguous(),
        moe_activation=moe_activation,
    )

    # -- compressor (BF16 wkv/wgate concatenated once), C4 also carries the index compressor.
    if ratio:
        kw = {}
        if ratio == 4:
            kw = dict(
                index_wkv=g("attn.indexer.compressor.wkv.weight").contiguous(),
                index_wgate=g("attn.indexer.compressor.wgate.weight").contiguous(),
                index_ape=g("attn.indexer.compressor.ape").float().contiguous(),
                index_norm=g("attn.indexer.compressor.norm.weight").contiguous(),
            )
            lw.indexer = dsv4_producer.pack_indexer_weights(
                g("attn.indexer.wq_b.weight"), _u8(g("attn.indexer.wq_b.scale")),
                g("attn.indexer.weights_proj.weight").contiguous(),
            )
        lw.compressor = dsv4_compressor.pack_weights(
            g("attn.compressor.wkv.weight").contiguous(), g("attn.compressor.wgate.weight").contiguous(),
            g("attn.compressor.ape").float().contiguous(), g("attn.compressor.norm.weight").contiguous(),
            **kw,
        )

    # -- output projection. Checkpoint wo_a is FP8 [8*1024, 4096] with 128x128 scales;
    #    the official convert.py dequantizes it to BF16 and model.py runs it as a BF16
    #    grouped einsum. b12x runs both wo_a and wo_b as MXFP8 GEMMs.
    wo_a, wo_a_s = g("attn.wo_a.weight"), g("attn.wo_a.scale")
    wo_b, wo_b_s = g("attn.wo_b.weight"), g("attn.wo_b.scale")
    group_width = cfg.n_heads * cfg.head_dim // cfg.o_groups
    if wo_mode == "b12x":
        lw.wo = wo_projection.pack_weights(
            wo_a, _u8(wo_a_s), wo_b, _u8(wo_b_s), groups=cfg.o_groups, group_width=group_width,
            rank=cfg.o_lora_rank, hidden=cfg.dim,
        )
    elif wo_mode == "reference":
        # TORCH FALLBACK (optional, --wo-mode reference): convert.py's BF16 wo_a, so the
        # grouped einsum matches model.py bit-for-bit up to summation order.
        deq = wo_a.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)) * wo_a_s.float()[:, None, :, None]
        lw.wo_a_bf16 = deq.flatten(2, 3).flatten(0, 1).bfloat16().view(cfg.o_groups, cfg.o_lora_rank, group_width)
        lw.wo_b_fp8 = block_fp8_linear.pack_weight(wo_b, _u8(wo_b_s))
    else:
        raise ValueError(f"unknown wo_mode {wo_mode!r}")
    del wo_a, wo_a_s, wo_b, wo_b_s

    # -- router
    lw.gate_weight = g("ffn.gate.weight").contiguous()
    if lw.hash_routing:
        lw.tid2eid = g("ffn.gate.tid2eid").to(torch.int32).contiguous()
    else:
        lw.gate_bias = g("ffn.gate.bias").float().contiguous()

    # -- shared expert: [w1(gate); w3(up)] share one activation quantization -> one GEMM.
    sw1, sw3 = g("ffn.shared_experts.w1.weight"), g("ffn.shared_experts.w3.weight")
    ss1, ss3 = _u8(g("ffn.shared_experts.w1.scale")), _u8(g("ffn.shared_experts.w3.scale"))
    lw.shared_w13 = block_fp8_linear.pack_weight(
        torch.cat((sw1.view(torch.uint8), sw3.view(torch.uint8))).view(torch.float8_e4m3fn),
        torch.cat((ss1, ss3)),
    )
    lw.shared_w2 = block_fp8_linear.pack_weight(g("ffn.shared_experts.w2.weight"),
                                                _u8(g("ffn.shared_experts.w2.scale")))
    del sw1, sw3, ss1, ss3

    lw.experts = load_routed_experts(ckpt, cfg, layer, device, moe_activation)
    return lw


def load_routed_experts(ckpt: Checkpoint, cfg: Config, layer: int, device, activation: str):
    """Stack the MXFP4 experts into the fused_moe PackedWeights ABI and prepare them.

    Checkpoint expert E: w1 (gate) / w3 (up) I8 [2048, 2048] = FP4 e2m1 packed low
    nibble first, E8M0 scale [2048, 128] per 32 along K; w2 I8 [4096, 1024], scale
    [4096, 64]. b12x W13 layout is [up; gate] rows ("w13" = kernel order), so
    w13[e] = [w3; w1]. Global/activation scales are 1 for MXFP4 (all scaling is E8M0).
    """
    from b12x.moe import fused_moe

    E, H, n = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
    w13 = torch.empty((E, 2 * n, H // 2), dtype=torch.uint8, device=device)
    s13 = torch.empty((E, 2 * n, H // 32), dtype=torch.uint8, device=device)
    w2 = torch.empty((E, H, n // 2), dtype=torch.uint8, device=device)
    s2 = torch.empty((E, H, n // 32), dtype=torch.uint8, device=device)
    for e in range(E):
        p = f"layers.{layer}.ffn.experts.{e}."
        w13[e, :n] = _u8(ckpt.get(p + "w3.weight"))
        w13[e, n:] = _u8(ckpt.get(p + "w1.weight"))
        s13[e, :n] = _u8(ckpt.get(p + "w3.scale"))
        s13[e, n:] = _u8(ckpt.get(p + "w1.scale"))
        w2[e] = _u8(ckpt.get(p + "w2.weight"))
        s2[e] = _u8(ckpt.get(p + "w2.scale"))
    ones = torch.ones(E, dtype=torch.float32, device=device)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
                                      w13_layout=fused_moe.W13Layout.W13),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity=activation,
                                            io_dtype=torch.bfloat16, swiglu_limit=cfg.swiglu_limit),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=n),
    )
    return fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=ones, w2_global_scales=ones.clone(),
    ))


# --------------------------------------------------------------------------- layer


class DeepseekV4Layer:
    """One V4 decoder layer as a b12x op sequence (prefill from position 0)."""

    def __init__(self, cfg: Config, weights: LayerWeights, device: torch.device, *,
                 ops: "OpPlans", wo_mode: str = "b12x"):
        self.cfg, self.w, self.device, self.ops, self.wo_mode = cfg, weights, device, ops, wo_mode
        self._moe_plans: dict[int, Any] = {}  # fused_moe execution plans bind this layer's experts

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def release(self) -> None:
        """Drop this layer's prepared fused_moe plans (they retain the layer's
        expert weights in b12x's lazy preparation session until released)."""
        from b12x.preparation.session import _LAZY_SESSIONS

        for plan in self._moe_plans.values():
            for session in list(_LAZY_SESSIONS.values()):
                try:
                    session.release(plan)
                except Exception:
                    pass
        self._moe_plans.clear()

    def forward(self, stream: torch.Tensor, token_ids: torch.Tensor, meta: PrefillMetadata,
                cos_sin: torch.Tensor, debug: dict | None = None) -> torch.Tensor:
        """stream [T, 4, dim] bf16 -> [T, 4, dim] bf16 (model.py Block.forward)."""
        cfg, w, ops = self.cfg, self.w, self.ops
        T = stream.shape[0]
        mix = dict(rms_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, sinkhorn_iters=cfg.hc_sinkhorn_iters,
                   norm_eps=cfg.norm_eps)

        # 1. mHC pre (attention): mixes = fn . flat(stream) * rsqrt(ms), Sinkhorn(20) -> pre/post/comb,
        #    x = sum_l pre_l * stream_l, then attn_norm (fused).  model.py Block.hc_pre + attn_norm.
        res_a, post_a, comb_a, x_attn = ops.mhc_pre(T).run(stream, *w.hc_attn, norm_weight=w.attn_norm, **mix)

        # 2. attention -> delta [T, dim]
        delta_attn = self.attention(x_attn, meta, cos_sin, debug)

        # 3. mHC post (attention) fused with mHC pre (ffn) + ffn_norm:
        #    stream' = post*delta + comb^T stream; then pre on stream'.
        res_f, post_f, comb_f, x_ffn = ops.mhc_post_pre(T).run(
            delta_attn, res_a, post_a, comb_a, *w.hc_ffn, norm_weight=w.ffn_norm, **mix)

        # 4. MoE -> delta [T, dim]
        delta_ffn = self.ffn(x_ffn, token_ids, debug)

        # 5. mHC post (ffn). An engine fuses this with the next layer's pre (run_post_pre)
        #    and the last layer's with run_head; kept separate so every layer output is
        #    the [T,4,dim] stream golden.py saves.
        out = torch.empty_like(stream)
        ops.mhc_post(T).run(delta_ffn, res_f, post_f, comb_f, out=out)
        if debug is not None:
            debug.update(x_attn=x_attn, delta_attn=delta_attn, x_ffn=x_ffn, delta_ffn=delta_ffn)
        return out

    # ------------------------------------------------------------------ attention
    def attention(self, x: torch.Tensor, meta: PrefillMetadata, cos_sin: torch.Tensor,
                  debug: dict | None) -> torch.Tensor:
        from b12x.attention import compressed_sparse_mla, dsa_indexer, dsv4_compressor, dsv4_producer

        cfg, w, ops, dev = self.cfg, self.w, self.ops, self.device
        T, ratio, led = x.shape[0], w.ratio, ops.ledger

        # 2a. producer: [wq_a|wkv] block-FP8 GEMM (act quant per 128, UE8M0) -> q_norm (q_rank,
        #     kept in producer scratch for the indexer) and kv_norm + RoPE + FP8 page pack into
        #     the window cache; wq_b GEMM into query; per-head RMS (no weight) + RoPE in place.
        main_cache = led.buffer("main_kv_cache", torch.zeros((meta.main_pages, MAIN_PAGE_BYTES),
                                                               dtype=torch.uint8, device=dev))
        query = led.buffer("query", torch.empty((T, cfg.n_heads, HEAD_DIM), dtype=torch.bfloat16, device=dev))
        pplan = ops.producer(T)
        producer = dsv4_producer.bind(
            pplan, scratch=led.scratch("dsv4_producer", pplan.scratch_specs()), hidden_states=x,
            positions=meta.positions, main_slots=meta.main_slots, cos_sin_cache=cos_sin,
            main_kv_cache=main_cache, query=query, weights=w.producer, eps=cfg.norm_eps, expected_m=T,
        )
        dsv4_producer.run(binding=producer)

        indexed_cache = indexed_idx = indexed_len = None
        indexed_page_rows = None
        if ratio:
            c = meta.compress[ratio]
            # 2b. compressor prefill: joint BF16 projection (torch.mm inside b12x), per-group
            #     fp32 softmax pooling (C4: 8 slots = prev group half A + this group half B,
            #     + ape), RMSNorm, RoPE at j*ratio, FP8 page pack; terminal state written
            #     for later continuation. C4 also pools the 128-d index key -> Hadamard ->
            #     FP4 QAT -> FP8 index page.
            comp_cache = led.buffer(f"c{ratio}_compressed_cache", torch.zeros(
                (c["pages"], compressed_page_bytes(ratio)), dtype=torch.uint8, device=dev))
            coff = 2 if ratio == 4 else 1
            state_rows = 2 * coff * ratio
            kv_state = led.buffer(f"c{ratio}_kv_state", torch.zeros((1, state_rows, coff * HEAD_DIM),
                                                                    dtype=torch.float32, device=dev))
            score_state = led.buffer(f"c{ratio}_score_state", torch.zeros_like(kv_state))
            index_kw = {}
            if ratio == 4:
                index_cache = led.buffer("c4_index_cache", torch.zeros((c["pages"], INDEX_PAGE_BYTES),
                                                                       dtype=torch.uint8, device=dev))
                ikv = led.buffer("c4_index_kv_state", torch.zeros((1, state_rows, 2 * INDEX_HEAD_DIM),
                                                                   dtype=torch.float32, device=dev))
                index_kw = dict(index_cache=index_cache, index_kv_state=ikv,
                                index_score_state=led.buffer("c4_index_score_state", torch.zeros_like(ikv)))
            cplan = ops.compressor(T, ratio)
            comp = dsv4_compressor.bind_prefill(
                cplan, scratch=led.scratch(f"dsv4_compressor_c{ratio}", cplan.scratch_specs()),
                hidden_states=x, active_groups=c["active_groups"],
                group_source_starts=c["group_source_starts"], group_rope_positions=c["group_rope_positions"],
                compressed_slots=c["compressed_slots"], active_sequences=c["active_sequences"],
                sequence_offsets=c["sequence_offsets"], state_sequence_ids=c["state_sequence_ids"],
                compressed_cos_sin_cache=cos_sin, compressed_main_cache=comp_cache,
                main_kv_state=kv_state, main_score_state=score_state, weights=w.compressor,
                eps=cfg.norm_eps, initial_prefill=True, **index_kw,
            )
            dsv4_compressor.run_prefill(binding=comp)
            indexed_cache, indexed_page_rows = comp_cache, compressed_page_rows(ratio)
            indexed_len = c["indexed_lengths"]

            if ratio == 4 and c["groups"] > 0:
                # 2c. index query: wq_b(q_rank) block-FP8 GEMM, RoPE, Hadamard(128), FP4 QAT
                #     stored as FP8; head weights = weights_proj(x) * 128^-0.5 * 64^-0.5 (BF16-rounded).
                index_q = led.buffer("index_query", torch.empty((T, INDEX_HEADS, INDEX_HEAD_DIM),
                                                                dtype=torch.float8_e4m3fn, device=dev))
                head_w = led.buffer("index_head_weights", torch.empty((T, INDEX_HEADS), dtype=torch.float32,
                                                                      device=dev))
                iplan = ops.index_producer(T)
                ib = dsv4_producer.bind_indexer(
                    iplan, scratch=led.scratch("dsv4_index_producer", iplan.scratch_specs()),
                    q_rank=producer.q_rank, hidden_states=x, positions=meta.positions, cos_sin_cache=cos_sin,
                    query=index_q, head_weights=head_w, weights=w.indexer, expected_m=T,
                )
                dsv4_producer.run_indexer(binding=ib)
                # 2d. top-k: score[t,j] = sum_h relu(q_h . k_j) * w_h over the causal
                #     completed groups j < (t+1)//4; top-512 as physical index-cache slots
                #     (== C4 compressed-cache slots: same pages, same 64 rows/page).
                selected = led.buffer("c4_selected_slots", torch.empty((T, cfg.index_topk), dtype=torch.int32,
                                                                       device=dev))
                operands = dict(q_fp8=index_q, query_weights=head_w, index_k_cache=index_cache,
                                page_table=c["index_page_table"], cache_lengths=c["index_cache_lengths"],
                                active_width=c["index_active_width"], output_indices=selected)
                splan = ops.index_topk(T, c["pages"], operands)
                sb = dsa_indexer.bind(splan, scratch=led.scratch("dsa_indexer", splan.scratch_specs()), **operands)
                dsa_indexer.run(sb)
                indexed_idx = selected
                if debug is not None:
                    debug.update(index_query=index_q, index_head_weights=head_w, c4_selected=selected)
            elif ratio == 128:
                indexed_idx = c["indexed_indices"]
            if c["groups"] == 0:
                indexed_cache = indexed_idx = indexed_len = None  # no completed group yet: window only

        # 2e. sparse MLA over [window slots | compressed slots] with per-head sink:
        #     softmax(q.k * 512^-0.5 ; sink) . v, v == k (512-d latent), output not de-rotated.
        attn_out = led.buffer("attn_out", torch.empty((T, cfg.n_heads, HEAD_DIM), dtype=torch.bfloat16, device=dev))
        aplan = ops.sparse_mla(T, query, main_cache, indexed_cache, w.attn_sink, attn_out,
                               swa_width=WINDOW,
                               indexed_width=0 if indexed_idx is None else int(indexed_idx.shape[1]),
                               indexed_page_size=indexed_page_rows or SOURCE_PAGE_TOKENS)
        ab = compressed_sparse_mla.bind(
            aplan, scratch=led.scratch("compressed_sparse_mla", aplan.scratch_specs()), q=query,
            swa_indices=meta.swa_indices, swa_lengths=meta.swa_lengths,
            indexed_indices=indexed_idx, indexed_lengths=indexed_len,
        )
        compressed_sparse_mla.run(plan=aplan, binding=ab, swa_k_cache=main_cache, indexed_k_cache=indexed_cache,
                                  attn_sink=w.attn_sink, sm_scale=HEAD_DIM ** -0.5, out=attn_out)
        if debug is not None:
            debug.update(query=query, main_kv_cache=main_cache, attn_out=attn_out,
                         compressed_cache=indexed_cache)

        # 2f. inverse RoPE on o[..., 448:], grouped wo_a (8 x [1024 x 4096]), wo_b.
        return self.output_projection(attn_out, meta, cos_sin)

    def output_projection(self, o: torch.Tensor, meta: PrefillMetadata, cos_sin: torch.Tensor) -> torch.Tensor:
        from b12x.gemm import wo_projection

        cfg, w, ops = self.cfg, self.w, self.ops
        T = o.shape[0]
        if self.wo_mode == "b12x":
            plan = ops.wo(T)
            b = wo_projection.bind_inv_rope(
                plan, scratch=ops.ledger.scratch("wo_projection", plan.scratch_specs()), o=o,
                positions=meta.positions, cos_sin_cache=cos_sin, weights=w.wo,
                heads_per_group=cfg.n_heads // cfg.o_groups, nope_dim=NOPE_DIM, rope_dim=ROPE_DIM, expected_m=T,
            )
            return wo_projection.run_inv_rope(binding=b, plan=plan)
        # TORCH FALLBACK (--wo-mode reference): model.py's BF16 inverse RoPE + BF16 grouped
        # wo_a einsum (no activation quantization), then b12x block-FP8 wo_b with the
        # checkpoint's per-128 activation quantization. Isolates the MXFP8 wo_a error.
        o = apply_rope_torch(o, cos_sin, meta.positions, inverse=True)
        tmp = torch.einsum("tgd,grd->tgr", o.view(T, cfg.o_groups, -1), w.wo_a_bf16).reshape(T, -1).contiguous()
        return ops.fp8_linear(tmp, w.wo_b_fp8, "wo_b")

    # ------------------------------------------------------------------ ffn
    def ffn(self, x: torch.Tensor, token_ids: torch.Tensor, debug: dict | None) -> torch.Tensor:
        from b12x.moe import fused_moe

        cfg, w, ops = self.cfg, self.w, self.ops
        T, dev = x.shape[0], self.device
        k = cfg.n_activated_experts

        # 4a. router. TORCH FALLBACK: the gate GEMM runs in FP32 like model.py
        #     (linear(x.float(), W.float())); b12x's route path consumes precomputed logits.
        logits = (x.float() @ w.gate_weight.float().t()).contiguous()
        topk_ids = torch.empty((T, k), dtype=torch.int32, device=dev)
        topk_weights = torch.empty((T, k), dtype=torch.float32, device=dev)
        if w.hash_routing:
            # TORCH FALLBACK: hash layers (layer < n_hash_layers) take expert ids from
            # tid2eid[token]; route_topk has no token-table mode. Weights are still the
            # sqrtsoftplus scores at those ids, sum-normalized (no eps), x route_scale.
            ids = w.tid2eid[token_ids.long()]
            scores = F.softplus(logits).sqrt()
            wts = scores.gather(1, ids.long())
            topk_weights.copy_(wts / wts.sum(dim=-1, keepdim=True) * cfg.route_scale)
            topk_ids.copy_(ids)
        else:
            # sqrtsoftplus scores; select top-6 of (score + bias); weights = unbiased scores at
            # the selected ids, normalized (b12x adds 1e-20), x 1.5. Ties -> larger expert id.
            # TORCH FALLBACK: b12x fused_moe.route_topk's prepared launcher passes BLOCK_E to a
            # compiled Triton kernel that no longer accepts it (sparkinfer 7fcc094e). model.py
            # semantics: select on score + bias, weight with the unbiased score, normalize, scale.
            scores = F.softplus(logits).sqrt()
            ids = (scores + w.gate_bias.float()).topk(k, dim=-1).indices
            wts = scores.gather(1, ids)
            topk_weights.copy_(wts / wts.sum(dim=-1, keepdim=True) * cfg.route_scale)
            topk_ids.copy_(ids.to(torch.int32))

        # 4b. routed experts: MXFP4 weights x MXFP8 activations (per-32 UE8M0), SwiGLU with
        #     clamp(up, +-10), clamp(gate, <=10), router weight applied (silu_v41: on the
        #     intermediate before FP8, as model.py), sum over top-6 -> BF16.
        routed = torch.empty((T, cfg.dim), dtype=torch.bfloat16, device=dev)
        if T not in self._moe_plans:
            self._moe_plans[T] = ops.moe(T, w.experts, w.layer)
        mplan = self._moe_plans[T]
        mb = fused_moe.bind(mplan, scratch=ops.ledger.scratch("fused_moe", mplan.scratch_specs()), a=x,
                            experts=w.experts, topk_weights=topk_weights, topk_ids=topk_ids, output=routed,
                            input_scales_static=True)
        fused_moe.run(binding=mb)

        # 4c. shared expert: [w1;w3] block-FP8 GEMM (act per-128) -> SwiGLU -> w2 block-FP8 GEMM.
        gate_up = ops.fp8_linear(x, w.shared_w13, "shared_w13")
        # TORCH FALLBACK: clamped SwiGLU on [T, 2x2048] (model.py Expert.forward, fp32 math,
        # BF16 out); b12x has no standalone clamped-SwiGLU for the dense FP8 path.
        gate, up = gate_up.float().split(cfg.moe_inter_dim, dim=-1)
        up = up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
        gate = gate.clamp(max=cfg.swiglu_limit)
        hidden = (F.silu(gate) * up).to(torch.bfloat16).contiguous()
        shared = ops.fp8_linear(hidden, w.shared_w2, "shared_w2")

        # TORCH FALLBACK: routed + shared in fp32 -> BF16 (model.py: y(fp32) += shared; type_as(x)).
        out = (routed.float() + shared.float()).to(torch.bfloat16)
        if debug is not None:
            debug.update(router_logits=logits, topk_ids=topk_ids, topk_weights=topk_weights,
                         routed=routed, shared=shared)
        return out


# --------------------------------------------------------------------------- plan cache


class OpPlans:
    """Declares and prepares b12x plans; reuses weight-independent plans per T."""

    def __init__(self, cfg: Config, device: torch.device):
        install_b12x_compat()
        self.cfg, self.device = cfg, device
        self.ledger = ScratchLedger()
        self._cache: dict[tuple, Any] = {}

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    # mHC ---------------------------------------------------------------------
    def _mhc_invocation(self, operation: str, **extra):
        from b12x.preparation import FrozenMapping

        cfg = self.cfg
        base = dict(operation=operation, output_mode="provided")
        if operation in ("pre", "post_pre"):
            base.update(has_norm_weight=True, norm_weight_dtype="bfloat16", rms_eps=cfg.norm_eps,
                        hc_eps=cfg.hc_eps, sinkhorn_iters=cfg.hc_sinkhorn_iters, norm_eps=cfg.norm_eps)
        base.update(extra)
        return FrozenMapping(base)

    def mhc_pre(self, T: int):
        return self._get(("mhc_pre", T), lambda: _MhcCall(self, T, "pre"))

    def mhc_post_pre(self, T: int):
        return self._get(("mhc_post_pre", T), lambda: _MhcCall(self, T, "post_pre"))

    def mhc_post(self, T: int):
        return self._get(("mhc_post", T), lambda: _MhcCall(self, T, "post"))

    # attention ---------------------------------------------------------------
    def producer(self, T: int):
        from b12x.attention import dsv4_producer

        cfg = self.cfg
        return self._get(("producer", T), lambda: dsv4_producer.plan(dsv4_producer.Caps(
            device=self.device, max_tokens=T, hidden=cfg.dim, q_lora_rank=cfg.q_lora_rank,
            heads=cfg.n_heads, cache_format="fp8")))

    def index_producer(self, T: int):
        from b12x.attention import dsv4_producer

        cfg = self.cfg
        return self._get(("index_producer", T), lambda: dsv4_producer.plan_indexer(dsv4_producer.IndexerCaps(
            device=self.device, max_tokens=T, hidden=cfg.dim, q_lora_rank=cfg.q_lora_rank)))

    def compressor(self, T: int, ratio: int):
        from b12x.attention import dsv4_compressor

        return self._get(("compressor", T, ratio), lambda: dsv4_compressor.plan(dsv4_compressor.Caps(
            device=self.device, max_tokens=T, hidden=self.cfg.dim, compress_ratio=ratio,
            with_indexer=ratio == 4, cache_format="fp8")))

    def index_topk(self, T: int, pages: int, operands: dict):
        from b12x.attention import dsa_indexer

        def build():
            caps = dsa_indexer.Caps(device=self.device, num_q_heads=INDEX_HEADS, max_q_rows=T,
                                    max_page_table_width=pages, topk=self.cfg.index_topk, mode="prefill",
                                    output_index_space="physical")
            plan = dsa_indexer.plan(caps, invocation=dsa_indexer.invocation_from_tensors(caps, **operands))
            return prepare(plan, f"dsa_indexer.T{T}")

        return self._get(("index_topk", T, pages), build)

    def sparse_mla(self, T, q, swa_cache, indexed_cache, sink, out, *, swa_width, indexed_width,
                   indexed_page_size):
        from b12x.attention import compressed_sparse_mla as mla

        key = ("sparse_mla", T, swa_cache.shape[0], None if indexed_cache is None else tuple(indexed_cache.shape),
               indexed_width, indexed_page_size)

        def build():
            width = swa_width + indexed_width
            caps = mla.Caps(
                device=self.device, num_q_heads=self.cfg.n_heads, max_q_rows=T, max_width=width,
                max_page_table_width=width, max_batch=T, max_kv_rows=T * width, mode="extend",
                swa_width=swa_width, indexed_width=indexed_width, swa_page_size=SOURCE_PAGE_TOKENS,
                indexed_page_size=indexed_page_size,
            )
            invocation = mla.invocation_from_tensors(q=q, swa_k_cache=swa_cache, indexed_k_cache=indexed_cache,
                                                     attn_sink=sink, out=out)
            return prepare(mla.plan(caps, invocation=invocation), f"compressed_sparse_mla.T{T}.w{indexed_width}")

        return self._get(key, build)

    def wo(self, T: int):
        from b12x.gemm import wo_projection
        from b12x.preparation import FrozenMapping

        cfg = self.cfg

        def build():
            caps = wo_projection.Caps(device=self.device, max_tokens=T, groups=cfg.o_groups,
                                      group_width=cfg.n_heads * HEAD_DIM // cfg.o_groups,
                                      rank=cfg.o_lora_rank, hidden=cfg.dim)
            inv = FrozenMapping(dict(operation="inv_rope", heads_per_group=cfg.n_heads // cfg.o_groups,
                                     nope_dim=NOPE_DIM, rope_dim=ROPE_DIM, positions_dtype="int64",
                                     cos_sin_dtype="float32"))
            return prepare(wo_projection.plan(caps, invocation=inv), f"wo_projection.T{T}")

        return self._get(("wo", T), build)

    # dense / moe ---------------------------------------------------------------
    def fp8_linear(self, x: torch.Tensor, weight, name: str) -> torch.Tensor:
        """Block-FP8 linear with the checkpoint's per-128 activation quantization."""
        from b12x.gemm import block_fp8_linear

        T = x.shape[0]
        key = ("fp8_linear", T, weight.in_features, weight.out_features)
        plan = self._get(key, lambda: prepare(block_fp8_linear.plan(block_fp8_linear.Caps(
            device=self.device, max_tokens=T, in_features=weight.in_features,
            out_features=weight.out_features, activation_block_size=128)), f"block_fp8_linear.{name}.T{T}"))
        out = torch.empty((T, weight.out_features, 1), dtype=torch.bfloat16, device=self.device)
        b = block_fp8_linear.bind(plan, scratch=self.ledger.scratch(f"block_fp8_linear.{name}", plan.scratch_specs()),
                                  source=x, packed_weight=weight, output=out, activation_block_size=128)
        block_fp8_linear.run(binding=b)
        return out[:, :, 0]

    def route(self, T: int):
        from b12x.moe import fused_moe

        cfg = self.cfg
        return self._get(("route", T), lambda: prepare(fused_moe.plan_route_topk(fused_moe.RouteTopKInvocation(
            num_tokens=T, num_experts=cfg.n_routed_experts, top_k=cfg.n_activated_experts,
            logits_dtype="float32", score_func="sqrtsoftplus", renormalize=True, has_correction_bias=True,
            routed_scaling_factor=cfg.route_scale)), f"route_topk.T{T}"))

    def moe(self, T: int, experts, layer: int):
        from b12x.moe import fused_moe

        # Execution plans bind the PreparedExperts object, so one per layer.
        plan = fused_moe.plan_execution(experts=experts, capacity=fused_moe.ExecutionCapacity(
            max_tokens=T, top_k=self.cfg.n_activated_experts))
        return prepare(plan, f"fused_moe.L{layer}.T{T}")


class _MhcCall:
    """One prepared mHC operation (pre | post_pre | post) with its bound outputs."""

    def __init__(self, ops: OpPlans, T: int, operation: str):
        from b12x.norm import mhc

        self.mhc, self.ops, self.T, self.operation = mhc, ops, T, operation
        extra = dict(expanded_residual=True) if operation == "pre" else {}
        caps = mhc.Caps(device=ops.device, max_tokens=T, hidden_size=ops.cfg.dim)
        self.plan = prepare(mhc.plan(caps, invocation=ops._mhc_invocation(operation, **extra)), f"mhc.{operation}.T{T}")

    def run(self, *args, out=None, **kwargs):
        mhc, T, H, dev = self.mhc, self.T, self.ops.cfg.dim, self.ops.device
        if self.operation == "post":
            x, residual, post, comb = args
            return mhc.run_post(x, residual, post, comb, plan=self.plan, out=out)
        binding = mhc.bind(
            self.plan, scratch=self.ops.ledger.scratch(f"mhc.{self.operation}", self.plan.scratch_specs()),
            tokens=T,
            out=torch.empty((T, 4, H), dtype=torch.bfloat16, device=dev),
            y=torch.empty((T, H), dtype=torch.bfloat16, device=dev),
            post=torch.empty((T, 4), dtype=torch.float32, device=dev),
            comb=torch.empty((T, 4, 4), dtype=torch.float32, device=dev),
        )
        fn = mhc.run_pre if self.operation == "pre" else mhc.run_post_pre
        return fn(*args, binding=binding, **kwargs)


# --------------------------------------------------------------------------- model driver


class DeepseekV4B12x:
    """Layer-by-layer V4 Flash prefill (one layer's weights resident at a time)."""

    def __init__(self, snapshot: Path = DEFAULT_SNAPSHOT, device: int | str = 0, *,
                 wo_mode: str = "b12x", moe_activation: str = "silu_v41"):
        self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        torch.cuda.set_device(self.device)
        self.snapshot = Path(snapshot)
        self.cfg = Config.from_snapshot(self.snapshot)
        self.ckpt = Checkpoint(self.snapshot)
        self.wo_mode, self.moe_activation = wo_mode, moe_activation
        self.ops = OpPlans(self.cfg, self.device)
        self._rope: dict[int, torch.Tensor] = {}
        self._meta: dict[int, PrefillMetadata] = {}

    def metadata(self, T: int) -> PrefillMetadata:
        if T not in self._meta:
            self._meta[T] = build_prefill_metadata(T, self.device, index_topk=self.cfg.index_topk)
        return self._meta[T]

    def cos_sin(self, ratio: int, T: int) -> torch.Tensor:
        key = (1 if ratio else 0, T)
        if key not in self._rope:
            self._rope[key] = rope_cos_sin(self.cfg, ratio, T, self.device)
        return self._rope[key]

    @torch.inference_mode()
    def embed(self, token_ids: list[int]) -> torch.Tensor:
        # TORCH FALLBACK: embedding is a row gather (read per token from the checkpoint).
        return self.ckpt.rows("embed.weight", list(token_ids), self.device).to(torch.bfloat16)

    @staticmethod
    def expand(h: torch.Tensor, hc: int = 4) -> torch.Tensor:
        """model.py: h.unsqueeze(2).repeat(1, 1, hc, 1). An engine can instead run the
        broadcast-lane mhc pre (residual [T,dim], lane-summed fn [24,dim]) for layer 0."""
        return h.unsqueeze(1).repeat(1, hc, 1).contiguous()

    def load_layer(self, layer: int) -> DeepseekV4Layer:
        weights = load_layer_weights(self.ckpt, self.cfg, layer, self.device, wo_mode=self.wo_mode,
                                     moe_activation=self.moe_activation)
        return DeepseekV4Layer(self.cfg, weights, self.device, ops=self.ops, wo_mode=self.wo_mode)

    @torch.inference_mode()
    def layer_forward(self, layer: int, stream: torch.Tensor, token_ids: torch.Tensor,
                      debug: dict | None = None, block: DeepseekV4Layer | None = None) -> torch.Tensor:
        T = stream.shape[0]
        block = block or self.load_layer(layer)
        return block.forward(stream, token_ids, self.metadata(T), self.cos_sin(self.cfg.ratio(layer), T), debug)

    @torch.inference_mode()
    def head(self, stream: torch.Tensor, chunk: int = 16_384) -> torch.Tensor:
        """hc_head collapse + final RMSNorm (b12x run_head), then FP32 logits [T, V]."""
        from b12x.norm import mhc

        cfg, dev = self.cfg, self.device
        T = stream.shape[0]
        y = torch.empty((T, cfg.dim), dtype=torch.bfloat16, device=dev)
        mhc.run_head(stream.contiguous(), self.ckpt.get("hc_head_fn", dev).float().contiguous(),
                     self.ckpt.get("hc_head_scale", dev).float().contiguous(),
                     self.ckpt.get("hc_head_base", dev).float().contiguous(),
                     self.ckpt.get("norm.weight", dev).contiguous(),
                     rms_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, norm_eps=cfg.norm_eps, out=y)
        # TORCH FALLBACK: model.py's head is F.linear(x.float(), W.float()) -> FP32 [T, V];
        # b12x bf16_vocab_projection is a decode-time BF16-output op. Chunked over V.
        W = self.ckpt.get("head.weight", dev)
        logits = torch.empty((T, cfg.vocab_size), dtype=torch.float32, device=dev)
        yf = y.float()
        for s in range(0, cfg.vocab_size, chunk):
            logits[:, s : s + chunk] = yf @ W[s : s + chunk].float().t()
        return logits

    @torch.inference_mode()
    def forward(self, token_ids: list[int], layers: range | None = None, on_layer=None) -> torch.Tensor:
        ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        stream = self.expand(self.embed(token_ids))
        for layer in layers or range(self.cfg.n_layers):
            stream = self.layer_forward(layer, stream, ids)
            if on_layer is not None:
                on_layer(layer, stream)
            torch.cuda.empty_cache()
        return self.head(stream)


# --------------------------------------------------------------------------- sizing


def describe_prefill_buffers(T: int, cfg: Config | None = None) -> dict[str, int]:
    """Closed-form bytes of every per-layer buffer this prefill binds (fp8 caches).

    Formulas follow the b12x plan layouts at sparkinfer 7fcc094e; op scratch that
    depends on a selected tactic (block-FP8 split-K workspace, fused_moe routing
    workspace, dsa_indexer tiles, wo MXFP8 staging) is printed exactly at run time by
    ScratchLedger.report() -- compare.py --ledger.
    """
    cfg = cfg or Config(compress_ratios=(0,))
    d, H, pages = cfg.dim, cfg.n_heads, math.ceil(T / SOURCE_PAGE_TOKENS)

    def fp8_linear_scratch(rows, k):  # MXFP8 staging: values + scale rows + MMA scale tiles
        return _align(_align(rows * k, 1024) + rows * (k // 32), 1024) + math.ceil(rows / 128) * math.ceil(k / 128) * 512

    producer = (_align(fp8_linear_scratch(T, d), 1024) + _align(fp8_linear_scratch(T, cfg.q_lora_rank), 1024)
                + _align(T * (cfg.q_lora_rank + HEAD_DIM) * 2, 1024) + _align(T * cfg.q_lora_rank * 2, 1024))
    out = {
        "stream [T,4,d] bf16": T * 4 * d * 2,
        "mhc partials [T,64,25] f32 (pre, post_pre)": T * 64 * 25 * 4,
        "mhc y/post/comb": T * d * 2 + T * 4 * 4 + T * 16 * 4,
        "main_kv_cache ceil(T/256) x 149760": pages * MAIN_PAGE_BYTES,
        "query [T,64,512] bf16": T * H * HEAD_DIM * 2,
        "dsv4_producer scratch (approx, no split-K)": producer,
        "C4 compressed cache ceil(T/256) x 37440": pages * compressed_page_bytes(4),
        "C4 index cache ceil(T/256) x 8448": pages * INDEX_PAGE_BYTES,
        "C4 compressor scratch T x 5120": _align(T * 2 * (1024 + 256) * 2, 1024),
        "C4 compressor state (1 seq)": 2 * 16 * 1024 * 4 + 2 * 16 * 256 * 4,
        "C4 index query [T,64,128] fp8 + weights [T,64] f32": T * INDEX_HEADS * INDEX_HEAD_DIM + T * INDEX_HEADS * 4,
        "C4 index producer scratch (approx)": _align(fp8_linear_scratch(T, cfg.q_lora_rank), 1024)
        + _align(T * 8192 * 2, 1024) + _align(T * 64 * 2, 1024),
        "C4 selected slots [T,512] i32": T * cfg.index_topk * 4,
        "C128 compressed cache ceil(T/256) x 1728": pages * compressed_page_bytes(128),
        "C128 compressor scratch T x 2048": _align(T * 2 * 512 * 2, 1024),
        "C128 compressor state (1 seq)": 2 * 256 * 512 * 4,
        "sparse MLA scratch (single pass)": _align(T * H * HEAD_DIM * 2, 1024) + 2 * _align(T * H * 4, 1024)
        + 3 * 1024 + _align(T * cfg.index_topk * 4, 1024),
        "attn_out [T,64,512] bf16": T * H * HEAD_DIM * 2,
        "wo MXFP8 staging (approx)": T * (4096 * 8 + 4096 * 8 // 32 + 8192 * 2 + 8192 + 8192 // 32 + d * 2),
        "router logits f32 + top-k": T * cfg.n_routed_experts * 4 + T * 6 * 12,
        "routed / shared outputs bf16": 2 * T * d * 2,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--describe", type=int, metavar="T", help="print closed-form buffer sizes for T tokens (CPU)")
    a = p.parse_args()
    if a.describe:
        for name, value in describe_prefill_buffers(a.describe).items():
            print(f"{name:55s} {value:>14,d} B  ({value / a.describe:,.0f} B/token)")


if __name__ == "__main__":
    main()
