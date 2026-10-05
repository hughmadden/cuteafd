#!/usr/bin/env python3
"""Golden activations for GLM 5.3 Flash (glm5_next) from transformers' reference.

Runs one Glm5NextTextDecoderLayer at a time on one GPU (the routed experts do
not fit): the four mHC streams go in and out of every layer, KDA layers take
the chunked (prefill) path with no cache, DSA layers run their own pooled
indexer and eager attention over its top-k mask (every token is selected up
to 2051 tokens, so shorter prompts are dense causal MLA). Final collapse is
the unweighted stream mean, then ``model.norm`` and ``lm_head`` in FP32.

Weights: the coordinator tensors come from ``--snapshot`` (BF16 as stored;
FP8 tensors times their FP32 128x128 block scales); the routed experts come
from ``--experts-snapshot`` (default the same snapshot), which must hold
FP8 or BF16 experts (the EXL3 checkpoints' dense tensors equal the official
BF16 release bit for bit, so ``--snapshot EXL3 --experts-snapshot FP8`` is
the official model with its FP8 experts). Routed experts are summed in FP32
with the SwiGLU clamp and rounded once (transformers' eager experts sum in
BF16, which moves routes on rounding-level changes: compare engines by NLL as
well as agreement). Run with ``PYTHONPATH=third_party/transformers/src`` (the
pinned transformers carries glm5_next) and ``USE_HUB_KERNELS=0``.

Writes the raw files ``glmf-golden`` reads:

  tokens.bin      i32  [T]
  layerNN.bin     bf16 [T, 4, hidden]   mHC streams after layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP [--experts-snapshot SNAP] --text-file prompt.txt --out DIR
            [--layers 0 1 ...] [--stop-after N] [--max-tokens T]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shape_invariant import install, qualify
from fidelity_windows import load_set, verify_snapshot, write_scored_logits, finish_golden

PREFIX = "model.language_model."


FP32_KEYS = ("conv1d", "dt_bias", "A_log", "e_score_correction_bias", "hc.base", "hc.scale")


def run_windows(a, config, ref, dense, experts_src):
    manifest = load_set(a.windows, "glm5_flash")
    identity = verify_snapshot(manifest, a.snapshot)
    proof = qualify(a, manifest, lambda probe: run_windows(probe, config, ref, dense, experts_src))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    started, rows, times, states = time.time(), [], [], []
    with torch.inference_mode():
        embed = dense.get(PREFIX + "embed_tokens.weight")
        for w in manifest["windows"]:
            ids = torch.tensor([w["tokens"]], device="cuda")
            h = torch.nn.functional.embedding(ids, embed)
            states.append((h.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous().cpu(), None))
        del embed, ids, h
        for layer_id in range(config.num_hidden_layers):
            start = time.time()
            kind = config.layer_types[layer_id]
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.Glm5NextTextDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda").eval()
            for name, param in list(layer.named_parameters()) + list(layer.named_buffers()):
                if any(key in name for key in FP32_KEYS):
                    param.data = param.data.float()
            load_layer(layer, dense, experts_src, f"{PREFIX}layers.{layer_id}.")
            for i, w in enumerate(manifest["windows"]):
                host_h, host_topk = states[i]
                h = host_h.cuda()
                topk = host_topk.cuda() if host_topk is not None else None
                positions = torch.arange(len(w["tokens"]), device="cuda")[None]
                mask = torch.ones(1, len(w["tokens"]), dtype=torch.bool, device="cuda")
                # Mirror the official model loop, including its cross-layer DSA indices.
                h, topk = layer(h, attention_mask=mask, position_ids=positions,
                                past_key_values=None, prev_topk_indices=topk)
                if a.layers is not None and layer_id in a.layers:
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
                states[i] = (h.cpu(), topk.cpu() if topk is not None else None)
                del h, topk, positions, mask
            del layer
            torch.cuda.empty_cache()
            times.append(time.time() - start)
            print(f"layer {layer_id} ({kind}) {times[-1]:.1f}s ({len(states)} windows)", flush=True)
        norm = ref.Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps).cuda().to(torch.bfloat16)
        norm.weight.copy_(dense.get(PREFIX + "norm.weight"))
        head = dense.get("lm_head.weight").float()
        for i, w in enumerate(manifest["windows"]):
            h = states[i][0][:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            logits = torch.nn.functional.linear(norm(h.mean(dim=2)).float()[0], head)
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        experts_snapshot=str(a.experts_snapshot or a.snapshot),
        reference="transformers glm5_next (eager, FP32 routed sum, fixed-M128 linears)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        prefix_qualification=proof)


class Weights:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def raw(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name)

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        """BF16/FP32 tensors as stored; FP8 weights times their 128x128 FP32 block scales."""
        value = self.raw(name).to(device)
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
        rows, cols = value.shape
        grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
        return (value.float() * grown).bfloat16()


def checkpoint_names(key: str) -> list[str]:
    """Checkpoint tensors (within a layer) behind one transformers parameter."""
    renames = {
        "attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.forget_gate.f_a_proj.weight": "self_attn.f_a_proj.weight",
        "self_attn.forget_gate.f_b_proj.weight": "self_attn.f_b_proj.weight",
        "self_attn.forget_gate.dt_bias": "self_attn.dt_bias",
        "self_attn.forget_gate.A_log": "self_attn.A_log",
    }
    if key == "self_attn.conv1d.weight":
        return [f"self_attn.{p}_conv1d.weight" for p in "qkv"]
    return [renames.get(key, key)]


def load_layer(layer: torch.nn.Module, dense: Weights, experts_src: Weights, prefix: str) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    loaded = set()
    if experts is not None:
        for e in range(experts.gate_up_proj.shape[0]):
            base = f"{prefix}mlp.experts.{e}."
            with torch.no_grad():
                experts.gate_up_proj[e].copy_(torch.cat([experts_src.get(base + "gate_proj.weight"),
                                                         experts_src.get(base + "up_proj.weight")], 0))
                experts.down_proj[e].copy_(experts_src.get(base + "down_proj.weight"))
        loaded |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    for key, param in list(layer.named_parameters()) + list(layer.named_buffers()):
        if key in loaded:
            continue
        names = [prefix + n for n in checkpoint_names(key)]
        missing = [n for n in names if n not in dense]
        if missing:
            raise KeyError(f"{missing}: no checkpoint tensor for {key}")
        value = torch.cat([dense.get(n) for n in names], 0) if len(names) > 1 else dense.get(names[0])
        with torch.no_grad():
            param.copy_(value.reshape(param.shape).to(param.dtype))


def experts_fp32(self, hidden_states, top_k_index, top_k_weights):
    """Glm5NextTextExperts.forward (clamped SwiGLU) with the routed sum in FP32, one BF16 rounding."""
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert = expert[0]
        slot, token = torch.where(mask[expert])
        current = self._apply_gate(torch.nn.functional.linear(hidden_states[token], self.gate_up_proj[expert]))
        out = torch.nn.functional.linear(current, self.down_proj[expert])
        final.index_add_(0, token, out.float() * top_k_weights[token, slot, None])
    return final.to(hidden_states.dtype)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True, help="coordinator weights, config and tokenizer")
    p.add_argument("--experts-snapshot", type=Path, help="routed experts (FP8 or BF16); default --snapshot")
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--windows", type=Path, help="pinned fidelity set; layer-major scored-row logits")
    p.add_argument("--prefix-only", action="store_true", help="qualify prefix arithmetic without the full panel")
    p.add_argument("--max-tokens", type=int, help="keep the first T tokens")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose streams to save (default all)")
    p.add_argument("--stop-after", type=int, help="run only layers 0..N (no logits)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if a.windows and (a.text or a.text_file or a.max_tokens or a.stop_after is not None):
        p.error("--windows cannot be combined with legacy text/truncation/stop options")

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.models.glm5_next import modeling_glm5_next as ref

    ref.Glm5NextTextExperts.forward = experts_fp32
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    install()
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(a.snapshot).text_config
    config._attn_implementation = "eager"
    if a.windows:
        a.out.mkdir(parents=True, exist_ok=True)
        dense = Weights(a.snapshot)
        experts_src = Weights(a.experts_snapshot) if a.experts_snapshot else dense
        run_windows(a, config, ref, dense, experts_src)
        return
    text = a.text_file.read_text() if a.text_file else a.text
    tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    if a.max_tokens:
        tokens = tokens[:a.max_tokens]
    dense = Weights(a.snapshot)
    experts_src = Weights(a.experts_snapshot) if a.experts_snapshot else dense
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tokens.bin").write_bytes(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    n_layers = config.num_hidden_layers
    layers = n_layers if a.stop_after is None else min(a.stop_after + 1, n_layers)
    save = set(range(layers)) if not a.layers else set(a.layers)
    t = len(tokens)
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(t, device="cuda")[None]
    mask = torch.ones(1, t, dtype=torch.bool, device="cuda")
    fp32_keys = FP32_KEYS
    with torch.inference_mode():
        embed = torch.nn.functional.embedding(ids, dense.get(PREFIX + "embed_tokens.weight"))
        h = embed.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
        for layer_id in range(layers):
            start = time.time()
            kind = config.layer_types[layer_id]
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.Glm5NextTextDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda")
            for name, param in list(layer.named_parameters()) + list(layer.named_buffers()):
                if any(k in name for k in fp32_keys):
                    param.data = param.data.float()
            load_layer(layer, dense, experts_src, f"{PREFIX}layers.{layer_id}.")
            h, _ = layer(h, attention_mask=mask, position_ids=positions, past_key_values=None,
                         prev_topk_indices=None)
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(
                    h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            torch.cuda.empty_cache()
            print(f"layer {layer_id} ({kind}) {time.time() - start:.1f}s", flush=True)
        if layers < n_layers:
            return
        norm = ref.Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps).cuda().to(torch.bfloat16)
        norm.weight.copy_(dense.get(PREFIX + "norm.weight"))
        final = norm(h.mean(dim=2))
        logits = torch.nn.functional.linear(final.float(), dense.get("lm_head.weight").float())
        (a.out / "logits.bin").write_bytes(logits[0].contiguous().cpu().numpy().tobytes())
    argmax = logits[0].argmax(-1)
    next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
    nll = -torch.log_softmax(logits[0, :-1].double(), -1).gather(1, ids[0, 1:, None]).mean().item()
    (a.out / "meta.json").write_text(json.dumps({
        "tokens": t, "snapshot": str(a.snapshot), "experts_snapshot": str(a.experts_snapshot or a.snapshot),
        "reference": "transformers glm5_next (eager, FP32 routed sum)", "argmax_last": int(argmax[-1]),
        "next_token_accuracy": next_ok, "mean_nll": nll,
    }, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}; mean NLL {nll:.4f}")


if __name__ == "__main__":
    main()
