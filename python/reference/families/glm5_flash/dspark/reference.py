#!/usr/bin/env python3
"""dSpark drafter for GLM 5.3 Flash (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
in plain PyTorch: the oracle the engine's drafter (glmf-golden --draft-oracle) is
compared with.

Semantics are those of the Speculators training model (DSparkDraftModel over
DFlashDraftModel, speculators 0.7) and vLLM's DSparkSpeculator, written as BF16
torch statements:

  taps     the target's aux_hidden_state_layer_ids are vLLM aux ids: id L is the
           input of layer L, i.e. the output of layer L - 1 (20, 28, 32, 36,
           40, 44 -> golden layer19 .. layer43), each the BF16 mean of the
           four mHC streams (FP32 sum in stream order, then / 4)
  context  fused = RMSNorm(fc(concat(taps)), hidden_norm); per draft layer
           K = rope(k_norm(k_proj(fused))), V = v_proj(fused); the block at
           anchor position p sees context positions p - 2048 .. p - 1 (the
           training mask's window, the same for every block row)
  block    [anchor, mask x 7] embeddings at positions p .. p + 7; Qwen3 layers
           (input norm, q/k/v, per-head q/k norm, rotate_half RoPE theta 1e4,
           attention over the context and the block, causal inside the block
           for sliding layers unless sliding_window_non_causal; o_proj;
           residual; post norm; SiLU MLP; residual); final norm
  heads    sample_from_anchor: block row k predicts token p + k + 1.
           base = lm_head(h) in FP32; Markov (vanilla): row k adds
           W2 . W1[prev] with prev the anchor for k = 0, else draft k - 1;
           greedy argmax (lowest index on ties). Confidence: sigmoid(w .
           [h_k, W1[prev_k]] + b) in FP32, the predicted acceptance of row k.

Writes, for every anchor position p in --positions (context = golden taps of
rows 0..p-1, anchor = tokens[p]):
  drafts.bin      u32  [N, 8]
  confidence.bin  f32  [N, 8]
  hidden.bin      bf16 [N, 8, hidden] final-norm output
  meta.json

The same oracle supports SpecForge DSparkDraftModel checkpoints with --target
and --streams 1: target_layer_ids already name post-block outputs, the target
supplies embedding/head, and full-context GQA attention uses checkpoint YaRN.

--score A:B replays every anchor in A..B against golden greedy picks. Beyond
the first proposal this is a conservative teacher-forced prefix diagnostic:
count a deeper match only while previous proposals also equal the recorded
input tokens. Diverged branches require live target verification, not these
teacher-forced logits; this diagnostic is not a serving acceptance benchmark.

  reference.py --draft SNAP --golden DIR --positions 64:1500:16 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: the unit-RMS value is rounded to BF16 before the weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """rotate_half RoPE on [n, heads, dim] with BF16 cos/sin, as transformers."""
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.int64).float() / dim))
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
    half = dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


def target_tensor(snapshot: Path, suffix: str, device: str) -> torch.Tensor:
    """Read only a borrowed, native BF16 embedding/head from a target shard."""
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for name in index if name == suffix or name.endswith("." + suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one target {suffix}, got {names}")
    with safe_open(str(snapshot / index[names[0]]), framework="pt", device=device) as weights:
        tensor = weights.get_tensor(names[0])
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"target {suffix} must be native BF16")
    return tensor


class Drafter:
    def __init__(self, draft: Path, device: str, tap_shift: int = 1, causal: bool | None = None,
                 target: Path | None = None):
        self.cfg = json.loads((draft / "config.json").read_text())
        speculators = self.cfg.get("speculators_model_type") == "dspark"
        if not speculators and "DSparkDraftModel" not in self.cfg.get("architectures", []):
            raise ValueError("not a dSpark checkpoint")
        t = self.cfg["transformer_layer_config"] if speculators else self.cfg
        d = self.cfg if speculators else self.cfg["dflash_config"]
        self.block, self.mask = self.cfg["block_size"], d["mask_token_id"]
        if self.cfg["markov_head_type"] != "vanilla" or not self.cfg["confidence_head_with_markov"]:
            raise ValueError("requires vanilla Markov and Markov confidence")
        if speculators and not self.cfg["sample_from_anchor"]:
            raise ValueError("requires sample_from_anchor")
        self.taps = ([i - tap_shift for i in d["aux_hidden_state_layer_ids"]] if speculators
                     else d["target_layer_ids"])
        self.heads, self.kv_heads, self.head_dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        if self.kv_heads <= 0 or self.heads % self.kv_heads:
            raise ValueError("invalid GQA grouping")
        self.theta, self.eps = float(t["rope_parameters"]["rope_theta"]), float(t["rms_norm_eps"])
        self.window, self.layers, self.hidden = t["sliding_window"], t["num_hidden_layers"], t["hidden_size"]
        sliding = all(kind == "sliding_attention" for kind in t.get("layer_types") or [])
        self.causal = (sliding and not self.cfg.get("sliding_window_non_causal", False)) if causal is None else causal
        self.rope_config = t["rope_parameters"]
        self.device = device
        f = safe_open(str(draft / "model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys() if k != "embed_tokens.weight"}
        if target is None:
            if "embed_tokens.weight" not in f.keys() or "lm_head.weight" not in self.w:
                raise ValueError("drafter borrows target embedding/head: pass --target SNAPSHOT")
            self.embed = f.get_tensor("embed_tokens.weight").cpu() if device != "cpu" else f.get_tensor("embed_tokens.weight")
            self.head = self.w["lm_head.weight"].float()
        else:
            self.embed = target_tensor(target, "embed_tokens.weight", "cpu")
            self.head = target_tensor(target, "lm_head.weight", device).float()
        if self.embed.shape != (t["vocab_size"], self.hidden) or self.head.shape != self.embed.shape:
            raise ValueError("target embedding/head geometry differs from drafter")
        self.w2 = self.w["markov_head.markov_w2.weight"].float()
        self.w1 = self.w["markov_head.markov_w1.weight"]
        self.conf_w = self.w["confidence_head.proj.weight"].float()[0]
        self.conf_b = self.w["confidence_head.proj.bias"].float()[0]

    def rotary(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.rope_config["rope_type"] == "default":
            return rope(x, positions, self.theta)
        if self.rope_config["rope_type"] != "yarn":
            raise ValueError("unsupported draft RoPE")
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        if not hasattr(self, "_rotary"):
            self._rotary = Qwen3RotaryEmbedding(Qwen3Config(**self.cfg)).to(self.device)
        cos, sin = self._rotary(x[None, :, 0], positions[None])
        cos, sin = cos[0, :, None], sin[0, :, None]
        half = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer context K (normed, roped) and V for tap rows [n, taps * hidden]."""
        fused = rms(F.linear(taps, self.w["fc.weight"]), self.w["hidden_norm.weight"], self.eps)
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = F.linear(fused, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            out.append((self.rotary(rms(k, self.w[p + "k_norm.weight"], self.eps), positions), v))
        return out

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Final-norm output [block, hidden] of one block at `position`."""
        b = self.block
        n_ctx = ctx[0][0].shape[0]
        mask = torch.ones(b, n_ctx + b, dtype=torch.bool, device=self.device)
        if self.causal:
            mask[:, n_ctx:] = torch.ones(b, b, dtype=torch.bool, device=self.device).tril()
        tokens = torch.tensor([anchor] + [self.mask] * (b - 1))
        h = self.embed[tokens].to(self.device)
        positions = torch.arange(position, position + b, device=self.device)
        for l in range(self.layers):
            p = f"layers.{l}."
            a = p + "self_attn."
            n = rms(h, self.w[p + "input_layernorm.weight"], self.eps)
            q = F.linear(n, self.w[a + "q_proj.weight"]).view(b, self.heads, self.head_dim)
            k = F.linear(n, self.w[a + "k_proj.weight"]).view(b, self.kv_heads, self.head_dim)
            v = F.linear(n, self.w[a + "v_proj.weight"]).view(b, self.kv_heads, self.head_dim)
            q = self.rotary(rms(q, self.w[a + "q_norm.weight"], self.eps), positions)
            k = self.rotary(rms(k, self.w[a + "k_norm.weight"], self.eps), positions)
            kc, vc = ctx[l]
            keys, values = torch.cat([kc, k]), torch.cat([vc, v])
            if self.heads != self.kv_heads:
                keys = keys.repeat_interleave(self.heads // self.kv_heads, dim=1)
                values = values.repeat_interleave(self.heads // self.kv_heads, dim=1)
            # HF Qwen3 casts FP32 softmax probabilities to the query dtype before PV.
            scores = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask[None], -math.inf)
            probs = scores.softmax(-1).to(q.dtype)
            attn = torch.einsum("hqk,khd->qhd", probs, values).to(h.dtype).reshape(b, -1)
            h = h + F.linear(attn, self.w[a + "o_proj.weight"])
            n = rms(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
            m = p + "mlp."
            act = F.silu(F.linear(n, self.w[m + "gate_proj.weight"])) * F.linear(n, self.w[m + "up_proj.weight"])
            h = h + F.linear(act, self.w[m + "down_proj.weight"])
        return rms(h, self.w["norm.weight"], self.eps)

    def heads_(self, anchor: int, hidden: torch.Tensor) -> tuple[list[int], list[float]]:
        """Markov-biased greedy chain over the block rows and the confidence per row."""
        logits = F.linear(hidden.float(), self.head)
        previous, tokens, confidence = anchor, [], []
        for k in range(self.block):
            e = self.w1[previous]
            score = logits[k] + F.linear(e.float(), self.w2)
            best = int(torch.argmax(score))  # first index of the maximum
            c = torch.dot(self.conf_w[:self.hidden], hidden[k].float()) + \
                torch.dot(self.conf_w[self.hidden:], e.float()) + self.conf_b
            confidence.append(float(torch.sigmoid(c)))
            tokens.append(best)
            previous = best
        return tokens, confidence


def parse_positions(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b, *s = (int(x) for x in part.split(":"))
            out.extend(range(a, b, s[0] if s else 1))
        else:
            out.append(int(part))
    return out


def golden_taps(golden: Path, layers: list[int], hidden: int, device: str, streams: int = 4) -> torch.Tensor:
    taps = []
    for layer in layers:
        raw = torch.frombuffer(bytearray((golden / f"layer{layer:02d}.bin").read_bytes()), dtype=torch.bfloat16)
        # FP32 sum of the four streams in order, then the division, rounded to BF16.
        if streams == 1:
            taps.append(raw.view(-1, hidden))
        elif streams == 4:
            values = raw.view(-1, 4, hidden).float()
            taps.append(((((values[:, 0] + values[:, 1]) + values[:, 2]) + values[:, 3]) / 4).to(torch.bfloat16))
        else:
            raise ValueError("expected one dense or four mHC streams")
    return torch.cat(taps, -1).to(device)


def greedy_picks(golden: Path, tokens: int, vocab: int) -> list[int]:
    meta = json.loads((golden / "meta.json").read_text()) if (golden / "meta.json").exists() else {}
    start = meta.get("logits_start", 0)
    rows, remainder = divmod((golden / "logits.bin").stat().st_size, vocab * 4)
    if remainder or start < 0 or rows > tokens - start:
        raise ValueError("golden logits do not match vocabulary/position metadata")
    out = [-1] * tokens
    with open(golden / "logits.bin", "rb") as f:
        for position in range(start, start + rows):
            row = torch.frombuffer(bytearray(f.read(vocab * 4)), dtype=torch.float32)
            out[position] = int(torch.argmax(row))
    return out


def live_decode(drafter, evaluate, prompt, max_new_tokens, eos_tokens, *, markov=True):
    """Exact causal full-prefix replay: no recurrent-cache crop approximation."""
    if not prompt or max_new_tokens < 1:
        raise ValueError("live decoding needs a nonempty prompt and positive token budget")
    prefix = list(prompt)
    taps, logits = evaluate(prefix, [len(prefix) - 1])
    anchor = int(logits[0].argmax())
    emitted, lengths = [], []
    while len(emitted) < max_new_tokens:
        position = len(prefix)
        first = max(0, position - drafter.window) if drafter.window is not None else 0
        rows = torch.arange(first, position, device=drafter.device)
        context = drafter.context(taps[first:position], rows)
        hidden = drafter.body(anchor, position, context)
        proposals = (drafter.heads_(anchor, hidden)[0] if markov else
                     F.linear(hidden, drafter.live_head).argmax(-1).tolist())
        width = min(drafter.block, max_new_tokens - len(emitted) - 1)
        verify_tokens = prefix + [anchor] + proposals[:width]
        verified_taps, logits = evaluate(verify_tokens, list(range(position, position + width + 1)))
        posterior = logits.argmax(-1).tolist()
        accepted = 0
        while accepted < width and proposals[accepted] == posterior[accepted]:
            accepted += 1
        committed = [anchor] + proposals[:accepted]
        stop = next((i + 1 for i, token in enumerate(committed) if token in eos_tokens), len(committed))
        committed = committed[:stop]
        lengths.append(len(committed))
        emitted.extend(committed)
        prefix.extend(committed)
        # Retain only causal target outputs of committed input rows. The next
        # replay recomputes the complete recurrent state, including GDN state.
        taps = verified_taps[:len(prefix)]
        if committed[-1] in eos_tokens:
            break
        anchor = posterior[accepted]
    return {"tokens": emitted, "accepted_lengths": lengths,
            "completion_tokens": len(emitted), "spec_verify_ct": len(lengths),
            "spec_accept_length": len(emitted) / len(lengths)}


def run_live(args):
    """Compare explicit Markov proposals and a bundled-style base-head chain."""
    import hashlib
    import importlib.util
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[5]
    path = root / "scripts/qualify/qwen3_5/reference.py"
    spec = importlib.util.spec_from_file_location("dense_live_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model, fp32 = reference.load_reference(args.target)
    from shape_invariant import install, install_eager
    from transformers.models.qwen3_5 import modeling_qwen3_5
    install()
    install_eager(modeling_qwen3_5)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    drafter = Drafter(args.draft, args.device, target=args.target)
    language = model.model.language_model
    drafter.live_head = model.lm_head.weight
    if drafter.hidden != language.embed_tokens.weight.shape[1]:
        raise ValueError("target language-model embedding differs from drafter")

    def evaluate(tokens, positions):
        captured, hooks = {}, []
        def hook(layer):
            def save(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if value.dtype != torch.bfloat16 or value.ndim != 3:
                    raise ValueError("live taps require BF16 post-block target outputs")
                captured[layer] = value[0].detach()
            return save
        for layer in drafter.taps:
            hooks.append(language.layers[layer].register_forward_hook(hook(layer)))
        try:
            ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
            keep = torch.tensor(positions, dtype=torch.long, device=args.device)
            output = model(input_ids=ids, use_cache=False, logits_to_keep=keep)
            logits = output.logits[0].float()
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite live target logits")
            return torch.cat([captured[layer] for layer in drafter.taps], -1), logits
        finally:
            for handle in hooks:
                handle.remove()

    if args.inspect_prefix:
        fixture = json.loads(args.inspect_prefix.read_text())
        ids = tokenizer.apply_chat_template([{"role": "user", "content": fixture["prompt"]}], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=True, return_dict=False)
        prefix = ids + fixture["prefix"]
        with torch.inference_mode():
            _, logits = evaluate(prefix, [len(prefix) - 1])
            values, indices = logits[0].topk(2)
        report = {"generated_index": len(fixture["prefix"]), "input_tokens": len(prefix),
                  "top2_tokens": indices.tolist(), "top2_logits": values.tolist(),
                  "top2_margin": float(values[0] - values[1]),
                  "base_token": fixture["base_token"], "markov_token": fixture["markov_token"]}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "margin.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    prompts = args.prompt or [
        "Write a Python function that returns the first n Fibonacci numbers. Explain edge cases.",
        "A shop has 12 boxes with 8 pencils each and sells 37 pencils. How many remain?",
        "Explain why a hash table has average constant-time lookups and give one limitation.",
    ]
    results = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                add_generation_prompt=True, enable_thinking=True, return_dict=False)
            arms = {}
            for markov in (False, True):
                arm = "markov" if markov else "base_head"
                arms[arm] = live_decode(drafter, evaluate, ids, args.max_new_tokens, eos, markov=markov)
                arms[arm]["text"] = tokenizer.decode(arms[arm]["tokens"])
                print(json.dumps({"prompt": len(results), "arm": arm, **arms[arm]}), flush=True)
            if arms["markov"]["tokens"] != arms["base_head"]["tokens"]:
                raise ValueError("live speculative arms differ in emitted greedy tokens")
            results.append({"prompt": prompt, "prompt_tokens": len(ids), "arms": arms})
    report = {"mode": "live", "thinking": True, "temperature": 0, "target_dtype": "bf16",
              "target_revision": args.target.name, "draft_revision": args.draft.name,
              "target_replay": "full-prefix-use_cache-false", "taps": drafter.taps,
              "native_fp32_parameters": fp32, "torch": torch.__version__,
              "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_head_semantics": "bundled-style proposals, explicit exact target replay (not spec_generate)",
              "requests": results}
    for arm in ("base_head", "markov"):
        completed = sum(result["arms"][arm]["completion_tokens"] for result in results)
        verifies = sum(result["arms"][arm]["spec_verify_ct"] for result in results)
        report[arm] = {"completion_tokens": completed, "spec_verify_ct": verifies,
                       "spec_accept_length": completed / verifies,
                       "request_mean": sum(result["arms"][arm]["spec_accept_length"] for result in results) / len(results)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "live.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({arm: report[arm] for arm in ("base_head", "markov")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--live", action="store_true", help="greedy chat/thinking proposals with exact target replay")
    parser.add_argument("--prompt", action="append", help="live user prompt; may repeat (default three short prompts)")
    parser.add_argument("--inspect-prefix", type=Path, help="live prompt/generated-prefix JSON for one target top-two margin")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="bounded live completion length")
    parser.add_argument("--target", type=Path, help="target snapshot supplying the borrowed embedding/head")
    parser.add_argument("--streams", type=int, choices=(1, 4), default=4, help="dense=1, GLM mHC=4")
    parser.add_argument("--positions", help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--score", help="A:B: teacher-forced acceptance vs the golden greedy picks")
    parser.add_argument("--tap-shift", type=int, default=1, help="target layer = aux id - shift (vLLM: 1)")
    parser.add_argument("--causal", type=int, choices=(0, 1), help="override in-block causality")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true", help="CPU header/config/golden preflight; no weights loaded")
    args = parser.parse_args()
    if args.live:
        if args.target is None or args.out is None or args.max_new_tokens < 1:
            parser.error("--live requires --target, --out and a positive --max-new-tokens")
        if args.golden or args.positions or args.score or args.check_only:
            parser.error("--live cannot be combined with golden replay options")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        run_live(args)
        return
    if args.golden is None:
        parser.error("golden replay requires --golden")
    if args.positions and args.out is None:
        parser.error("--positions requires --out")
    if args.check_only:
        cfg = json.loads((args.draft / "config.json").read_text())
        speculators = cfg.get("speculators_model_type") == "dspark"
        if not speculators and "DSparkDraftModel" not in cfg.get("architectures", []):
            raise ValueError("not a dSpark checkpoint")
        t = cfg["transformer_layer_config"] if speculators else cfg
        d = cfg if speculators else cfg["dflash_config"]
        taps = ([i - args.tap_shift for i in d["aux_hidden_state_layer_ids"]] if speculators else d["target_layer_ids"])
        if t["num_key_value_heads"] <= 0 or t["num_attention_heads"] % t["num_key_value_heads"]:
            raise ValueError("invalid GQA grouping")
        with safe_open(str(args.draft / "model.safetensors"), framework="pt", device="cpu") as weights:
            expected = {"fc.weight": [t["hidden_size"], len(taps) * t["hidden_size"]],
                        "markov_head.markov_w1.weight": [t["vocab_size"], cfg["markov_rank"]],
                        "markov_head.markov_w2.weight": [t["vocab_size"], cfg["markov_rank"]],
                        "confidence_head.proj.weight": [1, t["hidden_size"] + cfg["markov_rank"]]}
            for name, shape in expected.items():
                view = weights.get_slice(name)
                if view.get_shape() != shape or view.get_dtype() != "BF16":
                    raise ValueError(f"drafter tensor {name} disagrees with config")
            if not {"embed_tokens.weight", "lm_head.weight"}.issubset(weights.keys()) and args.target is None:
                raise ValueError("drafter borrows target embedding/head: pass --target")
        if args.target:
            index = json.loads((args.target / "model.safetensors.index.json").read_text())["weight_map"]
            for suffix in ("embed_tokens.weight", "lm_head.weight"):
                names = [name for name in index if name == suffix or name.endswith("." + suffix)]
                if len(names) != 1:
                    raise ValueError(f"expected one target {suffix}")
                with safe_open(str(args.target / index[names[0]]), framework="pt", device="cpu") as weights:
                    view = weights.get_slice(names[0])
                    if view.get_dtype() != "BF16" or view.get_shape() != [t["vocab_size"], t["hidden_size"]]:
                        raise ValueError(f"target tensor {suffix} disagrees with drafter")
        print(json.dumps({"preflight": "ok", "taps": taps, "head_dim": t["head_dim"],
                          "query_rows": t["num_attention_heads"] // t["num_key_value_heads"] * cfg["block_size"],
                          "rope": t["rope_parameters"], "window": t["sliding_window"]}))
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.draft, args.device, args.tap_shift, None if args.causal is None else bool(args.causal), args.target)
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    taps = golden_taps(args.golden, drafter.taps, drafter.hidden, args.device, args.streams)
    if taps.shape[0] != len(tokens):
        raise ValueError("golden tap rows differ from token count")
    rows = torch.arange(taps.shape[0], device=args.device)
    with torch.no_grad():
        ctx = drafter.context(taps, rows)

        def draft(p: int) -> tuple[list[int], list[float], torch.Tensor]:
            first = max(0, p - drafter.window) if drafter.window is not None else 0
            h = drafter.body(int(tokens[p]), p, [(k[first:p], v[first:p]) for k, v in ctx])
            t, c = drafter.heads_(int(tokens[p]), h)
            return t, c, h

        if args.score:
            a, b = (int(x) for x in args.score.split(":"))
            b = min(b, len(tokens) - drafter.block + 1)
            if not 0 <= a < b:
                raise ValueError("score range is empty or out of bounds")
            greedy = greedy_picks(args.golden, len(tokens), drafter.head.shape[0])
            if any(greedy[p] < 0 for p in range(a, b + drafter.block - 1)):
                raise ValueError("score range includes positions not present in golden logits")
            kept_total, first_ok, conf_sum, hist = 0, 0, [0.0] * drafter.block, [0] * (drafter.block + 1)
            for p in range(a, b):
                t, c, _ = draft(p)
                kept = 0
                while kept < drafter.block and t[kept] == greedy[p + kept] and \
                        (kept == 0 or t[kept - 1] == int(tokens[p + kept])):
                    kept += 1
                kept_total += kept
                hist[kept] += 1
                first_ok += int(t[0] == greedy[p])
                conf_sum = [s + x for s, x in zip(conf_sum, c)]
            n = b - a
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / "score.json").write_text(json.dumps(dict(anchors=n, block=drafter.block,
                    taps=drafter.taps, causal=drafter.causal, mean_accepted=kept_total / n,
                    first_draft=first_ok / n, histogram=hist, mean_confidence=[s / n for s in conf_sum],
                    score_from=a, score_until=b, markov=True, confidence_with_markov=True), indent=2) + "\n")
            print(f"score taps {drafter.taps} causal {drafter.causal}: {n} anchors, accepted vs greedy "
                  f"{kept_total / n:.3f} of {drafter.block}, first draft {100 * first_ok / n:.1f}%, histogram {hist}, "
                  f"mean confidence {[round(s / n, 3) for s in conf_sum]}")
        if args.positions:
            positions = parse_positions(args.positions)
            if not positions or any(p < 0 or p >= len(tokens) for p in positions):
                raise ValueError("anchor positions must be nonempty and within golden tokens")
            drafts, confidence, hidden = [], [], []
            for p in positions:
                t, c, h = draft(p)
                drafts.append(t)
                confidence.append(c)
                hidden.append(h.cpu())
            args.out.mkdir(parents=True, exist_ok=True)
            torch.tensor(drafts, dtype=torch.int32).numpy().tofile(args.out / "drafts.bin")
            torch.tensor(confidence, dtype=torch.float32).numpy().tofile(args.out / "confidence.bin")
            torch.stack(hidden).view(torch.int16).numpy().tofile(args.out / "hidden.bin")
            truth = [[int(tokens[p + 1 + i]) if p + 1 + i < len(tokens) else -1 for i in range(drafter.block)]
                     for p in positions]
            accepted = [next((i for i, (d, t) in enumerate(zip(ds, ts)) if d != t), len(ds))
                        for ds, ts in zip(drafts, truth)]
            (args.out / "meta.json").write_text(json.dumps({"positions": positions, "taps": drafter.taps,
                "causal": drafter.causal, "accepted_vs_tokens": accepted,
                "mean_accepted": sum(accepted) / len(accepted)}, indent=1))
            print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}")


if __name__ == "__main__":
    main()
