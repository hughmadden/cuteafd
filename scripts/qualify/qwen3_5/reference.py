#!/usr/bin/env python3
"""Official dense Qwen BF16 reference, MTP off, inside coordinator-dev.

Qualification only: this does not launch containers or acquire hardware.
Writes tokens.bin i32 [T], logits.bin f32 [T,V], layerNN.bin BF16 [T,H],
and meta.json, following the family golden file convention.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

DEFAULT_TEXT = """A model checkpoint is a contract between its tensors and its configuration.
A dense transformer applies attention, adds the residual, then applies a gated
feed-forward network and adds a second residual. Recurrent attention also keeps
state: restoring a cached prefix must restore that state exactly, not merely
produce similar output. In Python, a small function can check this invariant:

def restore_and_compare(saved, fresh):
    assert saved.keys() == fresh.keys()
    for name in saved:
        assert saved[name].equal(fresh[name]), name
    return True

The same input tokens should give the same next-token distribution.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")

    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

    config = json.loads((args.snapshot / "config.json").read_text())
    text_config = config.get("text_config", config)
    if config.get("quantization_config") or text_config.get("quantization_config"):
        raise ValueError("BF16 fidelity root required, not a quantized checkpoint")
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (args.snapshot / shard).is_file():
            raise ValueError(f"missing shard: {shard}")
    # Bind the gate-site decision to the executable reference, not a config alias.
    gated_source = inspect.getsource(reference.Qwen3_5RMSNormGated.forward)
    attention_source = inspect.getsource(reference.Qwen3_5Attention.forward)
    source = Path(reference.__file__).read_bytes()
    if ("silu" not in gated_source or "sigmoid(gate)" not in attention_source
            or b"output_gate_type" in source):
        raise ValueError("reference gate implementation changed; reconcile before running")
    if text_config.get("output_gate_type", "swish") not in ("swish", "silu"):
        raise ValueError("unreconciled output_gate_type")
    tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    text = args.text_file.read_text() if args.text_file else DEFAULT_TEXT
    tokens = tokenizer.encode(text, add_special_tokens=False).ids[:args.max_tokens]
    if len(tokens) < 2:
        raise ValueError("reference requires at least two input tokens")
    if args.check_only:
        print(json.dumps({"preflight": "ok", "transformers": transformers.__version__,
                          "tokens": len(tokens), "snapshot": str(args.snapshot)}))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # MTP tensors are not used by the Transformers target implementation.
    model = reference.Qwen3_5ForConditionalGeneration.from_pretrained(
        args.snapshot, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True)
    # HF's dtype=BF16 casts even native FP32 GDN parameters in 5.13.0.
    # Reload only these tiny tensors at their checkpoint dtype before moving.
    from safetensors import safe_open
    fp32_parameters = []
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(args.snapshot / shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if weights.get_slice(name).get_dtype() != "F32":
                    continue
                if not name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight")):
                    raise ValueError(f"unreconciled native FP32 tensor: {name}")
                parameter = model.get_parameter(name)
                parameter.data = weights.get_tensor(name)
                fp32_parameters.append(name)
    model = model.to(device="cuda:0").eval()
    assert all(model.get_parameter(name).dtype == torch.float32 for name in fp32_parameters)
    hooks = []
    def capture(layer_id):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden.detach().contiguous().cpu().view(torch.uint16).numpy().astype("<u2").tofile(
                args.out / f"layer{layer_id:02d}.bin")
        return save
    for layer_id, layer in enumerate(model.model.language_model.layers):
        hooks.append(layer.register_forward_hook(capture(layer_id)))
    ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        result = model(input_ids=ids, use_cache=False)
        logits = result.logits[0].float()
        nll = torch.nn.functional.cross_entropy(logits[:-1].double(), ids[0, 1:]).item()
    for hook in hooks:
        hook.remove()
    if not torch.isfinite(logits).all() or not 0 <= nll < float("inf"):
        raise ValueError("non-finite reference output")
    torch.tensor(tokens, dtype=torch.int32).numpy().astype("<i4").tofile(args.out / "tokens.bin")
    logits.cpu().numpy().astype("<f4").tofile(args.out / "logits.bin")
    meta = {"family": "qwen3_5", "snapshot": str(args.snapshot),
            "revision": args.snapshot.name, "tokens": len(tokens), "vocab": logits.shape[-1],
            "hidden": text_config["hidden_size"], "layers": text_config["num_hidden_layers"],
            "nll": nll, "scored_tokens": len(tokens) - 1, "mtp": False, "dtype": "bf16",
            "reference": "transformers.Qwen3_5ForConditionalGeneration", "transformers": transformers.__version__,
            "native_fp32_parameters": sorted(fp32_parameters),
            "torch": torch.__version__, "reference_sha256": hashlib.sha256(source).hexdigest(),
            "qualification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "gates": {"gdn": "silu", "full_attention": "sigmoid"},
            "attention": "eager", "use_cache": False}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
