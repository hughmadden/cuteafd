#!/usr/bin/env python3
"""Golden activations for DeepSeek V4 from the official inference/model.py.

Runs the reference one Block at a time on one GPU (the full model does not
fit), with kernel.py replaced by kernel_torch and fixed-width, position-ordered
index selection. Saves the embedding,
every requested layer's output stream [1, tokens, hc, dim] and the logits.

  golden.py --snapshot SNAP --text "..." --out DIR [--layers 0 1 2 ...] [--device 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from types import SimpleNamespace
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shape_invariant import install, qualify
from fidelity_windows import (CheckpointStorage, LayerCheckpoints, WindowCheckpoint,
                              admit_reference_memory, log_reference_cuda_memory, release_reference_page_cache,
                              rss_bytes, canonical, release_checkpoint, load_set,
                              verify_snapshot, write_scored_logits, finish_golden, log_checkpoint_reads)

HERE = Path(__file__).resolve().parent


def import_reference(snapshot: Path):
    spec = importlib.util.spec_from_file_location("kernel", HERE / "kernel_torch.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    sys.modules["kernel"] = kernel
    fht = types.ModuleType("fast_hadamard_transform")
    fht.hadamard_transform = kernel.hadamard_transform
    sys.modules["fast_hadamard_transform"] = fht
    spec = importlib.util.spec_from_file_location("v4_reference", snapshot / "inference" / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    from shape_invariant import install_index_topk
    install_index_topk(model)
    return model


class Weights:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}
        self.read_bytes = 0
        self.read_seconds = 0.0

    def get(self, name: str) -> torch.Tensor:
        start = time.monotonic()
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        value = self.files[shard].get_tensor(name).clone()
        self.read_bytes += value.numel() * value.element_size()
        self.read_seconds += time.monotonic() - start
        return value

    def names(self, prefix: str) -> list[str]:
        return [n for n in self.index if n.startswith(prefix)]


def reference_name(name: str) -> str:
    """The key the official convert.py would give a checkpoint tensor."""
    if name.startswith("model."):
        name = name[len("model."):]
    return (name.replace("self_attn", "attn").replace("mlp", "ffn")
            .replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias"))


def load_module(module: torch.nn.Module, weights: Weights, prefix: str) -> None:
    """Load `prefix`* tensors into `module` with convert.py's transformations:
    wo_a is dequantized to BF16 with its 128x128 scales (and the scale dropped),
    packed FP4 is reinterpreted, everything else is copied (with dtype casts)."""
    params = dict(module.named_parameters())
    missing = set(params)
    by_reference = {reference_name(n): n for n in weights.index}
    for ref, name in by_reference.items():
        if not ref.startswith(prefix):
            continue
        key = ref[len(prefix):]
        if key.endswith("wo_a.scale"):
            continue
        value = weights.get(name)
        if key.endswith("wo_a.weight"):
            scale = weights.get(name[: -len("weight")] + "scale").float()
            value = (value.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
                     * scale[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()
        if key not in params:
            raise KeyError(f"checkpoint tensor {name} has no parameter {key}")
        param = params[key]
        if value.dtype != param.dtype and value.element_size() == param.element_size() and value.shape == param.shape:
            value = value.view(param.dtype)  # packed fp4 (I8) and e8m0 bit patterns
        elif value.dtype in (torch.int8, torch.uint8) and param.dtype == torch.float4_e2m1fn_x2:
            value = value.view(torch.float4_e2m1fn_x2)
        with torch.no_grad():
            param.data.copy_(value.to(param.device) if value.dtype == param.dtype else value.to(param.device, param.dtype))
        missing.discard(key)
    if missing:
        raise KeyError(f"{prefix}: parameters without checkpoint tensors: {sorted(missing)[:8]}")
    release_checkpoint(torch.cuda, weights)


def install_compressed_slots(ref, max_tokens):
    """Keep non-indexed compressed attention reductions at a panel-fixed width."""
    original = getattr(ref, "_reference_compress_topk", ref.get_compress_topk_idxs)
    ref._reference_compress_topk = original

    def fixed(ratio, bsz, seqlen, start_pos, offset):
        ids = original(ratio, bsz, seqlen, start_pos, offset)
        slots = max_tokens // ratio
        if ids.shape[-1] > slots:
            raise ValueError("compressed attention exceeds panel slot extent")
        return torch.nn.functional.pad(ids, (0, slots - ids.shape[-1]), value=-1)

    fixed.cache_clear = original.cache_clear
    ref.get_compress_topk_idxs = fixed


def initial_runtime_buffers(block):
    return [(module, name, value.detach().cpu().clone())
            for module in block.modules()
            for name, value in module._buffers.items()
            if name in module._non_persistent_buffers_set and name != "freqs_cis" and value is not None]


def reset_runtime_buffers(initial):
    # Compressor aliases point into these buffers: reset in place, never replace them.
    for module, name, value in initial:
        module._buffers[name].copy_(value)


def run_windows(a, ref, args, weights):
    manifest = load_set(a.windows, "deepseek_v4")
    identity = verify_snapshot(manifest, a.snapshot)
    if not getattr(a, "_prefix_probe", False):
        a._compressed_slots_max_tokens = max(len(w["tokens"]) for w in manifest["windows"])
    install_compressed_slots(ref, a._compressed_slots_max_tokens)
    replay = getattr(a, "qualify_resume", None)
    proof = None if replay else qualify(a, manifest, lambda probe: run_windows(probe, ref, args, weights))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    args.max_seq_len = max(256, max(len(w["tokens"]) for w in manifest["windows"]))
    started, rows, times, states = time.time(), [], [], []
    checkpoints = None
    streaming = getattr(a, "stream_window_states", False) and not getattr(a, "_prefix_probe", False)
    peaks = json.loads(a.layer_memory_peaks.read_text()) if getattr(a, "layer_memory_peaks", None) else {}
    measured = {}
    if getattr(a, "resume_source_seal_sha256", None) and not replay and not getattr(a, "_prefix_probe", False):
        import hashlib
        receipt = json.loads(a.resume_qualification.read_text())
        pointer = json.loads((a.resume_layers / "latest.json").read_text())
        expected = {"set_sha256": manifest["set_sha256"], "snapshot_identity": identity,
                    "source_seal_sha256": a.resume_source_seal_sha256}
        if (receipt.get("complete") is not True or receipt.get("kind") != "resume replay diagnostic"
                or receipt.get("layer_id") != 23 or "b11" not in receipt.get("windows", {})
                or len(receipt["windows"]) != 2 or receipt.get("binding") != expected
                or receipt.get("recovery_source_seal_sha256") != a.source_seal_sha256
                or pointer.get("layer") != "layer23"
                or receipt.get("target_seal_sha256") != pointer.get("seal_sha256")
                or peaks != receipt.get("measured_peaks") or set(peaks) != {"4", "128"}):
            raise ValueError("qualified CUDA resume/source migration evidence differs")
    if replay:
        manifest = dict(manifest, windows=[w for w in manifest["windows"] if w["id"] in a.resume_windows])
        if len(manifest["windows"]) != len(a.resume_windows):
            raise ValueError("unknown resume qualification window")
    if getattr(a, "checkpoint_layers", False) and not getattr(a, "_prefix_probe", False):
        shapes = {w["id"]: [1, len(w["tokens"]), args.hc_mult, args.dim] for w in manifest["windows"]}
        binding = {"set_sha256": manifest["set_sha256"], "snapshot_identity": identity,
                   "source_seal_sha256": a.source_seal_sha256}
        resume_binding = dict(binding, source_seal_sha256=a.resume_source_seal_sha256) if getattr(a, "resume_source_seal_sha256", None) else binding
        checkpoints = LayerCheckpoints(a.out / "layer-checkpoints", binding, shapes,
                                       resume=getattr(a, "resume_layers", None), streaming=streaming,
                                       resume_binding=resume_binding, retain_previous=streaming)
    first_layer = 0
    with torch.inference_mode():
        if checkpoints is not None and checkpoints.resumed is not None:
            last, arrays, times = checkpoints.resumed
            if not 0 <= last < args.n_layers or len(times) != last + 1:
                raise ValueError("invalid resumed layer extent")
            states = arrays if streaming else [torch.from_numpy(bits).view(torch.bfloat16) for bits in arrays]
            checkpoints.resumed = None
            del arrays
            first_layer = last + 1
        else:
            embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
            load_module(embed, weights, "embed.")
            folder = checkpoints.begin(-1) if streaming else None
            for w in manifest["windows"]:
                ids = torch.tensor([w["tokens"]], dtype=torch.long)
                h = embed(ids).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
                if streaming:
                    cpu = h.cpu()
                    states.append(checkpoints.write_window(folder, w["id"], cpu.view(torch.uint16).numpy()))
                    del cpu
                    release_checkpoint(torch.cuda)
                else:
                    states.append(h.cpu())
            del embed, ids, h
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(first_layer, args.n_layers):
            start = time.time()
            read_start, read_before = time.monotonic(), weights.read_bytes
            category = str(args.compress_ratios[layer_id])
            baseline_rss = rss_bytes()
            guarded = streaming or replay or getattr(a, "page_cache_request", None)
            if guarded:
                memory.release()
                log_reference_cuda_memory(torch.cuda, f"layer {layer_id} before cache release")
                if getattr(a, "page_cache_request", None):
                    release_reference_page_cache(a.page_cache_request, a.page_cache_response,
                                                 f"layer {layer_id} admission")
                # Replay supplies measured worst-window peaks for both C4/C128 layers.
                required = peaks.get(category, measured.get(category, 90 * 2**30))
                log_reference_cuda_memory(torch.cuda, f"layer {layer_id} admission")
                admit_reference_memory(torch.cuda, required)
                torch.cuda.reset_peak_memory_stats()
            folder = checkpoints.begin(layer_id) if streaming else None
            block = ref.Block(layer_id, args)
            if guarded:
                import weakref
                retired_block = weakref.ref(block)
            load_module(block, weights, f"layers.{layer_id}.")
            log_checkpoint_reads(f"layer {layer_id} load", (weights,), read_before, read_start)
            if guarded and getattr(a, "page_cache_request", None):
                release_reference_page_cache(a.page_cache_request, a.page_cache_response,
                                             f"layer {layer_id} weights resident")
                log_reference_cuda_memory(torch.cuda, f"layer {layer_id} loaded")
            block.eval()
            initial = initial_runtime_buffers(block)
            for i, w in enumerate(manifest["windows"]):
                reset_runtime_buffers(initial)
                ids = torch.tensor([w["tokens"]], dtype=torch.long)
                cpu = torch.from_numpy(states[i].load()).view(torch.bfloat16) if streaming else states[i]
                h = block(cpu.cuda(), 0, ids)
                if a.layers is not None and layer_id in a.layers:
                    saved_folder = a.out / "windows" / w["id"]
                    saved_folder.mkdir(parents=True, exist_ok=True)
                    torch.save(h.cpu(), saved_folder / f"layer{layer_id:02d}.pt")
                if streaming:
                    result = h.cpu()
                    states[i] = checkpoints.write_window(folder, w["id"], result.view(torch.uint16).numpy())
                    del result
                else:
                    states[i] = h.cpu()
                if guarded:
                    measured[category] = max(measured.get(category, 0), torch.cuda.max_memory_reserved()
                                             + max(0, rss_bytes() - baseline_rss))
                del h, ids, cpu
                if streaming:
                    release_checkpoint(torch.cuda)
            # Top-k helper lru caches also own CUDA tensors from the last visit.
            ref.get_window_topk_idxs.cache_clear()
            ref.get_compress_topk_idxs.cache_clear()
            del initial, block
            memory.release()
            if guarded:
                if retired_block() is not None:
                    raise RuntimeError("reference layer remains live after retirement")
                log_reference_cuda_memory(torch.cuda, f"layer {layer_id} retired")
                if getattr(a, "page_cache_request", None):
                    release_reference_page_cache(a.page_cache_request, a.page_cache_response,
                                                 f"layer {layer_id} retired")
                    log_reference_cuda_memory(torch.cuda, f"layer {layer_id} retired cache released")
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            if checkpoints is not None:
                checkpoints.commit(layer_id, states if streaming else [h.view(torch.uint16).numpy() for h in states], times)
                memory.release()
                memory.check(f"layer {layer_id} checkpoint")
            print(f"layer {layer_id} {times[-1]:.1f}s ({len(states)} windows)", flush=True)
            if streaming or replay:
                (a.out / "layer-memory-peaks.json").write_bytes(canonical(measured) + b"\n")
            if replay and getattr(a, "resume_telemetry_layers", None) == layer_id + 1:
                print("Resume telemetry pilot complete; CUDA equality not qualified", flush=True)
                return
            if replay and layer_id == 23:
                import hashlib
                pointer = json.loads((replay / "latest.json").read_text())
                target = replay / pointer["layer"]
                seal_bytes = (target / "seal.json").read_bytes()
                if pointer["layer"] != "layer23" or hashlib.sha256(seal_bytes).hexdigest() != pointer["seal_sha256"]:
                    raise ValueError("resume qualification target seal differs")
                seal = json.loads(seal_bytes)
                expected_binding = {"set_sha256": manifest["set_sha256"], "snapshot_identity": identity,
                                    "source_seal_sha256": a.resume_source_seal_sha256}
                if seal["binding"] != expected_binding:
                    raise ValueError("resume qualification target identity differs")
                matches = {}
                for w, state in zip(manifest["windows"], states):
                    saved = WindowCheckpoint(target / (w["id"] + ".bin"), list(state.shape), seal["files"][w["id"]])
                    expected = saved.load()
                    actual = state.view(torch.uint16).numpy()
                    changed = int((actual != expected).sum())
                    delta = (state.float() - torch.from_numpy(expected).view(torch.bfloat16).float()).abs()
                    if not torch.isfinite(delta).all():
                        raise ValueError("nonfinite CUDA resume replay diagnostic: " + w["id"])
                    matches[w["id"]] = dict(saved.entry, changed_elements=changed,
                                            max_abs_diff=float(delta.max()), byte_exact=changed == 0)
                    del delta
                    del expected, actual
                (a.out / "resume-qualification.json").write_bytes(canonical(dict(
                    complete=True, kind="resume replay diagnostic",
                    byte_exact=all(entry["byte_exact"] for entry in matches.values()), layer_id=23, windows=matches,
                    target_seal_sha256=pointer["seal_sha256"], binding=expected_binding,
                    recovery_source_seal_sha256=a.source_seal_sha256, measured_peaks=measured)) + b"\n")
                return
        hc_fn = weights.get("hc_head_fn").cuda().float()
        hc_scale = weights.get("hc_head_scale").cuda().float()
        hc_base = weights.get("hc_head_base").cuda().float()
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        # hc_head needs only scalar config, not an unused full expert layer.
        tail = SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps)
        for i, w in enumerate(manifest["windows"]):
            cpu = torch.from_numpy(states[i].load()).view(torch.bfloat16) if streaming else states[i]
            h = cpu[:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            h = ref.Block.hc_head(tail, h, hc_fn, hc_scale, hc_base)
            logits = head(norm(h), full_logits=True)[0].float()
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits, cpu
            if streaming:
                release_checkpoint(torch.cuda)
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        reference="official inference/model.py (kernel_torch, fixed-M128 linears; official math, order/shape-invariant index top-k and panel-fixed masked compressed slots)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        prefix_qualification=proof,
        **({"resume_replay_diagnostic": receipt} if getattr(a, "resume_layers", None)
           and getattr(a, "resume_source_seal_sha256", None) and not getattr(a, "_prefix_probe", False) else {}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--tokens", type=int, nargs="+", help="explicit token ids")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--windows", type=Path, help="pinned fidelity set; layer-major scored logits")
    p.add_argument("--prefix-only", action="store_true")
    p.add_argument("--checkpoint-layers", action="store_true", help="rolling hash-sealed local layer states")
    p.add_argument("--resume-layers", type=Path, help="resume into a fresh output from a complete layer checkpoint tree")
    p.add_argument("--source-seal-sha256", help="verified immutable source seal for checkpoint/resume identity")
    p.add_argument("--stream-window-states", action="store_true", help="keep complete layer states on NVMe, not anonymous RAM")
    p.add_argument("--resume-source-seal-sha256", help="original checkpoint source identity for qualified source migration")
    p.add_argument("--qualify-resume", type=Path, help="replay through layer23 against this immutable checkpoint tree")
    p.add_argument("--resume-windows", nargs="+", default=["a00", "b11"])
    p.add_argument("--resume-telemetry-layers", type=int, help="nonqualifying bounded replay memory pilot")
    p.add_argument("--layer-memory-peaks", type=Path, help="measured unified-memory peaks by compression ratio")
    p.add_argument("--resume-qualification", type=Path, help="byte-exact CUDA replay receipt required for source migration")
    p.add_argument("--page-cache-request", type=Path, help="host runner cache-release request directory")
    p.add_argument("--page-cache-response", type=Path, help="host runner cache-release acknowledgement directory")
    a = p.parse_args()
    if bool(a.page_cache_request) != bool(a.page_cache_response) or (a.page_cache_request and not a.windows):
        p.error("page-cache coordination requires both directories and --windows")
    if a.resume_telemetry_layers is not None and (not a.qualify_resume or not 1 <= a.resume_telemetry_layers <= 23):
        p.error("resume telemetry requires --qualify-resume and 1..23 layers")
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if a.checkpoint_layers and (not a.windows or a.prefix_only or not a.source_seal_sha256
            or len(a.source_seal_sha256) != 64 or any(c not in "0123456789abcdef" for c in a.source_seal_sha256)):
        p.error("--checkpoint-layers requires full --windows and a verified source seal SHA256")
    if a.resume_layers and not a.checkpoint_layers:
        p.error("--resume-layers requires --checkpoint-layers")
    if a.stream_window_states and not a.checkpoint_layers:
        p.error("--stream-window-states requires --checkpoint-layers")
    if a.resume_source_seal_sha256 and not a.qualify_resume and (not a.resume_layers
            or not a.resume_qualification or not a.layer_memory_peaks or not a.stream_window_states):
        p.error("source migration requires streamed resume plus CUDA equality receipt and measured peaks")
    if a.qualify_resume and (not a.windows or not a.resume_source_seal_sha256 or not a.source_seal_sha256
                             or a.resume_layers or a.checkpoint_layers):
        p.error("--qualify-resume requires a panel and both source identities, without checkpoint/resume")
    if a.windows and (a.text is not None or a.tokens is not None):
        p.error("--windows cannot be combined with legacy text/tokens")

    torch.cuda.set_device(a.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    install()
    ref = import_reference(a.snapshot)
    from shape_invariant import bounded_sparse
    ref.sparse_attn = bounded_sparse(ref.sparse_attn)
    config = json.loads((a.snapshot / "inference" / "config.json").read_text())
    if a.windows:
        manifest = load_set(a.windows, "deepseek_v4")
        tokens = max(manifest["windows"], key=lambda w: len(w["tokens"]))["tokens"]
    elif a.tokens:
        tokens = a.tokens
    else:
        from tokenizers import Tokenizer
        tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(a.text, add_special_tokens=False).ids
    args = ref.ModelArgs(**config, max_batch_size=1, max_seq_len=max(len(tokens), 256))
    # What Transformer.__init__ sets for a single rank.
    ref.world_size, ref.rank = 1, 0
    ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    ref.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    ref.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

    weights = Weights(a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.windows:
        run_windows(a, ref, args, weights)
        return
    ids = torch.tensor([tokens], dtype=torch.long)
    save = set(range(args.n_layers)) if a.layers is None or not a.layers else set(a.layers)
    with torch.inference_mode():
        embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
        load_module(embed, weights, "embed.")
        h = embed(ids)
        torch.save(h.cpu(), a.out / "embed.pt")
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        del embed
        memory = CheckpointStorage(torch.cuda, weights)
        for layer in range(args.n_layers):
            start = time.time()
            block = ref.Block(layer, args)
            load_module(block, weights, f"layers.{layer}.")
            block.eval()
            h = block(h, 0, ids)
            if layer in save:
                torch.save(h.cpu(), a.out / f"layer{layer:02d}.pt")
            del block
            memory.release()
            memory.check(f"layer {layer}")
            print(f"layer {layer} {time.time() - start:.1f}s", flush=True)
        hc_fn = weights.get("hc_head_fn").cuda().float()
        hc_scale = weights.get("hc_head_scale").cuda().float()
        hc_base = weights.get("hc_head_base").cuda().float()
        tail = SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps)
        h = ref.Block.hc_head(tail, h, hc_fn, hc_scale, hc_base)
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        logits = head(norm(h), full_logits=True)
        torch.save(logits.float().cpu(), a.out / "logits.pt")
    (a.out / "meta.json").write_text(json.dumps({"tokens": tokens, "snapshot": str(a.snapshot),
                                                  "argmax_last": int(logits[0, -1].argmax())}, indent=1))
    print("argmax of last position:", int(logits[0, -1].argmax()))


if __name__ == "__main__":
    main()
