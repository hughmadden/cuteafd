#!/usr/bin/env python3
"""Compare the admitted native FP32 audio ABI with the pinned official tower.

The Rust example's arena dump is a diagnostic input, not a required serving
sidecar. This probe loads no LM and cannot qualify end-to-end audio serving.
"""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import hashlib
import json
import re
from pathlib import Path


class CodecBlock(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "q", "qb", "k", "v", "vb", "o", "ob", "norm1", "norm1b", "norm2", "norm2b", "fc1", "fc1b", "fc2", "fc2b")]


class PatchBlock(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "norm1", "norm2", "q", "qb", "k", "kb", "v", "vb", "o", "gate", "up", "down")]


class Spec(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("abi_version", "numerics", "max_samples", "output_width")]
    _fields_ += [(name, C.c_uint64) for name in ("weight_bytes", "conv1", "conv1b", "conv2", "conv2b", "downsample", "norm", "normb", "downnorm", "downnormb")]
    _fields_ += [("codec", CodecBlock * 24), ("codebooks", C.c_uint64 * 20),
                ("speech", C.c_uint64 * 20), ("patch", PatchBlock * 6)]
    _fields_ += [(name, C.c_uint64) for name in ("patch_norm", "projection1", "projection2")]


class Ledger(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "weights", "scratch", "blas_workspace", "fft_workspace", "device_allocations", "encodes")]


Observer = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_int32, C.c_int32)


def native_spec(plan, max_samples):
    if plan["numerics"] != "fp32_qualification_v1":
        raise ValueError("native probe requires the admitted FP32 arena")
    tensors = plan["tensors"]

    def offset(name):
        entry = tensors[name]
        if entry["resident_dtype"] != "F32" or entry["offset"] % 256:
            raise ValueError(f"{name}: invalid native resident tensor")
        return entry["offset"]

    spec = Spec(abi_version=1, numerics=1, max_samples=max_samples,
                output_width=plan["output_width"], weight_bytes=plan["weight_bytes"])
    for name, key in {"conv1": "conv1.weight", "conv1b": "conv1.bias",
        "conv2": "conv2.weight", "conv2b": "conv2.bias", "downsample": "down_sample_layer.0.weight",
        "norm": "layer_norm.weight", "normb": "layer_norm.bias",
        "downnorm": "down_sample_norm.weight", "downnormb": "down_sample_norm.bias"}.items():
        setattr(spec, name, offset("encoder." + key))
    codec = {"q": "self_attn.q_proj.weight", "qb": "self_attn.q_proj.bias",
        "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight", "vb": "self_attn.v_proj.bias",
        "o": "self_attn.out_proj.weight", "ob": "self_attn.out_proj.bias",
        "norm1": "self_attn_layer_norm.weight", "norm1b": "self_attn_layer_norm.bias",
        "norm2": "final_layer_norm.weight", "norm2b": "final_layer_norm.bias",
        "fc1": "fc1.weight", "fc1b": "fc1.bias", "fc2": "fc2.weight", "fc2b": "fc2.bias"}
    for i in range(24):
        for name, key in codec.items():
            setattr(spec.codec[i], name, offset(f"encoder.layers.{i}.{key}"))
    for i in range(20):
        spec.codebooks[i] = offset(f"encoder.quantizer.vq.layers.{i}._codebook.embed")
        spec.speech[i] = offset(f"speech_embeddings.{i}.weight")
    patch = {"norm1": "input_layernorm.weight", "norm2": "post_attention_layernorm.weight",
        "q": "self_attn.q_proj.weight", "qb": "self_attn.q_proj.bias",
        "k": "self_attn.k_proj.weight", "kb": "self_attn.k_proj.bias",
        "v": "self_attn.v_proj.weight", "vb": "self_attn.v_proj.bias", "o": "self_attn.o_proj.weight",
        "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
    for i in range(6):
        for name, key in patch.items():
            setattr(spec.patch[i], name, offset(f"audio_encoder.input_local_transformer.layers.{i}.{key}"))
    spec.patch_norm = offset("audio_encoder.input_local_transformer.norm.weight")
    spec.projection1 = offset("audio_encoder.projection.mlp.0.weight")
    spec.projection2 = offset("audio_encoder.projection.mlp.2.weight")
    return spec


def library(path):
    lib = C.CDLL(str(path))
    signatures = {
        "required": [C.POINTER(Spec), C.POINTER(Ledger)],
        "create": [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)],
        "upload": [C.c_void_p, C.c_void_p, C.c_uint64] + [C.c_void_p] * 4,
        "encode": [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64,
                   C.c_void_p, C.c_uint64, Observer, C.c_void_p],
        "get_ledger": [C.c_void_p, C.POINTER(Ledger)],
        "backend": [C.c_void_p, C.c_uint64], "destroy": [C.c_void_p]}
    for name, arguments in signatures.items():
        function = getattr(lib, "cuteafd_audio_" + name)
        function.restype = C.c_int32
        function.argtypes = arguments
    return lib


def check(status, operation):
    if status:
        raise RuntimeError(f"native audio {operation}: status {status}")


def metric(actual, expected):
    import torch
    actual = torch.as_tensor(actual, dtype=torch.float32).reshape_as(expected)
    expected = expected.detach().cpu().float()
    cosine = torch.nn.functional.cosine_similarity(actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1]), dim=-1)
    return {"max_abs": float((actual-expected).abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected)),
        "mean_cos": float(cosine.mean()), "worst_cos": float(cosine.min()),
        "finite": bool(torch.isfinite(actual).all())}


def stage_views(expected, blocks, geometry):
    """Split the official padded batch/packed rows using its segment lengths."""
    import torch
    views = {}
    mel_offset = codec_offset = code_offset = 0
    for index, frames in enumerate(geometry["segments"]):
        codec_rows = (frames + 1) // 2
        code_rows = (codec_rows + 1) // 2
        views[0, mel_offset] = expected["mel"][mel_offset:mel_offset+frames]
        for stage, name, count in ((1, "conv1", frames), (2, "conv2", codec_rows)):
            views[stage, mel_offset] = torch.nn.functional.gelu(expected[name][index, :, :count]).T
        for stage, tensor in blocks.items():
            views[stage, mel_offset] = tensor[codec_offset:codec_offset+codec_rows]
        views[27, mel_offset] = expected["tokenizer_norm"][codec_offset:codec_offset+codec_rows]
        views[28, mel_offset] = expected["pre_rvq"][code_offset:code_offset+code_rows]
        mel_offset += frames
        codec_offset += codec_rows
        code_offset += code_rows
    views[29, 0] = expected["speech_sum"].reshape(-1, 1024)
    views[30, 0] = expected["local_transformer"].reshape(-1, 1024)
    views[31, 0] = expected["projection"]
    return views


def embedded_tables(header, manifest):
    """Read exact generated F32 literals and verify the manifest's table bytes."""
    import numpy as np
    source = header.read_text()
    tables = {}
    for name, expected_count in (("hann", 960), ("filterbank", 481 * 128),
                                 ("codec_rotary", 3000 * 64), ("patch_rotary", 4 * 64)):
        match = re.search(r"static const float cuteafd_audio_" + name
                          + r"\[(\d+)\] = \{([^}]+)\};", source)
        if match is None or int(match[1]) != expected_count:
            raise ValueError(f"embedded {name}: missing or wrong extent")
        values = [float.fromhex(value.strip()[:-1]) for value in match[2].split(",") if value.strip()]
        table = np.asarray(values, dtype="<f4")
        if table.size != expected_count or not np.isfinite(table).all():
            raise ValueError(f"embedded {name}: invalid values")
        digest = hashlib.sha256(table.tobytes()).hexdigest()
        if digest != manifest["table_sha256"][name]:
            raise ValueError(f"embedded {name}: manifest mismatch")
        tables[name] = table
    if hashlib.sha256(header.read_bytes()).hexdigest() != manifest["artifacts"][header.name]:
        raise ValueError("embedded table header: manifest mismatch")
    return tables


def projection_limit(sm):
    # FP32 cuBLAS/reduction order varies by architecture; SM121's long-row
    # projection reaches 1.04e-6. Codes must still match 100% and tables exactly.
    if sm == 120:
        return 1e-6
    if sm == 121:
        return 2e-6
    raise ValueError("native audio qualification requires SM120 or SM121")


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=72001)
    parser.add_argument("--pcm", type=Path, action="append", default=[])
    parser.add_argument("--boundary-fixtures", action="store_true", help="compare 6000-frame transition and padded odd tails")
    parser.add_argument("--embedded-tables", type=Path,
                        help="qualify all-null native upload and attest this generated audio_tables.h")
    args = parser.parse_args()
    spec_import = importlib.util.spec_from_file_location("audio_cuda_reference", Path(__file__).with_name("audio_cuda_reference.py"))
    diagnostic = importlib.util.module_from_spec(spec_import)
    spec_import.loader.exec_module(diagnostic)
    ref = diagnostic.reference_module()
    if torch.__version__ != diagnostic.TORCH_VERSION or torch.version.cuda != diagnostic.CUDA_VERSION:
        raise ValueError("native probe requires the pinned NGC 26.05 CUDA environment")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((args.arena / "plan.json").read_text())
    spec = native_spec(plan, args.max_samples)
    lib = library(args.library)
    ledger = Ledger()
    check(lib.cuteafd_audio_required(C.byref(spec), C.byref(ledger)), "required")
    if torch.cuda.is_initialized():
        raise ValueError("native ledger query initialized a CUDA context")
    admitted = ledger.weights + ledger.scratch + ledger.blas_workspace + ledger.fft_workspace
    rejected = C.c_void_p()
    check_status = lib.cuteafd_audio_create(C.byref(spec), 0, admitted-1, C.byref(rejected))
    if not check_status or rejected.value or torch.cuda.is_initialized():
        raise ValueError("native admission must fail before context/allocation")
    sm = int("%d%d" % torch.cuda.get_device_capability())
    rel_l2_limit = projection_limit(sm)
    device = torch.device("cuda")
    official, ns = diagnostic.official_processor(args.source_dir, device)
    codec, patch, speech, definitions, provenance = ref.load_modules(args.snapshot)
    codec.to(device); patch.to(device); speech.to(device)
    official(torch.zeros(481, device=device))
    transform = ns["MEL_TRANSFORM"]
    def host(tensor):
        return tensor.detach().cpu().contiguous().numpy()
    hann, filterbank = host(transform.spectrogram.window), host(transform.mel_scale.fb)
    capacity = (min(6000, args.max_samples // 240 + 1) + 1) // 2
    with torch.inference_mode():
        ccos, csin = codec.position_embedding(torch.zeros(1, device=device), torch.arange(capacity, device=device))
        pcos, psin = patch.input_local_transformer.rotary_emb(torch.zeros(1, device=device), torch.arange(4, device=device)[None])
    codec_rotary = host(torch.cat((ccos[:, :32], csin[:, :32]), -1))
    patch_rotary = host(torch.cat((pcos[0, :, :32], psin[0, :, :32]), -1))
    weights = np.memmap(args.arena / "weights.f32", dtype=np.uint8, mode="r")
    if weights.nbytes != spec.weight_bytes:
        raise ValueError("native weight arena extent mismatch")
    owner = C.c_void_p()
    check(lib.cuteafd_audio_create(C.byref(spec), 0, admitted, C.byref(owner)), "create")
    rows = []
    owner_cases = []
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"kind": "native_fp32_tower_probe_not_end_to_end_qualification", "fixtures": rows,
        "sm": sm, "projection_rel_l2_limit": rel_l2_limit,
        "native_library_sha256": ref.digest(args.library.read_bytes()), "arena_plan_sha256": ref.digest((args.arena / "plan.json").read_bytes()),
        "source_sha256": ref.digest(Path(__file__).read_bytes()), "snapshot": str(args.snapshot),
        "admission": {name: getattr(ledger, name) for name, _ in Ledger._fields_}}
    try:
        table_pointers = [table.ctypes.data for table in (hann, filterbank, codec_rotary, patch_rotary)]
        report["table_policy"] = "official_oracle_override"
        if args.embedded_tables:
            manifest = json.loads(args.embedded_tables.with_name("audio_support.json").read_text())
            tables = embedded_tables(args.embedded_tables, manifest)
            report["table_policy"] = "attested_embedded_native_constants"
            report["export_manifest"] = manifest
            report["table_comparison"] = {}
            for name, expected in (("hann", hann), ("filterbank", filterbank),
                                   ("codec_rotary", codec_rotary), ("patch_rotary", patch_rotary)):
                actual = tables[name][:expected.size].reshape(expected.shape)
                report["table_comparison"][name] = {
                    "byte_exact": actual.tobytes() == expected.tobytes(),
                    "different_elements": int(np.count_nonzero(actual != expected)),
                    "max_abs": float(np.max(np.abs(actual - expected)))}
            if not all(row["byte_exact"] for row in report["table_comparison"].values()):
                raise ValueError("embedded constants must exactly match the official tables")
            # A partial table set must reject before publishing any upload.
            partial = lib.cuteafd_audio_upload(owner, weights.ctypes.data, weights.nbytes,
                                               hann.ctypes.data, None, None, None)
            if partial == 0:
                raise ValueError("native upload accepted a partial table set")
            report["partial_table_rejected_status"] = partial
            table_pointers = [None] * 4
        check(lib.cuteafd_audio_upload(owner, weights.ctypes.data, weights.nbytes,
                                      *table_pointers), "upload")
        backend = C.create_string_buffer(256)
        check(lib.cuteafd_audio_backend(backend, len(backend)), "backend")
        report["backend"] = backend.value.decode()
        if args.embedded_tables:
            identity = manifest["export_identity_sha256"]
            payload = {key: value for key, value in manifest.items() if key != "export_identity_sha256"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if identity != digest or not report["backend"].endswith("/export" + identity):
                raise ValueError("native backend does not attest the qualified export manifest")
        cudart = C.CDLL("libcudart.so")
        cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        cudart.cudaMemcpy.restype = C.c_int
        t = torch.arange(24000, dtype=torch.float32) / 24000
        fixtures = {"silence": np.zeros(481, dtype=np.float32),
            "impulse": np.concatenate((np.ones(1, dtype=np.float32), np.zeros(960, dtype=np.float32))),
            "tone": host(0.2 * torch.sin(2 * torch.pi * 440 * t))}
        fixtures.update({str(path): np.fromfile(path, dtype="<f4") for path in args.pcm})
        # Request composition must never become an unstated part of a cache key.
        # The native ABI intentionally encodes one clip at a time.
        batch_checks = []
        native_batch_checks = []
        def native_once(pcm):
            pcm = np.ascontiguousarray(pcm, dtype=np.float32)
            geometry = ref.token_geometry(pcm.size)
            output = np.empty((geometry["tokens"], spec.output_width), dtype=np.float32)
            codes = np.empty((geometry["codes"], 20), dtype=np.int32)
            check(lib.cuteafd_audio_encode(owner, pcm.ctypes.data, pcm.size, output.ctypes.data, output.nbytes,
                codes.ctypes.data, codes.nbytes, Observer(), None), "batch-invariance encode")
            return output, codes
        with torch.inference_mode():
            longer_pcm = torch.zeros(72000, device=device)
            longer_mel = official(longer_pcm)
            for samples in (481, 721, 961, 1201, 24000):
                positions = torch.arange(samples, device=device, dtype=torch.float32) / 24000
                pcm_probe = 0.2 * torch.sin(2 * torch.pi * 440 * positions)
                mel_probe = official(pcm_probe)
                alone = definitions["tokenize_audio_batch"]([mel_probe], codec)[0]
                batched = definitions["tokenize_audio_batch"]([mel_probe, longer_mel], codec)[0]
                batched_first_long = definitions["tokenize_audio_batch"]([longer_mel, mel_probe], codec)[1]
                differences = torch.nonzero(alone != batched).cpu().tolist()
                first_long_differences = torch.nonzero(alone != batched_first_long).cpu().tolist()
                first_rows, first_codes = native_once(host(pcm_probe))
                native_once(host(longer_pcm))
                after_rows, after_codes = native_once(host(pcm_probe))
                native_batch_checks.append({"samples": samples,
                    "codes_byte_exact_after_longer": first_codes.tobytes() == after_codes.tobytes(),
                    "rows_byte_exact_after_longer": first_rows.tobytes() == after_rows.tobytes(),
                    "official_single_clip_rvq_agreement": float(np.mean(first_codes == host(alone)))})
                batch_checks.append({"samples": samples, "mel_frames": mel_probe.shape[0],
                    "official_alone_vs_longer_flips": len(differences),
                    "official_alone_vs_longer_positions": differences,
                    "official_longer_first_flips": len(first_long_differences),
                    "official_longer_first_positions": first_long_differences})
        report["batch_composition"] = {"official": batch_checks,
            "native_policy": "single_clip_no_cross_clip_padding", "native_reuse": native_batch_checks}
        (args.output / "native-reference.json").write_text(json.dumps(report, indent=2) + "\n")
        print("official batch composition", batch_checks, flush=True)
        if args.boundary_fixtures:
            for frames in (6000, 6001, 6003, 6005):
                samples = (frames - 1) * 240
                positions = np.arange(samples, dtype=np.float64) / 24000
                fixtures[f"segment_frames_{frames}"] = (0.2 * np.sin(2 * np.pi * 440 * positions)).astype(np.float32)
            fixtures["short_after_long"] = np.zeros(481, dtype=np.float32)
        if any(pcm.size > args.max_samples for pcm in fixtures.values()):
            raise ValueError("fixture exceeds admitted native sample capacity")
        for name, pcm in fixtures.items():
            geometry = ref.token_geometry(pcm.size)
            expected_blocks = {}
            handles = []
            for i, block in enumerate(codec.layers):
                def capture(module, inputs, output, stage=3+i):
                    expected_blocks[stage] = output.detach().cpu()
                handles.append(block.register_forward_hook(capture))
            try:
                expected = ref.pipeline(torch.from_numpy(pcm).to(device), codec, patch, speech, definitions, mel_fn=official)
            finally:
                for handle in handles:
                    handle.remove()
            expected_stages = stage_views(expected, expected_blocks, geometry)
            stages, callback_errors = {}, []
            @Observer
            def observe(context, stage, segment, data, count, width):
                try:
                    actual = np.empty((count, width), dtype=np.float32)
                    check(cudart.cudaMemcpy(actual.ctypes.data, data, actual.nbytes, 2), "observer copy")
                    key = (stage, segment)
                    stages[f"{stage}@{segment}"] = metric(actual, expected_stages[key])
                    return 0
                except Exception as error:
                    callback_errors.append(str(error))
                    return -2000
            output = np.empty((geometry["tokens"], spec.output_width), dtype=np.float32)
            codes = np.empty((geometry["codes"], 20), dtype=np.int32)
            check(lib.cuteafd_audio_encode(owner, pcm.ctypes.data, pcm.size, output.ctypes.data, output.nbytes,
                codes.ctypes.data, codes.nbytes, observe, None), "encode " + name)
            repeated, repeated_codes = np.empty_like(output), np.empty_like(codes)
            check(lib.cuteafd_audio_encode(owner, pcm.ctypes.data, pcm.size, repeated.ctypes.data, repeated.nbytes,
                repeated_codes.ctypes.data, repeated_codes.nbytes, Observer(), None), "repeat " + name)
            check(lib.cuteafd_audio_get_ledger(owner, C.byref(ledger)), "ledger")
            differences = np.argwhere(codes != host(expected["codes"]))
            result = {"fixture": name, "geometry": geometry, "stages": stages,
                "pcm_sha256": ref.digest(pcm.tobytes()), "rvq_agreement": 1-float(differences.shape[0]/codes.size),
                "rvq_positions_frame_codebook": differences.tolist(), "projection": metric(output, expected["projection"]),
                "byte_deterministic": output.tobytes() == repeated.tobytes() and codes.tobytes() == repeated_codes.tobytes(),
                "callback_errors": callback_errors, "ledger": {key: getattr(ledger, key) for key, _ in Ledger._fields_}}
            p = result["projection"]
            result["passed"] = result["byte_deterministic"] and result["rvq_agreement"] >= 0.995 and p["finite"] and p["rel_l2"] <= 0.03 and p["mean_cos"] >= 0.9995 and p["worst_cos"] >= 0.99 and ledger.device_allocations == 2
            if args.embedded_tables:
                result["exact_embedded_target_passed"] = result["rvq_agreement"] == 1.0 and p["rel_l2"] <= rel_l2_limit
                result["passed"] = result["passed"] and result["exact_embedded_target_passed"]
            rows.append(result)
            if name in {"silence", "tone", "segment_frames_6005", "short_after_long"}:
                bits = output.view(np.uint32)
                bf16 = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
                owner_dir = args.output / "owner-fixtures"
                owner_dir.mkdir(exist_ok=True)
                pcm_name = f"{len(owner_cases)}.pcm.f32"
                (owner_dir / pcm_name).write_bytes(pcm.tobytes())
                owner_cases.append({"fixture": name, "pcm_file": pcm_name,
                                    "bf16_sha256": hashlib.sha256(bf16.tobytes()).hexdigest()})
                (owner_dir / "manifest.json").write_text(json.dumps({
                    "max_samples": args.max_samples, "native_backend": report["backend"],
                    "cases": owner_cases}, indent=2) + "\n")
            (args.output / "native-reference.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, result["rvq_agreement"], result["projection"], "passed", result["passed"], flush=True)
    finally:
        check(lib.cuteafd_audio_destroy(owner), "destroy")
    report["passed"] = bool(rows) and all(row["passed"] for row in rows) and all(
        row["codes_byte_exact_after_longer"] and row["rows_byte_exact_after_longer"]
        and row["official_single_clip_rvq_agreement"] >= (1.0 if args.embedded_tables else 0.995)
        for row in native_batch_checks)
    (args.output / "native-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
