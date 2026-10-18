"""Optional real-weight component fixtures for the native Step-5 dense chain."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "step5_component_golden", ROOT / "python/reference/families/step5/golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

SEED = 274
ROWS = (1, 7, 128)


def test_official_layer0_dense_fixtures():
    destination = os.environ.get("CUTEAFD_STEP5_FFN_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_STEP5_FFN_FIXTURE_DIR for real-weight CUDA fixtures")
    torch = pytest.importorskip("torch")
    snapshot = Path(os.environ["STEP5_REFERENCE_SNAPSHOT"])
    root = Path(destination).resolve()
    build_root = Path(os.environ.get("CARGO_TARGET_DIR", Path.home() / ".cache/cuteafd/builds")).resolve()
    assert root.is_relative_to(build_root), "fixtures require the guarded NVMe build bind mount"
    root.mkdir(parents=True, exist_ok=True)
    ref, _ = golden.load_release(snapshot)
    values = json.loads((snapshot / "config.json").read_text())
    config_values = dict(values.get("text_config", values))
    config_values["tp_size"] = 1
    config = ref.Step5Config(**config_values)
    with torch.device("meta"):
        layer = ref.Step5DecoderLayer(config, 0, dtype=torch.bfloat16)
    # Only this subchain is materialized. Attention remains meta, not a fake oracle.
    layer.post_attention_layernorm.to_empty(device="cuda")
    layer.mlp.to_empty(device="cuda")
    reader = golden.Weights(snapshot)
    sources = {}
    with torch.no_grad():
        for name, parameter in list(layer.named_parameters()):
            if not (name.startswith("mlp.") or name.startswith("post_attention_layernorm.")):
                continue
            value = reader.get("model.layers.0." + name)
            assert value.dtype == parameter.dtype and value.shape == parameter.shape
            sources[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            parameter.copy_(value)
    reader.files.clear()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    residual = torch.randn((max(ROWS), config.hidden_size), generator=generator, dtype=torch.float32).cuda()
    manifest = {
        "schema": 1, "checkpoint": golden.CHECKPOINT, "revision": golden.REVISION,
        "source_pins": golden.SOURCE_PINS, "seed": SEED, "layer": 0, "rows": list(ROWS),
        "epsilon": config.rms_norm_eps, "limit": layer.mlp.limit, "weights_sha256": sources,
        "torch": torch.__version__, "cuda": torch.version.cuda, "files": {},
        "primary": "official_native", "secondary": "official_f32",
        "secondary_activation": "official_cpu_f32_branch_cuda_bf16_only",
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }

    def save(name, tensor):
        value = tensor.detach().contiguous().cpu()
        data = value.view(torch.uint8).numpy().tobytes()
        (root / (name + ".bin")).write_bytes(data)
        manifest["files"][name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                   "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with torch.inference_mode():
        for oracle in ("official_native", "official_f32"):
            if oracle == "official_f32":
                layer.mlp.float()
            for rows in ROWS:
                x = residual[:rows].contiguous()
                hidden = layer.post_attention_layernorm(x).to(layer.mlp.gate_proj.weight.dtype)
                gate = layer.mlp.gate_proj(hidden)
                up = layer.mlp.up_proj(hidden)
                if oracle == "official_native":
                    activated = ref.clamped_swiglu(gate, up, layer.mlp.limit)
                else:
                    # The shipped CUDA activation explicitly accepts BF16 only.
                    # Its official CPU branch supplies the secondary F32 arithmetic.
                    activated = ref.clamped_swiglu(gate.cpu(), up.cpu(), layer.mlp.limit).cuda()
                down = layer.mlp.down_proj(activated)
                if oracle == "official_native":
                    # Boundary capture must reproduce the shipped module directly.
                    assert torch.equal(down, layer._feed_forward(hidden))
                    gate64 = torch.nn.functional.linear(hidden.double(), layer.mlp.gate_proj.weight.double())
                    up64 = torch.nn.functional.linear(hidden.double(), layer.mlp.up_proj.weight.double())
                    down64 = torch.nn.functional.linear(activated.double(), layer.mlp.down_proj.weight.double())
                    save(f"fp64_r{rows}_gate_up", torch.cat((gate64, up64), dim=-1))
                    save(f"fp64_r{rows}_down", down64)
                for name, tensor in {"residual": x, "norm": hidden,
                                     "gate_up": torch.cat((gate, up), dim=-1),
                                     "activation": activated, "down": down,
                                     "output": x + down.float()}.items():
                    save(f"{oracle}_r{rows}_{name}", tensor)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
