"""Qualification-floor checks; no CUDA/checkpoint needed."""
import importlib.util
from pathlib import Path


def _harness():
    path = Path(__file__).resolve().parents[2] / "python/tools/qualify/mimo_v2/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("mimo_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_keeps_strict_fail_and_demands_native_mean_floor():
    gate = _harness()
    native = {
        "relative_l2": 0.014, "mean_cosine": 0.99995,
        "worst_cosine": 0.962, "pass": False,
    }
    official = dict(relative_l2=0.054, mean_cosine=0.99918, worst_cosine=0.914)
    measured = gate.calibrated_metrics(native, official)
    assert measured["pass"] and not measured["strict_pass"]
    assert not gate.calibrated_metrics(dict(native, mean_cosine=0.99949), official)["pass"]
    assert native["pass"] is False


def test_calibration_is_no_worse_than_bf16_even_if_literal_bar_passes():
    gate = _harness()
    native = {
        "relative_l2": 0.011, "mean_cosine": 0.99996,
        "worst_cosine": 0.995, "pass": True,
    }
    official = dict(relative_l2=0.009, mean_cosine=0.99997, worst_cosine=0.999)
    assert not gate.calibrated_metrics(native, official)["pass"]
    assert not gate.calibrated_metrics(
        dict(native, relative_l2=0.02, worst_cosine=0.999), official
    )["pass"]


def test_allocation_gate_uses_ledger_not_cuda_free_memory():
    gate = _harness()
    initial = gate.Ledger(device_allocations=2)
    before = dict(free=1000, total=2000)
    after = dict(free=800, total=2000)
    for sm in ([12, 0], (12, 1)):
        result = gate.allocation_metrics(initial, gate.Ledger(device_allocations=2), before, after, sm)
        assert result["no_encode_device_allocation"]
        assert result["steady_cuda_memory_growth_bytes"] == 200
        assert result["unified_memory_observation"] == (tuple(sm) == (12, 1))
        assert not gate.allocation_metrics(initial, gate.Ledger(device_allocations=3), before, before, sm)["no_encode_device_allocation"]
    assert not gate.allocation_metrics(gate.Ledger(device_allocations=1), initial, before, before, (12, 1))["no_encode_device_allocation"]


def test_rectangular_rgb_patch_order_and_temporal_duplicate():
    import numpy as np

    gate = _harness()
    gh, gw, rgb = gate.fixture(256)
    lut = gate.normalization_lut()
    patches = gate.patches(rgb, lut).reshape(gh * gw, 3, 2, 16, 16)
    np.testing.assert_array_equal(patches[:, :, 0], patches[:, :, 1])
    for patch, y, x in [(0, 0, 0), (1, 0, 16), (2, 16, 0), (3, 16, 16), (4, 0, 32)]:
        expected = np.stack([lut[c, rgb[y:y + 16, x:x + 16, c]] for c in range(3)])
        np.testing.assert_array_equal(patches[patch, :, 0], expected)


def _remote_harness():
    path = Path(__file__).resolve().parents[2] / "python/tools/qualify/mimo_v2/qualify-remote-vision.py"
    spec = importlib.util.spec_from_file_location("mimo_remote_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_identity_sorts_headers_without_reading_tensor_payload(tmp_path):
    import json
    import struct

    gate = _remote_harness()
    headers = {
        "visual.z": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        "visual.a": {"dtype": "BF16", "shape": [3], "data_offsets": [4, 10]},
    }
    shard = tmp_path / "model.safetensors"

    def write(entries):
        data = json.dumps(entries).encode()
        shard.write_bytes(struct.pack("<Q", len(data)) + data)

    write(headers)
    identity = gate.encoder_identity(tmp_path)
    write(dict(reversed(list(headers.items()))))
    assert gate.encoder_identity(tmp_path) == identity
    headers["visual.a"]["data_offsets"] = [5, 11]
    write(headers)
    assert gate.encoder_identity(tmp_path) != identity


def test_remote_cli_rejects_incomplete_or_invalid_modes():
    import pytest

    gate = _remote_harness()
    for extra in ([], ["--address", "host:9000"],
                  ["--address", "host:0", "--output", "result.json"],
                  ["--plan-hash", "z" * 64]):
        with pytest.raises(SystemExit) as error:
            gate.parse_args(["--reference", "reference.npz", *extra])
        assert error.value.code == 2
    args = gate.parse_args(["--reference", "reference.npz", "--snapshot", "snapshot", "--library", "lib.so"])
    assert args.snapshot == Path("snapshot")


def test_remote_wire_checks_byte_identity_and_collects_three_samples(tmp_path, monkeypatch):
    import hashlib
    import json
    import struct
    from types import SimpleNamespace

    import numpy as np
    import pytest

    gate = _remote_harness()
    reference = tmp_path / "reference.npz"
    arrays = {str(n): np.arange(n * 2, dtype=np.uint16) for n in (256, 1024, 4096)}
    np.savez(reference, **arrays)
    reference.with_suffix(".id").write_text("cd" * 32)
    rgb = np.array([1, 2, 3], dtype=np.uint8)
    monkeypatch.setattr(gate, "harness", lambda: SimpleNamespace(fixture=lambda n: (1, n * 4, rgb)))
    header = struct.pack("<8s32s32s4I", b"CAFDVI01", bytes.fromhex("cd" * 32), bytes.fromhex("ab" * 32), 16384, 2, 16, 2)
    key = hashlib.sha256(rgb.tobytes()).digest()
    payload = bytearray(header + bytes(4))
    for repeat in range(4):
        for tokens in ((4096, 1024, 256) if repeat % 2 else (256, 1024, 4096)):
            output = arrays[str(tokens)].tobytes()
            payload.extend(struct.pack("<I32sQQ", 0, key, 1000000, len(output)) + output)

    class Socket:
        def __init__(self, data):
            self.data = bytearray(data)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self, count):
            chunk = bytes(self.data[:min(count, 17)])
            del self.data[:len(chunk)]
            return chunk

        def sendall(self, data):
            pass

        def setsockopt(self, level, option, value):
            assert (level, option, value) == (gate.socket.IPPROTO_TCP, gate.socket.TCP_NODELAY, 1)

    monkeypatch.setattr(gate.socket, "create_connection", lambda *a, **k: Socket(payload))
    args = gate.parse_args(["--reference", str(reference), "--address", "host:9000", "--output", str(tmp_path / "result.json")])
    gate.remote(args)
    result = json.loads(args.output.read_text())
    assert result["byte_exact"]
    assert all(len(samples) == 3 for samples in result["samples"].values())
    payload[-1] ^= 1
    with pytest.raises(AssertionError, match="bytes differ"):
        gate.remote(args)
