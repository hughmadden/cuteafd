"""WP-10a source/geometry guards; optional CPU tensor checks use the oracle venv."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "python/reference/families/mimo_v2/mimo_v26/audio_reference.py"
spec = importlib.util.spec_from_file_location("mimo_audio_reference", PATH)
audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio)


@pytest.mark.parametrize("samples", [0, 1, 479, 480])
def test_short_clips_are_rejected_without_inventing_padding(samples):
    with pytest.raises(ValueError, match=f"audio clip too short: {samples} samples, need > 480"):
        audio.token_geometry(samples)


def test_exact_five_minute_geometry_includes_centered_stft_tail():
    assert audio.token_geometry(7_200_000) == {
        "samples": 7_200_000, "mel_frames": 30_001,
        "segments": [6000, 6000, 6000, 6000, 6000, 1], "codes": 7501, "tokens": 1876,
    }
    with pytest.raises(ValueError, match="exceeds 300 seconds"):
        audio.token_geometry(7_200_001)


@pytest.mark.parametrize("samples,frames,codes,tokens", [
    (481, 3, 1, 1), (961, 5, 2, 1), (24_000, 101, 26, 7),
    (1_439_759, 5999, 1500, 375), (1_439_760, 6000, 1500, 375),
    (1_440_000, 6001, 1501, 376),
])
def test_geometry_segment_boundaries(samples, frames, codes, tokens):
    result = audio.token_geometry(samples)
    assert (result["mel_frames"], result["codes"], result["tokens"]) == (frames, codes, tokens)


def test_source_pins_are_commits_with_byte_digests():
    assert len(audio.SOURCES) == 4
    for source in audio.SOURCES.values():
        assert len(source["commit"]) == 40
        assert len(source["sha256"]) == 64
        assert source["license"] == "Apache-2.0"
    assert audio.MEL_PARAMETERS["power"] == 1.0
    assert audio.MEL_PARAMETERS["window"] == "hann_periodic"
    assert audio.MEL_PARAMETERS["pad_mode"] == "reflect"
    assert audio.MEL_PARAMETERS["mel_scale"] == "htk"
    assert audio.MEL_PARAMETERS["norm"] is None


def test_source_checks_fail_closed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def selected():\n    return 3\n\nraise AssertionError('not executed')\n")
    expected = audio.digest(source.read_bytes())
    text = audio.checked_source(source, expected)
    namespace = audio.definitions(text, ["selected"], {}, str(source))
    assert namespace["selected"]() == 3
    source.write_text("edited")
    with pytest.raises(ValueError, match="pin mismatch"):
        audio.checked_source(source, expected)
    with pytest.raises(ValueError, match="missing official definitions"):
        audio.definitions(text, ["absent"], {}, str(source))


def test_official_group_padding_repeats_last_code():
    torch = pytest.importorskip("torch")
    snapshot = Path("/mnt/sparknest/hf-home/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots") / audio.SNAPSHOTS["flash"]
    if not snapshot.exists():
        pytest.skip("official snapshot not mounted")
    ns = audio.definitions(audio.checked_source(snapshot / "modeling_mimo_v2.py", audio.MODEL_SHA256),
                           ["_pad_and_group_audio_codes"], {"torch": torch}, "official snapshot")
    codes = torch.arange(100).reshape(5, 20)
    grouped = ns["_pad_and_group_audio_codes"](codes, 20, 4)
    assert grouped.shape == (2, 4, 20)
    assert torch.equal(grouped[1], codes[-1:].expand(4, -1))


def test_cuda_geometry_gate_is_decisive_not_energy_error_only():
    path = PATH.with_name("audio_cuda_reference.py")
    spec = importlib.util.spec_from_file_location("mimo_audio_cuda_reference", path)
    cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cuda)
    good = {"mel_byte_exact": True, "byte_deterministic": True, "official_module_byte_exact": True,
            "rvq_agreement": 1.0, "rel_l2": 0.0, "mean_cos": 1.0, "worst_cos": 1.0}
    assert cuda.passes_gate(good)
    for field, value in [("rvq_agreement", 0.99), ("rel_l2", 0.031), ("mean_cos", 0.9994),
                         ("worst_cos", 0.989), ("byte_deterministic", False),
                         ("official_module_byte_exact", False), ("mel_byte_exact", False)]:
        assert not cuda.passes_gate({**good, field: value})
    assert set(cuda.TORCHAUDIO_SOURCES) == {"functional/functional.py", "transforms/_transforms.py"}
    assert all(len(sha) == 64 for sha in cuda.TORCHAUDIO_SOURCES.values())


def test_mp3_floor_acceptance_uses_speech_aggregate_and_clip_cap():
    path = PATH.with_name("audio_cuda_reference.py")
    spec = importlib.util.spec_from_file_location("mimo_audio_cuda_reference", path)
    cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cuda)

    def fixture(name, flips, floor=False, fmt="mp3", licensed=True):
        manifest = ({"kind": "decoder_noise_floor"} if floor else
                    {"format": fmt, "provenance": {"license": "CC-BY-4.0"} if licensed else {}})
        return {"fixture": f"/fixtures/{name}/pcm.f32", "fixture_manifest": manifest,
                "rvq_mismatches": flips, "geometry": {"codes": 76},
                "byte_deterministic": True, "official_module_byte_exact": True,
                "rvq_positions_frame_codebook": [], "projection_row_rel_l2": [0.0],
                "projection_row_cosine": [1.0]}

    rows = [fixture("wav", 0, fmt="wav"), fixture("speech-a", 1),
            fixture("speech-a-ffmpeg-floor", 0, floor=True), fixture("speech-b", 0),
            fixture("speech-b-ffmpeg-floor", 10, floor=True),
            fixture("synthetic-tone", 30, licensed=False)]
    result = cuda.decoded_pcm_acceptance(rows)
    assert result["passed"]
    assert result["speech_mp3_candidate_flips"] == 1
    assert result["speech_mp3_floor_flips"] == 10
    assert result["speech_mp3_clips"][0]["clip_cap"] == 1.52
    assert result["end_to_end_pending"]
    assert not cuda.decoded_pcm_acceptance([*rows[:1], {**rows[1], "rvq_mismatches": 2}, *rows[2:]])["passed"]
    assert not cuda.decoded_pcm_acceptance([{**rows[0], "rvq_mismatches": 1}, *rows[1:]])["passed"]
    assert not cuda.decoded_pcm_acceptance([*rows[:3], {**rows[3], "rvq_mismatches": 10}, *rows[4:]])["passed"]
    with pytest.raises(ValueError, match="missing mainstream MP3 decoder floor"):
        cuda.decoded_pcm_acceptance(rows[:2])


def test_cuda_geometry_uses_checked_official_processor_on_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    import os
    source_dir = Path(os.environ.get("CUTEAFD_MIMO_AUDIO_SOURCE_DIR", "/nonexistent"))
    # Optional installed-source test; no network in the ordinary test suite.
    if not (source_dir / "torchaudio/functional/functional.py").exists():
        pytest.skip("checked torchaudio diagnostic sources unavailable")
    path = PATH.with_name("audio_cuda_reference.py")
    spec = importlib.util.spec_from_file_location("mimo_audio_cuda_reference", path)
    cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cuda)
    tokenizer = source_dir / "tokenizer/mimo_audio_tokenizer/utils.py"
    if not tokenizer.exists():
        pytest.skip("checked Xiaomi diagnostic source unavailable")
    official, ns = cuda.official_processor(source_dir, torch.device("cpu"))
    pcm = torch.zeros(481)
    expected = official(pcm)
    transform = ns["MEL_TRANSFORM"]
    actual = cuda.framed_log_mel(pcm, transform.spectrogram.window, transform.mel_scale.fb)
    assert audio.tensor_bytes(expected) == audio.tensor_bytes(actual)
    assert audio.tensor_bytes(expected) == audio.tensor_bytes(audio.log_mel(pcm))


def load_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_native_abi_and_export_wrapper_fail_closed():
    import ctypes
    native = load_tool("audio_native_probe", PATH.with_name("audio_native_reference.py"))
    exporter = load_tool("audio_exporter", ROOT / "python/tools/aot/export_b12x_audio_aot.py")
    assert ctypes.sizeof(native.Spec) == 3896
    assert ctypes.sizeof(native.CodecBlock) == 120
    assert ctypes.sizeof(native.PatchBlock) == 96
    assert ctypes.sizeof(native.Ledger) == 48
    assert len(exporter.OPERATIONS) == 23
    symbol = "test_audio"
    args = [symbol + "_Kernel_Module_t *module"]
    args += ["void *" + name for name in ("x", "weight", "bias", "out", "aux", "codes")]
    args += ["int32_t rows", "int32_t length", "int32_t offset", "float parameter", "cudaStream_t stream"]
    header = "static inline int32_t cute_dsl_" + symbol + "_wrapper(" + ",".join(args) + ")"
    exporter.check_wrapper(header, symbol)
    with pytest.raises(ValueError, match="unexpected generated audio support ABI"):
        exporter.check_wrapper(header.replace("float parameter", "double parameter"), symbol)


def test_audio_export_identity_is_order_independent_and_covers_table_bytes(tmp_path):
    exporter = load_tool("audio_exporter", ROOT / "python/tools/aot/export_b12x_audio_aot.py")
    first = exporter.write_identity(tmp_path, {"artifacts": {"tables": "a"}, "capability": [12, 0]})
    reordered = exporter.write_identity(tmp_path, {"capability": [12, 0], "artifacts": {"tables": "a"}})
    changed = exporter.write_identity(tmp_path, {"artifacts": {"tables": "b"}, "capability": [12, 0]})
    assert len(first) == 64 and first == reordered and first != changed
    assert changed in (tmp_path / "audio_identity.h").read_text()


def test_audio_embedded_table_generation_is_deterministic_and_attested(tmp_path):
    pytest.importorskip("torch")
    import hashlib
    exporter = load_tool("audio_exporter", ROOT / "python/tools/aot/export_b12x_audio_aot.py")
    native = load_tool("audio_native_probe", PATH.with_name("audio_native_reference.py"))
    hashes = exporter.write_tables(tmp_path)
    header = tmp_path / "audio_tables.h"
    content = header.read_bytes()
    manifest = {"table_sha256": hashes, "artifacts": {header.name: hashlib.sha256(content).hexdigest()}}
    tables = native.embedded_tables(header, manifest)
    assert {name: values.size for name, values in tables.items()} == {
        "hann": 960, "filterbank": 481 * 128, "codec_rotary": 3000 * 64, "patch_rotary": 4 * 64}
    assert exporter.write_tables(tmp_path) == hashes
    assert header.read_bytes() == content
    assert tables["hann"][0] == 0.0 and tables["hann"][480] == 1.0
    manifest["table_sha256"]["hann"] = "0" * 64
    with pytest.raises(ValueError, match="manifest mismatch"):
        native.embedded_tables(header, manifest)


def test_audio_stage_views_split_padded_internal_segments():
    torch = pytest.importorskip("torch")
    native = load_tool("audio_native_probe", PATH.with_name("audio_native_reference.py"))
    geometry = {"segments": [6, 1]}
    expected = {"mel": torch.arange(7), "conv1": torch.zeros(2, 1024, 6),
                "conv2": torch.zeros(2, 1024, 3), "tokenizer_norm": torch.arange(4),
                "pre_rvq": torch.arange(3), "speech_sum": torch.zeros(4, 1024),
                "local_transformer": torch.zeros(4, 1024), "projection": torch.zeros(1, 4096)}
    views = native.stage_views(expected, {3: torch.arange(4)}, geometry)
    assert views[0, 6].tolist() == [6]
    assert views[3, 6].tolist() == [3]
    assert views[28, 6].tolist() == [2]
    assert views[1, 6].shape == (1, 1024)
    assert views[2, 6].shape == (1, 1024)


def test_cpu_mel_rejects_nonfinite_or_nonmono_pcm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    with pytest.raises(ValueError, match="nonfinite"):
        audio.log_mel(torch.full((481,), float("nan")))
    with pytest.raises(ValueError, match="mono float32"):
        audio.log_mel(torch.zeros(2, 481))
    with pytest.raises(ValueError, match="mono float32"):
        audio.log_mel(torch.zeros(481, dtype=torch.float64))
