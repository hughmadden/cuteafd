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


def test_cpu_mel_rejects_nonfinite_or_nonmono_pcm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    with pytest.raises(ValueError, match="nonfinite"):
        audio.log_mel(torch.full((481,), float("nan")))
    with pytest.raises(ValueError, match="mono float32"):
        audio.log_mel(torch.zeros(2, 481))
    with pytest.raises(ValueError, match="mono float32"):
        audio.log_mel(torch.zeros(481, dtype=torch.float64))
