"""Release images carry the audio tower by default (AUDIO=auto serves it), and
build.sh forwards the switch to both artifact containers."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD = (REPO / "build.sh").read_text()
ARTIFACTS = (REPO / "scripts/build/build-release-artifacts.sh").read_text()


def test_release_artifacts_configure_audio_from_the_switch():
    assert 'audio_aot="${CUTEAFD_RELEASE_AUDIO_AOT:-ON}"' in ARTIFACTS
    assert '-DCUTEAFD_ENABLE_AUDIO_AOT="$audio_aot"' in ARTIFACTS


def test_build_forwards_the_audio_switch_to_both_roles():
    assert 'audio_aot="${CUTEAFD_RELEASE_AUDIO_AOT:-ON}"' in BUILD
    assert BUILD.count('-e "CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot"') == 2



def test_release_artifacts_validate_the_audio_switch():
    assert 'case "$audio_aot" in ON|OFF) ;;' in ARTIFACTS
    assert 'case "$audio_aot" in ON|OFF) ;;' in BUILD
