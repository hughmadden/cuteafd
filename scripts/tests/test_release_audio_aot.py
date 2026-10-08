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


def test_spark_remote_leg_receives_the_audio_switch():
    # The Spark leg runs build.sh's REMOTE heredoc under `set -u` on the seed host:
    # every variable it reads must be assigned there or passed as an argument.
    block = BUILD.split('echo "== building Spark development and inference images natively on $seed_host =="', 1)[1]
    invocation, remote = block.split("<<'REMOTE'", 1)
    remote = remote.split("\nREMOTE\n", 1)[0]
    assert invocation.rstrip().endswith('"$audio_aot"'), "the Spark leg must pass the audio switch"
    assert 'audio_aot="${21:-ON}"' in remote
    assert remote.index('audio_aot="${21:-ON}"') < remote.index('CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot')
