"""An sm_89 coordinator slot builds the daemon and the base native library only."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()
ARTIFACTS = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()


def _run(script):
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)


def _arch_check():
    start = WIP.index('  local phase="${1:-all}" coordinator_arch=')
    end = WIP.index('must be 120 or 89"\n', start) + len('must be 120 or 89"\n')
    return ('release_die() { echo "DIE $*"; exit 9; }\ncheck() {\n' + WIP[start:end]
            + '  echo "$coordinator_arch"\n}\n')


@pytest.mark.parametrize('setting,expected', [(None, '120'), ('120', '120'), ('89', '89')])
def test_coordinator_arch_defaults_to_sm120(setting, expected):
    env = '' if setting is None else f'CUTEAFD_WIP_COORDINATOR_ARCH={setting} '
    result = _run(_arch_check() + env + 'check\n')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == expected


def test_unknown_coordinator_arch_is_refused():
    result = _run(_arch_check() + 'CUTEAFD_WIP_COORDINATOR_ARCH=86 check\n')
    assert result.returncode == 9 and 'must be 120 or 89' in result.stdout


def test_sm89_native_phase_takes_no_gpu_turn_and_builds_its_own_arch():
    build = WIP[WIP.index('build_coordinator() {'):]
    assert '/wip/source coordinator "$coordinator_arch" /wip/build/coordinator' in build
    assert '"${wip_export_locks:-off}" != on || "$coordinator_arch" == 89' in build


LOCALS = ['coordinator_aot', 'v41_expert_aot', 'exl3_aot', 'nvfp4_aot', 'audio_aot', 'exlr8_aot']
FAMILY_SWITCHES = ['DSV4', 'GLM', 'GLMF', 'MIMO', 'QWEN4', 'QWEN_DENSE', 'STEP5', 'KIMIK3', 'MUSE', 'ML4',
                   'VISION_ATTENTION']


@pytest.mark.parametrize('role,arch,disabled', [
    ('coordinator', 89, True), ('coordinator', 120, False), ('expert', 121, False)])
def test_sm89_coordinator_turns_every_aot_switch_off(role, arch, disabled):
    start = ARTIFACTS.index('v41_expert_aot=ON\n')
    end = ARTIFACTS.index('# <<< sm89 base-only coordinator', start)
    preset = ' '.join(f'{name}=ON' for name in LOCALS if name != 'v41_expert_aot')
    preset += ' ' + ' '.join(f'CUTEAFD_WIP_{name}_AOT=ON' for name in FAMILY_SWITCHES)
    report = 'printf "<%s> <%s>" "$expert_families" "$bf16_families"\n'
    report += ''.join(f'printf " %s" "${name}"\n' for name in LOCALS)
    report += ''.join(f'printf " %s" "$CUTEAFD_WIP_{name}_AOT"\n' for name in FAMILY_SWITCHES)
    script = (f'set -u\nrole={role} cuda_arch={arch} expert_families=dsv4f:fp8 bf16_families=glm {preset}\n'
              + ARTIFACTS[start:end] + report)
    result = _run(script)
    assert result.returncode == 0, result.stderr
    words = result.stdout.split()
    state = 'OFF' if disabled else 'ON'
    assert words[:2] == (['<>', '<>'] if disabled else ['<dsv4f:fp8>', '<glm>'])
    assert words[2:] == [state] * (len(LOCALS) + len(FAMILY_SWITCHES))
    assert ('base native library only' in result.stderr) == disabled


def test_v41_expert_aot_reaches_cmake_and_its_manifest_has_an_off_form():
    assert '-DCUTEAFD_ENABLE_V41_EXPERT_AOT="$v41_expert_aot"' in ARTIFACTS
    assert '-DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON' not in ARTIFACTS
    assert '\'{"schema":1,"enabled":false}\' >"$output_dir/V41_EXPERT_AOT.json"' in ARTIFACTS
