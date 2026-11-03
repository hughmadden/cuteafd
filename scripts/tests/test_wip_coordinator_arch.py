"""An sm_89 coordinator slot carries DeepSeek V4.1's SM89 exports, made on an SM89 GPU."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()
ARTIFACTS = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()


def _run(script, env=None):
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30,
                          env={**os.environ, **(env or {})})


def _arch_check():
    start = WIP.index('wip_coordinator_arch="${CUTEAFD_WIP_COORDINATOR_ARCH:-120}"\n')
    end = WIP.index('must be 120 or 89"\n', start) + len('must be 120 or 89"\n')
    return ('release_die() { echo "DIE $*"; exit 9; }\ncheck() {\n' + WIP[start:end]
            + '  echo "$wip_coordinator_arch"\n}\n')


@pytest.mark.parametrize('setting,expected', [(None, '120'), ('120', '120'), ('89', '89')])
def test_coordinator_arch_defaults_to_sm120(setting, expected):
    env = '' if setting is None else f'CUTEAFD_WIP_COORDINATOR_ARCH={setting} '
    result = _run(_arch_check() + env + 'check\n')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == expected


def test_unknown_coordinator_arch_is_refused_before_any_operation():
    result = _run(_arch_check() + 'CUTEAFD_WIP_COORDINATOR_ARCH=86 check\n')
    assert result.returncode == 9 and 'must be 120 or 89' in result.stdout
    assert WIP.index('must be 120 or 89') < WIP.index('if ((dry_run)); then')
    assert 'echo "  coordinator arch: sm_$wip_coordinator_arch"' in WIP


def _tools(tmp_path, caps=None):
    """docker records its calls; ssh answers nvidia-smi with CAPS (exit 255 when None)."""
    tools = tmp_path / 'tools'
    tools.mkdir()
    ssh = ('printf "ssh %s\\n" "$*" >> "$TRACE"\n'
           + ('exit 255\n' if caps is None else f'printf "{caps}"\n'))
    for name, body in {'docker': 'printf "docker %s\\n" "$*" >> "$TRACE"\nprintf "image-id\\n"\n',
                       'ssh': ssh}.items():
        (tools / name).write_text('#!/bin/sh\n' + body)
        (tools / name).chmod(0o755)
    return {'PATH': f'{tools}:{os.environ["PATH"]}', 'TRACE': str(tmp_path / 'trace')}


def _modes():
    return WIP[WIP.index('# wip_export_host_is_sm89 HOST'):WIP.index('\n# wip_remote_export:')]


def _build_coordinator(tmp_path, arch, host='', caps=None):
    function = WIP[WIP.index('build_coordinator() {'):WIP.index('\nbuild_expert() (')]
    script = f'''set -euo pipefail
repo_root={str(ROOT)!r}
sync_local_source() {{ :; }}
release_die() {{ echo "DIE $*" >&2; exit 2; }}
cuteafd_compiler_cache_warn() {{ :; }}
native_store_sync() {{ :; }}
native_store_has() {{ return 1; }}
wip_capped_export() {{ printf "capped %s\\n" "$*" >> "$TRACE"; }}
wip_remote_export() {{ printf "remote\\n" >> "$TRACE"; return 75; }}
slot=test coordinator_container=owned COORDINATOR_DOCKER_DEV=image bf16_families= audio_aot=OFF
wip_export_locks=on wip_coordinator_arch={arch} CUTEAFD_EXPORT_HOST={host}
''' + _modes() + function + '\nbuild_coordinator\n'
    env = _tools(tmp_path, caps)
    result = _run(script, env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result, Path(env['TRACE']).read_text().splitlines()


@pytest.mark.parametrize('arch', ['120', '89'])
def test_native_phase_takes_the_export_gpu_turn_for_its_own_arch(tmp_path, arch):
    result, lines = _build_coordinator(tmp_path, arch)
    command = f'/wip/source/scripts/build/build-wip-artifacts.sh /wip/source coordinator {arch} /wip/build/coordinator /wip/output/coordinator'
    rust = next(line for line in lines if line.startswith('docker exec') and line.endswith(command + ' rust'))
    assert 'CUTEAFD_WIP_EXPORT_LOCKS=off' in rust
    # The native phase runs only inside the capped GPU turn, with this arch's command.
    capped = [line for line in lines if line.startswith('capped ')]
    assert len(capped) == 1 and capped[0].endswith(command), capped
    assert not any(line.endswith(command + ' native') for line in lines)
    assert 'done by capped' in result.stdout and not any(line.startswith('ssh ') for line in lines)


@pytest.mark.parametrize('caps,remote', [('12.0\\n', False), (None, False), ('8.9\\n12.0\\n', False),
                                         ('8.9\\n8.9\\n', True)])
def test_sm89_export_uses_a_remote_host_only_when_it_is_sm89(tmp_path, caps, remote):
    result, lines = _build_coordinator(tmp_path, '89', host='maxq', caps=caps)
    assert any(line.startswith('ssh ') and 'compute_cap' in line for line in lines)
    assert ('remote' in lines) == remote
    assert ('not an SM89 export host' in result.stderr) != remote
    assert 'done by capped' in result.stdout


def test_sm120_export_host_is_used_without_an_arch_probe(tmp_path):
    result, lines = _build_coordinator(tmp_path, '120', host='maxq', caps='12.0\\n')
    assert lines.count('remote') == 1 and not any(line.startswith('ssh ') for line in lines)
    assert 'done by capped' in result.stdout


@pytest.mark.parametrize('arch,host,caps,refused', [
    ('89', '', None, True), ('89', 'maxq', '12.0\\n', True), ('89', 'ada', '8.9\\n', False),
    ('120', '', None, False)])
def test_remote_only_mode_needs_an_sm89_host_for_sm89(tmp_path, arch, host, caps, refused):
    script = ('release_die() { echo "DIE $*"; exit 9; }\n' + _modes()
              + f'\nCUTEAFD_WIP_COORDINATOR_EXPORT=remote CUTEAFD_EXPORT_HOST={host} wip_coordinator_export_modes {arch}\n')
    result = _run(script, _tools(tmp_path, caps))
    if refused:
        assert result.returncode == 9 and 'needs an SM89 export host' in result.stdout
    else:
        assert result.returncode == 0 and result.stdout.split() == ['remote', 'wait-remote']


SWITCHES = ['v41_candidate_aot', 'exl3_aot', 'nvfp4_aot', 'audio_aot', 'exlr8_aot']
FAMILY_SWITCHES = ['DSV4', 'GLM', 'GLMF', 'MIMO', 'QWEN4', 'QWEN_DENSE', 'STEP5', 'KIMIK3', 'MUSE', 'ML4',
                   'VISION_ATTENTION']


@pytest.mark.parametrize('role,arch', [('coordinator', 89), ('coordinator', 120), ('expert', 121)])
def test_sm89_coordinator_keeps_the_v41_exports_and_turns_the_rest_off(role, arch):
    start = ARTIFACTS.index('# The V4.1 coordinator candidates')
    end = ARTIFACTS.index('# <<< sm89 V4.1 coordinator', start)
    coordinator_aot = 'ON' if role == 'coordinator' else 'OFF'
    preset = ' '.join(f'{name}=ON' for name in SWITCHES if name != 'v41_candidate_aot')
    preset += ' ' + ' '.join(f'CUTEAFD_WIP_{name}_AOT=ON' for name in FAMILY_SWITCHES)
    report = 'printf "<%s> <%s> %s" "$expert_families" "$bf16_families" "$coordinator_aot"\n'
    report += ''.join(f'printf " %s" "${name}"\n' for name in SWITCHES)
    report += ''.join(f'printf " %s" "$CUTEAFD_WIP_{name}_AOT"\n' for name in FAMILY_SWITCHES)
    script = (f'set -u\nrole={role} cuda_arch={arch} coordinator_aot={coordinator_aot} '
              f'expert_families=dsv4f:fp8 bf16_families=glm {preset}\n' + ARTIFACTS[start:end] + report)
    result = _run(script)
    assert result.returncode == 0, result.stderr
    words = result.stdout.split()
    sm89 = (role, arch) == ('coordinator', 89)
    assert words[:2] == (['<>', '<>'] if sm89 else ['<dsv4f:fp8>', '<glm>'])
    # The FP8 bundle (with router and index scorer) and the local, TP2 and dSpark
    # TP2 experts follow coordinator_aot, which an sm_89 coordinator keeps ON.
    assert words[2] == coordinator_aot
    expected = 'OFF' if sm89 else 'ON'
    assert words[3:] == [coordinator_aot if not sm89 else 'OFF'] + [expected] * (len(SWITCHES) - 1 + len(FAMILY_SWITCHES))
    assert ('DeepSeek V4.1 exports only' in result.stderr) == sm89


def test_v41_switches_reach_cmake():
    cmake = ARTIFACTS[ARTIFACTS.index('\ncmake \\\n'):ARTIFACTS.index('-DCUTEAFD_CUDA_ARCHITECTURES="$cuda_arch"')]
    for flag in ('V41_FP8_AOT', 'V41_LOCAL_EXPERT_AOT', 'V41_TP2_EXPERT_AOT'):
        assert f'-DCUTEAFD_ENABLE_{flag}="$coordinator_aot"' in cmake, flag
    for flag in ('V41_ATTENTION_AOT', 'V41_HC_LAGGED_AOT', 'V41_NARROW_AOT'):
        assert f'-DCUTEAFD_ENABLE_{flag}="$v41_candidate_aot"' in cmake, flag
    assert '-DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON' in cmake
    assert '-DCUTEAFD_ENABLE_V41_NVFP4_AOT="$nvfp4_aot"' in cmake
    # Every slot ships the exported dSpark coordinator manifest; there is no disabled form.
    assert 'install -m 0644 "$build_dir/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"' in ARTIFACTS
    assert '"enabled":false}\' >"$output_dir/V41_EXPERT_AOT.json"' not in ARTIFACTS


def test_v41_cmake_admits_sm89_for_every_enabled_role_and_refuses_the_rest():
    cmake = ROOT / 'native/cmake/families/deepseek_v41'
    assert 'CUTEAFD_CUDA_ARCHITECTURES STREQUAL "89"' in (cmake / 'v41_fp8.cmake').read_text()
    experts = (cmake / 'v41_experts.cmake').read_text()
    assert 'elseif(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "89")\n' in experts
    for role in ('v41_local_experts', 'v41_tp2_experts', 'v41_dspark_tp2_experts'):
        assert 'SM120 or SM89 coordinator expert build' in (cmake / f'{role}.cmake').read_text(), role
    assert 'CUTEAFD_V41_ROUTER_CC' in (cmake / 'v41_router.cmake').read_text()
    # The switches this block turns off name SM120 (or SM121) only.
    assert 'requires SM120' in (cmake / 'v41_attention.cmake').read_text()
    assert 'single native SM120 or SM121 target' in (cmake / 'v41_nvfp4_experts.cmake').read_text()


def test_coordinator_build_key_names_a_non_default_arch_only():
    start = WIP.index('  wip_coordinator_key="$( {')
    tail = "} | sha256sum | awk '{print $1}')\"\n"
    statement = WIP[start:WIP.index(tail, start) + len(tail)].replace("sha256sum | awk '{print $1}'", 'cat')
    keys = {}
    for arch in ('120', '89'):
        result = _run('docker() { echo image; }\nwip_build_key=k COORDINATOR_DOCKER_DEV=dev '
                      f'wip_coordinator_arch={arch}\n' + statement + 'printf "%s" "$wip_coordinator_key"\n')
        assert result.returncode == 0, result.stderr
        keys[arch] = result.stdout
    # SM120 keeps its key material (and so its key) unchanged.
    assert keys['120'] == 'k coordinator image'
    assert keys['89'] == 'k coordinator image\narch 89'
