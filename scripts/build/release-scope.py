#!/usr/bin/env python3
"""Derive release exports from shipping cards and verify image inventories (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import io
import re
import tarfile

PREFIX = 'CUTEAFD_RELEASE_'
SWITCHES = ('DSV4', 'GLM', 'GLMF', 'MIMO', 'QWEN4', 'QWEN_DENSE', 'AUDIO', 'CONTEXT_SPLIT')
LISTS = ('EXPERT_FAMILIES', 'GENERIC_SPARK_COUNTS', 'SPARK_TP_ROLES', 'MIMO_GEOMETRIES')
# v2.0.0 shipped set: complete releases retain this previous-release floor.
PREVIOUS_RELEASE_FLOOR = {
    'EXPERT_FAMILIES': {'mimo:fp8'},
    'MIMO_GEOMETRIES': {'mimo', 'mimo2'},
    'MIMO_AOT': {'ON'},
    'SPARK_TP_ROLES': {'tp2', 'tp3', 'tp6'},
    'GENERIC_SPARK_COUNTS': {
        f'{family}={count}'
        for family, counts in {
            'dsv4f': (2, 3, 4, 6), 'dsv4p': (2, 3, 4, 6),
            'glm': (2, 3, 4, 6), 'glmf': (2, 3, 4, 6),
            'mimo': (2, 4, 6), 'mimof': (2, 4), 'mimop': (2, 6),
            'qwen4': (1, 2, 3, 4),
        }.items() for count in counts
    },
}
DEFAULT_MATRIX = 'scripts/bench/release-smoke-v3.json'


def scopes_module(repo):
    spec = importlib.util.spec_from_file_location('wip_aot_scopes', repo / 'scripts/lib/wip_aot_scopes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entries_for(repo, matrix, scope):
    payload = json.loads(matrix.read_text())
    defaults = payload.get('defaults', {})
    entries = []
    for raw in payload['entries']:
        entry = {**defaults, **raw}
        entry['set'] = {**defaults.get('set', {}), **raw.get('set', {})}
        entries.append(entry)
    if not entries or len({e['name'] for e in entries}) != len(entries):
        raise ValueError('release matrix must contain nonempty, uniquely named entries')
    if scope == 'explicit':
        return []
    if scope == 'all':
        return entries
    selectors = scope.split(',')
    selected = []
    for selector in selectors:
        matches = [e for e in entries if selector in (e['name'], e.get('family'))]
        if not selector or not matches:
            raise ValueError(f'unknown release scope selector {selector!r}')
        selected.extend(e for e in matches if e not in selected)
    return selected


def entry_scopes(repo, entry):
    wip = scopes_module(repo)
    values = wip.read_config(repo / entry.get('config', 'cuteafd.config'))
    values.update({key: str(value) for key, value in entry['set'].items()})
    # Resource reservations do not override smoke's config/set topology.
    if entry.get('family'):
        values['FAMILY'] = entry['family']
    family, scopes = wip.config_scopes(values)
    return family, values, scopes


def tokens(value):
    return set(filter(None, value.split(';')))


def derive(repo, matrix, scope='all', environ=None):
    environ = os.environ if environ is None else environ
    entries = entries_for(repo, matrix, scope)
    result = {PREFIX + key + '_AOT': 'OFF' for key in SWITCHES}
    result.update({PREFIX + key: '' for key in LISTS})
    for entry in entries:
        _, _, scopes = entry_scopes(repo, entry)
        for key, value in scopes.items():
            suffix = key.removeprefix('CUTEAFD_WIP_')
            if suffix == 'COORDINATOR_EXPERT_FAMILIES':
                suffix = 'EXPERT_FAMILIES'
            target = PREFIX + suffix
            if suffix.endswith('_AOT') and value == 'ON':
                result[target] = 'ON'
            elif suffix in LISTS:
                result[target] = ';'.join(sorted(tokens(result[target]) | tokens(value)))
    # TP4 is the always-built unsuffixed V4.1 Spark shard, not a supplemental role.
    if scope == 'all':
        for key, floor in PREVIOUS_RELEASE_FLOOR.items():
            target = PREFIX + key
            if key.endswith('_AOT'):
                result[target] = 'ON'
            else:
                result[target] = ';'.join(sorted(tokens(result[target]) | floor))
    # Context-split programs are required by V4 as well as GLM's config scope.
    if result[PREFIX + 'DSV4_AOT'] == 'ON':
        result[PREFIX + 'CONTEXT_SPLIT_AOT'] = 'ON'
    for key in result:
        if key not in environ:
            continue
        value = environ[key]
        if scope == 'explicit':
            result[key] = value
        elif key.endswith('_AOT'):
            if value not in ('ON', 'OFF'):
                raise ValueError(f'{key} must be ON or OFF')
            result[key] = 'ON' if 'ON' in (result[key], value) else 'OFF'
        else:
            if key.endswith('MIMO_GEOMETRIES'):
                value = value.replace(',', ';')
            result[key] = ';'.join(sorted(tokens(result[key]) | tokens(value)))
    return result, entries


def requirements(repo, matrix, scope):
    result = []
    for entry in entries_for(repo, matrix, scope):
        family, values, scopes = entry_scopes(repo, entry)
        experts = tokens(scopes['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'])
        dual = int(values.get('RTX_GPUS', '1').replace('auto', '2')) == 2
        count = int(values.get('SPARK_COUNT', '0'))
        tp = int(values.get('SPARK_TP') or count or 1)
        programs = set()
        if family == 'deepseek_v4':
            tag = 'dsv4p' if any(s.startswith('dsv4p:') for s in experts) else 'dsv4f'
            programs = {tag, tag + '2'} if dual else {tag}
        elif family in ('glm5', 'glm5_flash'):
            tag = 'glm' if family == 'glm5' else 'glmf'
            programs = {tag, tag + '2'} if dual else {tag}
        elif family == 'mimo_v2':
            programs = tokens(scopes['CUTEAFD_WIP_MIMO_GEOMETRIES'])
        elif family == 'qwen4':
            programs = {'qwen4'}
        elif family == 'qwen3_5':
            programs = {'qwen35' if '9b' in values['MODEL_ID'].lower() else 'qwen38'}
        result.append(dict(name=entry['name'], family=family, programs=sorted(programs),
                           experts=sorted(experts), spark_experts=sorted(tokens(scopes['CUTEAFD_WIP_EXPERT_FAMILIES'])),
                           spark_count=count, spark_tp=tp, dual=dual,
                           roles=sorted(tokens(scopes['CUTEAFD_WIP_SPARK_TP_ROLES'])),
                           nvfp4='nvfp4' in values.get('MODEL_ID', '').lower(),
                           audio=scopes.get('CUTEAFD_WIP_AUDIO_AOT') == 'ON',
                           context_split=family in ('deepseek_v4', 'glm5', 'glm5_flash') and dual))
    return result


def read_json(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('not an object')
        return value
    except (OSError, ValueError) as error:
        raise ValueError(f'missing or invalid {path.name}: {error}') from error


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def native_symbols(root):
    library = root / 'lib/libcuteafd_native.so'
    output = subprocess.run(['nm', '-D', '--defined-only', str(library)], check=True,
                            capture_output=True, text=True).stdout
    return {line.split()[-1] for line in output.splitlines() if line.split()}


def check_package(root, package, layouts, role, verified=None):
    path = root / 'lib' / package
    manifest = read_json(path / 'manifest.json')
    require(manifest.get('role') == ('spark' if role == 'expert' else role), f'{package}: wrong role')
    if package.startswith('exl3/'):
        require(manifest.get('schema') == 'cuteafd.exl3-package.v1', f'{package}: invalid EXL3 schema')
        present = {v['directory'].split('/')[0] for v in manifest.get('variants', [])}
    else:
        present = set(manifest.get('layouts', {}))
        # Both Spark and coordinator H128 slices require exact-width siblings.
        exact = []
        for layout in layouts:
            if '-w' in layout:
                continue
            info = manifest.get('layouts', {}).get(layout, {})
            intermediate = info.get('intermediate', 0)
            world = int(layout.removeprefix('tp'))
            if intermediate and intermediate % 128 == 0 and intermediate // 128 >= world:
                blocks = intermediate // 128
                widths = {(blocks // world + int(rank < blocks % world)) * 128 for rank in range(world)}
                if len(widths) > 1:
                    exact.extend(f'{layout}-w{width}' for width in widths)
        layouts = [*layouts, *exact]
    require(set(layouts) <= present, f'{package}: missing layouts {sorted(set(layouts) - present)}')
    files = manifest.get('files', {})
    require(bool(files), f'{package}: empty file inventory')
    for name, record in ([] if verified is not None and path in verified else files.items()):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, f'{package}: unsafe file {name}')
        file = path / relative
        require(file.is_file(), f'{package}: missing file {name}')
        require(file.stat().st_size == record['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256'],
                f'{package}: file hash mismatch {name}')
    for layout in layouts:
        require(any(name.startswith(layout + '/') and name.endswith('.so') for name in files),
                f'{package}: missing library for {layout}')
    if verified is not None:
        verified.add(path)


def check_inventory(root, role, wanted, symbols=None):
    symbols = native_symbols(root) if symbols is None else symbols
    share = root / 'share'
    verified = set()
    for entry in wanted:
        if role == 'expert' and not entry['spark_count']:
            continue
        try:
            if role == 'coordinator' and entry['programs']:
                manifest = read_json(share / 'PROGRAMS.json')
                programs = manifest.get('programs', [])
                present = {p.get('family') for p in programs}
                missing = set(entry['programs']) - present
                require(not missing, f'PROGRAMS.json missing program families {sorted(missing)}')
                if entry['context_split']:
                    require(any(p.get('op') == 'scored_index_topk'
                                for p in programs if p.get('family') == entry['programs'][0]),
                            'PROGRAMS.json missing context-split programs')
            if entry['family'] == 'deepseek_v41':
                require(bool(read_json(share / 'V41_EXPERT_AOT.json').get('variants')), 'V41_EXPERT_AOT.json has no variants')
                if role == 'expert':
                    roles = read_json(share / 'V41_EXPERT_TP_AOT.json')
                    require(set(entry['roles']) <= set(roles.get('spark_tp_roles', [])),
                            f'V41_EXPERT_TP_AOT.json missing roles {entry["roles"]}')
                    symbol_role = '' if entry['spark_tp'] == 4 else '_spark_tp' + str(entry['spark_tp'])
                    require(f'cuteafd_v41{symbol_role}_expert_launch' in symbols,
                            f'missing V4.1 Spark TP{entry["spark_tp"]} native package')
                    if entry['nvfp4']:
                        suffix = '' if entry['spark_tp'] == 4 else '_spark_tp' + str(entry['spark_tp'])
                        require(f'cuteafd_v41_nvfp4{suffix}_expert_launch' in symbols,
                                f'missing V4.1 NVFP4 Spark TP{entry["spark_tp"]} package')
                else:
                    require(bool(read_json(share / 'V41_FP8_AOT.json').get('variants')), 'V41_FP8_AOT.json has no variants')
                    for local in ('rtx_backbone', 'rtx_tp2') if entry['dual'] else ('rtx_backbone',):
                        require(bool(read_json(share / f'V41_EXPERT_WORKSPACES/mxfp4_{local}.json').get('variants')),
                                f'V41_EXPERT_WORKSPACES missing {local}')
                        if entry['nvfp4']:
                            require(bool(read_json(share / f'V41_EXPERT_WORKSPACES/nvfp4_{local}.json').get('variants')),
                                    f'V41_EXPERT_WORKSPACES missing NVFP4 {local}')
            scopes = entry['experts'] if role == 'coordinator' else entry['spark_experts']
            for item in scopes:
                family, kind = item.split(':')
                if kind.startswith('exl3-'):
                    layouts = [f'tp{entry["spark_tp"]}-rank{rank}' for rank in range(entry['spark_tp'])] if role == 'expert' else ['rtx-tp1']
                    if role == 'coordinator' and entry['dual']:
                        layouts.extend(['rtx-tp2-rank0', 'rtx-tp2-rank1'] if family == 'qwen4' else ['rtx-tp2'])
                    check_package(root, f'exl3/exl3-{family}-{kind[5:]}', layouts, role, verified)
                elif kind in ('fp8', 'nvfp4', 'nvfp4a4'):
                    suffix = '' if kind == 'fp8' else '-' + kind
                    layouts = [f'tp{entry["spark_tp"]}'] if role == 'expert' else ['tp1']
                    if role == 'coordinator' and entry['dual'] and family != 'glmfdense':
                        layouts.append('tp2')
                        if family == 'qwen4' and kind.startswith('nvfp4'):
                            layouts.extend(['tp2-w384', 'tp2-w256'])
                    check_package(root, f'fp8/fp8-{family}{suffix}', layouts, role, verified)
                    if kind == 'nvfp4':
                        check_package(root, f'fp8/fp8-{family}-nvfp4a4', layouts, role, verified)
                    if family == 'qwen4' and kind.startswith('nvfp4') and role == 'coordinator':
                        check_package(root, 'fp8/fp8-qwen4', ['tp1'], role, verified)
                elif (role == 'coordinator') == kind.startswith('rtx_'):
                    tag = {'rtx_backbone': '_local', 'rtx_tp2': '_tp2', 'spark': '', 'spark_tp2': '_spark_tp2'}[kind]
                    symbol = f'cuteafd_{family}{tag}_expert_launch'
                    require(symbol in symbols, f'missing native expert package {item} ({symbol})')
            if entry['audio']:
                require('cuteafd_audio_required' in symbols, 'missing audio tower package (cuteafd_audio_required)')
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'entry {entry["name"]} ({role}): {error}') from error


def check_image(image, role, wanted):
    parent = Path.home() / '.cache/cuteafd/builds/release-inventory'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        root = Path(directory)
        container = subprocess.check_output(['docker', 'create', image], text=True).strip()
        try:
            for name in ('share', 'lib'):
                subprocess.run(['docker', 'cp', f'{container}:/opt/cuteafd/{name}', str(root / name)], check=True)
            check_inventory(root, role, wanted)
        finally:
            subprocess.run(['docker', 'rm', '-f', container], check=True, stdout=subprocess.DEVNULL)


def revision_requirements(repo, revision):
    require(re.fullmatch(r'[0-9a-f]{40}', revision) is not None,
            'release publication requires a clean full Git revision')
    # Materialize the build revision's matrix, config and shared derivation, not
    # the publisher's potentially different checkout or environment override.
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', revision, '--',
                                       'scripts/lib', 'examples/configs', 'cuteafd.config', DEFAULT_MATRIX])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
            tree.extractall(root, filter='data')
        return requirements(root, root / DEFAULT_MATRIX, 'all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('derive', 'requirements', 'check-image', 'check-files'))
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--scope', default=os.environ.get(PREFIX + 'SCOPE', 'all'))
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('--role', choices=('coordinator', 'expert'))
    parser.add_argument('--image')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--requirements')
    args = parser.parse_args()
    try:
        # Inventory checks supplied with requirements work when sent over stdin,
        # without resolving __file__ or assuming a remote working directory.
        if args.action in ('derive', 'requirements') or not args.requirements:
            args.repo = args.repo or Path(__file__).resolve().parents[2]
        matrix = args.matrix or Path(os.environ.get(PREFIX + 'MATRIX', str(args.repo / DEFAULT_MATRIX) if args.repo else DEFAULT_MATRIX))
        if args.action == 'derive':
            env, entries = derive(args.repo, matrix, args.scope)
            print(f'Release scope: {args.scope}; {len(entries)} matrix entries', file=sys.stderr)
            if args.scope != 'all':
                print('NON-RELEASE scope: full shipping matrix coverage is not claimed', file=sys.stderr)
            if args.shell:
                for key, value in sorted(env.items()):
                    print(f'export {key}={shlex.quote(value)}')
            else:
                print(json.dumps(env, indent=2, sort_keys=True))
        elif args.action == 'requirements':
            wanted = revision_requirements(args.repo, args.revision) if args.revision else requirements(args.repo, matrix, args.scope)
            print(json.dumps(wanted, separators=(',', ':')))
        else:
            wanted = json.loads(args.requirements) if args.requirements else requirements(args.repo, matrix, args.scope)
            require(args.role is not None, '--role is required')
            if args.action == 'check-image':
                check_image(args.image, args.role, wanted)
            else:
                check_inventory(args.root, args.role, wanted)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'Release scope: {error}\n')


if __name__ == '__main__':
    main()
