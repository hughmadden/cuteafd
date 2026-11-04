#!/usr/bin/env python3
"""Export native V4.1 K32 activation quantization and 32x32-scale projections.

Targets native SM120 (block-scaled MMA, swizzled MMA-layout scales) and SM89
(per-K32 promoted E4M3 MMA reading plain scale layouts: activation row scales
[rows, K/32] as the row quantizer writes them, weight scales [N/32, K/32]
unpacked). The C ABI is identical; SM89 manifests alias the MMA-scale offset
to the row scales and size packed weight scales as the plain grid.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("lib",)]  # sibling tool dirs
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import _pinned_sparkinfer  # noqa: F401
from sm89_cards import CARD_SMS as SM89_CARD_SMS, WORKSPACE_SMS as SM89_WORKSPACE_SMS
from v41_capacities import PROJECTION_CAPACITIES


PROJECTIONS = (
    ('engram', 25600, 6144), ('ffn_up', 2304, 5120), ('ffn_down', 5120, 2304),
    ('main', 5120, 15360), ('q_a', 1280, 5120), ('q_b', 32768, 1280),
    ('kv', 512, 5120), ('o_b', 5120, 8192), ('o_a', 8192, 32768), ('index_q', 4096, 1280),
    # Output-channel shards preserve the full reduction dimension on each GPU.
    ('q_b_tp2', 16384, 1280), ('o_b_tp2', 2560, 8192),
    ('ffn_tp2_up', 1152, 5120), ('ffn_tp2_down', 5120, 1152),
)

def validate_abi(path: Path, label: str, kind: str) -> dict:
    header = path.read_text()
    pointers = ('source_ptr', 'positions_ptr', 'cos_sin_ptr', 'values_ptr', 'scale_rows_ptr', 'scale_mma_ptr') if kind == 'group_quant' else ('source_ptr', 'values_ptr', 'scale_rows_ptr', 'scale_mma_ptr') if kind == 'quant' else (
        'a_ptr', 'b_ptr', 'sfa_ptr', 'sfb_ptr', 'c_ptr', 'quant_c_values_ptr',
        'quant_c_scale_rows_ptr', 'quant_c_scale_mma_ptr', 'alpha_ptr')
    scalars = ('m', 'cos_sin_len', 'grid_x') if kind == 'group_quant' else ('m', 'grid_x') if kind == 'quant' else ('m',)
    stream = 'stream' if kind in ('quant', 'group_quant') else 'current_stream'
    if kind == 'hc_project':
        pointers, scalars, stream = ('r', 'w', 'p'), ('m',), 's'
    if kind == 'hc_lagged':
        pointers = ('residual','fn','scale','bias','incoming','norm','predicted','post','comb','normalized','scratch')
        scalars, stream = ('rows',), 'stream'
    expected = [f'cuteafd_{label}_Kernel_Module_t *module']
    expected += [f'void *{name}' for name in pointers]
    expected += [f'int32_t {name}' for name in scalars] + [f'cudaStream_t {stream}']
    signature = re.search(r'static inline int32_t cute_dsl_cuteafd_' + re.escape(label) + r'_wrapper\(([^)]*)\)', header)
    if not signature or re.sub(r'\s+', '', signature[1]) != re.sub(r'\s+', '', ','.join(expected)):
        raise ValueError(f'unexpected generated ABI: {path}')
    symbols = re.findall(r'void (_mlir_\w+)\(void \*\*args, int32_t num_args\);', header)
    count = len(pointers) + len(scalars) + 2
    if len(symbols) != 1 or f'{symbols[0]}(args, {count});' not in header:
        raise ValueError(f'unexpected generated dispatch ABI: {path}')
    arguments = re.search(r'void \*args\[' + str(count) + r'\] = \{([^}]*)\}', header)
    expected_args = ','.join('&' + name for name in (*pointers, *scalars, stream, 'ret'))
    if not arguments or re.sub(r'\s+', '', arguments[1]) != expected_args:
        raise ValueError(f'unexpected generated argument order: {path}')
    return {'symbol': symbols[0], 'pointers': pointers, 'i32': scalars, 'stream': stream, 'argument_count': count}


# Bundles size grids for their architecture's largest card (SM120: the RTX
# PRO 6000; SM89: sm89_cards); launches cap them by the runtime SM count.
SM120_SMS = 188
SM120_CARD_SMS = (170, 188)
# Shapes with one exact plan per card (label suffix _sm<SMs>); every other
# shape has one plan for any card of its architecture. SM120 plans q_a/kv
# decode for the RTX 5090 and PRO 6000. SM89 plans the same shapes for each
# supported card, so another SM89 card fails closed as on SM120, and every
# shape whose SM89 plan depends on the SM count (sm89_card_shapes).
CARD_PLAN_SHAPES = frozenset((name, rows) for name in ('q_a', 'kv') for rows in (1, 16))


def expected_rows(name: str, capacity: int, k: int, n: int, groups: int, sms: int) -> int:
    """The row regime a projection's kernels are planned for on ``sms`` SMs."""
    expected_m = capacity
    if os.environ.get('CUTEAFD_EXPORT_NARROW_AOT') == '1':
        from b12x._lib.dense_gemm import v41_fp8_aot_expected_m
        expected_m = v41_fp8_aot_expected_m(capacity=capacity, input_dim=k, output_dim=n,
            sm_count=sms, num_groups=groups)
    if name in ("o_b", "o_b_tp2") and capacity <= 80:
        # Decode and verify share the fast unsplit m16 arithmetic.
        expected_m = 16
    return expected_m


def sm89_card_shapes(projections, rows, sm89_block32_plan=None) -> set[tuple[str, int]]:
    """CARD_PLAN_SHAPES plus every shape whose SM89 kernels differ between the
    supported cards, planned exactly as compile_dense_gemm_mxfp8_aot plans SM89."""
    if sm89_block32_plan is None:
        from b12x._lib.sm89_fp8 import sm89_block32_plan
    shapes = set()
    for name, n, k in projections:
        groups = 8 if name == 'o_a' else 1
        for capacity in rows:
            plans = []
            for sms in SM89_CARD_SMS:
                expected_m = expected_rows(name, capacity, k, n, groups, sms)
                plans.append((expected_m, sm89_block32_plan(
                    size_m=capacity, size_n=n // groups, size_k=k // groups, num_groups=groups,
                    sm_count=sms, fixed_rows=expected_m < capacity)))
            if (name, capacity) in CARD_PLAN_SHAPES or any(plan != plans[0] for plan in plans):
                shapes.add((name, capacity))
    return shapes


def check_declared_card_plans(cards, shapes, projections, rows) -> None:
    """The build lists every exported file in advance: refuse a stale list."""
    declared_cards = os.environ.get('CUTEAFD_V41_FP8_CARD_PLANS')
    declared_shapes = os.environ.get('CUTEAFD_V41_FP8_CARD_SHAPES')
    if declared_cards is None and declared_shapes is None:
        return
    exported = {(name, capacity) for name, _, _ in projections for capacity in rows}
    listed = {(name, int(capacity)) for name, capacity in
              (item.split(':') for item in (declared_shapes or '').split(',') if item)}
    if tuple(int(sms) for sms in (declared_cards or '').split(',') if sms) != tuple(cards) \
            or listed & exported != shapes:
        raise ValueError(
            f'v41_fp8.cmake lists card plans {declared_cards} for {sorted(listed & exported)}; '
            f'this export plans {",".join(map(str, cards))} for {sorted(shapes)}')


def dispatch_header(output: Path, manifest: dict) -> None:
    from b12x._lib.quant.mxfp8_rows import mxfp8_rows_quant_aot_grid
    plain = manifest['capability'] == [8, 9]
    sms = SM89_WORKSPACE_SMS if plain else SM120_SMS
    lines = ['#pragma once', '#include <stdint.h>',
             f'#define CUTEAFD_V41_FP8_SMS {sms}']
    if plain:
        lines += ['#define CUTEAFD_V41_FP8_CC_MAJOR 8', '#define CUTEAFD_V41_FP8_CC_MINOR 9',
                  '#define CUTEAFD_V41_FP8_PLAIN_SCALES 1']
    lines.append('#include "v41_hc_project.h"')
    prefix = '_mlir_cuteafd_v41_hc_project'
    lines.append('#define CUTEAFD_V41_HC_PROJECT_MODULE {' + ','.join((
        prefix + '_cuda_init', prefix + '_cuda_load_to_device',
        manifest['hc_project']['abi']['symbol'])) + '}')
    if 'hc_lagged' in manifest:
        lines.append('#include "v41_hc_lagged.h"')
        prefix = '_mlir_cuteafd_v41_hc_lagged'
        lines.append('#define CUTEAFD_V41_HC_LAGGED_MODULE {' + ','.join((prefix + '_cuda_init', prefix + '_cuda_load_to_device', manifest['hc_lagged']['abi']['symbol'])) + '}')
    variants = []
    for variant in manifest['variants']:
        label, capacity = variant['label'], variant['capacity']
        n, k = variant['output_dim'], variant['input_dim']
        for kind in ('quant', 'gemm'):
            lines.append(f'#include "{label}_{kind}.h"')
        groups = variant.get('groups', 1)
        if groups > 1:
            lines.append(f'#include "{label}_quant_rope.h"')
        grids = [mxfp8_rows_quant_aot_grid(size_k=k, rows=rows, expected_m=variant.get('expected_m', capacity),
                                         sm_count=sms) for rows in range(1, capacity + 1)]
        if groups > 1:
            grids = [min(rows * 32, sms * 4) for rows in range(1, capacity + 1)]
        lines.append('static const uint32_t ' + label + '_grids[] = {' + ','.join(map(str, grids)) + '};')
        info = [1, capacity, k, n, variant['scratch_bytes'],
                variant['activation_values_offset'], variant['activation_row_scales_offset'],
                variant['activation_mma_scales_offset'], n * k // groups // (1024 if plain else 32)]
        modules = []
        for kind in ('quant', 'gemm', *(['quant_rope'] if groups > 1 else [])):
            prefix = '_mlir_cuteafd_' + label + '_' + kind
            modules.append('{' + ','.join((prefix + '_cuda_init', prefix + '_cuda_load_to_device',
                                           variant[kind + '_abi']['symbol'])) + '}')
        if groups == 1:
            modules.append('{nullptr,nullptr,nullptr}')
        variants.append('{{' + ','.join(map(str, info)) + '},' + ','.join(modules) + ',' + label + '_grids,' + str(variant['split_k_offset']) + ',' + str(variant['split_k_slices']) + ',' + str(groups) + ',' + str(variant.get('grouped_output_offset', 0)) + ',' + str(variant.get('plan_sms', 0)) + ',' + str(variant.get('expected_m', capacity)) + ',' + str(int(variant.get('all_rows', capacity == 1 or variant.get('split_k_all_m', False)))) + '}')
    lines.append('#define CUTEAFD_V41_FP8_VARIANTS ' + ','.join(variants))
    (output / 'v41_fp8_variants.h').write_text('\n'.join(lines) + '\n')


def export(output: Path, rows: tuple[int, ...], projections=PROJECTIONS) -> None:
    os.environ['B12X_COMPILE_DISK_CACHE'] = '0'
    os.environ['B12X_COMPILE_MEMORY_CACHE'] = '0'
    import torch
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot
    from b12x.gemm.wo_projection._quant_cute import compile_wo_grouped_quant_aot
    from b12x.gemm._shared.block_fp8 import _block_fp8_linear_scratch_layout

    device = torch.device('cuda', torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) not in ((12, 0), (8, 9)):
        raise ValueError('coordinator FP8 export requires native SM120 or SM89')
    plain = (props.major, props.minor) == (8, 9)
    cards = SM89_CARD_SMS if plain else SM120_CARD_SMS
    shared_sms = SM89_WORKSPACE_SMS if plain else SM120_SMS
    card_shapes = (sm89_card_shapes(projections, rows) if plain else
                   {(name, capacity) for name, _, _ in projections for capacity in rows
                    if (name, capacity) in CARD_PLAN_SHAPES})
    check_declared_card_plans(cards, card_shapes, projections, rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'v41_fp8.json').unlink(missing_ok=True)
    manifest = {'schema': 2, 'role': 'coordinator', 'capability': [props.major, props.minor],
                'physical_sms': props.multi_processor_count, 'device': _pinned_sparkinfer.export_target(props),
                'projections': [{'name': name, 'n': n, 'k': k, 'weight_block': [32,32],
                                 'groups': 8 if name == 'o_a' else 1,
                                 'activation_block': 32, 'activation_amax_floor': 0.0 if name == 'o_a' else 1e-4} for name,n,k in projections],
                'sparkinfer_revision': _pinned_sparkinfer.REVISION,
                'variants': []}
    for name, n, k in projections:
        for capacity in rows:
            dual = (name, capacity) in card_shapes
            for plan_sms in (cards if dual else (shared_sms,)):
                label = f'v41_{name}_fp8_m{capacity}' + (f'_sm{plan_sms}' if dual else '')
                groups = 8 if name == 'o_a' else 1
                expected_m = expected_rows(name, capacity, k, n, groups, plan_sms)
                fixed_output = name in ("o_b", "o_b_tp2") and capacity <= 80
                all_m = False
                quant = (compile_wo_grouped_quant_aot(groups=groups, group_width=k // groups)
                         if groups > 1 else compile_mxfp8_rows_quant_aot(
                             size_k=k, scale_block_size=32, expected_m=expected_m, amax_floor=1e-4))
                quant.export_to_c(str(output), label + '_quant', 'cuteafd_' + label + '_quant')
                if groups > 1:
                    rope_quant = compile_wo_grouped_quant_aot(groups=groups, group_width=k // groups, row_frequencies=True)
                    rope_quant.export_to_c(str(output), label + '_quant_rope', 'cuteafd_' + label + '_quant_rope')
                gemm, split_k = compile_dense_gemm_mxfp8_aot(size_m=capacity, size_n=n // groups, size_k=k // groups, num_groups=groups,
                                                  expected_m=expected_m, sfb_k_replicated=False, device=device,
                                                  return_split_k_metadata=True, sm_count=plan_sms,
                                                  **({"split_k_all_m": True} if all_m else {}))
                if fixed_output and split_k != 1:
                    raise ValueError('decode output projection requires the unsplit m16 recipe')
                gemm.export_to_c(str(output), label + '_gemm', 'cuteafd_' + label + '_gemm')
                layout = _block_fp8_linear_scratch_layout(tokens=capacity, in_features=k,
                                                        out_features=n, output_dtype=torch.bfloat16)
                split_offset = ((layout.nbytes + 255) // 256) * 256 if split_k > 1 else 0
                split_bytes = split_k * capacity * n * 4 if split_k > 1 else 0
                group_layout = {}
                if groups > 1:
                    values_bytes = capacity * k
                    row_scale_offset = (values_bytes + 255) // 256 * 256
                    mma_offset = (row_scale_offset + capacity * k // 32 + 255) // 256 * 256
                    mma_bytes = groups * ((capacity + 127) // 128) * 512 * (k // groups // 128)
                    if plain:
                        # SM89 GEMMs read the row scales; the MMA slot aliases them.
                        mma_offset, mma_bytes = row_scale_offset, capacity * k // 32
                    output_offset = (mma_offset + mma_bytes + 255) // 256 * 256
                    group_layout = dict(groups=groups, activation_values_offset=0,
                                        activation_row_scales_offset=row_scale_offset,
                                        activation_mma_scales_offset=mma_offset,
                                        grouped_output_offset=output_offset,
                                        activation_scratch_bytes=output_offset,
                                        scratch_bytes=output_offset + capacity * n * 2,
                                        activation_mma_scale_shape=[groups, capacity, k // groups // 32] if plain else [32, 4, (capacity + 127) // 128, 4, k // groups // 128, groups])
                manifest['variants'].append({'capacity': capacity, 'plan_sms': plan_sms if dual else 0, 'expected_m': expected_m, 'split_k_all_m': all_m, 'all_rows': fixed_output or capacity == 1, 'label': label, 'input_dim': k, 'output_dim': n,
                    'activation_scratch_bytes': layout.nbytes,
                    'scratch_bytes': split_offset + split_bytes if split_k > 1 else layout.nbytes,
                    'split_k_offset': split_offset, 'split_k_bytes': split_bytes,
                    'activation_values_offset': layout.x_values_offset_bytes,
                    'activation_row_scales_offset': layout.x_scale_rows_offset_bytes,
                    'activation_mma_scales_offset': layout.x_scale_rows_offset_bytes if plain else layout.x_scale_mma_offset_bytes,
                    'activation_mma_scale_shape': [capacity, k // 32] if plain else list(layout.x_scale_mma_physical_shape),
                    'split_k_slices': split_k, 'gemm_output_dtype': 'FP32' if split_k > 1 else 'BF16', 'output_dtype': 'BF16', 'output_bytes': capacity * n * 2,
                    'quant_abi': validate_abi(output / (label + '_quant.h'), label + '_quant', 'group_quant' if groups > 1 else 'quant'),
                    'gemm_abi': validate_abi(output / (label + '_gemm.h'), label + '_gemm', 'gemm'), **group_layout})
                if groups > 1:
                    manifest['variants'][-1]['quant_rope_abi'] = validate_abi(output / (label + '_quant_rope.h'), label + '_quant_rope', 'group_quant')
                if plain:
                    manifest['variants'][-1].update(
                        activation_scale_layout='rows', weight_scale_layout='plain',
                        weight_scale_shape=[groups, n // groups // 32, k // groups // 32],
                        packed_weight_scale_bytes=n * k // groups // 1024)
                print(f'exported {label}', flush=True)
    from b12x.norm.mhc._v41_project import compile_v41_mhc_project_aot
    compile_v41_mhc_project_aot().export_to_c(str(output), 'v41_hc_project', 'cuteafd_v41_hc_project')
    manifest['hc_project'] = {'split_k': 8, 'scratch_bytes_per_row': 1536,
        'abi': validate_abi(output / 'v41_hc_project.h', 'v41_hc_project', 'hc_project')}
    if os.environ.get('CUTEAFD_EXPORT_HC_LAGGED') == '1':
        from b12x.norm.mhc._v41_lagged_aot import compile_v41_lagged_aot
        compile_v41_lagged_aot().export_to_c(str(output), 'v41_hc_lagged', 'cuteafd_v41_hc_lagged')
        manifest['hc_lagged'] = {'scratch_bytes_per_row': 8000, 'max_rows': 80,
            'abi': validate_abi(output / 'v41_hc_lagged.h', 'v41_hc_lagged', 'hc_lagged')}
    dispatch_header(output, manifest)
    artifacts = {"v41_fp8_variants.h": hashlib.sha256((output / "v41_fp8_variants.h").read_bytes()).hexdigest()}
    for variant in manifest['variants']:
        for kind in ('quant', 'gemm', *(['quant_rope'] if variant.get('groups', 1) > 1 else [])):
            for suffix in ('.h', '.o'):
                path = output / (variant['label'] + '_' + kind + suffix)
                artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for suffix in ('.h', '.o'):
        path = output / ('v41_hc_project' + suffix)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'hc_lagged' in manifest:
        for suffix in ('.h', '.o'):
            path = output / ('v41_hc_lagged' + suffix)
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['artifacts'] = artifacts
    (output / 'v41_fp8.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', default=','.join(map(str, PROJECTION_CAPACITIES)))
    parser.add_argument('--projections', help='Comma-separated projection names; default exports all')
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(','))
    if not rows or len(set(rows)) != len(rows) or any(value not in PROJECTION_CAPACITIES for value in rows):
        parser.error(f'rows must be distinct native capacities: {PROJECTION_CAPACITIES}')
    selected = args.projections.split(',') if args.projections else [p[0] for p in PROJECTIONS]
    if len(selected) != len(set(selected)) or set(selected) - {p[0] for p in PROJECTIONS}:
        parser.error('projections must be distinct supported names')
    export(args.output_dir, rows, tuple(p for p in PROJECTIONS if p[0] in selected))


if __name__ == '__main__':
    main()
