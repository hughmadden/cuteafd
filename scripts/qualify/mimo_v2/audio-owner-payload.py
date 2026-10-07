#!/usr/bin/env python3
"""Retain admitted serving-library audio payloads; compare with sealed CPU features.

Capture runs in the matching architecture image under its hardware lock. Compare
is CPU-only. This exercises the serving library ABI, not the Rust resident-owner
thread, and does not by itself qualify LM scoring or prefix restores.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/reference'))
from families.mimo_v2.mimo_v26 import audio_native_reference as native
from families.mimo_v2.mimo_v26 import audio_reference as oracle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_round(values):
    bits = np.ascontiguousarray(values, dtype='<f4').view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')


def differences(actual, expected):
    if actual.dtype != np.uint16 or expected.dtype != np.uint16 or actual.shape != expected.shape:
        raise ValueError('BF16 shape/dtype mismatch')
    a = (actual.astype(np.uint32) << 16).view(np.float32)
    b = (expected.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite BF16 features')
    changed = actual != expected
    # Monotonic bit ordering allows distances across zero as well as negative values.
    def ordered(bits):
        return np.where(bits & 0x8000, 0xffff - bits.astype(np.int32), bits.astype(np.int32) + 0x8000)
    distance = np.abs(ordered(actual) - ordered(expected))
    counts, frequencies = np.unique(distance[changed], return_counts=True)
    return {'elements': int(actual.size), 'different_elements': int(changed.sum()),
            'max_abs': float(np.max(np.abs(a - b))), 'max_bf16_ulp': int(distance.max()),
            'ulp_histogram': {str(int(k)): int(v) for k, v in zip(counts, frequencies)},
            'different_elements_per_row': changed.sum(axis=1).tolist(),
            'positions_row_dim': np.argwhere(changed).tolist()}


def capture(args):
    panel = json.loads(args.panel.read_text())
    if args.output.exists():
        raise ValueError('capture output must be new')
    args.output.mkdir(parents=True)
    plan = json.loads((args.arena / 'plan.json').read_text())
    spec = native.native_spec(plan, args.max_samples)
    lib = native.library(args.library)
    ledger = native.Ledger()
    native.check(lib.cuteafd_audio_required(C.byref(spec), C.byref(ledger)), 'required')
    admitted = ledger.weights + ledger.scratch + ledger.blas_workspace + ledger.fft_workspace
    owner = C.c_void_p()
    native.check(lib.cuteafd_audio_create(C.byref(spec), 0, admitted, C.byref(owner)), 'create')
    report = {'library_sha256': sha256(args.library), 'arena_plan_sha256': sha256(args.arena / 'plan.json'),
              'kind': 'serving_library_abi_payload_capture', 'max_samples': args.max_samples, 'clips': []}
    try:
        weights = np.memmap(args.arena / 'weights.f32', mode='r', dtype=np.uint8)
        if weights.size != spec.weight_bytes:
            raise ValueError('weight arena extent mismatch')
        native.check(lib.cuteafd_audio_upload(owner, weights.ctypes.data, weights.size, None, None, None, None), 'upload')
        backend = C.create_string_buffer(256)
        native.check(lib.cuteafd_audio_backend(backend, len(backend)), 'backend')
        report['backend'] = backend.value.decode()
        for index, clip in enumerate(panel):
            pcm = np.fromfile(clip['pcm'], dtype='<f4')
            geometry = oracle.token_geometry(pcm.size)
            if (not np.isfinite(pcm).all() or pcm.size != clip['span']['samples']
                    or geometry['tokens'] != clip['span']['len'] or pcm.size > args.max_samples
                    or oracle.digest(pcm.tobytes()) != clip['span']['pcm_sha256']):
                raise ValueError('canonical PCM identity/capacity mismatch')
            output = np.empty((geometry['tokens'], spec.output_width), dtype='<f4')
            codes = np.empty((geometry['codes'], 20), dtype='<i4')
            native.check(lib.cuteafd_audio_encode(owner, pcm.ctypes.data, pcm.size, output.ctypes.data, output.nbytes,
                         codes.ctypes.data, codes.nbytes, native.Observer(), None), 'encode')
            if not np.isfinite(output).all() or np.any(codes < 0) or np.any(codes >= 1024):
                raise ValueError('nonfinite projection or invalid RVQ codes')
            prefix = args.output / str(index)
            output.tofile(str(prefix) + '.projection.f32')
            bf16_round(output).tofile(str(prefix) + '.bf16')
            codes.astype('<i8').tofile(str(prefix) + '.codes.i64')
            report['clips'].append({**clip, 'payload_prefix': str(index), 'geometry': geometry,
                'bf16_sha256': sha256(Path(str(prefix) + '.bf16')),
                'codes_sha256': sha256(Path(str(prefix) + '.codes.i64'))})
            (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        native.check(lib.cuteafd_audio_destroy(owner), 'destroy')


def compare(args):
    from mimo_media import snapshot_identity
    report = json.loads((args.capture / 'capture.json').read_text())
    identity = snapshot_identity(args.snapshot)
    hidden = json.loads((args.snapshot / 'config.json').read_text())['hidden_size']
    results = []
    for clip in report['clips']:
        prefix = args.capture / clip['payload_prefix']
        if not prefix.resolve().is_relative_to(args.capture.resolve()):
            raise ValueError('capture payload escapes directory')
        for suffix, field in (('.bf16', 'bf16_sha256'), ('.codes.i64', 'codes_sha256')):
            if sha256(Path(str(prefix) + suffix)) != clip[field]:
                raise ValueError('capture payload hash differs')
        expected, meta = oracle.read_probe_features(args.features, clip['span'], hidden, identity)
        actual = np.fromfile(str(prefix) + '.bf16', dtype='<u2').reshape(expected.shape)
        codes = np.fromfile(str(prefix) + '.codes.i64', dtype='<i8').reshape(-1, 20)
        ref_codes = np.fromfile(args.features / (clip['span']['key'] + '.codes.i64'), dtype='<i8').reshape(codes.shape)
        mismatch = np.argwhere(codes != ref_codes)
        metric = differences(actual, expected)
        results.append({'id': clip['id'], 'pcm_sha256': clip['span']['pcm_sha256'],
            'native_bf16_sha256': sha256(Path(str(prefix) + '.bf16')), 'official_bf16_sha256': meta['sha256'],
            'matches_e2e_owner_hash': sha256(Path(str(prefix) + '.bf16')) == clip.get('e2e_bf16_sha256'),
            'rvq_codes': int(codes.size), 'rvq_identical': not mismatch.size,
            'rvq_positions_frame_codebook': mismatch.tolist(),
            'rvq_values_native_official': [[int(codes[tuple(p)]), int(ref_codes[tuple(p)])] for p in mismatch],
            'bf16': metric})
    args.output.write_text(json.dumps({'kind': 'cpu_sealed_payload_comparison',
        'native_library_sha256': report['library_sha256'], 'backend': report['backend'],
        'results': results}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    for name in ('library', 'arena', 'panel', 'output'):
        cap.add_argument('--' + name, type=Path, required=True)
    cap.add_argument('--max-samples', type=int, default=7200000)
    cmp = sub.add_parser('compare')
    for name in ('capture', 'features', 'snapshot', 'output'):
        cmp.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    (capture if args.mode == 'capture' else compare)(args)


if __name__ == '__main__':
    main()
