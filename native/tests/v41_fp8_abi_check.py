"""Check every exported V4.1 coordinator FP8 projection through the native ABI.

Loads a native library built with the V4.1 FP8 bundle and, for each manifest
variant, runs pack_scales, initialize_scratch and launch (launch_rope for one
WO-A capacity) on random BF16 activations and E4M3 weights with UE8M0 32x32
scales. The quantized activations left in scratch must equal a bit-exact
torch replica of the K32 quantizer (E4M3 values, UE8M0 row scales, amax floor
1e-4, WO-A floor 0); outputs must match the FP64 product of the dequantized
operands to BF16 rounding (max relative error below 8e-3).

Usage: v41_fp8_abi_check.py NATIVE_LIB MANIFEST [--rows full|odd|R1,R2,..] [--labels L1,L2]

A row list runs every listed count up to each variant's capacity, plus
capacity - 3 (variants may split their kernels by live rows).
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch

P, I = C.c_void_p, C.c_int32


class Info(C.Structure):
    _fields_ = ([(n, C.c_uint32) for n in ('abi', 'capacity', 'k', 'n')] +
                [(n, C.c_uint64) for n in ('scratch', 'values', 'row_scales', 'mma_scales', 'weight_scales')])


def pow2_ceil_byte(scale):
    bits = scale.contiguous().view(torch.int32)
    bumped = torch.where((bits & 0x007FFFFF) != 0, (bits + 0x00800000) & 0x7F800000, bits)
    return ((bumped >> 23) & 0xFF).to(torch.uint8)


def quant_reference(x, floor):
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // 32, 32)
    amax = blocks.abs().amax(-1)
    if floor > 0:
        amax = amax.clamp_min(floor)
    byte = pow2_ceil_byte(amax * torch.tensor(1.0 / 448.0, device=x.device))
    byte = torch.where(amax == 0, torch.full_like(byte, 127), byte)
    inv = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    values = (blocks * inv[..., None]).to(torch.float8_e4m3fn).reshape(rows, k)
    return values, byte


def rope_reference(x, freqs):
    rows, k = x.shape
    v = x.float().reshape(rows, k // 512, 512).clone()
    tail = v[..., 448:].reshape(rows, k // 512, 32, 2)
    c, s = freqs[:, None, :, 0].double(), freqs[:, None, :, 1].float()
    v0, v1 = tail[..., 0], tail[..., 1]
    r0 = (v0.double() * c + (v1 * s).double()).float().bfloat16().float()
    r1 = (v1.double() * c + (-(v0 * s)).double()).float().bfloat16().float()
    v[..., 448:] = torch.stack((r0, r1), -1).reshape(rows, k // 512, 64)
    return v.reshape(rows, k).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_lib', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--rows', default='full')
    parser.add_argument('--labels')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lib = C.CDLL(str(args.native_lib))

    def bind(name, types):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, I

        def checked(*values):
            status = fn(*values)
            if status:
                raise RuntimeError(name, status)
        return checked

    info_fn = bind('cuteafd_v41_fp8_matrix_info', [I, I, I, C.POINTER(Info)])
    init = bind('cuteafd_v41_fp8_matrix_initialize', [I, I, I, C.POINTER(P)])
    init_scratch = bind('cuteafd_v41_fp8_initialize_scratch', [P, P, C.c_uint64, P, P])
    pack = bind('cuteafd_v41_fp8_matrix_pack_scales', [P, P, I, I, P])
    launch = bind('cuteafd_v41_fp8_launch', [P, P, P, P, P, C.c_uint64, P, P, I, P])
    launch_rope = bind('cuteafd_v41_fp8_launch_rope', [P, P, P, P, P, P, C.c_uint64, P, P, I, P])
    wanted = set(args.labels.split(',')) if args.labels else None
    stream = torch.cuda.current_stream().cuda_stream
    failures = 0
    for variant in manifest['variants']:
        label = variant['label']
        if wanted and label not in wanted:
            continue
        cap, k, n = variant['capacity'], variant['input_dim'], variant['output_dim']
        groups = variant.get('groups', 1)
        gk, gn = k // groups, n // groups
        if args.rows == 'full':
            row_counts = [cap]
        elif args.rows == 'odd':
            row_counts = [max(1, cap - 3)]
        else:
            listed = [int(r) for r in args.rows.split(',') if int(r) <= cap]
            row_counts = sorted(set(listed + [max(1, cap - 3)]))
        info = Info()
        info_fn(cap, k, n, C.byref(info))
        handle = P()
        init(cap, k, n, C.byref(handle))
        torch.manual_seed(cap * 7 + k + n)
        w = (torch.randn(groups, gn, gk, device='cuda') * 2).to(torch.float8_e4m3fn)
        scales = torch.randint(118, 128, (groups, gn // 32, gk // 32), device='cuda', dtype=torch.uint8)
        packed = torch.empty(info.weight_scales, device='cuda', dtype=torch.uint8)
        scratch = torch.empty(info.scratch, device='cuda', dtype=torch.uint8)
        alpha = torch.empty(1, device='cuda')
        pack(scales.data_ptr(), packed.data_ptr(), k, n, stream)
        init_scratch(handle, scratch.data_ptr(), info.scratch, alpha.data_ptr(), stream)
        rope = groups > 1 and cap == 80
        for rows in row_counts:
            x = (torch.randn(rows, k, device='cuda') * torch.rand(rows, 1, device='cuda') * 4).bfloat16()
            out = torch.empty(rows, n, device='cuda', dtype=torch.bfloat16)
            freqs = torch.randn(rows, 32, 2, device='cuda') if rope else None
            if rope:
                launch_rope(handle, x.data_ptr(), freqs.data_ptr(), w.data_ptr(), packed.data_ptr(),
                            scratch.data_ptr(), info.scratch, alpha.data_ptr(), out.data_ptr(), rows, stream)
            else:
                launch(handle, x.data_ptr(), w.data_ptr(), packed.data_ptr(), scratch.data_ptr(), info.scratch,
                       alpha.data_ptr(), out.data_ptr(), rows, stream)
            torch.cuda.synchronize()
            source = rope_reference(x, freqs) if rope else x
            floor = 0.0 if groups > 1 else 1e-4
            mismatch, worst = 0, 0.0
            values = scratch[info.values:info.values + groups * rows * gk].view(groups, rows, gk)
            row_scales = scratch[info.row_scales:info.row_scales + groups * rows * (gk // 32)].view(groups, rows, gk // 32)
            for g in range(groups):
                qv, qs = quant_reference(source[:, g * gk:(g + 1) * gk], floor)
                mismatch += int((values[g] != qv.view(torch.uint8)).sum()) + int((row_scales[g] != qs).sum())
                a = (qv.double().view(rows, gk // 32, 32) * torch.exp2(qs.double() - 127)[..., None]).view(rows, gk)
                b = (w[g].double().view(gn // 32, 32, gk // 32, 32)
                     * torch.exp2(scales[g].double() - 127)[:, None, :, None]).view(gn, gk)
                ref = a @ b.T
                got = out[:, g * gn:(g + 1) * gn].double()
                worst = max(worst, ((got - ref).abs().max() / ref.abs().max()).item())
            ok = mismatch == 0 and worst < 8e-3
            failures += not ok
            print(f'{label:28s} rows={rows:4d} quant_mismatch={mismatch} rel_err={worst:.2e}'
                  f'{" rope" if rope else ""} ok={ok}', flush=True)
    print('V41_FP8_ABI', 'PASS' if not failures else f'FAIL ({failures})')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
