"""Compact GQA ring writer bytes and direct-reader dequantized-oracle gate.

Run inside a matching CUDA container: python native/tests/draft_kv_selftest.py LIBRARY
"""
import argparse
import ctypes as C

import torch

P, I, F = C.c_void_p, C.c_int32, C.c_float


def bind(path):
    lib = C.CDLL(path)
    signatures = {
        "glm_dflash_qk_rope": ([P] * 8 + [I] * 3 + [F] * 2 + [P], I),
        "mimo_dflash_qk_rope": ([P] * 8 + [I] * 4 + [F] * 3 + [P], I),
        "block_dspark_qk_rope": ([P] * 9 + [I] * 4 + [F] * 3 + [P], I),
        "glm_dflash_attention": ([P] * 10 + [I] * 7 + [F, P], I),
        "mimo_dflash_attention": ([P] * 11 + [I] * 7 + [F, P], I),
        "block_dspark_attention": ([P] * 10 + [I] * 8 + [F, P], I),
    }
    for name, (args, result) in signatures.items():
        for suffix in ("", "_fp8"):
            function = getattr(lib, "cuteafd_" + name + suffix)
            function.argtypes, function.restype = args, result
    for name, count in (("block_dspark", 6), ("glm_dflash", 3), ("mimo_dflash", 5)):
        function = getattr(lib, "cuteafd_" + name + "_attention_workspace")
        function.argtypes, function.restype = [I] * count, C.c_uint64
    return lib


def plane_bytes(heads, dim):
    return (heads * (dim + 4) + 15) // 16 * 16


def pack(x):
    """CPU round-to-nearest division matches the shared Qwen/Step-5 writer."""
    x = x.cpu().float()
    rows, heads, dim = x.shape
    amax = x.abs().amax(-1)
    scales = torch.where(amax > 0, amax / 448.0, 1.0)
    inverse = 1.0 / scales
    codes = (x * inverse[..., None]).to(torch.float8_e4m3fn)
    out = torch.zeros(rows, plane_bytes(heads, dim), dtype=torch.uint8)
    out[:, :heads * dim] = codes.view(torch.uint8).reshape(rows, -1)
    out[:, heads * dim:heads * (dim + 4)] = scales.contiguous().view(torch.uint8).reshape(rows, -1)
    return out.cuda()


def unpack(x, heads, dim):
    codes = x[:, :heads * dim].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, heads, dim)
    scales = x[:, heads * dim:heads * (dim + 4)].contiguous().view(torch.float32)
    return (codes * scales[..., None]).bfloat16()


def writer_check(lib):
    stream = torch.cuda.current_stream().cuda_stream
    positions = torch.tensor([0, 63, 64, 2047, 2048, 262143], device="cuda", dtype=torch.int64)
    slots = torch.tensor([3, -1, 0, 5, 2, 4], device="cuda", dtype=torch.int32)
    for family, dim, heads, kvh in (("glm", 128, 8, 2), ("mimo", 128, 8, 2),
                                    ("dspark", 64, 8, 1), ("dspark", 128, 8, 2)):
        for queries in (0, heads):
            rows = len(positions)
            source = torch.randn(rows, queries + 2 * kvh, dim, device="cuda").bfloat16()
            source[0].zero_()
            qnorm = (torch.randn(dim, device="cuda") * .1 + 1).bfloat16()
            knorm = (torch.randn(dim, device="cuda") * .1 + 1).bfloat16()
            q = torch.empty(rows, queries, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros(rows, kvh, dim, device="cuda", dtype=torch.bfloat16)
            v = torch.zeros_like(k)
            cq = torch.empty_like(q)
            ck = torch.full((rows, plane_bytes(kvh, dim)), 0xA5, device="cuda", dtype=torch.uint8)
            cv = torch.full_like(ck, 0xA5)
            inv = torch.pow(10000., -torch.arange(dim // 2, device="cuda", dtype=torch.float32) * 2 / dim)
            name = {"glm": "glm_dflash", "mimo": "mimo_dflash", "dspark": "block_dspark"}[family]
            prefix = [source.data_ptr(), qnorm.data_ptr(), knorm.data_ptr(), positions.data_ptr(), slots.data_ptr()]
            tail = [rows, queries, kvh]
            if family == "dspark":
                prefix += [inv.data_ptr() if dim == 128 else None]
                tail += [dim, 10000., 1., 1e-6]
            elif family == "mimo":
                tail += [64, 10000., 1e-6, 0.75]
            else:
                tail += [10000., 1e-6]
            for suffix, outputs in (("", (q, k, v)), ("_fp8", (cq, ck, cv))):
                assert getattr(lib, "cuteafd_" + name + "_qk_rope" + suffix)(
                    *prefix, *[a.data_ptr() for a in outputs], *tail, stream) == 0
            torch.cuda.synchronize()
            assert torch.equal(q, cq), (family, dim, "query bytes")
            for expected, actual in ((pack(k), ck), (pack(v), cv)):
                written = torch.tensor([0, 2, 3, 4, 5], device="cuda")
                assert torch.equal(expected[written], actual[written]), (family, dim, "record bytes")
                assert torch.all(actual[1] == 0xA5), (family, dim, "negative slot wrote")
            print(f"writer {family} D{dim} Q{queries} H{kvh}: exact codes/scales/padding/query; negative slot intact")


def attention_check(lib):
    stream = torch.cuda.current_stream().cuda_stream
    for family, dim, heads, kvh, block in (("glm", 128, 8, 2, 8), ("mimo", 128, 16, 2, 8),
                                           ("dspark", 64, 2, 1, 8), ("dspark", 128, 4, 2, 7)):
        for sequences in (1, 3):
            for context in (0, 33, 1024, 1031):
                ring = 1088
                slots = torch.arange(sequences - 1, -1, -1, device="cuda", dtype=torch.int32)
                lengths = torch.tensor([max(0, context - 7 * i) for i in range(sequences)], device="cuda", dtype=torch.int32)
                ends = torch.tensor([2 * ring + context + 17 + i for i in range(sequences)], device="cuda", dtype=torch.int32)
                original_k = torch.randn(sequences * ring, kvh, dim, device="cuda").bfloat16()
                original_v = torch.randn_like(original_k)
                ck, cv = pack(original_k), pack(original_v)
                k, v = unpack(ck, kvh, dim), unpack(cv, kvh, dim)
                kb = torch.randn(sequences * block, kvh, dim, device="cuda").bfloat16()
                vb = torch.randn_like(kb)
                q = torch.empty(sequences * block, heads, dim, device="cuda", dtype=torch.bfloat16)
                output, compact = torch.empty_like(q), torch.empty_like(q)
                if family == "glm":
                    size = lib.cuteafd_glm_dflash_attention_workspace(sequences, kvh, context + block)
                elif family == "mimo":
                    size = lib.cuteafd_mimo_dflash_attention_workspace(sequences, heads, kvh, block, context + block)
                else:
                    size = lib.cuteafd_block_dspark_attention_workspace(sequences, heads, kvh, block, context + block, dim)
                assert size > 0
                scratch = torch.empty(size, device="cuda", dtype=torch.uint8)
                sinks = torch.randn(heads, device="cuda").bfloat16()
                name = {"glm": "glm_dflash", "mimo": "mimo_dflash", "dspark": "block_dspark"}[family]
                for amplitude in (.2, 4.):  # Flat AND peaked score regimes.
                    q.normal_().mul_(amplitude)
                    for masking in (0, 1):
                        common = [q.data_ptr(), kb.data_ptr(), vb.data_ptr()]
                        tables = [slots.data_ptr(), lengths.data_ptr(), ends.data_ptr()]
                        dims = [sequences, block, heads, kvh, ring, context + block]
                        dims += [masking, dim] if family == "dspark" else [128 * masking]
                        def launch(suffix, kr, vr, out):
                            pointers = common + [kr.data_ptr(), vr.data_ptr()] + tables
                            if family == "mimo":
                                pointers += [sinks.data_ptr()]
                            assert getattr(lib, "cuteafd_" + name + "_attention" + suffix)(
                                *pointers, out.data_ptr(), scratch.data_ptr(), *dims, dim ** -.5,
                                torch.cuda.current_stream().cuda_stream) == 0
                        launch("", k, v, output)
                        launch("_fp8", ck, cv, compact)
                        torch.cuda.synchronize()
                        assert torch.equal(output, compact), (family, dim, sequences, context, amplitude, masking,
                            (output.float() - compact.float()).abs().max().item())
                        if context == 33 and amplitude == 4. and masking == 1:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                launch("_fp8", ck, cv, compact)
                            compact.zero_()
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(output, compact), "graph replay changed compact reader"
                print(f"reader {family} D{dim} C{sequences} context={context}: byte-exact dequantized oracle, flat/peaked, masks/wrap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    args = parser.parse_args()
    torch.manual_seed(595)
    lib = bind(args.library)
    writer_check(lib)
    attention_check(lib)
    print("compact GQA ring kernel gate passed")


if __name__ == "__main__":
    main()
