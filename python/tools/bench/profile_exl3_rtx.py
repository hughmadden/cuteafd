#!/usr/bin/env python3
"""Profile resident SM120 EXL3 expert layers using the coordinator tile policy.

Run inside the coordinator development container with the pinned fork on
PYTHONPATH. Inputs and unique top-k routes are seeded synthetic data; weights
and all rotation/scale vectors come from the supplied MCG checkpoint. This
measures a layer, not model TTFT or accuracy. No policy or default is changed.
Use --profile with ncu --profile-from-start off or nsys --capture-range=cudaProfilerApi.
For --profile-graph, add nsys --cuda-graph-trace=node to expose kernel durations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def install_phase_clock(address):
    """Offline timestamps at existing grid barriers; never change kernel source.

    Embed a measurement-owned address in a fresh JIT specialization. The caller
    retains the tensor until all work drains. Five device-global nanosecond
    timestamps per CTA bracket input rotation, FC1, SwiGLU/rotation and FC2.
    These instrumented times must not be used as the unprofiled performance
    number. The trellis decode and MMA remain interleaved within each GEMM.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.base_dsl import dsl_user_op
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import W4A16MixedTrellisKernel

    @dsl_user_op
    def stamp(pointer, *, loc=None, ip=None):
        llvm.inline_asm(None,
            [cutlass.Int64(pointer).ir_value(loc=loc, ip=ip)],
            "{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$0], t; }",
            "l", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @cute.jit
    def timed_rotation(self, *args):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if tid == 0:
            stamp(cutlass.Int64(self._measurement_addr) + cutlass.Int64(cta) * 40)
        self._measurement_original_rotation(*args)

    @cute.jit
    def timed_barrier(self, locks, tid, grid_x):
        self._measurement_original_barrier(locks, tid, grid_x)
        cta, _, _ = cute.arch.block_idx()
        if tid == 0:
            # This coordinator path has exactly three grid barriers per wave.
            slot = (locks[self.barrier_sense_off] - 1) % 3 + 1
            stamp(cutlass.Int64(self._measurement_addr) + cutlass.Int64(cta) * 40
                  + cutlass.Int64(slot) * 8)

    @cute.jit
    def timed_body(self, *args, **kwargs):
        self._measurement_original_body(*args, **kwargs)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if tid == 0:
            stamp(cutlass.Int64(self._measurement_addr) + cutlass.Int64(cta) * 40 + 32)

    # Repeated row geometries update the embedded pointer, keeping the original
    # methods once. Disable executable caches since pointer identity is private.
    for cls, name, wrapper in (
        (W4A16MixedTrellisKernel, "rotation", timed_rotation),
        (W4A16FusedMoeKernel, "barrier", timed_barrier),
        (W4A16FusedMoeKernel, "body", timed_body),
    ):
        method = {"rotation": "_run_input_rotation_token_major",
                  "barrier": "_grid_barrier", "body": "_moe_body"}[name]
        original = f"_measurement_original_{name}"
        if not hasattr(cls, original):
            setattr(cls, original, getattr(cls, method))
        setattr(cls, method, wrapper)
        cls._measurement_addr = address


def load_weights(snapshot, layer, hidden, width, experts, bits):
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import (
        ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights,
    )

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    # Both checkpoints use model.language_model.layers. Infer the namespace
    # from the catalog, requiring a unique layer/expert/projection suffix.
    suffix = f"layers.{layer}.mlp.experts.0.gate_proj.trellis"
    matches = [name for name in index if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix}, got {matches}")
    prefix = matches[0][:-len("0.gate_proj.trellis")]
    raw, handles = {}, {}
    projections = ("gate_proj", "up_proj", "down_proj")
    for expert in range(experts):
        for projection in projections:
            base = f"{prefix}{expert}.{projection}"
            for field in ("trellis", "suh", "svh", "mcg"):
                name = f"{base}.{field}"
                shard = index[name]
                if shard not in handles:
                    handles[shard] = safe_open(snapshot / shard, framework="pt", device="cpu")
                value = handles[shard].get_tensor(name)
                if field == "mcg":
                    if value.numel() != 1 or (int(value.item()) & 0xffffffff) != 0xcbac1fed:
                        raise ValueError(f"unsupported MCG marker: {name}")
                else:
                    raw[expert, projection, field] = value.cuda()

    tier_bits = {(e, p): raw[e, p, "trellis"].shape[-1] // 16
                 for e in range(experts) for p in projections}
    if set(tier_bits.values()) != set(bits):
        raise ValueError(f"checkpoint tiers {set(tier_bits.values())} != {bits}")
    def rotation(projection, field):
        return torch.stack([raw[e, projection, field] for e in range(experts)])
    tiers = []
    for bit in bits:
        members = {p: tuple(e for e in range(experts) if tier_bits[e, p] == bit)
                   for p in projections}
        def packed(projection):
            shape = ((width // 16, hidden // 16, bit * 16) if projection == "down_proj"
                     else (hidden // 16, width // 16, bit * 16))
            values = [raw[e, projection, "trellis"] for e in members[projection]]
            if any(tuple(v.shape) != shape for v in values):
                raise ValueError(f"unexpected {projection} shape; expected {shape}")
            return (torch.stack(values) if values else
                    torch.empty((0, *shape), dtype=torch.int16, device="cuda"))
        tiers.append(ProjectionTrellisTierWeights(
            bit, torch.cat((packed("gate_proj"), packed("up_proj"))), packed("down_proj"),
            members["gate_proj"], members["up_proj"], members["down_proj"]))
    prepared = prepare_projection_native_trellis_weights(
        tuple(tiers), gate_suh=rotation("gate_proj", "suh"),
        up_suh=rotation("up_proj", "suh"),
        intermediate_rotations=torch.cat((rotation("gate_proj", "svh"),
            rotation("up_proj", "svh"), rotation("down_proj", "suh")), dim=1),
        down_svh=rotation("down_proj", "svh"), activation="silu",
        params_dtype=torch.bfloat16, num_experts=experts,
        hidden_size=hidden, intermediate_size=width)
    # Native ABI requires aligned dummy storage for a tier with no down planes.
    padded = []
    for bit, tier in zip(bits, prepared.tiers):
        changes = {}
        stride = (width // 16) * (hidden // 16) * (bit * 8)
        if tier.w13.numel() == 0:
            changes["w13"] = torch.zeros(stride, dtype=torch.int32, device="cuda")
        if tier.w2.numel() == 0:
            changes.update(w2=torch.zeros(stride, dtype=torch.int32, device="cuda"),
                           w2_global_scale=torch.ones(1, device="cuda"))
        padded.append(replace(tier, **changes))
    return replace(prepared, tiers=tuple(padded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--geometry", choices=("qwen4", "glmf"), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-graph", action="store_true", help="trace graph replays instead of eager launches")
    parser.add_argument("--profile-replays", type=int, default=1, help="launches per profiler range (1..64)")
    parser.add_argument("--phase-clock", action="store_true", help="instrument existing phase boundaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rows) != len(set(args.rows)):
        parser.error("row geometries must be unique")
    if args.replays < 3 or any(m < 2048 or m > 8192 for m in args.rows):
        parser.error("use at least three replays and 2048..8192 rows")
    if not 1 <= args.profile_replays <= 64 or ((args.profile_graph or args.profile_replays != 1) and not args.profile):
        parser.error("profiling options require --profile and 1..64 profile replays")
    if args.phase_clock:
        os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
        os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

    import torch
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers,
        bind_mixed_trellis, run_bound_mixed_trellis,
    )
    from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aot"))
    import package_exl3_aot as policy

    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise ValueError(f"requires SM120, got {props.name}")
    hidden, width, experts, topk = policy.GEOMETRIES[args.geometry]
    bits = (4, 5) if args.geometry == "qwen4" else (3, 4)
    started = time.monotonic()
    prepared = load_weights(args.snapshot, args.layer, hidden, width, experts, bits)
    record = dict(geometry=args.geometry, snapshot=str(args.snapshot), layer=args.layer,
                  measurement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sparkinfer_revision=_pinned_sparkinfer.REVISION,
                  sparkinfer_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
                  trellis_weight_bytes=sum(v.numel() * v.element_size()
                      for tier in prepared.tiers for v in (tier.w13, tier.w2)),
                  gpu=props.name, sms=props.multi_processor_count,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  weights_seconds=time.monotonic() - started, results=[])
    print(json.dumps({k: v for k, v in record.items() if k != "results"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        phase_clock = None
        if args.phase_clock:
            phase_clock = torch.zeros((props.multi_processor_count * 2, 5),
                                      dtype=torch.int64, device="cuda")
            install_phase_clock(phase_clock.data_ptr())
        block = policy.route_block(args.geometry, rows)
        tile = _projection_mixed_tile_config(None, hidden_size=hidden,
            intermediate_size=width, token_count=rows, direct_topk_routes=False)
        slots = route_pack_capacity(rows * topk, block, experts, topk=topk)[1]
        launch = compile_mixed_trellis(
            size_m=rows, hidden_size=hidden, intermediate_size=width,
            tier0_num_experts=experts, tier1_num_experts=experts,
            route_num_experts=experts, top_k=topk, max_m_blocks=(slots + block - 1) // block,
            sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
            force_tile_config=tile, tier0_bits=bits[0], tier1_bits=bits[1],
            trellis_codebook="mcg", swiglu_limit=policy.swiglu_limit(args.geometry),
            moe_block_size=block, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
            route_ids_dtype=torch.int32, direct_topk_routes=False,
            token_major_rotation=policy.token_major_rotation(args.geometry, rows),
            fused_input_rotation=policy.fused_input_rotation(args.geometry, "coordinator", width, rows),
            warp_specialized=policy.warp_specialized(args.geometry, "coordinator", width, rows))
        buffers = make_mixed_trellis_buffers(launch, device=torch.device("cuda", 0),
                                            sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*prepared.tiers, prepared.global_to_combined,
            prepared.descriptor_map, prepared.rotations, launch,
            gate_experts=prepared.gate_counts, up_experts=prepared.up_counts)
        generator = torch.Generator(device="cpu").manual_seed(7 + rows)
        x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16).cuda()
        ids = torch.stack([torch.randperm(experts, generator=generator)[:topk]
                           for _ in range(rows)]).to(torch.int32).cuda()
        weights = torch.softmax(torch.randn(rows, topk, generator=generator), 1).cuda()
        def run():
            return run_bound_mixed_trellis(x, weights, ids, binding, buffers)
        for _ in range(3):
            output = run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graph.replay()
        torch.cuda.synchronize()
        reference = output.clone()
        pointers = {name: value.data_ptr() for name, value in vars(buffers).items()
                    if isinstance(value, torch.Tensor)}
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        replay_allocations = torch.cuda.memory_stats()["allocation.all.allocated"] - allocations
        gate = dict(finite=bool(torch.isfinite(output).all()),
                    replay_equal=bool(torch.equal(reference, output)),
                    replay_allocations=replay_allocations,
                    stable_pointers=all(getattr(buffers, name).data_ptr() == ptr
                                        for name, ptr in pointers.items()))
        if not gate["finite"] or not gate["replay_equal"] or gate["replay_allocations"] or not gate["stable_pointers"]:
            raise RuntimeError(f"graph correctness gate failed: {gate}")
        samples, phase_samples = [], []
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"{args.geometry}/layer{args.layer}/rows{rows}")
            # Default eager launches let NCU inspect the cooperative kernel;
            # graph bursts are useful for Nsight Systems timing diagnostics.
            for _ in range(args.profile_replays):
                if args.profile_graph:
                    graph.replay()
                else:
                    run()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            # Device event nodes bracket the replay inside one graph launch,
            # excluding host submission gaps between separately recorded events.
            begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            timed_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed_graph):
                begin.record()
                run()
                end.record()
            for _ in range(args.replays):
                timed_graph.replay()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
                if phase_clock is not None:
                    phase_samples.append(phase_clock.cpu())
        gate["final_replay_equal"] = bool(torch.equal(reference, output))
        if not gate["final_replay_equal"]:
            raise RuntimeError(f"repeated replay changed the output: {gate}")
        result = dict(rows=rows, block=block, tile=list(tile),
                      timing_method="device_graph_event_nodes" if samples else None,
                      blocks_per_sm=launch.blocks_per_sm,
                      token_major_rotation=policy.token_major_rotation(args.geometry, rows),
                      graph_gate=gate, samples_ms=samples,
                      median_ms=statistics.median(samples) if samples else None)
        if phase_clock is not None:
            phases = []
            for clocks in phase_samples or [phase_clock.cpu()]:
                live = clocks[clocks[:, 0] > 0]
                boundaries = [int(live[:, 0].min()), *[int(live[:, j].max()) for j in range(1, 5)]]
                if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
                    raise RuntimeError(f"nonmonotonic phase clocks: {boundaries}")
                phases.append([(b - a) / 1e6 for a, b in zip(boundaries, boundaries[1:])])
            result["instrumented_phases_ms"] = dict(zip(
                ("input_rotation", "fc1_decode_mma", "swiglu_rotation", "fc2_decode_mma"),
                [statistics.median(p[j] for p in phases) for j in range(4)]))
            result["instrumented_phase_samples_ms"] = phases
            result["instrumented_ctas"] = len(live)
        record["results"].append(result)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not args.profile:
            del timed_graph, begin, end
        del graph, buffers, binding, output, reference


if __name__ == "__main__":
    main()
