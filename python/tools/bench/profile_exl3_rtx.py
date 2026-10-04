#!/usr/bin/env python3
"""Profile resident SM120/SM121 EXL3 expert layers using the export tile policy.

Run inside the matching architecture development container with the pinned fork on
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


def install_cooperative_diagnostic(kind):
    """Install a measurement-only decode or MMA elision with invalid numerics.

    This changes scheduling/register pressure and exposes overlap; its delta is
    a sensitivity measurement, not an additive fraction of production time.
    The measurement specialization never enters an executable cache.
    """
    import cutlass.cute as cute
    from cutlass.base_dsl import dsl_user_op
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16GemmKernel
    compile_hits = []

    @dsl_user_op
    def mark(*, loc=None, ip=None):
        compile_hits.append(1)

    @cute.jit
    def elided(self, frag, win_a, win_b, trellis_lut_addr, bits):
        mark()
        frag[0, 0] = win_a
        frag[0, 1] = win_b
        frag[1, 0] = win_a
        frag[1, 1] = win_b

    @cute.jit
    def no_mma(self, d0, d1, d2, d3, a0, a1, a2, a3, b0, b1):
        mark()
        return d0, d1, d2, d3

    if kind == "nodecode":
        W4A16GemmKernel._scaled_dequant_b_fragment_trellis256_bits = elided
    else:
        # Unused decode and operand loads may also disappear. This is a
        # combined consumer-path elision, never an isolated math-time fraction.
        W4A16GemmKernel._mma_m16n8k16_f32 = no_mma
        W4A16GemmKernel._mma_rhs_fragments_as_mma_a_m16n8k16_f32 = no_mma
    return compile_hits


def load_weights(snapshot, layer, hidden, width, experts, bits, *, start=0, full_width=None):
    full_width = full_width or width
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
                    if field == "trellis":
                        expected = ((full_width // 16, hidden // 16, value.shape[-1])
                                    if projection == "down_proj" else
                                    (hidden // 16, full_width // 16, value.shape[-1]))
                        if tuple(value.shape) != expected:
                            raise ValueError(f"unexpected checkpoint shape: {name}: {value.shape}")
                        value = (value[start // 16:(start + width) // 16] if projection == "down_proj"
                                 else value[:, start // 16:(start + width) // 16])
                    elif ((projection == "down_proj" and field == "suh") or
                          (projection != "down_proj" and field == "svh")):
                        value = value[start:start + width]
                    raw[expert, projection, field] = value.contiguous().cuda()

    tier_bits = {(e, p): raw[e, p, "trellis"].shape[-1] // 16
                 for e in range(experts) for p in projections}
    if not set(tier_bits.values()).issubset(bits):
        raise ValueError(f"checkpoint tiers {set(tier_bits.values())} outside supported {bits}")
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
    parser.add_argument("--geometry", choices=("qwen4", "glmf", "glm"), required=True)
    parser.add_argument("--role", choices=("coordinator", "spark"), default="coordinator")
    parser.add_argument("--width", type=int, help="Spark intermediate slice width (H128 blocks)")
    parser.add_argument("--start", type=int, default=0, help="Spark intermediate slice offset")
    parser.add_argument("--steady-ms", type=int, default=1000, help="sustained graph warm-up before timing")
    parser.add_argument("--diagnostic", choices=("nomma", "nodecode", "norot"), help="invalid-numerics ablation")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--kernel-times", action="store_true", help="collect a separate CUDA-only Torch profiler trace after timing")
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
    if args.kernel_times and args.profile:
        parser.error("kernel-times cannot be combined with an external profiler range")
    if args.phase_clock or args.diagnostic:
        os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
        os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

    if os.environ.get("B12X_WS_EXP") or os.environ.get("B12X_WS_A8"):
        parser.error("use --diagnostic for ablations; baseline must not inherit WS experiment/A8 knobs")
    if args.diagnostic:
        os.environ["B12X_WS_EXP"] = args.diagnostic
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
    expected_cc = (12, 0) if args.role == "coordinator" else (12, 1)
    if (props.major, props.minor) != expected_cc:
        raise ValueError(f"requires {expected_cc}, got {props.name}")
    if args.phase_clock and args.role != "coordinator":
        parser.error("phase clocks instrument only the cooperative coordinator kernel")
    decode_compile_hits = None
    if args.diagnostic and args.role == "coordinator":
        if args.diagnostic not in ("nodecode", "nomma"):
            parser.error("coordinator supports nodecode/nomma; WS norot requires Spark role")
        decode_compile_hits = install_cooperative_diagnostic(args.diagnostic)
    hidden, full_width, experts, topk = policy.GEOMETRIES[args.geometry]
    width = full_width if args.width is None else args.width
    if (width % 128 or args.start % 128 or width <= 0 or args.start < 0 or
            args.start + width > full_width or
            (args.role == "coordinator" and (args.start or width != full_width))):
        parser.error("slice must be whole H128 blocks inside the intermediate; coordinator uses full width")
    bits = (4, 5) if args.geometry in ("qwen4", "glm") else (3, 4)
    output_dtype = "fp32" if args.role == "coordinator" else "bf16"
    started = time.monotonic()
    if args.diagnostic == "norot" and policy.wire_input(args.geometry, args.role, width, args.rows[0]):
        parser.error("norot does not elide the FP8 wire rotation; use a dedicated wire-rotation probe")
    prepared = load_weights(args.snapshot, args.layer, hidden, width, experts, bits, start=args.start, full_width=full_width)
    record = dict(role=args.role, output_dtype=output_dtype, width=width, start=args.start, diagnostic=args.diagnostic, geometry=args.geometry, snapshot=str(args.snapshot), layer=args.layer,
                  measurement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sparkinfer_revision=_pinned_sparkinfer.REVISION,
                  sparkinfer_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
                  trellis_weight_bytes=sum(v.numel() * v.element_size()
                      for tier in prepared.tiers for v in (tier.w13, tier.w2)),
                  gpu=props.name, sms=props.multi_processor_count,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  weights_seconds=time.monotonic() - started, cuda_mem_free_total=list(torch.cuda.mem_get_info()), results=[])
    print(json.dumps({k: v for k, v in record.items() if k != "results"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        phase_clock = None
        if args.phase_clock:
            phase_clock = torch.zeros((props.multi_processor_count * 2, 5),
                                      dtype=torch.int64, device="cuda")
            install_phase_clock(phase_clock.data_ptr())
        ws = policy.warp_specialized(args.geometry, args.role, width, rows)
        wire_input = policy.wire_input(args.geometry, args.role, width, rows)
        block = 64 if ws else policy.route_block(args.geometry, rows)
        tile = _projection_mixed_tile_config(None, hidden_size=hidden,
            intermediate_size=width, token_count=rows, direct_topk_routes=False)
        tile = policy.ws_tile(args.geometry, args.role, width, rows) or tile
        slots = route_pack_capacity(rows * topk, block, experts, topk=topk)[1]
        launch = compile_mixed_trellis(
            size_m=rows, hidden_size=hidden, intermediate_size=width,
            tier0_num_experts=experts, tier1_num_experts=experts,
            route_num_experts=experts, top_k=topk, max_m_blocks=(slots + block - 1) // block,
            sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
            force_tile_config=tile, tier0_bits=bits[0], tier1_bits=bits[1],
            trellis_codebook="mcg", swiglu_limit=policy.swiglu_limit(args.geometry),
            moe_block_size=block, rotation_input_dtype="bf16", full_rotation_output_dtype=output_dtype,
            route_ids_dtype=torch.int32, direct_topk_routes=False,
            token_major_rotation=not ws and policy.token_major_rotation(args.geometry, rows),
            fused_input_rotation=policy.fused_input_rotation(args.geometry, args.role, width, rows),
            warp_specialized=ws,
            input_format="e4m3_k32" if wire_input else "bf16",
            ws_input_stages=policy.ws_input_stages(args.geometry, args.role, width, rows),
            ws_dynamic_tiles=policy.ws_dynamic_tiles(args.geometry, args.role, width, rows))
        if decode_compile_hits is not None and not decode_compile_hits:
            raise RuntimeError("cooperative elision did not reach this compiled kernel; discard the diagnostic")
        buffers = make_mixed_trellis_buffers(launch, device=torch.device("cuda", 0),
                                            sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*prepared.tiers, prepared.global_to_combined,
            prepared.descriptor_map, prepared.rotations, launch,
            gate_experts=prepared.gate_counts, up_experts=prepared.up_counts)
        generator = torch.Generator(device="cpu").manual_seed(7 + rows)
        x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16).cuda()
        if wire_input:
            groups = x.float().view(rows, hidden // 32, 32)
            exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(2.0 ** -100) / 448)).clamp(-127, 127)
            quantized = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
            x = torch.cat((quantized.view(torch.uint8).view(rows, hidden), (exponent + 127).to(torch.uint8)), 1).contiguous()
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
        if (gate["replay_allocations"] or not gate["stable_pointers"] or
                (not args.diagnostic and (not gate["finite"] or not gate["replay_equal"]))):
            raise RuntimeError(f"graph correctness gate failed: {gate}")
        # Keep the GPU occupied long enough to settle clocks before sampling.
        warm_until = time.monotonic() + max(0, args.steady_ms) / 1000
        while time.monotonic() < warm_until:
            for _ in range(16):
                graph.replay()
            torch.cuda.synchronize()
        print(json.dumps(dict(rows=rows, timing_start_unix=time.time())), flush=True)
        samples, phase_samples, sample_times = [], [], []
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
                sample_times.append(time.time())
                if phase_clock is not None:
                    phase_samples.append(phase_clock.cpu())
        kernel_times = None
        if args.kernel_times:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as trace:
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
            kernel_times = {}
            for event in trace.events():
                if event.device_type.name == "CUDA":
                    duration = event.device_time if hasattr(event, "device_time") else event.cuda_time
                    kernel_times[event.name] = kernel_times.get(event.name, 0) + duration / 10000
        gate["final_replay_equal"] = bool(torch.equal(reference, output))
        if not args.diagnostic and not gate["final_replay_equal"]:
            raise RuntimeError(f"repeated replay changed the output: {gate}")
        result = dict(rows=rows, block=block, tile=list(tile), warp_specialized=ws, wire_input=wire_input, sample_times_unix=sample_times,
                      profiled_kernel_times_ms=kernel_times,
                      timing_method="device_graph_event_nodes" if samples else None,
                      blocks_per_sm=launch.blocks_per_sm,
                      token_major_rotation=not ws and policy.token_major_rotation(args.geometry, rows),
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
