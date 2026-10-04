#!/usr/bin/env python3
"""Compare opt-in EXL3 A8 prefill with the A16 export policy on SM120/SM121.

Run inside the matching architecture container, with the verified pinned fork.
Loads real MCG checkpoint trellises and rotations, but uses seeded synthetic
inputs/routes. Layer output error is a component gate, not model NLL/KL.
Spark arms consume identical FP8 K32 wire values; packed A16 component rows
(<=80) are checked byte-for-byte. Timings use interleaved warmed CUDA graph event nodes.
No defaults change and no measurement-only kernel modifications are installed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


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



def gpu_state():
    """Clock/power evidence is outside graph timing; do not alter GPU settings."""
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,name,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10, check=False)
    return {"query": result.args[1], "returncode": result.returncode,
            "values": result.stdout.strip(), "error": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--geometry", choices=("glm", "glmf", "qwen4"), required=True)
    parser.add_argument("--role", choices=("coordinator", "spark"), required=True)
    parser.add_argument("--width", type=int, help="Spark intermediate slice width")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 80, 2752, 4096])
    parser.add_argument("--replays", type=int, default=7)
    parser.add_argument("--steady-ms", type=int, default=1000)
    parser.add_argument("--max-rel-error", type=float, default=0.05)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (len(set(args.rows)) != len(args.rows) or args.replays < 3 or
            any(m < 1 or m > 8192 or 80 < m < 256 for m in args.rows)):
        parser.error("unique rows must be 1..80 or 256..8192; at least 3 replays")
    if args.steady_ms < 0 or not 0 < args.max_rel_error < 1 or not 0 < args.min_cosine <= 1:
        parser.error("invalid warm-up or component error thresholds")
    if os.environ.get("B12X_WS_EXP") or os.environ.get("B12X_WS_A8"):
        parser.error("unset experimental B12X_WS_* knobs; use explicit activations")
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
    cc = (12, 0) if args.role == "coordinator" else (12, 1)
    if (props.major, props.minor) != cc:
        raise ValueError(f"requires SM{cc[0]}{cc[1]}, got {props.name}")
    hidden, full_width, experts, topk = policy.GEOMETRIES[args.geometry]
    width = args.width or full_width
    if (width <= 0 or width % 128 or args.start < 0 or args.start % 128 or
            args.start + width > full_width or
            (args.role == "coordinator" and (args.start or width != full_width))):
        parser.error("use whole H128 Spark slice blocks; coordinator uses full width")
    bits = (3, 4) if args.geometry == "glmf" else (4, 5)
    output_dtype = "fp32" if args.role == "coordinator" else "bf16"
    started = time.monotonic()
    prepared = load_weights(args.snapshot, args.layer, hidden, width, experts,
        bits, start=args.start, full_width=full_width)
    record = dict(scope="Synthetic layer component comparison; not served TTFT or model quality",
        role=args.role, geometry=args.geometry, snapshot=str(args.snapshot), layer=args.layer,
        width=width, start=args.start, output_dtype=output_dtype, gpu=props.name,
        sms=props.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda,
        sparkinfer_revision=_pinned_sparkinfer.REVISION,
        sparkinfer_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weights_seconds=time.monotonic() - started, gpu_state_before=gpu_state(),
        trellis_weight_bytes=sum(v.numel()*v.element_size() for tier in prepared.tiers
            for v in (tier.w13, tier.w2)), results=[], passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        generator = torch.Generator(device="cpu").manual_seed(7 + rows)
        x = torch.randn(rows, hidden, generator=generator).to(torch.bfloat16).cuda()
        wire = None
        if args.role == "spark":
            groups = x.float().reshape(rows, hidden // 32, 32)
            exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(2.0**-100)/448)).clamp(-127,127)
            scales = torch.exp2(exponent)
            quantized = (groups / scales[..., None]).to(torch.float8_e4m3fn)
            wire = torch.cat((quantized.view(torch.uint8).reshape(rows, hidden),
                (exponent + 127).to(torch.uint8)), 1).contiguous()
            x = (quantized.float()*scales[..., None]).reshape(rows,hidden).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(experts, generator=generator)[:topk]
            for _ in range(rows)]).to(torch.int32).cuda()
        weights = torch.softmax(torch.randn(rows, topk, generator=generator), 1).cuda()
        arms = {}
        for name in ("a16", "a8"):
            candidate = name == "a8" and rows >= 256
            ws = candidate or policy.warp_specialized(args.geometry,args.role,width,rows)
            block = 64 if ws else policy.route_block(args.geometry,rows)
            tile = policy.ws_tile(args.geometry,args.role,width,rows)
            tile = tile or _projection_mixed_tile_config(None,hidden_size=hidden,
                intermediate_size=width,token_count=rows,direct_topk_routes=False)
            use_wire = (candidate and args.role == "spark") or policy.wire_input(args.geometry,args.role,width,rows)
            slots = route_pack_capacity(rows*topk,block,experts,topk=topk)[1]
            options = dict(size_m=rows,hidden_size=hidden,intermediate_size=width,
                tier0_num_experts=experts,tier1_num_experts=experts,
                route_num_experts=experts,top_k=topk,max_m_blocks=(slots+block-1)//block,
                sms=props.multi_processor_count,max_shared_mem=props.shared_memory_per_block_optin,
                force_tile_config=tile,tier0_bits=bits[0],tier1_bits=bits[1],trellis_codebook="mcg",
                swiglu_limit=policy.swiglu_limit(args.geometry),moe_block_size=block,
                rotation_input_dtype="bf16",full_rotation_output_dtype=output_dtype,
                route_ids_dtype=torch.int32,direct_topk_routes=False,
                token_major_rotation=not ws and policy.token_major_rotation(args.geometry,rows),
                fused_input_rotation=not candidate and policy.fused_input_rotation(args.geometry,args.role,width,rows),
                warp_specialized=ws,input_format="e4m3_k32" if use_wire else "bf16",
                ws_input_stages=policy.ws_input_stages(args.geometry,args.role,width,rows),
                ws_dynamic_tiles=policy.ws_dynamic_tiles(args.geometry,args.role,width,rows))
            if candidate:
                options["activations"] = "a8"
            launch = compile_mixed_trellis(**options)
            buffers = make_mixed_trellis_buffers(launch,device=torch.device("cuda",0),sms=props.multi_processor_count)
            binding = bind_mixed_trellis(*prepared.tiers,prepared.global_to_combined,
                prepared.descriptor_map,prepared.rotations,launch,
                gate_experts=prepared.gate_counts,up_experts=prepared.up_counts)
            operand = wire if use_wire else x
            def run(operand=operand,binding=binding,buffers=buffers):
                return run_bound_mixed_trellis(operand,weights,ids,binding,buffers)
            for _ in range(3):
                output = run()
            torch.cuda.synchronize()
            begin,end = (torch.cuda.Event(enable_timing=True,external=True) for _ in range(2))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                begin.record()
                output = run()
                end.record()
            graph.replay()
            end.synchronize()
            reference = output.clone()
            pointers = {n:v.data_ptr() for n,v in vars(buffers).items() if isinstance(v,torch.Tensor)}
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            end.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"] - before
            gate = dict(finite=bool(torch.isfinite(output).all()),nonzero=bool(output.count_nonzero()),
                stable_pointers=all(getattr(buffers,n).data_ptr()==p for n,p in pointers.items()),
                replay_allocations=allocations,
                replay_relative_error=float(torch.linalg.vector_norm(output.float()-reference.float())/
                    torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)))
            if (not gate["finite"] or not gate["nonzero"] or not gate["stable_pointers"] or
                    gate["replay_allocations"] or gate["replay_relative_error"] > 1e-5):
                raise RuntimeError(f"graph correctness failed for {name}, rows={rows}: {gate}")
            arms[name] = dict(graph=graph,begin=begin,end=end,output=output,
                reference=reference,buffers=buffers,binding=binding,run=run,samples=[],
                config=dict(activations="a8" if candidate else "a16",warp_specialized=ws,
                    wire_input=use_wire,block=block,tile=list(tile)),graph_gate=gate)
        a,b = (arms[n]["reference"].float().flatten() for n in ("a16","a8"))
        relative_error = float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30))
        cosine = float(torch.nn.functional.cosine_similarity(a,b,dim=0))
        quality = dict(relative_error=relative_error,cosine=cosine,
            max_abs=float((a-b).abs().max()),short_rows_equal=bool(torch.equal(a,b)) if rows<=80 else None)
        passed = (all(math.isfinite(v) for v in (relative_error,cosine)) and
            relative_error<=args.max_rel_error and cosine>=args.min_cosine and
            (rows>80 or quality["short_rows_equal"]))
        warm_until = time.monotonic()+args.steady_ms/1000
        while time.monotonic()<warm_until:
            for name in ("a16","a8"):
                for _ in range(8):
                    arms[name]["graph"].replay()
            torch.cuda.synchronize()
        for iteration in range(args.replays):
            for name in (("a16","a8") if iteration%2==0 else ("a8","a16")):
                arm=arms[name]
                arm["graph"].replay()
                arm["end"].synchronize()
                arm["samples"].append(arm["begin"].elapsed_time(arm["end"]))
        measured = {}
        for name,arm in arms.items():
            drift=float(torch.linalg.vector_norm(arm["output"].float()-arm["reference"].float())/
                torch.linalg.vector_norm(arm["reference"].float()).clamp_min(1e-30))
            passed &= math.isfinite(drift) and drift <= 1e-5
            measured[name]=dict(**arm["config"],graph_gate=arm["graph_gate"],
                final_replay_relative_error=drift,samples_ms=arm["samples"],
                median_ms=statistics.median(arm["samples"]))
        result=dict(rows=rows,quality=quality,passed=passed,arms=measured,
            speedup=measured["a16"]["median_ms"]/measured["a8"]["median_ms"],
            timing_method="interleaved_device_graph_event_nodes",gpu_state_after=gpu_state())
        record["results"].append(result)
        record["passed"] &= passed
        args.output.write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        torch.cuda.synchronize()
        del arms,run,graph,binding,buffers,output,reference,arm
    if not record["passed"]:
        raise SystemExit("EXL3 A8 component gate failed; see output JSON")


if __name__ == "__main__":
    main()
