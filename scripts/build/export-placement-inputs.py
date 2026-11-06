#!/usr/bin/env python3
"""Publish CPU-compiled planner inputs for the live installed checkpoint inventory.

Run through scripts/build/cuteafd-dev.sh cpu, with --output and --staging on
NVMe. Publication never switches current; validate the candidate with
CUTEAFD_DEPLOYMENT_TEST_PROGRAMS=<output>/PROGRAMS.json before promoting it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("placement_compare", ROOT / "scripts/build/compare-sm-exports.py")
kernel_records = load("placement_kernel_records", ROOT / "scripts/build/kernel-records.py")


def live_configs(root=ROOT):
    retired = {entry["config"] for entry in json.loads((root / "benchmarks/current-matrix.json").read_text())["retired"]}
    return sorted(str(path.relative_to(root)) for path in (root / "examples/configs").glob("*.config")
                  if str(path.relative_to(root)) not in retired)


def source_hashes():
    # Keep identical to plan::testing::placement_input_source_hashes. Card report
    # rewrites do not affect compilation; only the selected example inventory does.
    paths = [ROOT / "scripts/build/export-placement-inputs.py", ROOT / "scripts/build/kernel-records.py",
             ROOT / "scripts/build/release-scope.py", ROOT / "scripts/lib/wip_aot_scopes.py",
             *(ROOT / config for config in live_configs(ROOT))]
    return {**compare.v4_placement_sources(),
            **{str(path.relative_to(ROOT)): compare.sha(path.read_bytes()) for path in paths},
            "installed_live_configs": compare.sha("\n".join(live_configs(ROOT)).encode())}


def export_jobs():
    scope = load("placement_scope", ROOT / "scripts/build/release-scope.py")
    experts = set()
    for config in live_configs():
        _, _, scopes = scope.entry_scopes(ROOT, {"config": config, "set": {}})
        experts.update(scope.tokens(scopes.get("CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES", "")))
    # The current release cells also include Qwen Next, whose old example path
    # has retired cards. Its live checkpoint still needs native workspace inputs.
    experts.add("qwen4:exl3-k45")
    geometries = [*compare.GEOMETRIES, "kimik3", "muse", "muse2", "ml4",
                  "dense_trellis_k6", "kimi_split", "expert_partial", "expert_reply",
                  "glm_stage", "glm2_stage", "glmf_stage", "glmf2_stage", "exlr8", "exlr8_dense"]
    jobs = [("programs", "export_b12x_dsv4_aot.py", [
        "--geometry", ",".join(geometries), "--max-context", "1048576",
        "--decode-rows", "64", "--prefill-rows", "4096", "--context-split", "on",
        "--dense-trellis-families", "kimik3,glm,glmf,step5,muse,ml4",
        "--exlr8-family", "glm,glmf", "--exlr8-role", "coordinator"])]
    for expert in sorted(experts):
        family, kind = expert.split(":")
        if kind.startswith("exl3-k"):
            bits = kind.removeprefix("exl3-k")
            jobs.append((f"exl3/exl3-{family}-k{bits}", "package_exl3_aot.py", [
                "--geometry", family, "--bits", ",".join(bits), "--capacities",
                compare.EXL3_CAPACITIES_KIMIK3 if family == "kimik3" else compare.EXL3_CAPACITIES,
                "--role", "coordinator"]))
        elif kind in ("fp8", "bf16"):
            geometry = family + ("_bf16" if kind == "bf16" else "")
            suffix = "-bf16-weights" if kind == "bf16" else ""
            jobs.append((f"fp8/fp8-{family}{suffix}", "package_fp8_moe_aot.py", [
                "--geometry", geometry, "--capacities", compare.EXL3_CAPACITIES,
                "--layouts", "tp1,tp2", "--role", "coordinator"]))
    return jobs


def verified_job_files(previous, name):
    metadata = json.loads((previous / "INPUTS.json").read_text())
    prefix = ".workers/" + name + "/"
    files = {path[len(prefix):]: checksum for path, checksum in metadata["files"].items()
             if path.startswith(prefix)}
    if not files:
        return None
    directory = previous / ".workers" / name
    for relative, checksum in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or compare.sha((directory / path).read_bytes()) != checksum:
            raise ValueError(f"invalid or corrupt replay file: {name}/{relative}")
    return directory, files


def refresh_replay_provenance(directory, lock):
    for path in directory.rglob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "sparkinfer_revision" in data:
            data["sparkinfer_revision"] = lock["revision"]
            if "source_tree_sha256" in data:
                data["source_tree_sha256"] = lock["source_tree_sha256"]
            path.write_text(json.dumps(data, indent=2) + "\n")
    for path in directory.rglob("v41_exl3.json"):
        data = json.loads(path.read_text())
        route = data.get("route_preparation")
        if route:
            route["sha256"] = compare.sha((path.parent / route["manifest"]).read_bytes())
            path.write_text(json.dumps(data, indent=2) + "\n")


def compile_job(job, staging, previous, lock, timeout, backend_cache=None, verify_skips=False):
    name, exporter, extra = job
    output = staging / name
    output.mkdir(parents=True)
    payload = dict(sms=188, exporter=exporter, args=extra, output=str(output), program_records=True)
    if backend_cache:
        payload["backend_cache"] = str(backend_cache)
    prior = verified_job_files(previous, name) if previous else None
    skipped = False
    if prior:
        prior_directory, prior_files = prior
        prior_context = json.loads((prior_directory / "comparison-context.json").read_text())
        current_inputs = kernel_records.stage0_inputs(ROOT, payload, (12, 0))
        triggers = kernel_records.job_candidates(ROOT, prior_context, current_inputs)
        print(f"Placement job {name}: candidate triggers={triggers}", flush=True)
        skipped = not triggers
        if skipped and not verify_skips:
            for relative in prior_files:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prior_directory / relative, target)
            refresh_replay_provenance(output, lock)
            summary = json.loads((output / "comparison-context.json").read_text())
            summary["execution_stats"] = {"stage0_skipped": True, "verified_skip": False,
                                          "backend_hits": 0, "backend_misses": 0}
            return summary
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTEAFD_SOURCE_ROOT=str(ROOT),
               CUTEAFD_SPARKINFER_SOURCE_DIR=str(ROOT / "third_party/sparkinfer"),
               CUTEAFD_SPARKINFER_LOCK_FILE=str(ROOT / "third_party/sparkinfer.lock.json"))
    print(f"Placement export {name}: {extra}", flush=True)
    with (output / "export.log").open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"),
                        "--_worker", json.dumps(payload)], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)
    if skipped:
        old_records = json.loads((prior_directory / "program-records.json").read_text())
        new_records = json.loads((output / "program-records.json").read_text())
        if ({key: record["ir_sha256"] for key, record in old_records.items()}
                != {key: record["ir_sha256"] for key, record in new_records.items()}):
            raise ValueError(f"--verify-skips changed IR in skipped job {name}")
    summary = json.loads((output / "comparison-context.json").read_text())
    reuse = summary.get("backend_reuse", {})
    summary["execution_stats"] = {"stage0_skipped": skipped, "verified_skip": skipped and verify_skips,
                                  "backend_hits": reuse.get("hits", 0), "backend_misses": reuse.get("misses", 0)}
    return summary


def publish(destination, staging, timeout, workers, previous=None, expect=None,
            backend_cache=None, verify_skips=False):
    filesystem = compare.load_file("placement_filesystem", ROOT / "scripts/build/assert-build-filesystem.py")
    if backend_cache:
        filesystem.check_path(str(backend_cache))
    verify_skips = verify_skips or os.environ.get("CUTEAFD_BUILD_KIND") == "release"
    for path in (destination, staging):
        filesystem.check_path(str(path))
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"output is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = source_hashes()
    lock = json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text())
    engine = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    jobs = export_jobs()

    def compile_one(job):
        return compile_job(job, staging, previous, lock, timeout, backend_cache, verify_skips)

    report_previous = previous
    with ThreadPoolExecutor(max_workers=workers) as pool:
        summaries = list(pool.map(compile_one, jobs))
    if sources != source_hashes() or lock != json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text()):
        raise ValueError("placement export sources changed during compilation")
    if any(summary.get("sparkinfer_revision") != lock["revision"]
           or summary.get("source_tree_sha256") != lock["source_tree_sha256"]
           or summary.get("cuda_initialized") is not False for summary in summaries):
        raise ValueError("placement worker provenance mismatch or initialized CUDA")
    shutil.copy2(staging / "programs/dsv4_programs.json", destination / "PROGRAMS.json")
    for name, exporter, _ in jobs[1:]:
        target = destination / name
        if exporter == "package_exl3_aot.py":
            for source in (staging / name).rglob("v41_exl3.json"):
                copied = target / source.relative_to(staging / name)
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copied)
        else:
            # These capacities are the production compiler's exact scratch
            # query, including the maximum across each capacity's prefill forms.
            records = json.loads((staging / name / "routed.json").read_text())["programs"]
            layouts = {}
            for record in records:
                capacities = layouts.setdefault(record["layout"], {})
                rows = record["capacity"]
                capacities[rows] = max(capacities.get(rows, 0), record["scratch_bytes"])
            manifest = {"layouts": {layout: {"capacities": [
                {"capacity": rows, "scratch_bytes": scratch} for rows, scratch in sorted(capacities.items())]}
                for layout, capacities in layouts.items()}}
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Preserve only the exact metadata needed to replay a job, not heavy objects.
    for name, _, _ in jobs:
        for source in (staging / name).rglob("*.json"):
            if source.name == "export.log":
                continue
            target = destination / ".workers" / name / source.relative_to(staging / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    records = publication_records(staging, jobs)
    if records:
        programs_path = destination / "PROGRAMS.json"
        programs = json.loads(programs_path.read_text())
        programs["kernel_records"] = records
        programs_path.write_text(json.dumps(programs, indent=2) + "\n")
    files = {str(path.relative_to(destination)): compare.sha(path.read_bytes())
             for path in sorted(destination.rglob("*.json"))}
    provenance = dict(schema=1, engine_commit=engine, sparkinfer_revision=lock["revision"],
                      source_tree_sha256=lock["source_tree_sha256"], engine_sources_sha256=sources,
                      exports=summaries, files=files, wall_seconds=round(time.monotonic() - started, 3))
    provenance["program_records"] = records
    (destination / "INPUTS.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if report_previous:
        old = json.loads((report_previous / "INPUTS.json").read_text()).get("program_records")
        if old is None:
            raise ValueError("previous publication has no program records; seed it with a cold export")
        expected = json.loads(expect.read_text()) if expect else None
        if isinstance(expected, dict):
            expected = expected["groups"]
        if expected is not None and (not isinstance(expected, list) or not all(isinstance(x, str) for x in expected)):
            raise ValueError("--expect must contain a JSON list of export groups, or {groups: [...]}")
        report = kernel_records.change_report(old, records, expected)
        kernel_records.write_report(destination, report)
        if report["leaks"]:
            raise ValueError(f"UNEXPECTED KERNEL CHANGES outside --expect: {report['leaks']}")
    print(f"Published placement inputs: {destination}; {provenance['wall_seconds']} s; "
          f"{sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())} bytes", flush=True)


def publication_records(staging, jobs):
    records = {}
    for name, _, _ in jobs:
        directory = staging / name
        path = directory / "program-records.json"
        if not path.exists():
            continue  # Test fixtures can represent metadata-only workers.
        job_records = json.loads(path.read_text())
        if (directory / "dsv4_programs.json").exists():
            manifest = json.loads((directory / "dsv4_programs.json").read_text())
            for program in manifest.get("programs", []):
                for plan in program.get("card_plans", [program]):
                    label = plan.get("label", program["name"])
                    record = job_records[label + ".o"]
                    record["scratch_bytes_at_capacity"] = {
                        str(program["capacity_rows"]): plan.get("scratch_bytes_at_capacity", {})}
                    record["family"] = program["family"]
                    record["planner_metadata"] = {key: value for key, value in program.items()
                                                  if key not in ("object_sha256", "card_plans")}
                    record["planner_metadata"]["plan"] = plan
        elif name.startswith("fp8/"):
            for program in json.loads((directory / "routed.json").read_text())["programs"]:
                job_records[program["name"] + ".o"]["scratch_bytes_at_capacity"] = {
                    str(program["capacity"]): program["scratch_bytes"]}
                job_records[program["name"] + ".o"]["planner_metadata"] = program
        else:
            for manifest_path in directory.rglob("v41_exl3.json"):
                manifest = json.loads(manifest_path.read_text())
                for record_path, record in job_records.items():
                    if Path(record_path).parent == manifest_path.parent.relative_to(directory):
                        record["scratch_bytes_at_capacity"] = {
                            str(manifest["capacity"]): manifest["unique_execution_buffer_bytes"]}
                        record["planner_metadata"] = {key: value for key, value in manifest.items()
                                                      if key not in ("sparkinfer_revision", "gpu", "physical_sms", "objects")}
                        record["planner_metadata"]["objects"] = manifest["objects"]
        records.update({f"{name}/{path}": {**record, "group": name} for path, record in job_records.items()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--previous", type=Path, help="previous publication to compare; writes CHANGES.json and CHANGES.md")
    parser.add_argument("--expect", type=Path, help="JSON list of export job groups expected to change")
    parser.add_argument("--backend-cache", type=Path, help="content-addressed lowered-MLIR cache on NVMe")
    parser.add_argument("--verify-skips", action="store_true", help="run skipped frontends and reject changed IR")
    args = parser.parse_args()
    if args.expect and not args.previous:
        parser.error("--expect requires --previous")
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    publish(args.output.expanduser().resolve(), args.staging.expanduser().resolve(), args.timeout, args.workers,
            args.previous.expanduser().resolve() if args.previous else None,
            args.expect.expanduser().resolve() if args.expect else None,
            args.backend_cache.expanduser().resolve() if args.backend_cache else None, args.verify_skips)


if __name__ == "__main__":
    main()
