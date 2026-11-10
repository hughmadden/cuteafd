"""Native MXFP4 resident geometry and hardware-qualified slice recipes."""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

LAYOUTS_PATH = Path(__file__).resolve().parents[3] / "scripts/build/mxfp4-layouts.json"


def layouts() -> dict:
    data = json.loads(LAYOUTS_PATH.read_text())
    if data["schema"] != 1:
        raise ValueError("unsupported MXFP4 layout schema")
    validate(data)
    return data


def validate(data: dict) -> None:
    decode = data["decode_capacities"]
    if not decode or decode[0] != 1 or decode != sorted(set(decode)) or decode[-1] > 80 or data["prefill_capacities"] != [256, 1024, 4096]:
        raise ValueError("invalid native MXFP4 capacity recipe")
    sms = set()
    for row in data["stream_prefill"]:
        cutoff = row["min_work"]
        if (row["sm"] not in (120, 121) or row["sm"] in sms
                or cutoff is not None and (type(cutoff) is not int or cutoff <= 0)):
            raise ValueError("invalid native MXFP4 STREAM policy")
        sms.add(row["sm"])
    if sms != {120, 121}:
        raise ValueError("native MXFP4 STREAM policy must name both hardware classes")
    hidden = set()
    for row in data["atomic_slices"]:
        if (row["hidden"] <= 0 or row["hidden"] % 256 or row["hidden"] in hidden
                or row["min_capacity"] not in (256, 1024, 4096)):
            raise ValueError("invalid native MXFP4 atomic policy")
        hidden.add(row["hidden"])
    keys = set()
    for row in data["native_slices"]:
        key = (row["hidden"], row["width"], row["sm"])
        prefill = row.get("prefill", [])
        if key in keys or len(set(prefill)) != len(prefill) or any(c not in (256, 1024, 4096) for c in prefill):
            raise ValueError("invalid native MXFP4 prefill recipe")
        keys.add(key)
        if not row["capacities"] and not prefill:
            raise ValueError("empty native MXFP4 recipe")
        recipes = dict(zip(row["capacities"], row["slices"], strict=True))
        if len(recipes) != len(row["capacities"]):
            raise ValueError("native MXFP4 capacities must pair uniquely with slices")
        if recipes and len({width for capacity, width in recipes.items() if capacity <= 80}) != 1:
            raise ValueError("native MXFP4 decode capacities must share one slice recipe")
        if any(width not in (32, 64, 128, 192) for width in recipes.values()):
            raise ValueError("invalid native MXFP4 slice width")
        splits = row.get("fc1_k_splits", [1] * len(recipes))
        split_recipes = dict(zip(row["capacities"], splits, strict=True))
        if any(split not in (1, 2, 4, 8) for split in splits):
            raise ValueError("invalid native MXFP4 FC1 K split")
        if split_recipes and len({split for capacity, split in split_recipes.items() if capacity <= 80}) != 1:
            raise ValueError("native MXFP4 decode capacities must share one FC1 K split")
        if any(split != 1 for capacity, split in split_recipes.items() if capacity > 80):
            raise ValueError("native MXFP4 prefill FC1 K splitting is unqualified")

    for row in data["ownership"]:
        if (row["unit"] not in (32, 64, 128) or row["intermediate"] % row["unit"]
                or not 1 <= row["world"] <= 8 or row["intermediate"] // row["unit"] < row["world"]):
            raise ValueError("invalid measured MXFP4 ownership")


def ownership_unit(intermediate: int, world: int) -> int:
    return next((row["unit"] for row in layouts()["ownership"]
                 if (row["intermediate"], row["world"]) == (intermediate, world)), 32)


def generic_stored_width(hidden: int, width: int) -> int:
    return next((row["stored"] for row in layouts()["generic_storage_holds"]
                 if (row["hidden"], row["width"]) == (hidden, width)), width)


def stored_width(hidden: int, width: int) -> int:
    if hidden <= 0 or hidden % 256 or width <= 0 or width % 32 or width > 8192:
        raise ValueError("invalid native MXFP4 geometry")
    return next((row["stored"] for row in layouts()["storage_holds"]
                 if (row["hidden"], row["width"]) == (hidden, width)), width)


def native_slice_width(hidden: int, width: int, sm: int, capacity: int,
                       matching_width: int) -> int:
    for row in layouts()["native_slices"]:
        if (row["hidden"], row["width"], row["sm"]) == (hidden, width, sm):
            return dict(zip(row["capacities"], row["slices"], strict=True)).get(
                capacity, matching_width)
    return matching_width


def _default_prefill_unit_rows(source: str) -> int:
    def constant(name):
        match = re.search(rf"^pub const {name}: u64 = ([0-9][0-9_]*);$", source, re.MULTILINE)
        if match is None:
            raise ValueError(f"cannot parse Rust literal {name} for native MXFP4 prefill policy")
        return int(match[1].replace("_", ""))
    reserve, depth, page, max_waves = (constant("PREFILL_" + name) for name in
                                      ("RESERVE_ROWS", "DEPTH", "ROW_PAGE", "MAX_WAVES"))
    if reserve <= 0 or page <= 0 or max_waves < 2:
        raise ValueError("invalid Rust prefill capacity constants")
    waves = min(max(depth, 2), max_waves)
    return ((reserve + waves - 1) // waves + page - 1) // page * page


@lru_cache(maxsize=1)
def default_prefill_unit_rows() -> int:
    """Use the planner's literal shared contract, including its wave clamp."""
    source = (LAYOUTS_PATH.parents[2] / "rust/crates/cuteafd-core/src/serving_capacity.rs").read_text()
    return _default_prefill_unit_rows(source)


def native_prefill_work(sm: int, rows: int, width: int, experts: int, topk: int) -> bool:
    cutoff = next((row["min_work"] for row in layouts()["stream_prefill"] if row["sm"] == sm), None)
    return (cutoff is not None and rows > 0 and width > 0 and experts > 0 and topk > 0
            and rows * topk * width >= cutoff * experts)


def native_prefill(hidden: int, width: int, sm: int, capacity: int,
                   experts: int, topk: int) -> bool:
    """Static export route tuned to the default admitted wave, not compiled padding.

    Larger custom waves retain this correct, default-tuned route. Hot-only screens
    cannot qualify one: the crossover uses the checkpoint expert population.
    """
    rows = min(capacity, default_prefill_unit_rows())
    return native_prefill_work(sm, rows, width, experts, topk) and any(
        (row["hidden"], row["width"], row["sm"]) == (hidden, width, sm)
        and capacity in row.get("prefill", []) for row in layouts()["native_slices"])


def native_prefill_scratch(capacity: int, hidden: int, width: int,
                           experts: int, topk: int) -> tuple[list[dict], int]:
    """Pure counterpart of the pinned STREAM's seven caller-owned planes."""
    if capacity not in (256, 1024, 4096) or hidden <= 0 or hidden % 512 or width <= 0 or width % 32 or experts <= 0 or topk <= 0:
        raise ValueError("invalid native V4 prefill geometry")
    pairs = capacity * topk
    tiles = (pairs + 127) // 128 + min(experts, pairs)
    row_bytes = (width + width // 32 + 15) // 16 * 16
    specs = [("meta", "i32", (4 + 3 * experts + tiles,)),
             ("pair_row", "i32", (pairs,)), ("pair_pos", "i32", (pairs,)),
             ("act", "u8", (tiles * 128 * row_bytes,)),
             ("grouped_routing", "f32", (pairs,)), ("grouped_route", "i32", (pairs,)),
             ("output", "f32", (pairs, hidden))]
    offset, planes = 0, []
    for name, dtype, shape in specs:
        size = 1 if dtype == "u8" else 4
        for extent in shape:
            size *= extent
        offset = (offset + 15) // 16 * 16
        planes.append(dict(name=name, dtype=dtype, shape=shape, offset=offset, nbytes=size))
        offset += size
    return planes, offset


def native_atomic_slices(hidden: int, role: str, capacity: int) -> bool:
    return (role in ("spark", "spark_tp2", "spark_tp3", "spark_tp4", "spark_tp6", "spark_tp8",
                     "rtx_backbone", "rtx_tp2") and any(
        row["hidden"] == hidden and capacity >= row["min_capacity"]
        for row in layouts()["atomic_slices"]))


def native_scratch_bytes(hidden: int, width: int, experts: int, topk: int,
                         capacity: int, sm: int, role: str, coded: bool) -> int:
    """Exact standard export census, excluding the owner's shared I/O planes."""
    stored_width(hidden, width)
    if experts <= 0 or topk <= 0 or capacity not in (1, 16, 64, 80, 256, 1024, 4096):
        raise ValueError("invalid native scratch geometry")
    if native_prefill(hidden, width, sm, capacity, experts, topk):
        end = native_prefill_scratch(capacity, hidden, width, experts, topk)[1]
        for _ in range(2):
            end = -(-end // 16) * 16 + 4 * experts
        return end
    matching = 192 if coded and (not role.startswith("spark_w") or width % 192 == 0) else 128
    tile = native_slice_width(hidden, width, sm, capacity, matching) if sm == 121 else matching
    atomic = native_atomic_slices(hidden, role, capacity)
    pairs = capacity * topk
    partial = 4 if atomic else -(-width // tile) * pairs * (tile + tile // 8)
    end = 0
    for size in (4, experts * pairs * 4, experts * 4, experts * 8,
                 pairs * 76, pairs * 4, pairs * 4, partial,
                 (capacity if atomic else pairs) * hidden * 4, experts * 4, experts * 4):
        end = -(-end // 16) * 16 + size
    return end


def assert_native_scratch_variant(variant: dict, hidden: int, width: int, experts: int,
                                  topk: int, sm: int, role: str, coded: bool) -> None:
    capacity = variant["capacity_rows"]
    stream = native_prefill(hidden, width, sm, capacity, experts, topk)
    atomic = not stream and native_atomic_slices(hidden, role, capacity)
    expected = native_scratch_bytes(hidden, width, experts, topk, capacity, sm, role, coded)
    if (variant["core_scratch_nbytes"] != expected
            or variant["output_kind"] != ("fp32_tokens" if atomic else "fp32_routes")):
        raise ValueError("native exporter scratch/output differs from planner census")


def native_export_capacities(family: str, role: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("native_v41_capacities", Path(__file__).with_name("v41_capacities.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DECODE_CAPACITIES = module.DECODE_CAPACITIES
    decode = (DECODE_CAPACITIES if family == "v41" and role not in ("coordinator", "dspark_tp2")
              else tuple(layouts()["decode_capacities"]))
    draft = (40,) if family == "v41" and role in ("coordinator", "dspark_tp2") else ()
    decode = tuple(sorted((*decode, *draft)))
    return (*decode, *layouts()["prefill_capacities"]), decode


def native_fc1_k_splits(hidden: int, width: int, sm: int, capacity: int) -> int:
    for row in layouts()["native_slices"]:
        if (row["hidden"], row["width"], row["sm"]) == (hidden, width, sm):
            splits = row.get("fc1_k_splits", [1] * len(row["capacities"]))
            return dict(zip(row["capacities"], splits, strict=True)).get(capacity, 1)
    return 1


def header() -> str:
    lines = ["#pragma once", "#include <stdint.h>",
             "/* Generated from scripts/build/mxfp4-layouts.json. */",
             "static inline uint32_t cuteafd_mxfp4_stored_width(uint32_t hidden, uint32_t width) {",
             "  if (!hidden || hidden % 256 || !width || width % 32 || width > 8192) return 0;"]
    for row in layouts()["storage_holds"]:
        lines.append(f"  if (hidden == {row['hidden']} && width == {row['width']}) return {row['stored']};")
    lines += ["  return width;", "}",
              "static inline uint32_t cuteafd_mxfp4_ownership_unit(uint32_t intermediate, uint32_t world) {"]
    for row in layouts()["ownership"]:
        lines.append(f"  if (intermediate == {row['intermediate']} && world == {row['world']}) return {row['unit']};")
    return "\n".join([*lines, "  return 32;", "}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    args.header.write_text(header())
