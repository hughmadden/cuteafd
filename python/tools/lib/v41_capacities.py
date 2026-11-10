"""V4.1 export capacities from the engine's authoritative Rust constants."""
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _ROOT / "rust/crates/cuteafd-core/src/dspark_verify.rs"


def _constant(name: str) -> int:
    match = re.search(rf"pub const {name}: usize = (\d+);", _SOURCE.read_text())
    if match is None:
        raise ValueError(f"missing literal V4.1 capacity {name} in {_SOURCE}")
    return int(match[1])


VERIFY_ROWS = _constant("V41_VERIFY_ROWS")
DRAFT_LEGACY_ROWS = _constant("V41_DRAFT_LEGACY_ROWS")
DRAFT_WIDE_ROWS = _constant("V41_DRAFT_WIDE_ROWS")
DECODE_CAPACITIES = (1, 16, VERIFY_ROWS)
TARGET_CAPACITIES = (*DECODE_CAPACITIES, 256, 1024, 4096)
DRAFT_CAPACITIES = (1, 16, 40, DRAFT_LEGACY_ROWS, DRAFT_WIDE_ROWS, 1024, 4096)
PROJECTION_CAPACITIES = tuple(sorted(set(TARGET_CAPACITIES) | {DRAFT_LEGACY_ROWS}))
