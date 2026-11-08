"""Owned allocation kinds are mandatory; scope strings cannot supply them."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
DELETED = re.compile(
    r"\b(?:DeviceAllocation|HostAllocation|Allocation)::(?:untagged|named)\b"
    r"|\balloc_(?:device|host)_buffer_untagged\b"
    r"|\bLEGACY_UNTAGGED\b|\brelabel_other\b|\bCategory::of_scope\b"
)
RAW_ALLOC = re.compile(
    r"\b(?:cudaMalloc(?:Managed|Host|Async)?|cudaHostAlloc|cuMemAlloc(?:_v2|Async)?|cudaHostRegister)\s*\("
)


def owned_sources():
    for directory in ("rust", "native"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in {".rs", ".cc", ".cu", ".h", ".cuh"}:
                yield path


def test_deleted_allocation_apis_cannot_return():
    # No allowlist: even Rust test fixtures use cfg(test) TEST, not legacy APIs.
    violations = []
    for path in owned_sources():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if DELETED.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("spelling", [
    "DeviceAllocation::untagged(lib, 1)", "HostAllocation::named(lib, 1, name)",
    "Allocation::untagged(device, 1)", "alloc_device_buffer_untagged(1)",
    "alloc_host_buffer_untagged(1)", "LEGACY_UNTAGGED", "relabel_other()",
    "Category::of_scope(scope)",
])
def test_deleted_spelling_detector(spelling):
    assert DELETED.search(spelling)


def test_ledger_entries_require_plain_tags():
    ledger = (ROOT / "rust/crates/cuteafd-ffi/src/memory_ledger.rs").read_text()
    key = ledger.split("pub struct Key {", 1)[1].split("}", 1)[0]
    assert re.search(r"pub tag:\s*Tag\s*,", key)
    for function in ("current_key", "record_alloc", "record_native_alloc"):
        signature = ledger.split(f"fn {function}(", 1)[1].split(")", 1)[0]
        assert re.search(r"\btag:\s*Tag\b", signature)
        assert "Option<" not in signature
    ffi = (ROOT / "rust/crates/cuteafd-ffi/src/lib.rs").read_text()
    for function in ("alloc_device_buffer_with", "alloc_host_buffer_with"):
        signature = ffi.split(f"fn {function}(", 1)[1].split(")", 1)[0]
        assert re.search(r"\btag:\s*memory_ledger::Tag\b", signature)
    for relative in ("memory.rs", "memory/device.rs"):
        source = (ROOT / "rust/crates/cuteafd-daemon/src/shared" / relative).read_text()
        assert not re.search(r"fn\s+(?:untagged|named)\s*\(", source)


def test_scope_strings_never_classify_owned_memory():
    audit = (ROOT / "scripts/bench/memory-audit.py").read_text()
    assert "def category(scope" not in audit
    assert 'dev["by_kind"]' in audit
    for relative in (
        "cuteafd-daemon/src/shared/memory/budget.rs",
        "cuteafd-ffi/src/memory_ledger/fault.rs",
    ):
        source = (ROOT / "rust/crates" / relative).read_text()
        assert ".tag.kind.category()" in source


def test_projection_scratch_keeps_the_callers_target_or_drafter_tag():
    daemon = ROOT / "rust/crates/cuteafd-daemon/src"
    source = (daemon / "shared/dense_trellis.rs").read_text()
    allocate = source.split("fn allocate<'a>", 1)[1].split("pub(crate) fn", 1)[0]
    assert "tag:Tag" in allocate.replace(" ", "")
    assert "Tag::workspace(" not in allocate
    head = source.split("impl<'a> CoreHead<'a>", 1)[1].split("pub(crate) fn launch", 1)[0]
    assert "tag: Tag" in head
    assert "layout.allocate(device, tag)" in head
    for family, relative in (("glm", "glm5/engine.rs"), ("glmf", "glm5_flash/engine/core.rs")):
        caller = (daemon / "families" / relative).read_text()
        assert f'Tag::workspace("{family}/target-head-projection")' in caller
        assert f'Tag::drafter_workspace("{family}/draft-head-projection")' in caller
    draft = (daemon / "families/kimi_k3/draft.rs").read_text()
    assert 'Tag::drafter_workspace("kimi/draft-head-projection")' in draft
    mtp = (daemon / "families/step5/mtp.rs").read_text()
    assert 'Tag::drafter_workspace("step5/mtp-projection")' in mtp


def test_raw_cuda_allocations_have_a_tagged_boundary():
    # Only ABI/media allocators record in Rust rather than the native callback.
    rust_recorded = {
        "native/shared/src/cuteafd_native.cc": (
            "cuteafd_alloc_device_buffer", "cuteafd_alloc_managed_device_buffer",
            "cuteafd_alloc_host_buffer", "cuteafd_register_host_mapping",
        ),
        "native/shared/include/cuteafd_media_backings.h": ("cuteafd_media_allocate",),
    }
    violations = []
    for path in owned_sources():
        relative = str(path.relative_to(ROOT))
        if relative.startswith("native/tests/"):
            continue  # Standalone CUDA correctness fixtures are not serving owners.
        source = path.read_text()
        for match in RAW_ALLOC.finditer(source):
            statement_start = max(source.rfind(";", 0, match.start()), source.rfind("{", 0, match.start()))
            statement = source[statement_start:match.start()]
            if "CUTEAFD_TRACK_ALLOC(" in statement:
                continue
            # The allowed entry point must validate the tag before its raw call.
            allowed = False
            for function in rust_recorded.get(relative, ()):
                start = source.find(function + "(")
                if start < 0:
                    continue
                body = source.find("{", start)
                depth, end = 1, body + 1
                while depth and end < len(source):
                    depth += (source[end] == "{") - (source[end] == "}")
                    end += 1
                if body < match.start() < end:
                    prefix = source[body:match.start()]
                    allowed = "check_tag(kind, owner)" in prefix or "cuteafd_tag_check(kind, owner)" in prefix
            if not allowed:
                line = source.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {match.group()}")
    assert not violations, "\n".join(violations)
