"""CPU-only kernel identity and change reporting regressions."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kernel_records", ROOT / "scripts/build/kernel-records.py")
records = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(records)


def test_change_report_classifies_and_flags_exact_groups(tmp_path):
    old = {"programs/glmf_mhc.o": {"group": "programs", "ir_sha256": "a", "object_sha256": "x",
        "source_reads_sha256": {"kernel.py": "old"}, "resources": [{"kernels": {"k": {"registers": 32}}}],
        "scratch_bytes_at_capacity": {"64": {"workspace": 16}}},
        "exl3/old.o": {"group": "exl3/old"}}
    new = {"programs/glmf_mhc.o": {**old["programs/glmf_mhc.o"], "object_sha256": "y",
        "source_reads_sha256": {"kernel.py": "new"}, "resources": [{"kernels": {"k": {"registers": 40}}}],
        "scratch_bytes_at_capacity": {"64": {"workspace": 32}}},
        "exl3/new.o": {"group": "exl3/new"}}
    report = records.change_report(old, new, ["programs"])
    assert len(report["changed"]) == len(report["added"]) == len(report["removed"]) == 1
    changed = report["changed"][0]
    assert changed["object_changed"] and not changed["ir_changed"]
    assert changed["triggering_sources"] == ["kernel.py"]
    assert changed["resource_deltas"]["[0].kernels.k.registers"]["delta"] == 8
    assert changed["scratch_deltas"]["64.workspace"]["delta"] == 16
    assert report["leaks"] == ["exl3/new.o", "exl3/old.o"]
    records.write_report(tmp_path, report)
    assert "UNEXPECTED CHANGES" in (tmp_path / "CHANGES.md").read_text()
    assert json.loads((tmp_path / "CHANGES.json").read_text()) == report


def test_source_reads_include_actual_frames_not_every_import(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("SCALE = 3\ndef kernel():\n    return SCALE\n")
    unused = tmp_path / "unused.py"
    unused.write_text("UNUSED = True\n")
    spec = importlib.util.spec_from_file_location("read_test_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    import sys
    recorder.imported = set(sys.modules)
    previous = sys.getprofile()
    sys.setprofile(recorder._profile)
    try:
        assert module.kernel() == 3
    finally:
        sys.setprofile(previous)
    assert recorder.reads() == {"kernel.py": records.digest(source.read_bytes())}
    assert ("kernel.py", "kernel") in recorder.functions


def cuda_elf():
    names = b"\0.shstrtab\0.strtab\0.symtab\0.text.kernel\0.nv.info\0.nv.info.kernel\0.nv.shared.kernel\0"
    symbol_names = b"\0kernel\0"
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 0, 0, 4, 0, 0)
    infos = struct.pack("<BBHII", 4, 47, 8, 1, 64)
    thread_info = struct.pack("<BBHIII", 4, 16, 12, 128, 1, 1)
    sections = [("", 0, b"", 0, 0), (".shstrtab", 3, names, 0, 0),
                (".strtab", 3, symbol_names, 0, 0), (".symtab", 2, symbols, 2, 0),
                (".text.kernel", 1, b"code", 0, 1), (".nv.info", 1, infos, 0, 0),
                (".nv.info.kernel", 1, thread_info, 0, 0), (".nv.shared.kernel", 8, bytes(1024), 0, 0)]
    blob = bytearray(64 + 64 * len(sections))
    blob[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", blob, 40, 64)
    struct.pack_into("<HHH", blob, 58, 64, len(sections), 1)
    for index, (name, kind, data, link, info) in enumerate(sections):
        struct.pack_into("<IIQQQQIIQQ", blob, 64 + index * 64,
                         names.index(name.encode() + b"\0") if name else 0, kind, 0, 0,
                         len(blob), len(data), link, info, 1, 0)
        if kind != 8:
            blob += data
    return bytes(blob)


def test_cuda_resources_match_elf_symbol_regcount_and_thread_attributes():
    assert records.elf_resources(cuda_elf()) == {
        "kernel": {"registers": 64, "static_smem_bytes": 1024, "max_threads": 128}}
    with pytest.raises(ValueError, match="ELF64"):
        records.elf_resources(b"not a cubin")


@pytest.mark.parametrize("field", ["abi_sha256", "plan", "planner_metadata"])
def test_metadata_only_changes_are_reported_and_checked(field):
    report = records.change_report({"k": {field: "old"}}, {"k": {field: "new"}}, [])
    assert len(report["changed"]) == 1
    assert report["leaks"] == ["k"]
    assert not report["changed"][0]["ir_changed"]


def test_construct_captures_recipe_and_scratch_callback_sources(tmp_path):
    source = tmp_path / "recipe.py"
    source.write_text("def construct():\n    return 5\n")
    spec = importlib.util.spec_from_file_location("record_recipe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    assert recorder.construct(module.construct) == 5
    assert "recipe.py" in recorder.reads()


def test_identity_and_reports_are_deterministic():
    assert records.stable_hash({"ir": "a", "target": [12, 0]}) == records.stable_hash({"target": [12, 0], "ir": "a"})
    assert records.stable_hash({"ir": "a"}) != records.stable_hash({"ir": "b"})
    assert records.change_report({"k": {}}, {"k": {}})["changed"] == []
