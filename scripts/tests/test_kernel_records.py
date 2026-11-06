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
        "source_reads_sha256": {"kernel.py": "old", "unchanged.py": "same"}, "resources": [{"kernels": {"k": {"registers": 32}}}],
        "scratch_bytes_at_capacity": {"64": {"workspace": 16}}},
        "exl3/old.o": {"group": "exl3/old"}}
    new = {"programs/glmf_mhc.o": {**old["programs/glmf_mhc.o"], "object_sha256": "y",
        "source_reads_sha256": {"kernel.py": "new", "unchanged.py": "same"}, "resources": [{"kernels": {"k": {"registers": 40}}}],
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
    markdown = (tmp_path / "CHANGES.md").read_text()
    assert "UNEXPECTED CHANGES" in markdown
    assert "cumulative" in markdown and "Sources: kernel.py." in markdown
    assert "unchanged.py" not in markdown
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
    recorder.start_monitoring()
    try:
        assert recorder.construct(module.construct) == 5
        assert "recipe.py" in recorder.reads()
    finally:
        recorder.stop_monitoring()


def test_local_package_policy_is_a_recorded_input(tmp_path):
    import sys
    import types
    package = types.ModuleType("package_policy")
    package_path = tmp_path / "package_policy.py"
    package_path.write_text("TILE = 64\n")
    package.__file__ = str(package_path)
    recipe = tmp_path / "recipe.py"
    recipe.write_text("def caller(package):\n    import sys\n    return sys._getframe()\n")
    spec = importlib.util.spec_from_file_location("local_recipe", recipe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    recorder.imported = set(sys.modules)
    recorder._profile(module.caller(package), "call", None)
    assert "package_policy.py" in recorder.reads()


def test_package_tuning_and_arguments_change_job_global_identity():
    context = {key: None for key in (
        "exporter", "exporter_sha256", "args", "comparison_script_sha256", "export_sources_sha256",
        "sms", "capability", "toolchain", "environment", "build_kind")}
    context.update(exporter="package_exl3_aot.py", exporter_sha256="old", args=["--bits", "4"])
    before = records.stable_hash(records.job_global_inputs(context))
    assert before != records.stable_hash(records.job_global_inputs({**context, "exporter_sha256": "new"}))
    assert before != records.stable_hash(records.job_global_inputs({**context, "args": ["--bits", "3"]}))


def test_identity_and_reports_are_deterministic():
    assert records.stable_hash({"ir": "a", "target": [12, 0]}) == records.stable_hash({"target": [12, 0], "ir": "a"})
    assert records.stable_hash({"ir": "a"}) != records.stable_hash({"ir": "b"})
    assert records.change_report({"k": {}}, {"k": {}})["changed"] == []


@pytest.fixture
def fake_cutlass(monkeypatch):
    import sys
    import types
    class JitCompiledFunction:
        def export_to_c(self, file_path, file_name, function_prefix=""):
            (Path(file_path) / (file_name + ".o")).write_bytes(cuda_elf())
    class BaseDSL:
        def generate_original_ir(*args, **kwargs):
            return ("same IR",)
        def compile_and_cache(*args, **kwargs):
            return JitCompiledFunction()
    for name in ("cutlass", "cutlass.base_dsl", "cutlass.base_dsl.dsl", "cutlass.base_dsl.jit_executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["cutlass.base_dsl.dsl"].BaseDSL = BaseDSL
    sys.modules["cutlass.base_dsl.jit_executor"].JitCompiledFunction = JitCompiledFunction
    return BaseDSL


def load_recipe(path, text):
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_recorder(recorder):
    recorder.stop_monitoring()
    recorder.active = False
    for cls, method, original in reversed(recorder.restores):
        setattr(cls, method, original)


@pytest.mark.parametrize("binding", ["local", "global", "cell"])
def test_each_program_rescans_recipe_tuning_modules(tmp_path, monkeypatch, fake_cutlass, binding):
    import sys
    import types
    bodies = {
        "local": "def recipe(policy, compile):\n    return compile(None)\n",
        "global": "def recipe(unused, compile):\n    return compile(policy)\n",
        "cell": "def recipe(value, compile):\n    policy = value\n    def closure():\n        return policy\n    return compile(None)\n",
    }
    recipe_module = load_recipe(tmp_path / "recipe.py", bodies[binding])
    policies = []
    for name in ("tuning_a", "tuning_b"):
        path = tmp_path / (name + ".py")
        path.write_text("TILE = 64\n")
        policy = types.ModuleType(name)
        policy.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, policy)
        policies.append(policy)
    del policy
    recorder = records.Recorder(tmp_path, tmp_path / "output").install()
    try:
        recipe_module.policy = policies[0]
        recipe_module.recipe(policies[0], fake_cutlass.generate_original_ir)
        first = next(iter(recorder.modules.values()))["source_reads_sha256"]
        assert "tuning_a.py" in first and "tuning_b.py" not in first
        recipe_module.policy = policies[1]
        recipe_module.recipe(policies[1], fake_cutlass.generate_original_ir)
        second = next(iter(recorder.modules.values()))["source_reads_sha256"]
        assert "tuning_b.py" in second
        assert "tuning_a.py" in second  # Program records are conservative cumulative observations.
        assert {"tuning_a.py", "tuning_b.py"} <= recorder.reads().keys()
    finally:
        stop_recorder(recorder)


def test_monitoring_restart_observes_same_code_in_later_recorder(tmp_path):
    module = load_recipe(tmp_path / "recipe.py", "def recipe():\n    return 5\n")
    for _ in range(2):
        recorder = records.Recorder(tmp_path, tmp_path / "output")
        recorder.start_monitoring()
        try:
            assert recorder.monitoring_id in (3, 4)
            assert module.recipe() == 5
            assert "recipe.py" in recorder.reads()
        finally:
            recorder.stop_monitoring()


def test_monitoring_disables_repeated_code_entry_and_releases_tool(tmp_path):
    source = tmp_path / "recipe.py"
    source.write_text("def recipe():\n    return 5\n")
    spec = importlib.util.spec_from_file_location("one_shot_recipe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = records.Recorder(tmp_path, tmp_path / "output")
    recorder.start_monitoring()
    tool_id = recorder.monitoring_id
    try:
        assert module.recipe() == 5
        recorder.frames.clear()
        assert module.recipe() == 5
        assert "recipe.py" not in recorder.frames
    finally:
        recorder.stop_monitoring()
    import sys
    assert sys.monitoring.get_tool(tool_id) is None


@pytest.fixture
def fake_worker(tmp_path, monkeypatch, fake_cutlass):
    import importlib.metadata
    import sys
    import types
    spec = importlib.util.spec_from_file_location("installed_worker", ROOT / "scripts/build/compare-sm-exports.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    for name in ("b12x", "b12x._lib", "b12x._lib.compile_pool", "b12x._lib.compiler",
                 "b12x.integration", "b12x.integration.cuteafd", "_pinned_sparkinfer", "torch"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["b12x._lib.compile_pool"]._initialize_worker = lambda *args: None
    sys.modules["b12x._lib.compiler"].b12x_package_fingerprint = lambda: None
    class AotProgram:
        def export_to_c(*args, **kwargs):
            pass
    sys.modules["b12x.integration.cuteafd"].AotProgram = AotProgram
    sys.modules["_pinned_sparkinfer"].REVISION = "pin"
    sys.modules["_pinned_sparkinfer"].LOCK_DATA = {"source_tree_sha256": "tree"}
    sys.modules["torch"].cuda = types.SimpleNamespace(
        get_device_properties=lambda ordinal: types.SimpleNamespace(multi_processor_count=188),
        is_initialized=lambda: False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fake")
    monkeypatch.setattr(worker, "load_file", lambda name, path:
                        records if name == "placement_kernel_records" else types.SimpleNamespace(
                            cuda_images=lambda blob: [(0, blob)]))
    exporter = tmp_path / "python/tools/aot/x.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# Fake backend; real worker and recorder install.\n")
    recipe = load_recipe(tmp_path / "recipe.py", "def recipe(compile, output):\n    ir = compile.generate_original_ir(None)[0]\n    result = compile.compile_and_cache(None, ir)\n    result.export_to_c(output, 'kernel')\n")
    after = load_recipe(tmp_path / "after.py", "def validate():\n    return 7\n")
    def run_path(*args, **kwargs):
        assert worker.PROGRAM_RECORDER.monitoring_id in (3, 4)
        recipe.recipe(fake_cutlass, worker.PROGRAM_RECORDER.output)
        assert after.validate() == 7
    monkeypatch.setattr(worker.runpy, "run_path", run_path)
    def run(payload):
        try:
            worker.worker(payload)
        finally:
            if worker.PROGRAM_RECORDER is not None:
                stop_recorder(worker.PROGRAM_RECORDER)
                worker.PROGRAM_RECORDER = None
    return run


def test_job_reads_include_helpers_after_last_program(tmp_path, fake_worker):
    output = tmp_path / "output"
    output.mkdir()
    fake_worker({"sms": 188, "exporter": "x.py", "args": [],
                 "output": str(output), "program_records": True})
    summary = json.loads((output / "comparison-context.json").read_text())
    assert {"recipe.py", "after.py"} <= summary["source_reads_sha256"].keys()
