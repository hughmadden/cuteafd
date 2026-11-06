#!/usr/bin/env python3
"""Content identities and measured inputs for CPU-only AOT publications."""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import weakref


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def source_name(path, root):
    try:
        resolved = Path(path).resolve()
        relative = resolved.relative_to(root.resolve())
    except (ValueError, TypeError, OSError):
        return None
    return relative.as_posix() if resolved.is_file() else None


def elf_resources(blob):
    """Read CUDA ELF section metadata without initializing a CUDA context."""
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 CUDA image")
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, strings_index = struct.unpack_from("<HHH", blob, 58)
    if size < 64 or strings_index >= count or offset + size * count > len(blob):
        raise ValueError("invalid CUDA ELF section table")
    strings_offset, strings_size = struct.unpack_from("<QQ", blob, offset + strings_index * size + 24)
    strings = blob[strings_offset:strings_offset + strings_size]
    kernels = {}
    symbols = {}
    for index in range(count):
        entry = offset + index * size
        kind = struct.unpack_from("<I", blob, entry + 4)[0]
        if kind == 2:  # SHT_SYMTAB; EIATTR_REGCOUNT references symbol indices.
            start, length = struct.unpack_from("<QQ", blob, entry + 24)
            link = struct.unpack_from("<I", blob, entry + 40)[0]
            names_start, names_size = struct.unpack_from("<QQ", blob, offset + link * size + 24)
            names = blob[names_start:names_start + names_size]
            for symbol_index, symbol_offset in enumerate(range(start, start + length, 24)):
                name_offset = struct.unpack_from("<I", blob, symbol_offset)[0]
                symbols[symbol_index] = names[name_offset:names.find(b"\0", name_offset)].decode()
    register_counts = {}
    for index in range(count):
        entry = offset + index * size
        name_offset = struct.unpack_from("<I", blob, entry)[0]
        end = strings.find(b"\0", name_offset)
        name = strings[name_offset:end if end >= 0 else len(strings)].decode()
        data_offset, data_size = struct.unpack_from("<QQ", blob, entry + 24)
        if name.startswith(".text."):
            kernel = kernels.setdefault(name[6:], {})
            count_from_header = struct.unpack_from("<I", blob, entry + 44)[0] >> 24
            if count_from_header:
                kernel["registers"] = count_from_header
        elif name.startswith(".nv.shared.") and name != ".nv.shared.reserved.0":
            kernels.setdefault(name[11:], {})["static_smem_bytes"] = data_size
        elif name == ".nv.info" or name.startswith(".nv.info."):
            kernel = kernels.setdefault(name[9:], {}) if name != ".nv.info" else None
            data = blob[data_offset:data_offset + data_size]
            position = 0
            while position + 4 <= len(data):
                form, attribute, value = struct.unpack_from("<BBH", data, position)
                position += 4
                if form == 4:  # EIFMT_SVAL; MAX_THREADS is a three-dimensional extent.
                    payload = data[position:position + value]
                    position += value
                    if kernel is not None and attribute in (5, 16) and len(payload) == 12:
                        x, y, z = struct.unpack("<III", payload)
                        kernel["max_threads"] = x * y * z
                    elif name == ".nv.info" and attribute == 47 and len(payload) == 8:
                        symbol, registers = struct.unpack("<II", payload)
                        register_counts[symbols[symbol]] = registers
    for name, registers in register_counts.items():
        if name in kernels:
            kernels[name]["registers"] = registers
    for kernel in kernels.values():
        kernel.setdefault("static_smem_bytes", 0)
        kernel.setdefault("registers", None)
        kernel.setdefault("max_threads", None)
    return {name: kernels[name] for name in sorted(kernels)}


class Recorder:
    """Observe the existing compiler pass; never invoke a second compilation.

    Read sets contain executed frontend functions, referenced module globals,
    newly imported modules and opened inputs. Whole files are fingerprinted,
    retaining module-level constants without making eager unused imports global.
    """

    def __init__(self, root, output):
        self.root = Path(root).resolve()
        self.output = Path(output)
        self.executors = {}
        self.modules = {}
        self.exports = {}
        self.opened = set()
        self.frames = set()
        self.active = False
        self.restores = []
        self.frame_paths = {}
        self.functions = set()
        self.seen_codes = set()
        self.imported = set()

    def _audit(self, event, args):
        if self.active and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = source_name(args[0], self.root)
            if path and not path.startswith(str(self.output)):
                self.opened.add(path)

    def _profile(self, frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename not in self.frame_paths:
                self.frame_paths[filename] = source_name(filename, self.root)
            path = self.frame_paths[filename]
            if path:
                self.frames.add(path)
                self.functions.add((path, frame.f_code.co_qualname))
                if frame.f_code not in self.seen_codes:
                    self.seen_codes.add(frame.f_code)
                    # A referenced imported class/function/module can contribute
                    # constants without executing a function in its source file.
                    for name in frame.f_code.co_names:
                        value = frame.f_globals.get(name)
                        module = sys.modules.get(getattr(value, "__module__", ""))
                        for dependency in (module, value):
                            source = source_name(getattr(dependency, "__file__", None), self.root)
                            if source:
                                self.frames.add(source)

    def reads(self):
        paths = self.opened | self.frames
        for name in set(sys.modules) - self.imported:
            module = sys.modules.get(name)
            path = source_name(getattr(module, "__file__", None), self.root)
            if path:
                paths.add(path)
        self.active = False
        try:
            return {path: digest((self.root / path).read_bytes()) for path in sorted(paths)}
        finally:
            self.active = True

    def patch(self, cls, method, replacement):
        original = getattr(cls, method)
        self.restores.append((cls, method, original))
        setattr(cls, method, replacement(original))

    def install(self):
        from cutlass.base_dsl.dsl import BaseDSL
        from cutlass.base_dsl.jit_executor import JitCompiledFunction
        sys.addaudithook(self._audit)
        self.active = True

        def frontend(original):
            @functools.wraps(original)
            def wrapped(dsl, *args, **kwargs):
                self.frames, self.functions, self.seen_codes, self.opened = set(), set(), set(), set()
                self.imported = set(sys.modules)
                previous = sys.getprofile()
                sys.setprofile(self._profile)
                try:
                    result = original(dsl, *args, **kwargs)
                finally:
                    sys.setprofile(previous)
                module = result[0]
                # Do not normalize symbols or locations: both can carry semantics.
                self.modules[id(module)] = {
                    "ir_kind": "original_mlir", "ir_sha256": digest(str(module).encode()),
                    "source_reads_sha256": self.reads(),
                    "source_functions": [list(pair) for pair in sorted(self.functions)],
                }
                return result
            return wrapped

        def compiled(original):
            @functools.wraps(original)
            def wrapped(dsl, module, *args, **kwargs):
                record = self.modules.pop(id(module), None)
                if record is None:
                    record = {"ir_kind": "mlir", "ir_sha256": digest(str(module).encode()),
                              "source_reads_sha256": self.reads()}
                result = original(dsl, module, *args, **kwargs)
                self.executors[id(result)] = (weakref.ref(result), record)
                return result
            return wrapped

        def exported(original):
            @functools.wraps(original)
            def wrapped(executor, file_path, file_name, function_prefix=""):
                result = original(executor, file_path, file_name, function_prefix)
                found = self.executors.get(id(executor))
                if found is None or found[0]() is not executor:
                    raise ValueError(f"{file_name}: export has no recorded compiler IR")
                path = (Path(file_path) / (file_name + ".o")).relative_to(self.output).as_posix()
                self.exports[path] = {**found[1], "export_prefix": function_prefix or file_name}
                return result
            return wrapped

        self.patch(BaseDSL, "generate_original_ir", frontend)
        self.patch(BaseDSL, "compile_and_cache", compiled)
        self.patch(JitCompiledFunction, "export_to_c", exported)
        return self

    def finish(self, inspector, context, plans):
        self.active = False
        for cls, method, original in reversed(self.restores):
            setattr(cls, method, original)
        records = {}
        for path in sorted(self.output.rglob("*")):
            if path.suffix not in (".o", ".cubin"):
                continue
            relative = path.relative_to(self.output).as_posix()
            record = self.exports.get(relative)
            if record is None and path.suffix == ".cubin" and path.with_suffix(".ptx").exists():
                record = {"ir_kind": "ptx", "ir_sha256": digest(path.with_suffix(".ptx").read_bytes()),
                          "source_reads_sha256": self.reads()}
            if record is None:
                raise ValueError(f"{relative}: missing IR identity; refusing incomplete publication")
            blob = path.read_bytes()
            images = [image for _, image in inspector.cuda_images(blob)]
            if not images:
                raise ValueError(f"{relative}: missing CUDA image")
            header = path.with_suffix(".h")
            abi = digest(header.read_bytes()) if header.exists() else None
            resources = [{"cubin_sha256": digest(image), "kernels": elf_resources(image)} for image in images]
            key = {"ir_sha256": record["ir_sha256"], "target": context["capability"],
                   "toolchain": context["toolchain"], "environment": context["environment"],
                   "abi_sha256": abi, "build_kind": context["build_kind"]}
            records[relative] = {**record, "object_sha256": digest(blob),
                                 "cubin_sha256": [digest(image) for image in images],
                                 "resources": resources, "abi_sha256": abi,
                                 "compile_key": stable_hash(key), "compile_inputs": key,
                                 "plan": plans.get(path.stem)}
        (self.output / "program-records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        return records


def change_report(previous, current, expected=None):
    changes = {"schema": 1, "added": [], "removed": [], "changed": [], "leaks": []}
    for name in sorted(previous.keys() | current.keys()):
        old, new = previous.get(name), current.get(name)
        group = (new or old).get("group", name.split("/", 1)[0])
        if old is None:
            category, entry = "added", {"program": name, "group": group}
        elif new is None:
            category, entry = "removed", {"program": name, "group": group}
        else:
            fields = ("object_sha256", "cubin_sha256", "ir_sha256", "resources", "scratch_bytes_at_capacity")
            if all(old.get(field) == new.get(field) for field in fields):
                continue
            reads = old.get("source_reads_sha256", {})
            new_reads = new.get("source_reads_sha256", {})
            entry = {"program": name, "group": group,
                     "object_changed": old.get("object_sha256") != new.get("object_sha256"),
                     "ir_changed": old.get("ir_sha256") != new.get("ir_sha256"),
                     "triggering_sources": sorted(path for path in reads.keys() | new_reads.keys()
                                                  if reads.get(path) != new_reads.get(path)),
                     "resource_deltas": numeric_deltas(old.get("resources"), new.get("resources")),
                     "scratch_deltas": numeric_deltas(old.get("scratch_bytes_at_capacity"),
                                                       new.get("scratch_bytes_at_capacity"))}
            category = "changed"
        changes[category].append(entry)
        if expected is not None and group not in expected:
            changes["leaks"].append(name)
    return changes


def numeric_deltas(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        return {key: value for field in sorted(old.keys() | new.keys())
                for key, value in numeric_deltas(old.get(field), new.get(field),
                                                 f"{prefix}.{field}" if prefix else field).items()}
    if isinstance(old, list) and isinstance(new, list):
        return {key: value for index in range(max(len(old), len(new)))
                for key, value in numeric_deltas(old[index] if index < len(old) else None,
                                                 new[index] if index < len(new) else None,
                                                 f"{prefix}[{index}]").items()}
    if type(old) in (int, float) and type(new) in (int, float) and old != new:
        return {prefix: {"before": old, "after": new, "delta": new - old}}
    if old != new and (old is None or new is None):
        return {prefix: {"before": old, "after": new, "delta": None}}
    return {}


def write_report(destination, report):
    (destination / "CHANGES.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Kernel Changes", "", "Program groups are the publication's export job names.", ""]
    if report["leaks"]:
        lines += ["## UNEXPECTED CHANGES OUTSIDE DECLARED TARGETS", "", *[f"- `{name}`" for name in report["leaks"]], ""]
    for category in ("added", "removed", "changed"):
        lines += [f"## {category.title()} ({len(report[category])})", ""]
        for entry in report[category]:
            lines += [f"- `{entry['program']}`"]
            if category == "changed":
                lines += [f"  Object changed: {entry['object_changed']}; IR changed: {entry['ir_changed']}.",
                          f"  Sources: {', '.join(entry['triggering_sources']) or '(no recorded source delta)' }.",
                          f"  Resources: {json.dumps(entry['resource_deltas'], sort_keys=True)}.",
                          f"  Scratch: {json.dumps(entry['scratch_deltas'], sort_keys=True)}."]
        lines.append("")
    (destination / "CHANGES.md").write_text("\n".join(lines))
