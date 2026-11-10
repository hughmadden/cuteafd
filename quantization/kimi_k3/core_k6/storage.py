"""Committed phase files between two rolling Kimi layer boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from kimi_k3.uniform_k2.storage import atomic_json, fsync_directory, verify_receipt


class StageStore:
    """Trust a phase only after its complete manifest has been published.

    Callers can discard earlier phases once their successor contains everything
    needed to resume. Packed projection receipts live outside these directories
    until the layer's export and frontier have both committed.
    """
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def directory(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError("invalid phase name")
        path = self.root / name
        if path.is_symlink():
            raise ValueError("refuse symlink phase directory")
        return path

    @staticmethod
    def _validate_entries(entries):
        names = [entry["filename"] for entry in entries]
        if not names or len(set(names)) != len(names):
            raise ValueError("empty or duplicate phase inventory")
        if any(not isinstance(name, str) or Path(name).name != name
               or name in (".", "..", "manifest.json") for name in names):
            raise ValueError("phase filename must be a local basename")

    def load(self, name, *, checksum=False):
        directory = self.directory(name)
        path = directory / "manifest.json"
        if not path.exists():
            return None
        manifest = json.loads(path.read_text())
        if manifest.get("identity") != self.identity or manifest.get("stage") != name:
            raise ValueError("phase identity differs")
        self._validate_entries(manifest["entries"])
        for entry in manifest["entries"]:
            target = directory / entry["filename"]
            if target.is_symlink():
                raise ValueError("refuse symlink phase payload")
            verify_receipt(target, entry, checksum=checksum)
        return manifest

    def commit(self, name, entries, metadata=None):
        self._validate_entries(entries)
        directory = self.directory(name)
        for entry in entries:
            target = directory / entry["filename"]
            if target.is_symlink():
                raise ValueError("refuse symlink phase payload")
            verify_receipt(target, entry)
        record = dict(identity=self.identity, stage=name, entries=entries,
                      metadata={} if metadata is None else metadata)
        prior = self.load(name)
        if prior is not None:
            if prior != record:
                raise ValueError("refuse to replace a committed phase")
            return prior
        atomic_json(directory / "manifest.json", record)
        return record

    def retire(self, *names):
        for name in names:
            directory = self.directory(name)
            if directory.exists():
                shutil.rmtree(directory)
        fsync_directory(self.root)

    def clear(self):
        self.retire(*(path.name for path in self.root.iterdir() if path.is_dir()))
