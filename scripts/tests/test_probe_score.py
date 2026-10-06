"""scripts/bench/probe-score.py against a stand-in server that writes probe row dumps."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench" / "probe-score.py"
VOCAB = 5


def _module():
    spec = importlib.util.spec_from_file_location("probe_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(position: int) -> list[float]:
    """The stand-in engine's log-probabilities of the row predicting ``position``."""
    return [-float(position) - v / 8 for v in range(VOCAB)]


def _write_dump(leaf: Path, positions: list[int]) -> None:
    """A dump as the engine writes it (cuteafd-api probe/rows.rs): one safetensors per row and a manifest."""
    mod = _module()
    leaf.mkdir()
    with open(leaf / "manifest.jsonl", "w") as manifest:
        for index, position in enumerate(positions):
            name = f"row-{index:08}.safetensors"
            mod.write_safetensors(str(leaf / name), [("log_probs", "F32", [VOCAB],
                                                      struct.pack(f"<{VOCAB}f", *_row(position)))], {})
            manifest.write(json.dumps({"position": position, "vocab_size": VOCAB, "file": name,
                                       "tensor": "log_probs", "dtype": "F32", "byte_order": "little"}) + "\n")


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "dumps"
    root.mkdir()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            spec = payload["spec"]
            seen.append(spec)
            wanted = [p for p in spec["score_positions"] if p >= spec["score_from"]]
            _write_dump(root / spec["dump_rows"], wanted)
            body = json.dumps({"probe": {"scored": len(wanted), "score_path": spec["score_path"], "error": None},
                               "server": {"build": "test", "settings": {"kda_state": "bf16"}}}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root, seen
    httpd.shutdown()


def _plan(path: Path) -> dict:
    windows = []
    for wid, tokens, rows in (("final-0000", [3, 1, 4, 1, 5, 9, 2], [0, 1, 4, 5]), ("final-0001", [2, 7, 1, 8], [2])):
        windows.append({"window_id": wid, "tokens": tokens, "positions": rows, "role": "final",
                        "tokens_sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()})
    plan = {"schema": "test", "vocab": VOCAB, "windows": windows}
    path.write_text(json.dumps(plan))
    return plan


@pytest.mark.parametrize("path", ["decode", "prefill"])
def test_rows_become_one_klgate_window_file_each(tmp_path, server, path):
    url, root, seen = server
    plan = _plan(tmp_path / "plan.json")
    out = tmp_path / "engine"
    run = subprocess.run([sys.executable, str(SCRIPT), "--url", url, "--plan", str(tmp_path / "plan.json"),
                          "--dump-root", str(root), "--out", str(out), "--path", path, "--verify-rows", "8"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    mod = _module()
    plan_sha = hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest()
    for window, spec in zip(plan["windows"], seen):
        # The plan's row r is the probe position r + 1, cold, from position 1, on the asked path.
        assert spec["score_positions"] == [r + 1 for r in window["positions"]]
        assert (spec["score_from"], spec["score_path"], spec["cold"], spec["no_speculation"]) == (1, path, True, True)
        assert spec.get("verify_rows") == (8 if path == "decode" else None)
        meta, header, data = mod.read_safetensors(str(out / f"{window['window_id']}.safetensors"))
        assert meta["window_id"] == window["window_id"] and meta["tokens_sha256"] == window["tokens_sha256"]
        assert meta["plan_sha256"] == plan_sha and meta["score_path"] == path
        assert json.loads(meta["engine"])["settings"] == {"kda_state": "bf16"}
        k = len(window["positions"])
        begin, end = header["positions"]["data_offsets"]
        assert header["positions"]["dtype"] == "I32" and list(struct.unpack(f"<{k}i", data[begin:end])) == window["positions"]
        begin, end = header["logits"]["data_offsets"]
        assert header["logits"]["shape"] == [k, VOCAB]
        values = struct.unpack(f"<{k * VOCAB}f", data[begin:end])
        expected = [v for r in window["positions"] for v in _row(r + 1)]
        assert list(values) == pytest.approx(expected, abs=0)
    assert not any(root.iterdir()), "every dump leaf is removed once converted"
    record = json.loads((out / "run.json").read_text())
    assert record["complete"] and [w["window_id"] for w in record["windows"]] == ["final-0000", "final-0001"]
    # A rerun skips the finished windows; another path's run into the same directory is refused.
    again = subprocess.run([sys.executable, str(SCRIPT), "--url", url, "--plan", str(tmp_path / "plan.json"),
                            "--dump-root", str(root), "--out", str(out), "--path", path], capture_output=True, text=True,
                           timeout=60)
    assert again.returncode == 0 and len(seen) == 2
    other = "prefill" if path == "decode" else "decode"
    refused = subprocess.run([sys.executable, str(SCRIPT), "--url", url, "--plan", str(tmp_path / "plan.json"),
                              "--dump-root", str(root), "--out", str(out), "--path", other], capture_output=True,
                             text=True, timeout=60)
    assert refused.returncode != 0 and "another plan's or path's run" in refused.stderr


def test_missing_rows_and_vocabulary_changes_are_refused(tmp_path):
    mod = _module()
    _write_dump(tmp_path / "dump", [1, 3])
    with pytest.raises(ValueError, match="no row for 1 plan rows"):
        mod.collect(str(tmp_path / "dump"), [0, 1], VOCAB)
    assert len(mod.collect(str(tmp_path / "dump"), [0, 2], VOCAB)) == 2 * VOCAB * 4
    with pytest.raises(ValueError, match="vocabulary"):
        mod.collect(str(tmp_path / "dump"), [0], VOCAB + 1)
