"""Fidelity recipe and probe contract tests; no CUDA or external data."""
import collections
import importlib.util
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fidelity_set_builder", ROOT / "scripts/bench/fidelity-set.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[(ord(text[i]) % 61) + 1 for i in range(0, len(text), 3)])


@pytest.fixture
def fake_server():
    state = {"requests": [], "mode": "good", "generated": 600, "family": "deepseek_v41", "tool_once": False, "called": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/bench/probe"
            assert self.headers["x-cuteafd-bench"] == "test-token"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            body, probe_spec = request["body"], request["spec"]
            assert probe_spec["cold"] and probe_spec["no_speculation"]
            assert body["temperature"] == 0 and body["reasoning_effort"] == "high"
            if state["mode"] == "http_error":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"native library not found"}')
                return
            text = json.dumps(body["messages"], sort_keys=True)
            ids = FakeTokenizer().encode(text).ids
            probe = {"engine": "fake-bf16", "cold": True, "no_speculation": True, "cached_tokens": 0,
                     "prompt_ids": ids, "generated": [4] * min(body["max_tokens"], state["generated"])}
            if state["mode"] == "warm":
                probe["cold"] = False
            if state["mode"] == "missing":
                probe["generated"] = []
            if state["mode"] == "error":
                probe["error"] = "unsupported probe"
            response = {"probe": probe, "content": "a fixture answer", "reasoning": "because", "tool_calls": [],
                        "server": {"model": "test-model", "family": state["family"], "snapshot": "pinned-revision"}}
            if state["tool_once"] and not state["called"] and body["max_tokens"] == 1024 and body.get("tools"):
                response["tool_calls"] = [{"id": "fixture_call", "type": "function",
                    "function": {"name": "list_dir", "arguments": "{\"path\":\".\"}"}}]
                state["called"] = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["client"] = builder.ProbeClient(f"http://127.0.0.1:{server.server_port}", "test-token", 10)
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def policy():
    return {"checkpoint": "test-model", "snapshot_revision": "pinned-revision", "activations": "bf16",
            "head": "bf16", "kv": "bf16", "state": "bf16", "speculation": False, "prefix_cache": False}


def make(state, **kwargs):
    return builder.build_set(family=state["family"], model="test-model", checkpoint="test-model",
        version="test1", arm=policy(), tokenizer=FakeTokenizer(), probe=state["client"], **kwargs)


def test_64_window_recipe_is_deterministic_and_hashed(fake_server):
    manifest = make(fake_server)
    again = make(fake_server)
    assert manifest == again
    assert manifest["set_sha256"] == builder.set_hash(manifest)
    assert collections.Counter(w["block"] for w in manifest["windows"]) == {"A": 26, "B": 12, "C": 10, "D": 6, "E": 10}
    assert len(manifest["quick_windows"]) == 12
    assert sum(len(w["tokens"]) - w["score_from"] for w in manifest["windows"]) == 32768
    for window in manifest["windows"]:
        assert len(window["roles"]) == len(window["tokens"]) <= 16384
        assert window["bucket"] == builder.bucket(window["score_from"])
        if window["block"] == "C" or window["id"] == "legacy":
            assert set(window["roles"]) == {"ctx"}
        else:
            assert window["roles"][window["score_from"]:] == ["gen"] * 512
    a_buckets = collections.Counter(w["bucket"] for w in manifest["windows"] if w["block"] == "A")
    assert a_buckets == {"0-2K": 8, "2-8K": 12, "8-16K": 6}
    assert {w["bucket"] for w in manifest["windows"] if w["block"] == "B"} == {"8-16K"}
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])
    changed = json.loads(json.dumps(manifest))
    changed["windows"][0]["tokens"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.validate_set(changed)


@pytest.mark.parametrize("family", ["mimo_v2", "qwen4"])
def test_other_family_recipes_keep_internal_names(fake_server, family):
    fake_server["family"] = family
    manifest = make(fake_server)
    assert manifest["family"] == family
    assert len(manifest["windows"]) == 64


def test_live_fixture_tools_execute_and_reasoning_is_echoed(fake_server):
    fake_server["tool_once"] = True
    manifest = make(fake_server)
    assert len(manifest["windows"]) == 64
    histories = [request["body"]["messages"] for request in fake_server["requests"]]
    continued = next(messages for messages in histories if any(m.get("role") == "tool" for m in messages))
    tool = next(m for m in continued if m.get("role") == "tool")
    assistant = next(m for m in continued if m.get("role") == "assistant")
    assert tool["tool_call_id"] == "fixture_call" and "README.md" in tool["content"]
    assert assistant["reasoning_content"] == "because"


def test_short_assistant_turns_get_context_padding_not_fake_gen(fake_server):
    fake_server["generated"] = 73
    manifest = make(fake_server)
    for w in manifest["windows"]:
        if w["id"].startswith(("a", "b", "d", "e")):
            assert w["roles"][w["score_from"]:] == ["gen"] * 73 + ["ctx"] * 439
            assert w["provenance"]["trailing_ctx"] == 439


@pytest.mark.parametrize("mode", ["warm", "missing", "error"])
def test_probe_fails_closed(fake_server, mode):
    fake_server["mode"] = mode
    with pytest.raises(ValueError):
        make(fake_server)


def test_probe_reports_server_rejection(fake_server):
    fake_server["mode"] = "http_error"
    with pytest.raises(ValueError, match="HTTP 400: .*native library not found"):
        make(fake_server)


def test_arm_provenance_and_sources_fail_closed():
    arm = policy()
    arm["head"] = "fp8"
    with pytest.raises(ValueError, match="BF16"):
        builder.arm_policy(arm, "test-model")
    with pytest.raises(ValueError, match="checkpoint"):
        builder.arm_policy(policy(), "other-model")
    with pytest.raises(ValueError, match="inside"):
        builder.source(ROOT, "../secret.txt")


def test_recording_model_mismatch(fake_server):
    with pytest.raises(ValueError, match="selected model"):
        make(fake_server, recordings=[{"model": "external-teacher", "mode": "record", "sessions": []}])


def test_recording_history_above_16k_is_not_truncated(fake_server):
    recording = {"model": "test-model", "mode": "record", "sessions": [
        {"messages": [{"role": "user", "content": "x" * 60000}], "turns": [{"messages": 1}]}]}
    with pytest.raises(ValueError, match="16K"):
        make(fake_server, recordings=[recording])
