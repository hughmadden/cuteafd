import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "bench-agentic-session.py"


def load():
    spec = importlib.util.spec_from_file_location("bench_agentic_session", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sse(events):
    return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events) + b"data: [DONE]\n")


def test_fixture_fails_exactly_the_planted_tests_and_the_fixes_pass():
    bench = load()
    workspace = bench.Workspace()
    assert 40 <= len(workspace.files) <= 60
    failing = sorted(line.split()[1] for line in workspace.run_tests().splitlines() if line.startswith("FAILED"))
    assert failing == ["tests/test_dates.py::test_december_month_end", "tests/test_money.py::test_negative_rounding",
                       "tests/test_parser.py::test_doubled_quotes", "tests/test_parser.py::test_quoted_commas",
                       "tests/test_rates.py::test_expired_rate_refetched", "tests/test_report.py::test_december_report"]
    assert not any(workspace.passing(task["tests"]) for task in bench.TASKS)
    workspace.edit("ledger/money.py", "        return -quotient - (1 if remainder else 0)", "        return -quotient")
    workspace.edit("ledger/rates.py", "now - cached.fetched_at > self.ttl", "now - cached.fetched_at <= self.ttl")
    workspace.edit("ledger/dates.py", "date(day.year, day.month % 12 + 1, 1)",
                   "date(day.year + day.month // 12, day.month % 12 + 1, 1)")
    assert [workspace.passing(task["tests"]) for task in bench.TASKS] == [True, False, True, True]
    # The run is deterministic: no timings, no temporary paths.
    assert workspace.run_tests("tests/test_money.py") == workspace.run_tests("tests/test_money.py")
    assert "<repo>" not in workspace.run_tests() and "tmp" not in workspace.run_tests()


def test_tools_are_deterministic_and_validated():
    bench = load()
    workspace = bench.Workspace()
    text, problem = workspace.execute("read_file", json.dumps({"path": "ledger/dates.py", "start_line": 1, "end_line": 3}))
    assert problem is None and text.splitlines()[1] == '    1\t"""Calendar helpers for monthly reporting periods."""'
    assert "[lines 4-" in text
    assert workspace.execute("list_dir", '{"path": "."}')[0].splitlines()[:3] == ["CHANGELOG.md", "CONTRIBUTING.md", "README.md"]
    assert "ledger/rates.py:" in workspace.execute("grep", '{"pattern": "class RateCache"}')[0]
    assert workspace.execute("edit", json.dumps({"path": "ledger/tags.py", "old_string": "import", "new_string": "x"}))[0] \
        .startswith("error: old_string is not unique")
    assert workspace.execute("read_file", '{"path": "../etc/passwd"}')[0].startswith("error: path")
    assert workspace.execute("write_file", '{"path": "notes.txt", "content": "hi"}')[0] == "wrote 2 bytes to notes.txt"
    # Invalid calls are answered with an error and counted as invalid.
    assert workspace.execute("read_file", "{not json")[1].startswith("arguments are not JSON")
    assert workspace.execute("read_file", '{"file": "x"}')[1].startswith("arguments do not match")
    assert workspace.execute("delete_everything", "{}")[1] == "unknown tool 'delete_everything'"


def test_stream_accumulates_tool_call_deltas_and_times_first_output(monkeypatch):
    bench = load()
    events = [
        {"choices": [{"delta": {"reasoning_content": "Read the test first."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "read_", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '{"path": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"tests/test_money.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 9000, "prompt_cache_hit_tokens": 8500, "completion_tokens": 41}},
    ]
    monkeypatch.setattr(bench, "open_request", lambda *a, **k: sse(events))
    ticks = iter([0, 2, 3, 4, 5, 6, 12, 13, 14])
    result = bench.stream_turn("unused", {}, clock=lambda: next(ticks))
    assert result["calls"] == [{"id": "c0", "name": "read_file", "arguments": '{"path": "tests/test_money.py"}'},
                               {"id": "c1", "name": "run_tests", "arguments": "{}"}]
    assert (result["first_output_seconds"], result["finish_seconds"], result["done"]) == (2, 12, True)
    previous = {"prompt_tokens": 8000, "completion_tokens": 501}
    metrics = bench.turn_metrics(result, previous, 1, 12.5)
    assert metrics["decode_tps"] == 40 / 10
    assert (metrics["tool_calls"], metrics["invalid_tool_calls"], metrics["prefill_tokens"]) == (2, 0, 500)
    assert (metrics["prompt_reused"], metrics["full_turn_reused"]) == (True, True)
    previous["completion_tokens"] = 502
    assert bench.turn_metrics(result, previous, 1, 1)["full_turn_reused"] is False


def scripted_server(bench, monkeypatch, replies):
    """A fake API: returns scripted replies and records every request body."""
    bodies = []

    def fake(base, body, api_key=None, clock=None):
        bodies.append(json.loads(json.dumps(body)))
        reply = replies[min(len(bodies), len(replies)) - 1]
        prompt = sum(len(json.dumps(m)) for m in body["messages"]) // 4
        return dict(reasoning=reply.get("reasoning", "thinking"), content=reply.get("content", ""),
                    calls=reply.get("calls", []), finish_reason="tool_calls" if reply.get("calls") else "stop",
                    first_output_seconds=0.5, first_content_seconds=None, finish_seconds=1.5, done=True,
                    usage={"prompt_tokens": prompt, "prompt_cache_hit_tokens": 0, "completion_tokens": 20})

    monkeypatch.setattr(bench, "stream_turn", fake)
    return bodies


def test_record_then_replay_sends_identical_prompts_per_repeat(tmp_path, monkeypatch):
    bench = load()
    fix = {"path": "ledger/money.py", "old_string": "        return -quotient - (1 if remainder else 0)",
           "new_string": "        return -quotient"}
    replies = [
        {"calls": [{"id": "a", "name": "read_file", "arguments": '{"path": "tests/test_money.py"}'},
                   {"id": "", "name": "grep", "arguments": '{"pattern": "def round_half_even"}'}]},
        {"calls": [{"id": "b", "name": "edit", "arguments": json.dumps(fix)}]},
        {"calls": [{"id": "c", "name": "run_tests", "arguments": '{"path": "tests/test_money.py"}'}]},
        {"content": "Fixed the negative rounding."},
    ]
    bodies = scripted_server(bench, monkeypatch, replies)
    recording = tmp_path / "record.json"
    assert bench.main(["record", "--output", str(recording), "--seeds", "0", "--label", "t"]) == 0
    report = json.loads(recording.read_text())
    session = report["sessions"][0]
    assert (session["task"], session["success"], session["final_answer"]) == ("money", True, True)
    assert [t["messages"] for t in session["turns"]] == [2, 5, 7, 9]
    assert [m["role"] for m in session["messages"][2:5]] == ["assistant", "tool", "tool"]
    assert session["messages"][3]["tool_call_id"] == "a" and session["messages"][4]["tool_call_id"] == "call_0_1"
    assert session["messages"][2]["reasoning_content"] == "thinking"
    assert "{{NONCE}}" in session["messages"][0]["content"]
    assert bodies[0]["tools"] == bench.TOOLS and bodies[0]["temperature"] == 0 and bodies[0]["max_tokens"] == 8192
    assert bodies[0]["reasoning_effort"] == "high" and bodies[0]["messages"][0]["content"].startswith("session ")

    bodies.clear()
    replay = lambda label, output: bench.main(["replay", "--recording", str(recording), "--output", str(tmp_path / output),
                                               "--label", label, "--repeats", "2", "--concurrency", "1", "4",
                                               "--stagger", "0"])
    assert replay("ab", "a.json") == 0
    first = [json.dumps(b["messages"]) for b in bodies]
    bodies.clear()
    assert replay("ab", "b.json") == 0
    assert [json.dumps(b["messages"]) for b in bodies] == first, "both arms see identical prompts"
    # Four turns per run; runs (repeat x concurrency) never share a nonce.
    systems = [b["messages"][0]["content"].splitlines()[0] for b in bodies]
    assert len(set(systems)) == 4 and all(systems.count(s) == 4 for s in set(systems))
    assert json.dumps(bodies[3]["messages"][1:]) == json.dumps(session["messages"][1:9])
    summary = json.loads((tmp_path / "a.json").read_text())["summary"]
    assert set(summary) == {"c1", "c4"} and summary["c1"]["turns"] == 8 and summary["c1"]["tool_call_validity"] == 1.0
    assert "ttft_later_s" in bench.render(summary)


def test_echo_reasoning_off_keeps_reasoning_out_of_the_history(tmp_path, monkeypatch):
    bench = load()
    bodies = scripted_server(bench, monkeypatch, [{"calls": [{"id": "a", "name": "list_dir", "arguments": '{"path": "."}'}]},
                                                  {"content": "done"}])
    bench.main(["record", "--output", str(tmp_path / "r.json"), "--seeds", "2", "--echo-reasoning", "off"])
    assert "reasoning_content" not in bodies[1]["messages"][2]
    assert json.loads((tmp_path / "r.json").read_text())["sessions"][0]["success"] is False


def test_bench_ab_agentic_battery_rows_and_summary(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bench_ab", SCRIPT.with_name("bench-ab.py"))
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    commands = []

    def fake_run(cmd, cwd=None, log=None, timeout=0):
        commands.append(cmd)
        output = Path(cmd[cmd.index("--output") + 1])
        value = 2.0 if "off" in str(tmp_path / output.parent.name) else 0.5
        output.write_text(json.dumps({"summary": {"c1": {"ttft_later_s": value, "decode_tps": 30.0},
                                                  "c4": {"ttft_later_s": 2 * value, "decode_tps": 90.0}}}))

    monkeypatch.setattr(ab, "run", fake_run)
    args = type("Args", (), dict(agentic_recording=tmp_path / "rec.json", base_url="http://x", concurrency=[1, 4],
                                 agentic_arg=["--model=m"], nonce_seed=7, repeats=3))()
    rows = []
    for arm in ("off", "on"):
        out = tmp_path / arm
        out.mkdir()
        rows.append(ab.agentic_session(arm, 1, 12.0, out, out / "log", "python", args))
    assert commands[0][commands[0].index("--label") + 1] == "ab-7-warmup"
    assert commands[1][commands[1].index("--repeats") + 1] == "3"
    assert rows[0]["c1_ttft_later_s"] == 2.0 and rows[1]["c4_ttft_later_s"] == 1.0
    table = ab.summarize(rows, ["off", "on"])
    line = next(l for l in table.splitlines() if l.startswith("c1_ttft_later_s"))
    assert line.split()[-1] == "0.250"
