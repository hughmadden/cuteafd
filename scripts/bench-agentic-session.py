#!/usr/bin/env python3
"""Agentic coding sessions against the OpenAI-compatible API: the prefix cache benchmark.

A coding agent (system prompt, six tools: read_file, list_dir, grep, edit, write_file,
run_tests) works a seeded task in a fixed fixture repository (scripts/fixtures/agentic-repo,
held in memory; four planted bugs, each with failing tests) for at most --max-turns turns.
Thinking is on (reasoning_effort high), temperature 0, max_tokens 8192 per turn; with
--echo-reasoning on the assistant's reasoning goes back into the history, as agents that keep
it do. Contexts grow from ~8K to 60-100K tokens.

  record  Live agent loop. Tool calls run against the in-memory repository (deterministic,
          tests run in a subprocess by a tiny runner with stable output). Records every turn's
          request history, checks tool-call validity (JSON + jsonschema) and whether the task's
          tests pass at the end.
  replay  Sends the recorded histories turn by turn, so every arm sees identical prompts
          (timing only; the generated text is scored for tool-call validity but not fed back).
          A turn snapshot is reused only when the server regenerates the recorded completion
          token for token; speculative verify widths and suffix-prefill chunk boundaries flip
          greedy near-ties, so replay usually measures the prompt-snapshot regime (hits up to
          the previous prompt) and understates a prefix cache. Record mode (the live loop) is the
          real agent flow: compare a cache-on and a cache-off record run for session times.
  summary Prints the per-concurrency table of one or more result files.

Per turn: TTFT (first reasoning, content or tool-call delta; scripts/tests/
test_reasoning_stream_timing.py semantics), decode tok/s, prompt_cache_hit_tokens, prefill
tok/s of the uncached suffix, tool-call validity, and whether the previous turn was reused
whole (`full_turn_reused`: hit >= previous prompt + completion - 1, the Turn snapshot) or at
least its prompt (`prompt_reused`). Sessions run at C1 (one after another) and C4 (four
distinct seeds, starts staggered 2 s). Every session starts its system message with a nonce
(token zero) derived from --label, seed and repeat, so sessions never share a prefix with each
other or with an earlier repeat, while both arms of an A/B see the same prompts.

  scripts/bench-agentic-session.py record --base-url http://127.0.0.1:8000 \\
      --output runs/agentic/mimo-pro-record.json
  scripts/bench-agentic-session.py replay --recording runs/agentic/mimo-pro-record.json \\
      --base-url http://127.0.0.1:8000 --repeats 3 --concurrency 1 4 --output runs/agentic/on.json
  scripts/bench-ab.py --battery agentic --agentic-recording REC --arm on=CHECKOUT --arm off=CHECKOUT \\
      --arm-arg 'off=--env PREFIX_CACHE_ENTRIES=0' ...   (ABBA; cache-off is the baseline arm)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).resolve().with_name("fixtures") / "agentic-repo"
NONCE = "{{NONCE}}"
MAX_READ_LINES = 400
MAX_GREP_MATCHES = 100
TEST_TIMEOUT_S = 30

TOOL_SCHEMAS = {
    "read_file": {
        "description": "Read a text file of the repository. Lines are numbered from 1. Reads at most "
                       f"{MAX_READ_LINES} lines; use start_line/end_line for longer files.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        }, "required": ["path"], "additionalProperties": False},
    },
    "list_dir": {
        "description": "List a directory of the repository (directories end with '/').",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory relative to the root; '.' for the root."},
        }, "required": ["path"], "additionalProperties": False},
    },
    "grep": {
        "description": "Search file contents with a Python regular expression. Returns 'path:line: text' "
                       f"for at most {MAX_GREP_MATCHES} matches.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "File or directory to search; the whole repository by default."},
        }, "required": ["pattern"], "additionalProperties": False},
    },
    "edit": {
        "description": "Replace exactly one occurrence of old_string with new_string in a file. Fails if "
                       "old_string is missing or occurs more than once; include enough context to be unique.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        }, "required": ["path", "old_string", "new_string"], "additionalProperties": False},
    },
    "write_file": {
        "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"], "additionalProperties": False},
    },
    "run_tests": {
        "description": "Run the test suite (plain test functions under tests/). Optionally a file "
                       "('tests/test_x.py') or one test ('tests/test_x.py::test_name').",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": [], "additionalProperties": False},
    },
}
TOOLS = [{"type": "function", "function": {"name": name, **schema}} for name, schema in TOOL_SCHEMAS.items()]

TASKS = [
    {"name": "money", "tests": ["tests/test_money.py"], "prompt":
        "Our auditors report that splitting a negative expense in half is off by a cent: "
        "`Money(-1005).scale(1, 2)` gives -503 cents where -502 is expected, and "
        "`tests/test_money.py::test_negative_rounding` fails. Find the root cause, fix it without "
        "breaking anything else, and run the tests to confirm."},
    {"name": "parser", "tests": ["tests/test_parser.py"], "prompt":
        "Importing `data/export-2024.csv` fails with `ParseError: line 2: expected 4 fields, found 5`. "
        "The bank quotes descriptions that contain commas (and doubles quotes inside them), and "
        "`tests/test_parser.py` has failing tests for this. Fix the CSV handling so the whole export "
        "imports, keep the other importers working, and run the tests."},
    {"name": "rates", "tests": ["tests/test_rates.py"], "prompt":
        "Currency conversions keep using a stale exchange rate for days, while the rate provider is "
        "called on almost every lookup. `tests/test_rates.py::test_expired_rate_refetched` fails. "
        "Fix the rate cache and run the tests."},
    {"name": "dates", "tests": ["tests/test_dates.py", "tests/test_report.py"], "prompt":
        "The December monthly report comes out empty even though there were rent and dinner expenses "
        "on 2024-12-20 and 2024-12-31. `tests/test_report.py::test_december_report` and "
        "`tests/test_dates.py::test_december_month_end` fail. Find and fix the cause, then run the "
        "whole suite."},
]

SYSTEM_TEMPLATE = """session {nonce}

You are a careful software engineer working in a Python repository through tools. You cannot
see files until you read them. Work in small verified steps:

1. Understand the report: read the failing tests and the code they exercise; use grep and
   list_dir to find related code instead of guessing paths.
2. Find the root cause before editing. Prefer the smallest correct fix; do not rewrite files
   wholesale when an edit will do, and never weaken or delete a test to make it pass.
3. After editing, run the relevant tests, then the whole suite. If something fails, read the
   output and iterate.
4. When the work is done, reply without tool calls: summarize the root cause, the change and
   the test results in a few sentences.

Call tools with arguments that match their schemas exactly. Paths are relative to the
repository root. One or more tool calls per turn are fine when they are independent.

Repository layout:
{tree}

README.md:
{readme}

CONTRIBUTING.md:
{contributing}

docs/architecture.md:
{architecture}
"""

RUNNER = r'''
import importlib.util, sys, traceback
from pathlib import Path
root = Path(sys.argv[1]).resolve()
selection = sys.argv[2] if len(sys.argv) > 2 else ""
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
target, _, only = selection.partition("::")
files = sorted(p for p in (root / "tests").glob("test_*.py"))
if target:
    files = [p for p in files if p.relative_to(root).as_posix() == target.strip().lstrip("./")]
    if not files:
        print(f"no test file matches {selection!r}")
        sys.exit(4)
passed = failed = 0
def where(tb):
    frames = [f for f in traceback.extract_tb(tb) if f.filename.startswith(str(root))]
    return "; ".join(f"{Path(f.filename).relative_to(root).as_posix()}:{f.lineno}: {f.line}" for f in frames[-3:])
for path in files:
    rel = path.relative_to(root).as_posix()
    spec = importlib.util.spec_from_file_location("tests." + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        failed += 1
        print(f"ERROR {rel} - {type(error).__name__}: {error} [{where(error.__traceback__)}]")
        continue
    for name in sorted(n for n in vars(module) if n.startswith("test_") and callable(vars(module)[n])):
        if only and name != only:
            continue
        try:
            vars(module)[name]()
        except BaseException as error:
            failed += 1
            message = str(error).splitlines()[0] if str(error) else ""
            print(f"FAILED {rel}::{name} - {type(error).__name__}: {message} [{where(error.__traceback__)}]")
        else:
            passed += 1
            print(f"PASSED {rel}::{name}")
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
'''


def load_repo(root: Path = FIXTURE) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = path.read_text()
    return files


def tree(files: dict[str, str]) -> str:
    return "\n".join(f"  {name}" for name in sorted(files))


def system_prompt(files: dict[str, str]) -> str:
    return SYSTEM_TEMPLATE.format(nonce=NONCE, tree=tree(files), readme=files["README.md"].strip(),
                                  contributing=files["CONTRIBUTING.md"].strip(),
                                  architecture=files["docs/architecture.md"].strip())


class ToolError(Exception):
    pass


class Workspace:
    """The fixture repository in memory; every tool is deterministic."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = dict(load_repo() if files is None else files)

    def _path(self, path: str) -> str:
        clean = os.path.normpath(path.strip()).replace("\\", "/")
        if clean.startswith("../") or clean == ".." or clean.startswith("/"):
            raise ToolError(f"path {path!r} is outside the repository")
        return "" if clean == "." else clean

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        name = self._path(path)
        if name not in self.files:
            raise ToolError(f"no such file: {path}")
        lines = self.files[name].splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line if end_line is not None else len(lines), start + MAX_READ_LINES - 1)
        body = "\n".join(f"{number:>5}\t{lines[number - 1]}" for number in range(start, end + 1))
        more = f"\n[lines {end + 1}-{len(lines)} not shown]" if end < len(lines) else ""
        return f"{name} ({len(lines)} lines)\n{body}{more}"

    def list_dir(self, path: str = ".") -> str:
        prefix = self._path(path)
        prefix = prefix + "/" if prefix else ""
        entries = set()
        for name in self.files:
            if name.startswith(prefix):
                rest = name[len(prefix):]
                entries.add(rest.split("/", 1)[0] + ("/" if "/" in rest else ""))
        if not entries:
            raise ToolError(f"no such directory: {path}")
        return "\n".join(sorted(entries))

    def grep(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as error:
            raise ToolError(f"bad pattern: {error}") from None
        scope = self._path(path)
        names = [n for n in sorted(self.files) if not scope or n == scope or n.startswith(scope + "/")]
        matches = []
        for name in names:
            for number, line in enumerate(self.files[name].splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{name}:{number}: {line}")
        shown = matches[:MAX_GREP_MATCHES]
        tail = f"\n[{len(matches) - len(shown)} more matches]" if len(matches) > len(shown) else ""
        return ("\n".join(shown) + tail) if matches else "no matches"

    def edit(self, path: str, old_string: str, new_string: str) -> str:
        name = self._path(path)
        if name not in self.files:
            raise ToolError(f"no such file: {path}")
        count = self.files[name].count(old_string) if old_string else 0
        if count != 1:
            raise ToolError("old_string not found" if count == 0 else f"old_string is not unique ({count} matches)")
        self.files[name] = self.files[name].replace(old_string, new_string, 1)
        return f"edited {name}: -{old_string.count(chr(10)) + 1} +{new_string.count(chr(10)) + 1} lines"

    def write_file(self, path: str, content: str) -> str:
        name = self._path(path)
        if not name:
            raise ToolError("a file path is required")
        self.files[name] = content
        return f"wrote {len(content.encode())} bytes to {name}"

    def run_tests(self, path: str = "") -> str:
        with tempfile.TemporaryDirectory(prefix="agentic-repo-") as root:
            for name, text in self.files.items():
                target = Path(root) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            try:
                done = subprocess.run([sys.executable, "-I", "-B", "-c", RUNNER, root, path], capture_output=True,
                                      text=True, timeout=TEST_TIMEOUT_S, env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
            except subprocess.TimeoutExpired:
                return f"test run timed out after {TEST_TIMEOUT_S} s"
        output = (done.stdout + done.stderr).replace(root, "<repo>")
        return output.strip() or f"test runner exited with status {done.returncode}"

    def passing(self, tests: list[str]) -> bool:
        return all(self.run_tests(test).splitlines()[-1].endswith(" 0 failed") for test in tests)

    def execute(self, name: str, arguments: str) -> tuple[str, str | None]:
        """Run one tool call; returns (tool message, validity problem or None)."""
        problem = validate_call(name, arguments)
        if problem is not None:
            return f"error: {problem}", problem
        args = json.loads(arguments)
        try:
            return getattr(self, name)(**args), None
        except ToolError as error:
            return f"error: {error}", None


def validate_call(name: str, arguments: str) -> str | None:
    if name not in TOOL_SCHEMAS:
        return f"unknown tool {name!r}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as error:
        return f"arguments are not JSON: {error}"
    try:
        import jsonschema
    except ImportError:
        schema = TOOL_SCHEMAS[name]["parameters"]
        missing = [key for key in schema["required"] if key not in args]
        return f"missing {missing}" if missing else None
    try:
        jsonschema.validate(args, TOOL_SCHEMAS[name]["parameters"])
    except jsonschema.ValidationError as error:
        return f"arguments do not match the schema: {error.message}"
    return None


def open_request(base: str, body: dict, api_key: str | None = None, timeout: float = 1800):
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + api_key} if api_key else {})}
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                                         data=json.dumps(body).encode(), headers=headers), timeout=timeout)


def stream_turn(base: str, body: dict, api_key: str | None = None, clock=time.perf_counter) -> dict:
    """Stream one completion: reasoning, content and tool calls (accumulated by index, as
    qualify-ds41-tool-serving.py does), usage, and the first-output / finish times."""
    start = clock()
    result = dict(reasoning="", content="", calls={}, usage=None, finish_reason=None, first_output_seconds=None,
                  first_content_seconds=None, finish_seconds=None, done=False)
    with open_request(base, dict(body, stream=True, stream_options={"include_usage": True}), api_key) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            elapsed = clock() - start
            data = line[6:].strip()
            if data == b"[DONE]":
                result["done"] = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"SSE inference error: {event['error']}")
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                thought = delta.get("reasoning_content") or ""
                text = delta.get("content") or ""
                pieces = delta.get("tool_calls") or []
                if (thought or text or pieces) and result["first_output_seconds"] is None:
                    result["first_output_seconds"] = elapsed
                if text and result["first_content_seconds"] is None:
                    result["first_content_seconds"] = elapsed
                result["reasoning"] += thought
                result["content"] += text
                for piece in pieces:
                    item = result["calls"].setdefault(piece.get("index", 0), dict(id="", name="", arguments=""))
                    item["id"] = item["id"] or piece.get("id") or ""
                    function = piece.get("function") or {}
                    for key in ("name", "arguments"):
                        item[key] += function.get(key) or ""
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
                    result["finish_seconds"] = elapsed
    result["calls"] = [result["calls"][index] for index in sorted(result["calls"])]
    return result


def turn_metrics(result: dict, previous: dict | None, turn: int, wall: float) -> dict:
    usage = result.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    hit = usage.get("prompt_cache_hit_tokens")
    completion = usage.get("completion_tokens")
    first, finish = result.get("first_output_seconds"), result.get("finish_seconds")
    decode = (completion - 1) / (finish - first) if completion and first is not None and finish and finish > first else None
    prefill = prompt - hit if prompt is not None and hit is not None else None
    problems = [problem for call in result["calls"] if (problem := validate_call(call["name"], call["arguments"]))]
    metrics = dict(turn=turn, prompt_tokens=prompt, hit_tokens=hit, completion_tokens=completion,
                   hit_ratio=hit / prompt if prompt and hit is not None else None, prefill_tokens=prefill,
                   ttft_s=first, prefill_tps=prefill / first if prefill and first else None, decode_tps=decode,
                   finish_reason=result.get("finish_reason"), tool_calls=len(result["calls"]),
                   invalid_tool_calls=len(problems), tool_problems=problems, wall_s=wall,
                   reasoning_chars=len(result["reasoning"]), content_chars=len(result["content"]),
                   prompt_reused=None, full_turn_reused=None)
    if previous and previous.get("prompt_tokens") and hit is not None:
        metrics["prompt_reused"] = hit >= previous["prompt_tokens"]
        if previous.get("completion_tokens"):
            metrics["full_turn_reused"] = hit >= previous["prompt_tokens"] + previous["completion_tokens"] - 1
    return metrics


def nonce(label: str, seed: int, repeat: int) -> str:
    return hashlib.sha256(f"{label}/{seed}/{repeat}".encode()).hexdigest()[:12]


def with_nonce(messages: list[dict], value: str) -> list[dict]:
    out = copy.deepcopy(messages)
    out[0]["content"] = out[0]["content"].replace(NONCE, value, 1)
    return out


def request_body(args, messages: list[dict], max_tokens: int) -> dict:
    body = dict(model=args.model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0,
                max_tokens=max_tokens, reasoning_effort=args.reasoning_effort)
    body.update(json.loads(args.extra_body) if args.extra_body else {})
    return body


def assistant_message(result: dict, turn: int, echo_reasoning: bool) -> dict:
    message: dict = {"role": "assistant", "content": result["content"]}
    if echo_reasoning and result["reasoning"]:
        message["reasoning_content"] = result["reasoning"]
    if result["calls"]:
        message["tool_calls"] = [{"id": call["id"] or f"call_{turn}_{index}", "type": "function",
                                  "function": {"name": call["name"], "arguments": call["arguments"]}}
                                 for index, call in enumerate(result["calls"])]
    return message


def record_session(args, seed: int, repeat: int) -> dict:
    task = TASKS[seed % len(TASKS)]
    workspace = Workspace()
    messages = [{"role": "system", "content": system_prompt(workspace.files)},
                {"role": "user", "content": task["prompt"]}]
    value = nonce(args.label + "/record", seed, repeat)
    turns, previous = [], None
    for turn in range(args.max_turns):
        started = time.perf_counter()
        result = stream_turn(args.base_url, request_body(args, with_nonce(messages, value), args.max_tokens), args.api_key)
        metrics = turn_metrics(result, previous, turn, time.perf_counter() - started)
        metrics["messages"] = len(messages)
        turns.append(metrics)
        previous = metrics
        assistant = assistant_message(result, turn, args.echo_reasoning == "on")
        messages.append(assistant)
        if not result["calls"]:
            break
        for call, sent in zip(result["calls"], assistant["tool_calls"]):
            output, _ = workspace.execute(call["name"], call["arguments"])
            messages.append({"role": "tool", "tool_call_id": sent["id"], "content": output})
        print(f"[{task['name']}#{seed}] turn {turn}: prompt {metrics['prompt_tokens']} hit {metrics['hit_tokens']} "
              f"ttft {metrics['ttft_s']} calls {metrics['tool_calls']}", flush=True)
    return dict(seed=seed, task=task["name"], messages=messages, turns=turns,
                success=workspace.passing(task["tests"]), final_answer=not turns or turns[-1]["tool_calls"] == 0)


def replay_session(args, session: dict, repeat: int) -> dict:
    value = nonce(args.label, session["seed"], repeat)
    turns, previous = [], None
    for turn, recorded in enumerate(session["turns"]):
        messages = with_nonce(session["messages"][: recorded["messages"]], value)
        max_tokens = args.max_tokens if args.max_tokens_from != "recording" else max(1, recorded["completion_tokens"] or 1)
        started = time.perf_counter()
        result = stream_turn(args.base_url, request_body(args, messages, max_tokens), args.api_key)
        metrics = turn_metrics(result, previous, turn, time.perf_counter() - started)
        metrics["same_tool_calls"] = metrics["tool_calls"] == recorded["tool_calls"]
        turns.append(metrics)
        previous = metrics
    return dict(seed=session["seed"], task=session["task"], turns=turns)


def run_sessions(fn, items: list, concurrency: int, stagger_s: float) -> list:
    if concurrency <= 1:
        return [fn(item) for item in items]
    results: list = [None] * len(items)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        def start(index_item):
            index, item = index_item
            time.sleep(stagger_s * (index % concurrency))
            value = fn(item)
            with lock:
                results[index] = value
        for batch in range(0, len(items), concurrency):
            list(pool.map(start, list(enumerate(items))[batch: batch + concurrency]))
    return results


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def rate(values):
    values = [v for v in values if v is not None]
    return sum(1 for v in values if v) / len(values) if values else None


def summarize(runs: list[dict]) -> dict:
    """Medians per concurrency over every turn of every session and repeat; turn 0 (cold
    prompt) and later turns (the cache's work) are kept apart."""
    out = {}
    for concurrency in sorted({run["concurrency"] for run in runs}):
        turns = [t for run in runs if run["concurrency"] == concurrency for s in run["sessions"] for t in s["turns"]]
        later = [t for t in turns if t["turn"] > 0]
        sessions = [s for run in runs if run["concurrency"] == concurrency for s in run["sessions"]]
        out[f"c{concurrency}"] = dict(
            sessions=len(sessions), turns=len(turns),
            ttft_first_s=median(t["ttft_s"] for t in turns if t["turn"] == 0),
            ttft_later_s=median(t["ttft_s"] for t in later),
            hit_ratio_later=median(t["hit_ratio"] for t in later),
            hit_tokens_total=sum(t["hit_tokens"] or 0 for t in turns),
            prompt_tokens_total=sum(t["prompt_tokens"] or 0 for t in turns),
            prefill_tps_later=median(t["prefill_tps"] for t in later),
            decode_tps=median(t["decode_tps"] for t in turns),
            full_turn_reused=rate(t["full_turn_reused"] for t in later),
            prompt_reused=rate(t["prompt_reused"] for t in later),
            tool_call_validity=1 - (sum(t["invalid_tool_calls"] for t in turns) / max(1, sum(t["tool_calls"] for t in turns))),
            session_wall_s=median(sum(t["wall_s"] for t in s["turns"]) for s in sessions),
            max_prompt_tokens=max((t["prompt_tokens"] or 0 for t in turns), default=0),
            success=rate(s.get("success") for s in sessions),
        )
    return out


def render(summary: dict) -> str:
    keys = ["sessions", "turns", "ttft_first_s", "ttft_later_s", "hit_ratio_later", "prefill_tps_later", "decode_tps",
            "full_turn_reused", "prompt_reused", "tool_call_validity", "session_wall_s", "max_prompt_tokens", "success"]
    columns = list(summary)
    lines = [f"{'metric':<22}" + "".join(f"{c:>12}" for c in columns)]
    for key in keys:
        cells = []
        for c in columns:
            v = summary[c].get(key)
            cells.append(f"{'-':>12}" if v is None else f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}")
        lines.append(f"{key:<22}" + "".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("record", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8000")
        p.add_argument("--model", default=os.environ.get("CUTEAFD_API_MODEL", "default"))
        p.add_argument("--api-key")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--label", default="agentic", help="nonce namespace (same label = same prompts)")
        p.add_argument("--concurrency", type=int, nargs="+", default=[1])
        p.add_argument("--stagger", type=float, default=2.0, help="seconds between concurrent session starts")
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--max-tokens", type=int, default=8192)
        p.add_argument("--reasoning-effort", default="high")
        p.add_argument("--extra-body", help="JSON merged into every request body")
        p.add_argument("--echo-reasoning", choices=["on", "off"], default="on")
        if name == "record":
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
            p.add_argument("--max-turns", type=int, default=12)
        else:
            p.add_argument("--recording", type=Path, required=True)
            p.add_argument("--max-tokens-from", choices=["flag", "recording"], default="flag",
                           help="cap each replayed turn at --max-tokens or at its recorded completion length")
    s = sub.add_parser("summary")
    s.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.mode == "summary":
        for path in args.results:
            print(f"{path}:\n{render(json.loads(path.read_text())['summary'])}\n")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(mode=args.mode, base_url=args.base_url, model=args.model, label=args.label,
                  echo_reasoning=args.echo_reasoning, max_tokens=args.max_tokens, runs=[], started=time.time())
    if args.mode == "replay":
        recording = json.loads(args.recording.read_text())
        report["recording"] = str(args.recording)
        items = recording["sessions"]
        run_one = lambda repeat: (lambda session: replay_session(args, session, repeat))
    else:
        items = list(args.seeds)
        report["tools"] = TOOLS
        run_one = lambda repeat: (lambda seed: record_session(args, seed, repeat))
    for repeat in range(args.repeats):
        for concurrency in args.concurrency:
            # Distinct nonces per repeat and concurrency: no run reuses an earlier run's prefixes.
            offset = 1000 * repeat + concurrency
            sessions = run_sessions(run_one(offset), items, concurrency, args.stagger)
            report["runs"].append(dict(repeat=repeat, concurrency=concurrency, sessions=sessions))
            report["summary"] = summarize(report["runs"])
            args.output.write_text(json.dumps(report, indent=1) + "\n")
    if args.mode == "record":
        # The recording replay consumes: the first run's sessions (C1 unless told otherwise).
        report["sessions"] = report["runs"][0]["sessions"]
    report["summary"] = summarize(report["runs"])
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(render(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
