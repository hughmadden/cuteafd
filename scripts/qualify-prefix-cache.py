#!/usr/bin/env python3
"""Prefix cache qualification over the OpenAI-compatible API (serve-mimo, later every family).

  reference  Run the conversation set on a cache-off server (PREFIX_CACHE_ENTRIES=0) and save
             every greedy reply.
  check      Run the same set on a cache-on server. Every reply must equal the reference
             (restores are exact, so greedy text is identical to prefilling from cold), a
             repeated prompt must report prompt_cache_hit_tokens == prompt_tokens, and a next
             turn must reuse at least the previous prompt. Then the torture phase and the idle
             accounting.
  torture    (part of check, or alone) W workers send prompts that share long prefixes and
             cancel a random fraction during prefill (the connection closes after a random
             delay) or during decode (after the first output), interleaved with probe requests
             whose replies must equal the reference. Afterwards, idle, /v1/stats must show no
             leaked pages (pages - pages_free == pages_retained) and one mark per entry.

  scripts/qualify-prefix-cache.py reference --base-url URL --output runs/prefix/ref.json
  scripts/qualify-prefix-cache.py check --base-url URL --reference runs/prefix/ref.json \\
      --output runs/prefix/check.json

Conversations: a long shared system prompt (the agentic fixture's docs) with a per-conversation
tail, three user turns each, thinking on, temperature 0, reasoning echoed back.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).resolve().with_name("fixtures") / "agentic-repo"
TURNS = [
    "Summarize what this library does in three sentences.",
    "Which module would you change to support a new bank export format, and why?",
    "Write a short Python function that sums a list of Money amounts in one currency.",
]
TOPICS = ["budgets and envelopes", "currency conversion", "CSV quoting", "month-end reporting"]


def system_prompt(topic: str) -> str:
    docs = "\n\n".join((FIXTURE / name).read_text() for name in
                       ("README.md", "CONTRIBUTING.md", "docs/architecture.md", "docs/importers.md",
                        "ledger/money.py", "ledger/parser.py", "ledger/journal.py", "ledger/report.py"))
    return f"You are a concise assistant for the ledger project.\n\n{docs}\n\nFocus on {topic}."


def open_request(base: str, body: dict, timeout: float = 1800):
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                                         data=json.dumps(body).encode(),
                                                         headers={"Content-Type": "application/json"}), timeout=timeout)


def stream(base: str, body: dict, cancel_after_s: float | None = None, cancel_on_output: bool = False) -> dict:
    """Stream one reply; with a cancel option the connection is closed early (the result is
    then partial and marked cancelled)."""
    result = dict(reasoning="", content="", usage=None, finish_reason=None, cancelled=False)
    started = time.monotonic()
    with open_request(base, dict(body, stream=True, stream_options={"include_usage": True})) as response:
        for line in response:
            if cancel_after_s is not None and time.monotonic() - started > cancel_after_s:
                result["cancelled"] = True
                return result
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"SSE error: {event['error']}")
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                result["reasoning"] += delta.get("reasoning_content") or ""
                result["content"] += delta.get("content") or ""
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
                if cancel_on_output and (result["reasoning"] or result["content"]):
                    result["cancelled"] = True
                    return result
    return result


def body(args, messages: list[dict]) -> dict:
    return dict(model=args.model, messages=messages, temperature=0, max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort)


def conversation(args, index: int, send=stream) -> list[dict]:
    """One conversation's turns: each prompt, reply and usage; the second request of turn 0
    repeats the first prompt exactly (a whole-prompt hit)."""
    messages = [{"role": "system", "content": system_prompt(TOPICS[index % len(TOPICS)])}]
    turns = []
    for turn, text in enumerate(TURNS[: args.turns]):
        messages.append({"role": "user", "content": text})
        reply = send(args.base_url, body(args, messages))
        record = dict(turn=turn, reasoning=reply["reasoning"], content=reply["content"], usage=reply["usage"],
                      finish_reason=reply["finish_reason"])
        if turn == 0:
            again = send(args.base_url, body(args, messages))
            record["repeat"] = dict(reasoning=again["reasoning"], content=again["content"], usage=again["usage"])
        turns.append(record)
        messages.append({"role": "assistant", "content": reply["content"], "reasoning_content": reply["reasoning"]})
    return turns


def compare(reference: list[list[dict]], observed: list[list[dict]]) -> list[str]:
    problems = []
    for c, (ref, got) in enumerate(zip(reference, observed)):
        for r, g in zip(ref, got):
            where = f"conversation {c} turn {g['turn']}"
            for key in ("reasoning", "content"):
                if r[key] != g[key]:
                    at = next((i for i, (x, y) in enumerate(zip(r[key], g[key])) if x != y), min(len(r[key]), len(g[key])))
                    problems.append(f"{where}: {key} differs from the reference at character {at}")
            if "repeat" in g:
                for key in ("reasoning", "content"):
                    if g["repeat"][key] != r[key]:
                        problems.append(f"{where}: the repeated prompt's {key} differs")
                usage = g["repeat"]["usage"] or {}
                if usage.get("prompt_cache_hit_tokens") != usage.get("prompt_tokens"):
                    problems.append(f"{where}: the repeated prompt hit {usage.get('prompt_cache_hit_tokens')} of "
                                    f"{usage.get('prompt_tokens')} tokens")
    for c, got in enumerate(observed):
        for previous, turn in zip(got, got[1:]):
            hit = (turn["usage"] or {}).get("prompt_cache_hit_tokens") or 0
            if hit < ((previous["usage"] or {}).get("prompt_tokens") or 0):
                problems.append(f"conversation {c} turn {turn['turn']}: hit {hit} is below the previous prompt")
    return problems


def stats(base: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/v1/stats", timeout=30) as response:
        return json.load(response)


def torture(args, reference: list[list[dict]] | None, send=stream, get_stats=stats) -> dict:
    """Random cancellations and evictions under concurrency, with exactness probes."""
    rng = random.Random(args.seed)
    plan = []
    for i in range(args.torture_requests):
        c = rng.randrange(len(TOPICS))
        turns = rng.randrange(1, args.turns + 1)
        mode = rng.choices(["complete", "cancel-prefill", "cancel-decode", "probe"], weights=[3, 3, 3, 2])[0]
        plan.append(dict(conversation=c, turns=turns, mode=mode, delay=rng.uniform(0.05, 1.5), index=i))
    results, lock, cursor = [], threading.Lock(), iter(plan)

    def worker():
        while True:
            with lock:
                item = next(cursor, None)
            if item is None:
                return
            messages = [{"role": "system", "content": system_prompt(TOPICS[item["conversation"]])}]
            ref = reference[item["conversation"]] if reference else None
            for t in range(item["turns"]):
                messages.append({"role": "user", "content": TURNS[t]})
                if t + 1 < item["turns"]:
                    if ref is None:
                        break
                    messages.append({"role": "assistant", "content": ref[t]["content"],
                                     "reasoning_content": ref[t]["reasoning"]})
            outcome = dict(item)
            try:
                reply = send(args.base_url, body(args, messages),
                             cancel_after_s=item["delay"] if item["mode"] == "cancel-prefill" else None,
                             cancel_on_output=item["mode"] == "cancel-decode")
                outcome["cancelled"] = reply["cancelled"]
                if item["mode"] == "probe" and ref is not None:
                    expected = ref[item["turns"] - 1]
                    outcome["exact"] = (reply["content"], reply["reasoning"]) == (expected["content"], expected["reasoning"])
            except Exception as error:  # a cancelled stream may surface as a transport error
                outcome["error"] = f"{type(error).__name__}: {error}"
            with lock:
                results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    time.sleep(args.settle_s)
    idle = get_stats(args.base_url).get("prefix_cache") or {}
    problems = [f"probe {r['index']} (conversation {r['conversation']}, {r['turns']} turns) differs from the reference"
                for r in results if r.get("exact") is False]
    problems += [f"request {r['index']} ({r['mode']}): {r['error']}" for r in results
                 if "error" in r and not r["mode"].startswith("cancel")]
    if idle:
        if idle.get("pages", 0) - idle.get("pages_free", 0) != idle.get("pages_retained"):
            problems.append(f"idle pages leaked: {idle}")
        if idle.get("marks_in_use") != idle.get("entries_prompt", 0) + idle.get("entries_turn", 0):
            problems.append(f"idle marks do not match entries: {idle}")
    else:
        problems.append("/v1/stats has no prefix_cache section")
    return dict(results=results, idle=idle, problems=problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["reference", "check", "torture"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--conversations", type=int, default=4)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torture-requests", type=int, default=60)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = dict(mode=args.mode, base_url=args.base_url, problems=[])
    reference = json.loads(args.reference.read_text())["conversations"] if args.reference else None
    if args.mode in ("reference", "check"):
        report["conversations"] = [conversation(args, c) for c in range(args.conversations)]
    if args.mode == "check":
        if reference is None:
            raise SystemExit("check needs --reference")
        report["problems"] += compare(reference, report["conversations"])
    if args.mode in ("check", "torture"):
        report["torture"] = torture(args, reference)
        report["problems"] += report["torture"]["problems"]
    report["passed"] = not report["problems"]
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for problem in report["problems"]:
        print("FAIL", problem)
    print("PASS" if report["passed"] else f"{len(report['problems'])} problem(s)")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
