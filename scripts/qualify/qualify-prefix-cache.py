#!/usr/bin/env python3
"""Prefix cache qualification over the OpenAI-compatible API (MiMo V2, GLM 5.3 Flash, GLM 5.3, Qwen 3.8, DeepSeek V4).

  reference  Run the conversation set on a cache-off server (PREFIX_CACHE_ENTRIES=0) and save
             every greedy reply.
  check      Run the same set on a cache-on server. A repeated prompt must report
             prompt_cache_hit_tokens == prompt_tokens and reply exactly as the first time (an
             exact-length restore takes the retained logits), and a next turn must reuse at
             least the previous prompt. Replies that differ from the cache-off reference are
             reported as informational: restores are byte-exact (mimo-golden --resume-at is that
             gate), but prefilling only the suffix changes chunk boundaries and so FP order,
             which can flip greedy near-ties. Then the torture phase and the idle accounting.
  torture    (part of check, or alone) W workers send prompts that share long prefixes and
             cancel a random fraction during prefill (cold prompts; the connection closes after
             a random delay, so the server parks the prefilled chunks) or during decode (after
             the first output), interleaved with probe requests compared with the reference. Afterwards, idle, /v1/stats must show no
             leaked pages (pages - pages_free == pages_retained) and one mark per entry (none for a
             pages-only family).

  scripts/qualify/qualify-prefix-cache.py reference --base-url URL --output runs/prefix/ref.json
  scripts/qualify/qualify-prefix-cache.py check --base-url URL --reference runs/prefix/ref.json \\
      --output runs/prefix/check.json

Conversations: a long shared system prompt (the agentic fixture's docs) with a per-conversation
tail, three user turns each, thinking on, temperature 0, reasoning echoed back.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "agentic-repo"
TURNS = [
    "Summarize what this library does in three sentences.",
    "Which module would you change to support a new bank export format, and why?",
    "Write a short Python function that sums a list of Money amounts in one currency.",
]
TOPICS = ["budgets and envelopes", "currency conversion", "CSV quoting", "month-end reporting"]


def system_prompt(topic: str, doc_chars: int = 0) -> str:
    """`doc_chars` > 0 keeps that many characters of the docs: conversations that stay under 2048
    tokens, below the DSA top-k (GLM 5.x), whose selection among exactly tied scores varies run
    to run, so replies there are deterministic."""
    docs = "\n\n".join((FIXTURE / name).read_text() for name in
                       ("README.md", "CONTRIBUTING.md", "docs/architecture.md", "docs/importers.md",
                        "ledger/money.py", "ledger/parser.py", "ledger/journal.py", "ledger/report.py"))
    if doc_chars > 0:
        docs = docs[:doc_chars]
    return f"You are a concise assistant for the ledger project.\n\n{docs}\n\nFocus on {topic}."


def open_request(base: str, body: dict, timeout: float = 1800):
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                                         data=json.dumps(body).encode(),
                                                         headers={"Content-Type": "application/json"}), timeout=timeout)


def hang_up(response) -> None:
    """Close the connection under a blocked read (nothing arrives while the server prefills)."""
    import socket
    try:
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except (AttributeError, OSError):
        response.close()


def stream(base: str, body: dict, cancel_after_s: float | None = None, cancel_on_output: bool = False) -> dict:
    """Stream one reply; with a cancel option the connection is closed early (the result is
    then partial and marked cancelled): `cancel_after_s` hangs up after that long whether or
    not anything arrived (a cancel during prefill), `cancel_on_output` at the first output."""
    result = dict(reasoning="", content="", usage=None, finish_reason=None, cancelled=False)
    with open_request(base, dict(body, stream=True, stream_options={"include_usage": True})) as response:
        timer = threading.Timer(cancel_after_s, hang_up, [response]) if cancel_after_s is not None else None
        if timer is not None:
            timer.start()
        try:
            read_events(response, result, cancel_on_output)
        except (OSError, ValueError, http.client.HTTPException):
            if timer is None or timer.is_alive():
                raise
        finally:
            if timer is not None:
                timer.cancel()
        hung_up = timer is not None and timer.finished.is_set() and not timer.is_alive() and result["finish_reason"] is None
        result["cancelled"] = result["cancelled"] or hung_up
        return result


def read_events(response, result: dict, cancel_on_output: bool) -> None:
    for line in response:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            return
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
                return


def body(args, messages: list[dict]) -> dict:
    return dict(model=args.model, messages=messages, temperature=0, max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort)


def conversation(args, index: int, send=stream) -> list[dict]:
    """One conversation's turns: each prompt, reply and usage; the second request of turn 0
    repeats the first prompt exactly (a whole-prompt hit)."""
    messages = [{"role": "system", "content": system_prompt(TOPICS[index % len(TOPICS)], args.doc_chars)}]
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


def compare(reference: list[list[dict]], observed: list[list[dict]]) -> tuple[list[str], list[str]]:
    """(problems, informational differences from the cache-off reference)."""
    problems, notes = [], []
    for c, (ref, got) in enumerate(zip(reference, observed)):
        for r, g in zip(ref, got):
            where = f"conversation {c} turn {g['turn']}"
            for key in ("reasoning", "content"):
                if r[key] != g[key]:
                    at = next((i for i, (x, y) in enumerate(zip(r[key], g[key])) if x != y), min(len(r[key]), len(g[key])))
                    notes.append(f"{where}: {key} differs from the cache-off reference at character {at}")
            if "repeat" in g:
                for key in ("reasoning", "content"):
                    if g["repeat"][key] != g[key]:
                        problems.append(f"{where}: the repeated prompt's {key} differs from its first reply")
                usage = g["repeat"]["usage"] or {}
                if usage.get("prompt_cache_hit_tokens") != usage.get("prompt_tokens"):
                    problems.append(f"{where}: the repeated prompt hit {usage.get('prompt_cache_hit_tokens')} of "
                                    f"{usage.get('prompt_tokens')} tokens")
    for c, got in enumerate(observed):
        for previous, turn in zip(got, got[1:]):
            hit = (turn["usage"] or {}).get("prompt_cache_hit_tokens") or 0
            if hit < ((previous["usage"] or {}).get("prompt_tokens") or 0):
                problems.append(f"conversation {c} turn {turn['turn']}: hit {hit} is below the previous prompt")
    return problems, notes


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
        plan.append(dict(conversation=c, turns=turns, mode=mode, delay=rng.uniform(args.cancel_min_s, args.cancel_max_s),
                         index=i))
    results, lock, cursor = [], threading.Lock(), iter(plan)

    def worker():
        while True:
            with lock:
                item = next(cursor, None)
            if item is None:
                return
            system = system_prompt(TOPICS[item["conversation"]], args.doc_chars)
            if item["mode"] == "cancel-prefill":
                # A cold prompt: its prefill spans many chunks, so the hang-up lands inside it.
                system = f"request {item['index']}\n{system}"
            messages = [{"role": "system", "content": system}]
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
    notes = [f"probe {r['index']} (conversation {r['conversation']}, {r['turns']} turns) differs from the reference"
             for r in results if r.get("exact") is False]
    problems = [f"request {r['index']} ({r['mode']}): {r['error']}" for r in results
                if "error" in r and not r["mode"].startswith("cancel")]
    if idle:
        if idle.get("pages", 0) - idle.get("pages_free", 0) != idle.get("pages_retained"):
            problems.append(f"idle pages leaked: {idle}")
        # A pages-only family (GLM 5.3) has no mark arena: its snapshots hold no mark.
        marks = idle.get("entries_prompt", 0) + idle.get("entries_turn", 0) if idle.get("mark_slots", 1) else 0
        if idle.get("marks_in_use") != marks:
            problems.append(f"idle marks do not match entries: {idle}")
    else:
        problems.append("/v1/stats has no prefix_cache section")
    return dict(results=results, idle=idle, problems=problems, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["reference", "check", "torture"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--conversations", type=int, default=4)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--doc-chars", type=int, default=0,
                        help="keep this many characters of the system prompt's docs (0: all; GLM 5.x: e.g. 2500 "
                             "keeps conversations below the DSA top-k's 2048 tokens)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torture-requests", type=int, default=60)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--cancel-min-s", type=float, default=0.02, help="prefill cancellations: earliest hang-up")
    parser.add_argument("--cancel-max-s", type=float, default=0.6, help="prefill cancellations: latest hang-up")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = dict(mode=args.mode, base_url=args.base_url, problems=[], notes=[])
    reference = json.loads(args.reference.read_text())["conversations"] if args.reference else None
    if args.mode in ("reference", "check"):
        report["conversations"] = [conversation(args, c) for c in range(args.conversations)]
    if args.mode == "check":
        if reference is None:
            raise SystemExit("check needs --reference")
        problems, notes = compare(reference, report["conversations"])
        report["problems"] += problems
        report["notes"] += notes
    if args.mode in ("check", "torture"):
        report["torture"] = torture(args, reference)
        report["problems"] += report["torture"]["problems"]
        report["notes"] += report["torture"]["notes"]
    report["passed"] = not report["problems"]
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for note in report["notes"]:
        print("INFO", note)
    for problem in report["problems"]:
        print("FAIL", problem)
    print("PASS" if report["passed"] else f"{len(report['problems'])} problem(s)")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
