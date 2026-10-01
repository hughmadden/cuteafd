import hashlib
import importlib.util
import json
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "qualify/qualify-prefix-cache.py"


def load():
    spec = importlib.util.spec_from_file_location("qualify_prefix_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeServer:
    """Greedy replies are a function of the prompt; the cache reports the longest seen prefix."""

    def __init__(self, cache=True, corrupt=()):
        self.cache, self.seen, self.corrupt, self.calls = cache, [], set(corrupt), 0

    def __call__(self, base, body, cancel_after_s=None, cancel_on_output=False):
        self.calls += 1
        text = json.dumps(body["messages"])[:-1]  # an open list: earlier prompts are string prefixes
        tokens = len(text) // 4
        hit = max((len(p) for p in self.seen if text.startswith(p)), default=0) // 4 if self.cache else 0
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.calls in self.corrupt:
            digest = digest[::-1]
        reply = dict(reasoning="r" + digest[:8], content="c" + digest[8:24], finish_reason="stop", cancelled=False,
                     usage=dict(prompt_tokens=tokens, prompt_cache_hit_tokens=hit, completion_tokens=9))
        if cancel_after_s is not None or cancel_on_output:
            reply["cancelled"] = True
        self.seen.append(text)
        return reply


def args(module, **extra):
    namespace = type("A", (), {})()
    defaults = dict(base_url="u", model="m", max_tokens=16, reasoning_effort="high", turns=3, workers=3,
                    torture_requests=12, settle_s=0, seed=1, cancel_min_s=0.02, cancel_max_s=0.6, doc_chars=0)
    for key, value in {**defaults, **extra}.items():
        setattr(namespace, key, value)
    return namespace


def test_exact_cache_passes_and_a_divergence_is_reported():
    q = load()
    a = args(q)
    reference = [q.conversation(a, c, send=FakeServer(cache=False)) for c in range(2)]
    server = FakeServer()
    observed = [q.conversation(a, c, send=server) for c in range(2)]
    assert q.compare(reference, observed) == ([], [])
    # A reply that differs from the cache-off reference is informational (FP order); a repeat that
    # differs from its own first reply is a failure.
    # Calls: 1 turn 0, 2 its repeat, 3 turn 1, 4 turn 2.
    corrupted = [q.conversation(a, c, send=FakeServer(corrupt={2, 3})) for c in range(2)]
    problems, notes = q.compare(reference, corrupted)
    assert any("cache-off reference" in n for n in notes)
    assert any("repeated prompt" in p for p in problems)


def test_repeat_without_a_full_hit_is_a_failure():
    q = load()
    a = args(q)
    reference = [q.conversation(a, 0, send=FakeServer(cache=False))]
    observed = [q.conversation(a, 0, send=FakeServer(cache=False))]
    problems, _ = q.compare(reference, observed)
    assert any("repeated prompt hit 0" in p for p in problems)
    assert any("below the previous prompt" in p for p in problems)


def test_torture_checks_probes_and_idle_accounting():
    q = load()
    a = args(q)
    reference = [q.conversation(a, c, send=FakeServer(cache=False)) for c in range(4)]
    good = dict(prefix_cache=dict(pages=100, pages_free=60, pages_retained=40, marks_in_use=3, entries_prompt=2,
                                  entries_turn=1))
    report = q.torture(a, reference, send=FakeServer(), get_stats=lambda base: good)
    assert report["problems"] == [] and report["notes"] == [] and len(report["results"]) == 12
    assert any(r["mode"] == "probe" and r["exact"] for r in report["results"])
    leaked = dict(prefix_cache=dict(good["prefix_cache"], pages_free=59))
    report = q.torture(a, reference, send=FakeServer(), get_stats=lambda base: leaked)
    assert any("idle pages leaked" in p for p in report["problems"])
    # A pages-only family (GLM 5.3) has no mark arena and holds no marks.
    pages_only = dict(prefix_cache=dict(good["prefix_cache"], mark_slots=0, marks_in_use=0))
    assert q.torture(a, reference, send=FakeServer(), get_stats=lambda base: pages_only)["problems"] == []
    stray = dict(prefix_cache=dict(good["prefix_cache"], mark_slots=8, marks_in_use=4))
    report = q.torture(a, reference, send=FakeServer(), get_stats=lambda base: stray)
    assert any("idle marks do not match" in p for p in report["problems"])


def test_a_prefill_cancel_hangs_up_while_the_server_is_silent():
    import http.server
    import socketserver
    q = load()
    gone = threading.Event()

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            # "Prefill": nothing is sent for a while; the client must hang up anyway.
            for _ in range(50):
                time.sleep(0.02)
                try:
                    self.wfile.write(b": keep\n")
                    self.wfile.flush()
                except OSError:
                    gone.set()
                    return

        def log_message(self, *args):
            pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Slow) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        started = time.monotonic()
        result = q.stream(base, {"messages": []}, cancel_after_s=0.1)
        assert result["cancelled"] and time.monotonic() - started < 0.8
        assert gone.wait(2), "the server saw the hang-up"
        server.shutdown()
