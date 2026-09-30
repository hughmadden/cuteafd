import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "qualify-prefix-cache.py"


def load():
    spec = importlib.util.spec_from_file_location("qualify_prefix_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeServer:
    """Greedy replies are a function of the prompt; the cache reports the longest seen prefix."""

    def __init__(self, cache=True, corrupt_after=None):
        self.cache, self.seen, self.corrupt_after, self.calls = cache, [], corrupt_after, 0

    def __call__(self, base, body, cancel_after_s=None, cancel_on_output=False):
        self.calls += 1
        text = json.dumps(body["messages"])[:-1]  # an open list: earlier prompts are string prefixes
        tokens = len(text) // 4
        hit = max((len(p) for p in self.seen if text.startswith(p)), default=0) // 4 if self.cache else 0
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.corrupt_after is not None and self.calls > self.corrupt_after:
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
                    torture_requests=12, settle_s=0, seed=1)
    for key, value in {**defaults, **extra}.items():
        setattr(namespace, key, value)
    return namespace


def test_exact_cache_passes_and_a_divergence_is_reported():
    q = load()
    a = args(q)
    reference = [q.conversation(a, c, send=FakeServer(cache=False)) for c in range(2)]
    server = FakeServer()
    observed = [q.conversation(a, c, send=server) for c in range(2)]
    assert q.compare(reference, observed) == []
    corrupted = [q.conversation(a, c, send=FakeServer(corrupt_after=3)) for c in range(2)]
    problems = q.compare(reference, corrupted)
    assert any("differs from the reference" in p for p in problems)


def test_repeat_without_a_full_hit_is_a_failure():
    q = load()
    a = args(q)
    reference = [q.conversation(a, 0, send=FakeServer(cache=False))]
    observed = [q.conversation(a, 0, send=FakeServer(cache=False))]
    problems = q.compare(reference, observed)
    assert any("repeated prompt hit 0" in p for p in problems)
    assert any("below the previous prompt" in p for p in problems)


def test_torture_checks_probes_and_idle_accounting():
    q = load()
    a = args(q)
    reference = [q.conversation(a, c, send=FakeServer(cache=False)) for c in range(4)]
    good = dict(prefix_cache=dict(pages=100, pages_free=60, pages_retained=40, marks_in_use=3, entries_prompt=2,
                                  entries_turn=1))
    report = q.torture(a, reference, send=FakeServer(), get_stats=lambda base: good)
    assert report["problems"] == [] and len(report["results"]) == 12
    assert any(r["mode"] == "probe" and r["exact"] for r in report["results"])
    leaked = dict(prefix_cache=dict(good["prefix_cache"], pages_free=59))
    report = q.torture(a, reference, send=FakeServer(), get_stats=lambda base: leaked)
    assert any("idle pages leaked" in p for p in report["problems"])
