#!/usr/bin/env python3
"""CPU-only usage UI fixture server. Synthetic records, never real prompts."""
import argparse
import base64
import hashlib
import json
import math
import random
import struct
import time
from collections import defaultdict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ["chat", "messages", "responses", "realtime", "completions"]
SETTINGS = dict(metadata_days=7, metadata_cap_mb=256, log_enabled=True,
                log_hours=24, log_cap_mb=1024, log_media=True, daily_days=90,
                client_ip=False, record_bench=False)
USAGE = dict(recorded=840, dropped=2, log_recorded=42, log_dropped=1,
             db_bytes=425984, log_bytes=1048576, media_bytes=68, media_files=1)
PIXEL = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jTioAAAAASUVORK5CYII=")
MEDIA_HASH = hashlib.sha256(PIXEL).hexdigest()


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    return values[math.floor((len(values) - 1) * p + .5)] if values else None


def histogram(values):
    bins = [0] * 24
    for v in values:
        if v is not None and v >= 0:
            bins[min(23, 0 if v <= 1 else int(math.log2(v)))] += 1
    return bins


def latency(rows):
    return {name: dict(p50=percentile(values, .5), p95=percentile(values, .95),
                       p99=percentile(values, .99), histogram=histogram(values))
            for name, key in [("ttft", "t_ttft_ms"), ("decode", "decode_tps"),
                              ("prefill", "prefill_tps")]
            for values in [[r.get(key) for r in rows]]}


def ratio(n, d):
    return n / d if d else None


def perf(rows):
    computed = cached = decoded = 0
    prefill_ms = decode_ms = queue_total = prefill_total = decode_total = 0
    prefill_rates, decode_rates = [], []
    for r in rows:
        inp = r.get("tokens_in") or 0
        hit = min(r.get("tokens_cached") or 0, inp)
        cached += hit
        admit, first = r.get("t_admit_ms"), r.get("t_ttft_ms")
        if admit is not None:
            queue_total += admit
        if admit is not None and first is not None:
            ms = max(0, first - admit)
            prefill_total += ms
            if ms > 0 and r.get("tokens_in") is not None:
                computed += inp - hit
                prefill_ms += ms
                prefill_rates.append((inp - hit) * 1000 / ms)
        output = r.get("tokens_out")
        if first is not None and output is not None:
            end = r.get("t_retire_ms")
            if end is None:
                end = r.get("t_total_ms")
            ms = max(0, (first if end is None else end) - first)
            decode_total += ms
            if ms > 0 and output > 1:
                decoded += output - 1
                decode_ms += ms
                decode_rates.append((output - 1) * 1000 / ms)
    return dict(prefill=dict(computed_tokens=computed, cached_tokens=cached, ms=prefill_ms,
                            total_tps=ratio(computed * 1000, prefill_ms), median_tps=percentile(prefill_rates, .5)),
                decode=dict(tokens=decoded, ms=decode_ms, total_tps=ratio(decoded * 1000, decode_ms), median_tps=percentile(decode_rates, .5)),
                cache_hit=ratio(cached, max(sum(r.get("tokens_in") or 0 for r in rows), computed + cached)),
                time_ms=dict(queue=queue_total, prefill=prefill_total, decode=decode_total))


def sort_options(query):
    key, order = query.get("sort", "time"), query.get("order", "desc")
    if key not in ["time", "decode_tps", "prefill_tps"]:
        raise ValueError("sort must be time, decode_tps or prefill_tps")
    if order not in ["asc", "desc"]:
        raise ValueError("order must be asc or desc")
    return key, order == "desc"


def sorted_rates(rows, get, descending):
    # Missing rates stay last in both directions; stable ties retain newest first.
    return sorted([r for r in rows if get(r) is not None], key=get, reverse=descending) + [r for r in rows if get(r) is None]


def summary(rows):
    sums = {key: sum(r.get(field) or 0 for r in rows) for key, field in
            [("tokens_in", "tokens_in"), ("tokens_cached", "tokens_cached"),
             ("tokens_out", "tokens_out")]}
    errors = sum(r["outcome"] != "ok" for r in rows)
    return dict(requests=len(rows), errors=errors, error_rate=ratio(errors, len(rows)),
                **sums, cache_hit=ratio(sums["tokens_cached"], sums["tokens_in"]),
                acceptance=ratio(sum(r["draft_accepted"] for r in rows),
                                 sum(r["draft_proposed"] for r in rows)),
                latency=latency(rows),
                concurrency_max=max((r["concurrency_engine"] for r in rows), default=None),
                concurrency_mean=ratio(sum(r["concurrency_engine"] for r in rows), len(rows)))


def groups(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return out


def series(rows, query):
    bucket = {"1m": 60000, "5m": 300000, "1h": 3600000, "1d": 86400000}[query.get("bucket", "1h")]
    split = {"protocol": "protocol", "client": "client_kind", "model": "model_served"}[query.get("split", "protocol")]
    return [dict(summary(rs), ts_ms=ts, split=name) for (ts, name), rs in
            sorted(groups(rows, lambda r: (r["ts_ms"] // bucket * bucket, r[split])).items())]


def cache_facets(rows):
    def key(r):
        fraction = ratio(r["tokens_cached"], r["tokens_in"]) or 0
        return "0" if fraction == 0 else "<50%" if fraction < .5 else ">=50%"
    return [dict(fraction=k, requests=len(rs), latency=latency(rs))
            for k, rs in sorted(groups(rows, key).items())]


def prompt_band(row):
    n = row.get("tokens_in") or 0
    return "<1K" if n < 1024 else "1-8K" if n < 8192 else "8-32K" if n < 32768 else "32K+"


def cached_band(row):
    f = (row.get("tokens_cached") or 0) / max(1, row.get("tokens_in") or 0)
    return "0" if f == 0 else "<50%" if f < .5 else ">=50%"


def filter_rows(rows, query):
    for key, field in [("client", "client_kind"), ("protocol", "protocol"),
                       ("model", "model_served"), ("key", "key_label"),
                       ("session", "session_id"), ("route", "route"),
                       ("outcome", "outcome"), ("stop", "stop_reason")]:
        if key in query:
            rows = [r for r in rows if r.get(field) == query[key]]
    if "error" in query:
        rows = [r for r in rows if r["outcome"] != "ok" and
                (r["error_class"] if r["error_class"] is not None else r["outcome"]) == query["error"]]
    for key, valid, get in [("prompt", ["<1K", "1-8K", "8-32K", "32K+"], prompt_band),
                            ("cached", ["0", "<50%", ">=50%"], cached_band)]:
        if key in query:
            if query[key] not in valid:
                raise ValueError("invalid " + key + " bucket")
            rows = [r for r in rows if get(r) == query[key]]
    if "metric" in query:
        metric = query["metric"]
        if metric not in ["t_ttft_ms", "decode_tps", "prefill_tps", "t_total_ms", "tokens_per_round"]:
            raise ValueError("invalid metric")
        if "min" not in query and "max" not in query:
            raise ValueError("a metric filter needs min or max")
        bounds = {k: float(query[k]) for k in ["min", "max"] if k in query}
        if any(not math.isfinite(n) for n in bounds.values()):
            raise ValueError("metric bounds must be finite")
        def matches(row):
            n = ratio(row.get("tokens_out"), row.get("rounds")) if metric == "tokens_per_round" else row.get(metric)
            return n is not None and ("min" not in bounds or n >= bounds["min"]) and ("max" not in bounds or n < bounds["max"])
        rows = [r for r in rows if matches(r)]
    return rows


def records():
    rng = random.Random(41)
    now = int(time.time() * 1000)
    result = []
    for i in range(840):
        protocol = PROTOCOLS[i % 5]
        client = ["code-agent", "workspace", "terminal"][i % 3]
        inp = rng.choice([512, 2048, 6144, 16384, 49152]) + rng.randrange(200)
        cached = int(inp * rng.choice([0, .25, .72, .9]))
        outcome = "engine_error" if i % 29 == 0 else "cancelled" if i % 41 == 0 else "timeout" if i % 67 == 0 else "ok"
        error = outcome != "ok"
        output = 64 + rng.randrange(900)
        admit = rng.randrange(30)
        prefill_ms = 90 + (inp - cached) / (4 + rng.random() * 20)
        first = admit + prefill_ms
        retire = first + (output - 1) * 1000 / (35 + rng.random() * 95)
        result.append(dict(rid=f"req-{i:04}", ts_ms=now - (839 - i) * 180000,
            route="/v1/" + {"chat": "chat/completions", "messages": "messages", "responses": "responses", "realtime": "realtime", "completions": "completions"}[protocol],
            protocol=protocol, client_kind=client, user_agent="synthetic-fixture/1.0",
            model_requested="fixture-model", model_served=["GLM Flash", "MiMo Pro"][i % 2],
            session_id=f"session-{i // 12:03}", session_source="prefix" if i % 3 == 0 else "metadata.user_id",
            key_label="local", client_ip=None, bench=i % 17 == 0,
            tokens_in=inp, tokens_cached=cached, tokens_out=output,
            t_queue_ms=admit, t_admit_ms=admit, t_ttft_ms=first, t_retire_ms=retire,
            t_total_ms=retire + 25, prefill_tps=(inp - cached) * 1000 / prefill_ms,
            decode_tps=(output - 1) * 1000 / (retire - first), concurrency_engine=1 + i % 12,
            draft_proposed=100, draft_accepted=58 + i % 37, rounds=30 + i % 50,
            status=503 if error else 200, outcome=outcome,
            error_class="overloaded" if outcome == "engine_error" else None,
            stop_reason="error" if error else "tool_use" if i % 7 == 0 else "end_turn"))
    return result


ROWS = records()
LOGS = []


def display(index, role, text=None, kind="text", **kwargs):
    return dict(index=index, role=role, kind=kind, text=text, name=None,
                arguments=None, call_id=None, media=[], echo=False, **kwargs)


# Deliberately exercise all delta markers, escaped markup, tools, echo and media.
for session in range(8):
    entries = []
    previous = None
    for turn, kind in enumerate(["base", "append", "edit", "rebased", "chained"]):
        row = ROWS[-1 - session * 12 - (4 - turn)]
        item = display(0, "user", f"Synthetic conversation {session + 1}: inspect the parser and add a regression test. <script>not executable</script>")
        if turn == 1:
            item["media"] = [dict(sha256=MEDIA_HASH, mime="image/png", bytes=len(PIXEL), stored=True, url="/console/usage/media/" + MEDIA_HASH), dict(sha256="b" * 64, mime="audio/wav", bytes=4096, stored=False, url=None)]
        response = [display(0, "assistant", "I will inspect the existing tests before changing the parser.", kind="reasoning"),
                    display(1, "assistant", kind="tool_call")]
        response[1].update(name="read_file", arguments='{"path":"src/parser.py"}', call_id="call-fixture")
        items = [item] if turn == 0 else [display(1, "tool", "The parser accepts the existing fixture.", kind="tool_result"), item]
        if turn == 1:
            echo = display(0, "assistant", "I will inspect the existing tests before changing the parser.")
            echo["echo"] = True
            items.insert(0, echo)
        entries.append(dict(rid=row["rid"], ts_ms=row["ts_ms"], protocol=row["protocol"], kind=kind,
            parent_rid=previous, parent_count=turn * 2, divergence=2 if kind == "edit" else None,
            diverged_from=previous if kind == "edit" else None, n_items=len(items),
            system=dict(changed=turn in [0, 2], text="You are a careful coding assistant." if turn in [0, 2] else None),
            tools=dict(changed=turn == 0, count=1, names=["read_file"]), items=items, response=response,
            usage=dict(input_tokens=row["tokens_in"], cached_tokens=row["tokens_cached"], output_tokens=row["tokens_out"]),
            stop=row["stop_reason"], truncated=turn == 4 and session == 1, bytes=1400,
            status=row["status"], outcome=row["outcome"], model=row["model_served"], client=row["client_kind"]))
        previous = row["rid"]
    LOGS.append(dict(vsid=f"virtual-{session:03}", first_ms=entries[0]["ts_ms"], last_ms=entries[-1]["ts_ms"],
        entries=entries, protocols=list({e["protocol"] for e in entries}), client=row["client_kind"],
        model=row["model_served"], session_id=row["session_id"], session_source="metadata.user_id",
        title=f"Synthetic conversation {session + 1}: inspect the parser and add a regression test", edits=1, bytes=7000))


def expert_geometry(config):
    data = json.loads(config.read_text())
    data = data.get("text_config", data)
    layers = data["num_hidden_layers"]
    frequency = data.get("moe_layer_freq", 1)
    dense = data.get("first_k_dense_replace", 0)
    routed = [i for i in range(layers) if (frequency[i] if isinstance(frequency, list) else i >= dense and i % frequency == 0)]
    return dict(layers=layers, experts=data.get("num_experts", data.get("n_routed_experts")),
                topk=data.get("num_experts_per_token", data.get("num_experts_per_tok")), routed_layers=routed)


def expert_snapshot(tick, geometry=None):
    geometry = geometry or dict(layers=12, experts=32, topk=2, routed_layers=list(range(1, 12)))
    experts = geometry["experts"]
    popularity = {}
    for layer in geometry["routed_layers"]:
        for phase in ["prefill", "decode"]:
            rng = random.Random(f"expert-fixture/{layer}/{phase}/{experts}")
            weights = [(rank + 6) ** -0.8 for rank in range(experts)]
            rng.shuffle(weights)
            popularity[layer, phase] = weights

    def window(scale):
        prefill, decode = [0] * (geometry["layers"] * experts), [0] * (geometry["layers"] * experts)
        for layer in geometry["routed_layers"]:
            for expert in range(experts):
                index = layer * experts + expert
                prefill[index] = scale * int((tick + layer) * 80 * popularity[layer, "prefill"][expert])
                decode[index] = scale * int((tick + 2) * 56 * popularity[layer, "decode"][expert])

        def summarize(values):
            layers = [values[layer * experts:(layer + 1) * experts] for layer in geometry["routed_layers"]]
            touched = sum(value > 0 for row in layers for value in row)
            active = [(row, sum(row)) for row in layers if any(row)]
            possible = len(layers) * experts
            return dict(touched=touched, possible=possible, coverage=touched / possible,
                        routed_rows=sum(values) // geometry["topk"],
                        concentration=sum(sum((value / total) ** 2 for value in row) for row, total in active) / len(active))

        return dict(elapsed_s=min(60.0, tick * .5) * scale,
                    prefill_routed_rows=prefill, decode_routed_rows=decode,
                    prefill=summarize(prefill), decode=summarize(decode),
                    combined=summarize([a + b for a, b in zip(prefill, decode)]))

    return dict(enabled=True, semantics="executed_routed_rows", geometry=geometry,
                sampled_s=tick * .5, error=None, fixture="Synthetic expert fixture",
                windows={name: window(scale) for name, scale in [("short", 1), ("long", 4), ("total", 12)]})


def expert_binary(snapshot, window, phase):
    g, w = snapshot["geometry"], snapshot["windows"][window]
    rows, experts = g["routed_layers"], g["experts"]
    counts, totals, peak = [], [], 0.0
    for layer in rows:
        values = []
        for i in range(layer * experts, (layer + 1) * experts):
            a, b = w["prefill_routed_rows"][i], w["decode_routed_rows"][i]
            values.append(a if phase == "prefill" else b if phase == "decode" else a + b)
        total = sum(values)
        totals.append(total)
        counts.extend(values)
        peak = max(peak, max(values) / total if total else 0)
    wide = window == "total"
    totals_offset = (128 + len(rows) * 4 + 7) & ~7
    offset = totals_offset + len(rows) * 8
    data = bytearray(offset + len(counts) * (8 if wide else 4))
    struct.pack_into("<4sHH5I4BddQQdQQddQI", data, 0, b"EU01", 1, 128,
                     g["layers"], experts, len(rows), g["topk"], rows[0],
                     ["short", "long", "total"].index(window), ["combined", "prefill", "decode"].index(phase), 2 if wide else 1, 0,
                     snapshot["sampled_s"], w["elapsed_s"], max(counts), int(snapshot["sampled_s"] * 2), peak,
                     w[phase]["touched"], w[phase]["possible"], w[phase]["coverage"], w[phase]["concentration"], w[phase]["routed_rows"], offset)
    struct.pack_into(f"<{len(rows)}I", data, 128, *rows)
    struct.pack_into(f"<{len(rows)}Q", data, totals_offset, *totals)
    struct.pack_into(f"<{len(counts)}{'Q' if wide else 'I'}", data, offset, *counts)
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def send(self, body, status=200, mime="application/json", headers=None):
        data = json.dumps(body).encode() if mime == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            self.get_route()
        except (ValueError, KeyError) as error:
            self.send({"error": {"type": "usage_error", "message": str(error)}}, 400)

    def get_route(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        unlocked = cookies.get("cuteafd_console") is not None and cookies["cuteafd_console"].value == "mock-unlock"
        if path == "/console/status":
            return self.send(dict(unlocked=unlocked, link_command=None))
        if path in ["/console/expert-usage", "/console/expert-usage.bin"]:
            if not unlocked:
                return self.send({"error": {"type": "console_locked"}}, 401)
            binary = path.endswith(".bin")
            if self.server.expert_usage == "disabled":
                return self.send(b"", 204, "application/octet-stream") if binary else self.send({"enabled": False})
            if not binary:
                return self.send(expert_snapshot(next(self.server.expert_ticks), getattr(self.server, "expert_geometry", None)))
            window, phase = query.get("window", "short"), query.get("phase", "combined")
            if window not in ["short", "long", "total"] or phase not in ["combined", "prefill", "decode"]:
                raise ValueError("invalid expert selection")
            tick = 1 + int(time.monotonic() - self.server.expert_started)
            etag = f'"fixture-{tick}-{window}-{phase}"'
            headers = {"ETag": etag}
            if self.headers.get("If-None-Match") == etag:
                return self.send(b"", 304, "application/octet-stream", headers)
            snapshot = expert_snapshot(tick, getattr(self.server, "expert_geometry", None))
            return self.send(expert_binary(snapshot, window, phase), mime="application/octet-stream", headers=headers)
        assets = {"/experts": "rust/crates/cuteafd-api/assets/experts.html",
                  "/usage": "rust/crates/cuteafd-usage/assets/usage.html",
                  "/": "rust/crates/cuteafd-api/assets/console.html",
                  "/bench": "rust/crates/cuteafd-bench/assets/bench.html",
                  "/assets/cuteafd-ui.css": "rust/crates/cuteafd-api/assets/cuteafd-ui.css",
                  "/assets/cuteafd-ui.js": "rust/crates/cuteafd-api/assets/cuteafd-ui.js",
                  "/assets/cuteafd-logo.svg": "assets/brand/cuteafd-logo-color-dark.svg",
                  "/assets/cuteafd-mark.svg": "assets/brand/cuteafd-mark-color-dark.svg"}
        if path in assets:
            file = ROOT / assets[path]
            mime = {".css": "text/css", ".js": "application/javascript", ".html": "text/html", ".svg": "image/svg+xml"}[file.suffix]
            return self.send(file.read_bytes(), mime=mime)
        if path == "/v1/stats":
            return self.send({"usage": USAGE})
        if path == "/v1/console/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            data = dict(type="snapshot", now=10000, text="locked" if self.server.locked else "on", config=dict(text="locked" if self.server.locked else "on", model="Synthetic fixture", concurrency=16, started_unix_ms=int(time.time() * 1000)), g=dict(active=3, queued=0), requests=[], recent=[])
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
            return
        if path == "/v1/bench/usage":
            return self.send(dict(record_bench=SETTINGS["record_bench"], available=True))
        if path == "/v1/bench/status":
            return self.send(dict(active=None, model="Synthetic fixture", auth="none"))
        if path == "/v1/bench/panels":
            return self.send(dict(panels=[]))
        if path == "/v1/bench/profiles":
            return self.send(dict(profiles=[]))
        if path == "/v1/bench/runs":
            return self.send(dict(runs=[]))
        if path == "/v1/bench/events":
            return self.send(b"", mime="text/event-stream")
        if path == "/bench/banner.js":
            return self.send(b"", mime="application/javascript")
        if not path.startswith("/console/usage/"):
            return self.send({"error": {"type": "not_found"}}, 404)
        if self.server.locked and "cuteafd_console=mock-unlock" not in self.headers.get("Cookie", ""):
            return self.send({"error": {"type": "console_locked"}}, 401)
        route = path.removeprefix("/console/usage/")
        if route == "media/" + MEDIA_HASH:
            return self.send(PIXEL, mime="image/png")
        if route == "settings":
            return self.send(dict(settings=SETTINGS, usage=USAGE, warning="With the full log on, user prompts and model outputs are stored in plain text for the retention period."))
        if route.startswith("log/sessions/"):
            s = next((s for s in LOGS if s["vsid"] == route.split("/")[-1]), None)
            return self.send(dict(vsid=s["vsid"], entries=s["entries"], perf=perf([r for r in ROWS if r["rid"] in {e["rid"] for e in s["entries"]}]))) if s else self.send({"error": {"type": "not_retained"}}, 404)
        if route.startswith("log/"):
            rid = route.split("/")[-1]
            found = next(((s, e) for s in LOGS for e in s["entries"] if e["rid"] == rid), None)
            if not found:
                return self.send({"error": {"type": "not_retained"}}, 404)
            s, e = found
            return self.send(dict(e, vsid=s["vsid"], request=dict(messages=e["items"]), response=dict(output=e["response"]), display=dict(items=e["items"], response=e["response"])))
        now = int(time.time() * 1000)
        to = int(query.get("to", now))
        from_ms = int(query.get("from", to - {"1h": 3600000, "6h": 21600000, "24h": 86400000, "7d": 604800000}.get(query.get("range", "24h"), 86400000)))
        if route.startswith("requests/") or route.startswith("sessions/"):
            from_ms = to - 604800000
        rows = [r for r in ROWS if from_ms <= r["ts_ms"] <= to and (query.get("bench") == "true" or not r["bench"])]
        rows = filter_rows(rows, query)
        if route == "summary":
            return self.send(dict(summary(rows), series=series(rows, query), usage=USAGE))
        if route == "series":
            return self.send(series(rows, query))
        if route == "latency":
            return self.send(dict(latency(rows), prompt_facets=[dict(prompt=k, latency=latency(rs)) for k, rs in groups(rows, prompt_band).items()], cache_facets=cache_facets(rows)))
        if route == "cache":
            return self.send(dict(series=series(rows, query), facets=cache_facets(rows), tokens_saved=sum(r["tokens_cached"] for r in rows), hit_rate=summary(rows)["cache_hit"]))
        if route == "flow":
            return self.send([dict(client=c, protocol=p, model=m, requests=len(rs), tokens=sum(r["tokens_in"] + r["tokens_out"] for r in rs)) for (c, p, m), rs in groups(rows, lambda r: (r["client_kind"], r["protocol"], r["model_served"])).items()])
        if route == "speculation":
            return self.send(dict(series=series(rows, query), by_model=[dict(model=k, summary=summary(rs), tokens_per_round=ratio(sum(r["tokens_out"] for r in rs), sum(r["rounds"] for r in rs)), histogram=histogram([ratio(r["tokens_out"], r["rounds"]) for r in rs])) for k, rs in groups(rows, lambda r: r["model_served"]).items()], by_client=[dict(client=k, summary=summary(rs)) for k, rs in groups(rows, lambda r: r["client_kind"]).items()]))
        if route == "errors":
            bucket = {"1m": 60000, "5m": 300000, "1h": 3600000, "1d": 86400000}[query.get("bucket", "1h")]
            outcomes = [dict(ts_ms=ts, counts={k: len(v) for k, v in groups(rs, lambda r: r["outcome"]).items()})
                        for ts, rs in sorted(groups(rows, lambda r: r["ts_ms"] // bucket * bucket).items())]
            return self.send(dict(outcomes=outcomes, series=series([r for r in rows if r["outcome"] != "ok"], query), classes=[dict(class_route_client="|".join(k), count=len(rs), last_ms=rs[-1]["ts_ms"]) for k, rs in groups([r for r in rows if r["outcome"] != "ok"], lambda r: (r["error_class"] if r["error_class"] is not None else r["outcome"], r["route"], r["client_kind"])).items()], stops=[dict(reason=k, count=len(rs)) for k, rs in groups(rows, lambda r: r["stop_reason"]).items()]))
        if route.startswith("sessions/"):
            sid = route.split("/")[-1]
            turns = [r for r in rows if r["session_id"] == sid]
            return self.send(dict(session_id=sid, turns=turns, perf=perf(turns)))
        if route == "sessions":
            key, descending = sort_options(query)
            sessions = [dict(session_id=k, source=rs[0]["session_source"], client=rs[0]["client_kind"], model=rs[0]["model_served"], first_ms=rs[0]["ts_ms"], last_ms=rs[-1]["ts_ms"], summary=summary(rs), perf=perf(rs), turns=[{key: r[key] for key in ["rid", "ts_ms", "tokens_in", "tokens_cached", "tokens_out"]} for r in rs]) for k, rs in groups(rows, lambda r: r["session_id"]).items()]
            sessions.sort(key=lambda s: -s["last_ms"])
            if key == "time":
                if not descending:
                    sessions.reverse()
            else:
                field = "prefill" if key == "prefill_tps" else "decode"
                sessions = sorted_rates(sessions, lambda s: s["perf"][field]["median_tps"], descending)
            return self.send(sessions)
        if route.startswith("requests/"):
            row = next((r for r in ROWS if r["rid"] == route.split("/")[-1]), None)
            return self.send(row) if row else self.send({"error": {"type": "not_retained"}}, 404)
        if route == "requests":
            key, descending = sort_options(query)
            rows = sorted(rows, key=lambda r: (r["ts_ms"], r["rid"]), reverse=True)
            limit = max(1, min(500, int(query.get("limit", 100))))
            offset = 0
            if key == "time" and descending:
                if "cursor" in query:
                    ts, rid = query["cursor"].split(":", 1)
                    rows = [r for r in rows if ((r["ts_ms"], r["rid"]) < (int(ts), rid) if descending else (r["ts_ms"], r["rid"]) > (int(ts), rid))]
            else:
                rows = list(reversed(rows)) if key == "time" else sorted_rates(rows, lambda r: r.get(key), descending)
                if "cursor" in query:
                    prefix, value = query["cursor"].split(":", 1)
                    if prefix != "o" or not value.isdigit():
                        raise ValueError("invalid offset cursor")
                    offset = int(value)
                rows = rows[offset:]
            page = rows[:limit]
            cursor = None
            if len(rows) > limit:
                cursor = f'{page[-1]["ts_ms"]}:{page[-1]["rid"]}' if key == "time" and descending else f'o:{offset + limit}'
            return self.send(dict(requests=page, next_cursor=cursor))
        if route == "log":
            logs = [s for s in LOGS if from_ms <= s["last_ms"] <= to]
            for key, field in [("client", "client"), ("model", "model"), ("session", "session_id")]:
                if key in query:
                    logs = [s for s in logs if s[field] == query[key]]
            if "protocol" in query:
                logs = [s for s in logs if query["protocol"] in s["protocols"]]
            logs.sort(key=lambda s: (-s["last_ms"], s["vsid"]))
            if "cursor" in query:
                ids = [s["vsid"] for s in logs]
                logs = logs[ids.index(query["cursor"]) + 1:] if query["cursor"] in ids else []
            limit = int(query.get("limit", 30))
            return self.send(dict(sessions=[dict(s, entries=len(s["entries"])) for s in logs[:limit]], next_cursor=logs[limit - 1]["vsid"] if len(logs) > limit else None))
        return self.send({"error": {"type": "not_found"}}, 404)

    def do_PUT(self):
        if self.server.locked and self.path.startswith("/console/") and "cuteafd_console=mock-unlock" not in self.headers.get("Cookie", ""):
            return self.send({"error": {"type": "console_locked"}}, 401)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/v1/bench/usage":
                SETTINGS["record_bench"] = bool(body["record_bench"])
                return self.send(dict(record_bench=SETTINGS["record_bench"], available=True))
            if self.path != "/console/usage/settings":
                return self.send({"error": {"type": "not_found"}}, 404)
            for k, current in SETTINGS.items():
                value = body[k]
                if isinstance(current, bool):
                    if not isinstance(value, bool):
                        raise ValueError(k + " must be boolean")
                elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(k + " must be a non-negative integer")
            SETTINGS.update(body)
            return self.send(dict(settings=SETTINGS, usage=USAGE, warning="With the full log on, user prompts and model outputs are stored in plain text for the retention period."))
        except (ValueError, KeyError) as error:
            return self.send({"error": {"type": "usage_error", "message": str(error)}}, 400)

    def do_POST(self):
        if self.server.locked and "cuteafd_console=mock-unlock" not in self.headers.get("Cookie", ""):
            return self.send({"error": {"type": "console_locked"}}, 401)
        if self.path not in ["/console/usage/log/clear", "/console/usage/clear"]:
            return self.send({"error": {"type": "not_found"}}, 404)
        LOGS.clear()
        if self.path == "/console/usage/clear":
            ROWS.clear()
        return self.send(dict(cleared=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--expert-usage", choices=["live", "disabled"], default="live")
    parser.add_argument("--expert-config", type=Path, help="Use a real checkpoint config's geometry with synthetic counts")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.locked = args.locked
    server.expert_usage = args.expert_usage
    server.expert_ticks = count(1)
    server.expert_started = time.monotonic()
    server.expert_geometry = expert_geometry(args.expert_config) if args.expert_config else None
    print(f"Synthetic usage server: http://{args.host}:{args.port}/usage", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
