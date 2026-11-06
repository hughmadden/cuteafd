import importlib.util
import json
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from itertools import count
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_renderer_installs_netscape_cookie_jar_without_logging_values(tmp_path, capsys):
    renderer = module("expert_renderer", "scripts/usage/render-smoke.py")
    jar = tmp_path / "console.cookies"
    jar.write_text("# Netscape HTTP Cookie File\n"
                   "#HttpOnly_127.0.0.1\tFALSE\t/\tFALSE\t0\tcuteafd_console\tfixture-session\n"
                   ".example.test\tTRUE\t/console\tTRUE\t4102444800\tscoped\tfixture-scoped\n"
                   "127.0.0.1\tFALSE\t/\tFALSE\t1\texpired\tfixture-expired\n")
    jar.chmod(0o600)
    calls = []

    class Client:
        def call(self, method, params):
            calls.append((method, params))
            return {"success": True}

    renderer.install_cookie_jar(Client(), jar, "http://127.0.0.1:8765")
    assert calls == [
        ("Network.setCookie", dict(name="cuteafd_console", value="fixture-session",
                                  url="http://127.0.0.1:8765", path="/", secure=False, httpOnly=True)),
        ("Network.setCookie", dict(name="scoped", value="fixture-scoped",
                                  url="http://127.0.0.1:8765", domain=".example.test",
                                  path="/console", secure=True, httpOnly=False, expires=4102444800)),
    ]
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


def test_expert_fixture_auth_live_values_and_disabled_variant():
    fixture = module("expert_fixture", "scripts/usage/mock-server.py")
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture.Handler)
    server.locked = True
    server.expert_usage = "live"
    server.expert_ticks = count(1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path, cookie=None):
        headers = {"Cookie": cookie} if cookie else {}
        with urlopen(Request(base + path, headers=headers), timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            return json.load(response)

    try:
        for cookie in [None, "cuteafd_console=mock-unlock-extra"]:
            with pytest.raises(HTTPError) as error:
                get("/console/expert-usage", cookie)
            assert error.value.code == 401
        assert not get("/console/status")["unlocked"]
        cookie = "cuteafd_console=mock-unlock"
        assert get("/console/status", cookie)["unlocked"]
        first, second = get("/console/expert-usage", cookie), get("/console/expert-usage", cookie)
        assert second["sampled_s"] > first["sampled_s"]
        assert first["windows"] != second["windows"]
        assert first["geometry"]["routed_layers"] == list(range(1, 12))
        server.expert_usage = "disabled"
        assert get("/console/expert-usage", cookie) == {"enabled": False}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for browser-independent UI behavior")
def test_expert_dashboard_refresh_is_serial_and_updates_all_views():
    page = (ROOT / "rust/crates/cuteafd-api/assets/experts.html").read_text()
    script = page.split("<script>\n", 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const elements = new Map();
const context2d = new Proxy({}, {get: () => () => {}});
class Element {
  constructor() { this.textContent = ''; this.hidden = true; this.value = ''; this.children = []; this.listeners = {}; this.style = {}; this.clientWidth = 1000; }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; if (children[0]?.value) this.value = children[0].value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  get selectedOptions() { return [{text: this.value}]; }
  getContext() { return context2d; }
}
const el = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
el('phase').value = 'combined'; el('window').value = 'short';
const document = {hidden: false, getElementById: el, createElement: () => new Element(), createDocumentFragment: () => new Element(), addEventListener: () => {}};
const header = {status: new Element(), meta: new Element()};
let interval, intervalMs, timer, calls = 0, pending;
const window = {devicePixelRatio: 1, addEventListener: () => {}, CuteUI: {
  header: () => header, adminGate: async () => true, token: () => '#111111',
  fmt: {n0: String}, facts: (target, facts) => target.replaceChildren(...facts),
}};
const sandbox = {document, window, performance: {now: () => 0}, AbortController,
  setInterval: (f, ms) => { interval = f; intervalMs = ms; },
  setTimeout: f => {timer = f; return 1;}, clearTimeout: () => {},
  fetch: (_url, options) => { calls++; return new Promise((resolve, reject) => {
    pending = resolve;
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error(), {name: 'AbortError'})));
  }); },
};
const tick = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const sample = n => {
  const summary = {touched: n, possible: 2, coverage: n / 2, concentration: .5, routed_rows: 3*n};
  const window = {elapsed_s: n, prefill_routed_rows: [n, 0], decode_routed_rows: [0, 2*n], combined: summary, prefill: summary, decode: summary};
  return {enabled: true, sampled_s: n, geometry: {layers: 1, experts: 2, topk: 1, routed_layers: [0]}, windows: {short: window, long: window, total: window}};
};
const respond = async data => { pending({ok: true, json: async () => data}); await tick(); };
(async () => {
  vm.runInNewContext(SCRIPT, sandbox); await tick();
  assert.equal(intervalMs, 500);
  interval(); interval(); await tick(); assert.equal(calls, 1, 'no overlapping requests');
  await respond(sample(1));
  assert.equal(el('rows').textContent, '3');
  el('matrix').listeners.keydown({key: 'ArrowRight', preventDefault() {}});
  assert.match(el('readout').textContent, /2 routed selections/);
  interval(); await respond(sample(2));
  assert.equal(el('rows').textContent, '6');
  assert.match(el('readout').textContent, /4 routed selections/);
  assert.equal(el('table-body').children[0].children[1].children[1].textContent, '4');
  interval(); el('pause').onclick(); await respond(sample(3));
  assert.equal(el('rows').textContent, '6', 'pause discards an in-flight response');
  const before = calls; interval(); await tick(); assert.equal(calls, before);
  el('pause').onclick(); await respond(sample(4));
  assert.equal(el('rows').textContent, '12', 'resume immediately refreshes');
  el('phase').value = 'prefill'; el('phase').listeners.change();
  assert.match(el('readout').textContent, /0 routed selections/);
  interval(); await respond({enabled: false});
  for (const id of ['summary', 'plot', 'table-panel']) assert.equal(el(id).hidden, true);
  assert.equal(el('unavailable').hidden, false);
  assert.equal(el('sample').textContent, ''); assert.equal(header.meta.children.length, 0);
  assert.equal(el('table-body').children.length, 0); assert.equal(header.status.textContent, 'UNAVAILABLE');
  interval(); timer(); await tick(); assert.equal(header.status.textContent, 'OFFLINE');
  interval(); await respond(sample(5)); assert.equal(el('unavailable').hidden, true);
  assert.equal(header.status.textContent, 'LIVE');
  document.hidden = true; const hiddenCalls = calls; interval(); await tick(); assert.equal(calls, hiddenCalls);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    subprocess.run([shutil.which("node"), "-e", "const SCRIPT = " + json.dumps(script) + ";\n" + harness],
                   capture_output=True, text=True, timeout=15, check=True)
    ui = (ROOT / "rust/crates/cuteafd-api/assets/cuteafd-ui.js").read_text()
    assert "{ id: 'experts', href: '/experts', label: 'EXPERTS' }" in ui
    assert "EXPERT_USAGE=on" in page
