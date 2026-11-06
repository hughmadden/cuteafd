import importlib.util
import json
import shutil
import subprocess
import threading
import time
import struct
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


def test_renderer_software_webgl_flags_are_opt_in():
    renderer = module("expert_webgl_renderer", "scripts/usage/render-smoke.py")
    assert renderer.webgl_flags(False) == ["--disable-gpu"]
    assert renderer.webgl_flags(True) == ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


def test_real_checkpoint_fixture_geometry_accepts_nested_and_flat_config(tmp_path):
    fixture = module("expert_geometry_fixture", "scripts/usage/mock-server.py")
    for data, expected in [
        ({"text_config": {"num_hidden_layers": 93, "num_experts": 896, "num_experts_per_token": 16, "first_k_dense_replace": 1}}, (93, 896, 92)),
        ({"num_hidden_layers": 48, "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_layer_freq": [0] + [1] * 47}, (48, 256, 47)),
    ]:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        geometry = fixture.expert_geometry(path)
        assert (geometry["layers"], geometry["experts"], len(geometry["routed_layers"])) == expected
        snapshot = fixture.expert_snapshot(1, geometry)
        payload = fixture.expert_binary(snapshot, "total", "combined")
        offset = struct.unpack_from("<I", payload, 112)[0]
        assert payload[30] == 2
        assert len(payload) == offset + expected[1] * expected[2] * 8


def test_expert_fixture_auth_live_values_and_disabled_variant():
    fixture = module("expert_fixture", "scripts/usage/mock-server.py")
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture.Handler)
    server.locked = True
    server.expert_usage = "live"
    server.expert_ticks = count(1)
    server.expert_started = time.monotonic()
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
        with pytest.raises(HTTPError) as locked:
            get("/console/expert-usage.bin")
        assert locked.value.code == 401
        with urlopen(Request(base + "/console/expert-usage.bin?window=short&phase=decode", headers={"Cookie": cookie}), timeout=5) as response:
            payload = response.read()
            tag = response.headers["ETag"]
        assert payload[:8] == b"EU01\x01\x00\x80\x00"
        assert payload[28:31] == bytes([0, 2, 1])
        assert struct.unpack_from("<I", payload, 16)[0] == 11
        offset = struct.unpack_from("<I", payload, 112)[0]
        assert len(payload) == offset + 11 * 32 * 4
        with pytest.raises(HTTPError) as unchanged:
            urlopen(Request(base + "/console/expert-usage.bin?window=short&phase=decode", headers={"Cookie": cookie, "If-None-Match": tag}), timeout=5)
        assert unchanged.value.code == 304
        assert unchanged.value.headers["Cache-Control"] == "no-store"
        server.expert_usage = "disabled"
        assert get("/console/expert-usage", cookie) == {"enabled": False}
        with urlopen(Request(base + "/console/expert-usage.bin", headers={"Cookie": cookie}), timeout=5) as response:
            assert response.status == 204
            assert response.read() == b""
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
const elements = new Map(); let uploads = 0, draws = 0;
const gl = new Proxy({getShaderParameter: () => true, getProgramParameter: () => true, isContextLost: () => false,
  texSubImage2D: () => uploads++, drawArrays: () => draws++}, {get: (target, key) => target[key] ?? (() => ({}))});
class Element {
  constructor() { this.textContent = ''; this.hidden = true; this.value = ''; this.children = []; this.listeners = {}; this.style = {}; this.clientWidth = 1000; }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; if (children[0]?.value) this.value = children[0].value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  get selectedOptions() { return [{text: this.value}]; }
  getContext(type) { assert.equal(type, 'webgl2'); return gl; }
}
const el = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
el('phase').value = 'combined'; el('window').value = 'short';
const document = {hidden: false, getElementById: el, createElement: () => new Element(), createDocumentFragment: () => new Element(), addEventListener: () => {}};
const header = {status: new Element(), meta: new Element()};
let interval, intervalMs, timer, calls = 0, pending, requestOptions;
const window = {devicePixelRatio: 1, addEventListener: () => {}, CuteUI: {
  header: () => header, adminGate: async () => true, token: () => '#111111',
  fmt: {n0: String}, facts: (target, facts) => target.replaceChildren(...facts),
}};
const sandbox = {document, window, location: {search: ''}, URLSearchParams, console, Intl,
  innerHeight: 1000, innerWidth: 1200, performance: {now: () => 0}, AbortController,
  setInterval: (f, ms) => { interval = f; intervalMs = ms; },
  setTimeout: f => {timer = f; return 1;}, clearTimeout: () => {},
  fetch: (url, options) => { assert.match(url, /^\/console\/expert-usage.bin\?/); requestOptions = options; calls++; return new Promise((resolve, reject) => {
    pending = resolve;
    options.signal.addEventListener('abort', () => reject(Object.assign(new Error(), {name: 'AbortError'})));
  }); },
};
const tick = async () => { for (let i = 0; i < 18; i++) await Promise.resolve(); };
const sample = (n, wide = false) => {
  const buffer = new ArrayBuffer(wide ? 160 : 152), d = new DataView(buffer);
  const u32 = (o, v) => d.setUint32(o, v, true), u64 = (o, v) => d.setBigUint64(o, BigInt(v), true), f64 = (o, v) => d.setFloat64(o, v, true);
  u32(0, 0x31305545); d.setUint16(4, 1, true); d.setUint16(6, 128, true);
  u32(8, 1); u32(12, 2); u32(16, 1); u32(20, 1);
  d.setUint8(28, ['short', 'long', 'total'].indexOf(el('window').value));
  d.setUint8(29, ['combined', 'prefill', 'decode'].indexOf(el('phase').value)); d.setUint8(30, wide ? 2 : 1);
  f64(32, n); f64(40, n); u64(48, 2*n); u64(56, n); f64(64, 2/3);
  u64(72, 2); u64(80, 2); f64(88, 1); f64(96, .5); u64(104, 3*n); u32(112, 144); u64(136, 3*n);
  if (wide) { u64(144, 9007199254740993n); u64(152, 2*n); } else { u32(144, n); u32(148, 2*n); }
  return buffer;
};
const respond = async (data, status = 200) => { pending({ok: status === 200, status, headers: {get: () => '"tag"'}, arrayBuffer: async () => data}); await tick(); };
(async () => {
  vm.runInNewContext(SCRIPT, sandbox); await tick();
  assert.equal(intervalMs, 500);
  interval(); interval(); await tick(); assert.equal(calls, 1, 'no overlapping requests');
  await respond(sample(1)); assert.equal(el('rows').textContent, '3');
  el('matrix').listeners.keydown({key: 'ArrowRight', preventDefault() {}});
  assert.match(el('readout').textContent, /2 routed selections/);
  const initialUploads = uploads, initialDraws = draws;
  interval(); assert.equal(requestOptions.headers['If-None-Match'], '"tag"'); await respond(null, 304);
  assert.equal(uploads, initialUploads); assert.equal(draws, initialDraws, '304 never redraws');
  interval(); await respond(sample(2)); assert.equal(el('rows').textContent, '6');
  assert.match(el('readout').textContent, /4 routed selections/);
  assert.equal(el('table-body').children[0].children[1].children[1].textContent, '4');
  interval(); el('pause').onclick(); await respond(sample(3)); assert.equal(el('rows').textContent, '6', 'pause discards response');
  const before = calls; interval(); await tick(); assert.equal(calls, before);
  el('pause').onclick(); await respond(sample(4)); assert.equal(el('rows').textContent, '12');
  el('window').value = 'total'; el('window').listeners.change(); await tick(); await respond(sample(5, true));
  el('matrix').listeners.keydown({key: 'Home', preventDefault() {}});
  assert.match(el('readout').textContent, /9,007,199,254,740,993 routed selections/, 'u64 hover stays exact');
  interval(); await respond(null, 204);
  for (const id of ['summary', 'plot', 'table-panel']) assert.equal(el(id).hidden, true);
  assert.equal(el('unavailable').hidden, false); assert.equal(header.status.textContent, 'UNAVAILABLE');
  interval(); timer(); await tick(); assert.equal(header.status.textContent, 'OFFLINE');
  interval(); await respond(sample(6, true)); assert.equal(el('unavailable').hidden, true);
  document.hidden = true; const hiddenCalls = calls; interval(); await tick(); assert.equal(calls, hiddenCalls);
  const count = calls; sandbox.location.search = '?webgl=off';
  vm.runInNewContext(SCRIPT, sandbox); await tick();
  assert.match(el('unavailable-reason').textContent, /WebGL2 is unavailable/); assert.equal(calls, count);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    subprocess.run([shutil.which("node"), "-e", "const SCRIPT = " + json.dumps(script) + ";\n" + harness],
                   capture_output=True, text=True, timeout=15, check=True)
    ui = (ROOT / "rust/crates/cuteafd-api/assets/cuteafd-ui.js").read_text()
    assert "{ id: 'experts', href: '/experts', label: 'EXPERTS' }" in ui
    assert "EXPERT_USAGE=on" in page
