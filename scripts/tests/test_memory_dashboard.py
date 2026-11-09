"""Memory UI consumes real-card stats and keeps scope/unknown semantics intact."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "scripts/tests/fixtures/memory-glm53-final-pair.json"
JS = ROOT / "rust/crates/cuteafd-api/assets/memory.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for UI model tests")
def test_memory_model_real_glm53_stats_and_missing_values():
    subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const m = require(process.argv[1]);
const fixture = JSON.parse(fs.readFileSync(process.argv[2]));
const data = m.normalize(fixture);
assert.equal(data.graphs[0].device, 0);
assert.equal(data.graphs[0].charged, 335544320);
assert.ok(data.domains.some(d => d.id === 'rtx0'));
assert.ok(data.domains.some(d => d.id.startsWith('spark')));
const gpu = data.domains.find(d => d.id === 'rtx0');
assert.ok(gpu.items.some(i => i.name === 'graph reserve'));
assert.ok(!gpu.items.some(i => i.name === 'unnamed other_bytes'));
assert.equal(gpu.items.reduce((s,i)=>s+i.bytes,0), 97832460152);
assert.ok(data.domains.find(d => d.id === 'host').reason.includes('not published'));
assert.equal(m.normalize({}).domains[0].cap, undefined);
assert.equal(m.bytes(null), 'unavailable');
assert.equal(m.bytes(0), '0.00 GiB');
assert.equal(m.exact(-100), '-100 B');
const folded = m.segments({cap:1000,items:[{category:'weights',name:'large',bytes:900},{category:'transport',name:'ring',bytes:5},{category:'runtime',name:'tail',bytes:4}]});
assert.equal(folded[1].category,'other-small');
assert.equal(folded[1].bytes,9);
assert.deepEqual(folded[1].members.map(i=>i.name),['ring','tail']);
const rates=m.rates({decode:{hits:20,captures:4}},{decode:{hits:10,captures:5}},2);
assert.equal(rates.decode.hits,5);assert.equal(rates.decode.captures,null);assert.equal(rates.decode.misses,null);
assert.equal(m.rates({decode:{hits:20}},null,2).decode.hits,null);
const h = new m.Rolling();for(let i=0;i<1000;i++)h.push(i*1000,{a:i});
assert.equal(h.samples.length,301);assert.equal(h.samples[0].time,699000);
assert.equal(h.series('missing')[0].value,undefined);
// Detailed admitted names replace aggregates rather than charging both.
const detailed = m.normalize({inventory:{plan:{domains:[{id:'gpu0',budget:{locations:[{device:{Coordinator:0}}],physical_bytes:1000,logical_bytes:900,headroom:10},items:[{category:'drafter',group:'draft weights',bytes:100}]}]},admitted:{'device.gpu0':{reserve:[{name:'release padding',bytes:10}]}}}});
assert.equal(detailed.domains.find(d=>d.id==='gpu0').items.reduce((s,i)=>s+i.bytes,0),110);
// Host budget combines every accepted owner, not system occupancy or RSS.
const host = m.normalize({inventory:{host_admission:{total_bytes:1000,host_budget_bytes:100,reserve_bytes:100,items:[{category:'tables',group:'engram',bytes:200}]}}}).domains[0];
assert.equal(host.items.reduce((s,i)=>s+i.bytes,0),400);
// New graph-pin fields remain intact, old builds need not supply them.
const pins = m.normalize({graph_cache:[{device:3,top_hits:[{hit_histogram:{counts:[0,3],interval_ms:10000},permanent_bytes:123}]}]});
assert.equal(pins.graphs[0].top_hits[0].permanent_bytes,123);
console.log('memory model: real-card fixture and edge cases pass');
""", str(JS), str(FIXTURE)], check=True, timeout=20)


def test_memory_fixture_endpoint_requires_cookie_and_serves_real_stats():
    spec = importlib.util.spec_from_file_location("memory_fixture", ROOT / "scripts/usage/mock-server.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as locked:
            urlopen(base + "/console/memory", timeout=5)
        assert locked.value.code == 401
        request = Request(base + "/console/memory", headers={"Cookie": "cuteafd_console=mock-unlock"})
        with urlopen(request, timeout=5) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response)["stats"]["graph_cache"][0]["charged"] == 335544320
        for route, needle in [("/memory", b"Reservations"), ("/assets/memory.js", b"memory_reconciliation")]:
            with urlopen(base + route, timeout=5) as response:
                assert needle in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
