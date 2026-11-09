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
const domain={cap:1000,items:[{category:'weights',name:'large',bytes:900},{category:'transport',name:'ring',bytes:1},{category:'runtime',name:'tail',bytes:1},{category:'runtime',name:'named runtime',bytes:4}]};
const folded = m.segments(domain);
const small=folded.find(i=>i.category==='other-small');
assert.equal(small.bytes,1);
assert.deepEqual(small.members.map(i=>i.name),['tail']);
assert.ok(folded.some(i=>i.name==='ring'&&i.bytes===1));
assert.ok(folded.some(i=>i.name==='named runtime'&&i.bytes===4));
const unfolded=m.segments(domain,false);
assert.ok(!unfolded.some(i=>i.category==='other-small'));
assert.ok(unfolded.some(i=>i.name==='tail'&&i.bytes===1));
assert.equal(unfolded.reduce((n,i)=>n+i.bytes,0),domain.items.reduce((n,i)=>n+i.bytes,0));
const capped=m.segments({cap:100*2**30,items:[{category:'runtime',name:'above fixed cap',bytes:20*2**20},{category:'runtime',name:'below fixed cap',bytes:15*2**20}]});
assert.ok(capped.some(i=>i.name==='above fixed cap'));
assert.equal(capped.find(i=>i.category==='other-small').bytes,15*2**20);
assert.ok(fs.readFileSync(process.argv[1],'utf8').includes('segments(d, false).filter'));
assert.ok(m.segments(gpu).some(i=>i.name==='graph reserve'&&i.group==='graph'));
const spark=data.domains.find(d=>d.id==='spark0');
assert.ok(spark.items.some(i=>i.name==='device workspace'&&i.bytes===450652136));
assert.ok(spark.items.some(i=>i.name==='future endpoint mapped rings'&&i.bytes===547356672));
assert.equal(spark.items.reduce((n,i)=>n+i.bytes,0),89711968256+450652136+536576+547356672+33992704+11008);
// Empty coordinator Spark forecasts cannot hide the real worker owners.
const withPlan=m.normalize({...fixture,inventory:{plan:{domains:[{id:'spark0',budget:{physical_bytes:100e9,logical_bytes:100e9},items:[]}]}}});
assert.ok(withPlan.domains.find(d=>d.id==='spark0').items.length>1);
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
assert.equal(m.ownerLabel('GLM process ordinary RSS steady: Estimated { derivation: "long" }'), 'GLM process ordinary RSS');
assert.equal(m.ownerLabel('glm.decode: Spark intake transport envelope; long provenance'), 'glm.decode Spark intake');
assert.equal(m.ownerLabel('graph reserve: glm.graphs.0'), 'graph reserve');
const key={site:'glm5.segment',bank:1,key_id:3,shape:'GlmGraphKey { layer: 0, rows: 16, table_width: 16, table_stride: 16, previous: Full, head: false }'};
assert.equal(m.graphIdentity(key).label,'L0 r16 w16 prev-full');
assert.equal(m.graphIdentity(key).geometry,m.graphIdentity({...key,shape:key.shape.replace('layer: 0','layer: 8')}).geometry);
assert.ok(m.graphIdentity(key).full.includes(key.shape));
assert.equal(m.graphIdentity({...key,shape:key.shape.replace('Full','Planes(4)')}).label,'L0 r16 w16 prev-planes4');
const provenance=m.segments({cap:1000,items:[{category:'runtime',name:'module touch: measured graphs-before-serving delta',bytes:500}]});
assert.equal(provenance[0].group,'runtime');
// Remove weights before folding: even tiny weight owners must not leak into Other small.
const filterDomain={cap:1000,items:[...domain.items,{category:'embedding',name:'tiny embedding',bytes:1},{category:'tables',name:'tables',bytes:2},{category:'experts',name:'experts',bytes:3},{category:'drafter',name:'draft weights',bytes:20}]};
const before=JSON.stringify(filterDomain), visible=m.reservationView(filterDomain,true), full=m.reservationView(filterDomain);
assert.equal(visible.hidden,906);
assert.equal(visible.total,26);
assert.equal(visible.scale,26);
assert.equal(full.scale,1000);
assert.equal(full.total,932);
assert.equal(JSON.stringify(filterDomain),before);
assert.ok(visible.parts.every(s=>s.group!=='weights'&&s.members.every(i=>!['weights','embedding','tables','experts'].includes(i.category))));
assert.ok(visible.parts.some(s=>s.name==='draft weights'&&s.group==='workspace'));
for(const s of visible.parts) assert.equal(s.variant,full.parts.find(p=>p.key===s.key).variant);
assert.equal(m.reservationView({items:[]},true).scale,1);
assert.equal(m.reservationView({items:[]},true).total,0);
assert.equal(m.reservationView({items:[],reason:'not published'},true).hidden,0);
assert.equal(m.segmentLabel(2**30,13.99),'');
assert.equal(m.segmentLabel(2**30,14),'1.00 GiB');
assert.equal(m.labelInk('#000000'),'#ffffff');
assert.equal(m.labelInk('#ffffff'),'#000000');
// Both themes and all owner shades select the stronger WCAG text contrast.
for(const color of ['#3987e5','#d95926','#199e70','#c98500','#d55181','#008300','#9085e9','#2a78d6','#eb6834','#159569','#bd8100','#cc608b','#6250d6']) for(const variant of [0,1,2]) {
  const overlay=variant===1?255:0,alpha=variant===0?0:.1;
  const rgb=color.slice(1).match(/../g).map(c=>(parseInt(c,16)*(1-alpha)+overlay*alpha)/255);
  const linear=rgb.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);
  const l=linear[0]*.2126+linear[1]*.7152+linear[2]*.0722;
  const contrast=m.labelInk(color,variant)==='#000000'?(l+.05)/.05:1.05/(l+.05);
  assert.ok(contrast>=4.5,`${color} shade ${variant}: ${contrast}`);
}
const stored=new Map(),viewer={localStorage:{getItem:k=>stored.get(k),setItem:(k,v)=>stored.set(k,v)}};
assert.equal(m.weightPreference(viewer),false);
assert.equal(m.weightPreference(viewer,true),true);
assert.equal(m.weightPreference(viewer),true);
assert.equal(m.weightPreference(viewer,false),false);
assert.equal(m.weightPreference(viewer),false);
const denied={get localStorage(){throw Error('denied');}};
assert.equal(m.weightPreference(denied),false);
assert.equal(m.weightPreference(denied,true),true);
assert.equal(m.weightPreference({}),false);
assert.equal(m.weightPreference({},true),true);
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
