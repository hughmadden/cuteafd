/* Memory telemetry has independent scopes. Never add aliases, cached stage
   samples, quotas and physical occupancy into a single measured total. */
(function(root) {
'use strict';
const numeric = x => typeof x === 'number' && Number.isFinite(x);
const arr = x => Array.isArray(x) ? x : [];
const bytes = n => !numeric(n) ? 'unavailable' : `${n < 0 ? '-' : ''}${(Math.abs(n) / 2 ** 30).toFixed(2)} GiB`;
const exact = n => numeric(n) ? `${n.toLocaleString('en-US')} B` : 'unavailable';
const number = n => numeric(n) ? n.toLocaleString('en-US', {maximumFractionDigits: 2}) : 'unavailable';
const CATEGORIES = ['weights','embedding','experts','drafter','kv','prefix','workspace','transport','staging','tables','runtime','reserved','unnamed','other-small'];
// Composite encoding: three validated hues with five stable, unique textures.
const categoryIndex = name => Math.max(0, CATEGORIES.indexOf(name));
const categoryName = name => name === 'other-small' ? 'Other small' : name === 'unnamed' ? 'Unnamed owner' : name;
const owner = (category, name, value) => ({category, name, bytes: value});
function stageRows(stats) {
  const r = stats.memory_reconciliation || {};
  const ready = arr(r.ready?.devices), first = arr(r.first_request?.devices);
  return [...new Set([...ready, ...first].map(d => d.device))].map(id => {
    const latest = first.find(d => d.device === id), previous = ready.find(d => d.device === id);
    // Spark first-request has no control-plane sample: retain the labelled ready
    // observation rather than turning an unavailable value into zero.
    return {...(latest || previous), ready_sample: previous, sample_stage: latest ? 'first_request' : 'ready'};
  });
}
function breakdownItems(b) {
  return [owner('weights','weights (aggregate)',b.weights_bytes), owner('workspace','workspace (aggregate)',b.workspace_bytes),
    owner('kv','KV + prefix (aggregate)',b.kv_bytes), owner('unnamed','unnamed other_bytes',b.other_bytes),
    ...arr(b.reserve).map(r => owner(r.name === 'release padding' ? 'reserved' : 'runtime',r.name,r.bytes))].filter(i => numeric(i.bytes) && i.bytes > 0);
}
function normalize(payload) {
  const stats = payload.stats || payload, inventory = payload.inventory || {}, stages = stageRows(stats);
  const domains = arr(inventory.plan?.domains).map(d => {
    const host = arr(d.budget?.locations).some(l => l.device === 'Host');
    const id = host ? 'host' : d.id;
    const admitted = inventory.admitted?.[`device.${d.id}`];
    const items = arr(d.items).map(i => owner(i.category,i.group,i.bytes)).filter(i => numeric(i.bytes) && i.bytes > 0);
    const headroom = d.budget?.headroom;
    if (numeric(headroom) && headroom > 0) {
      // Policy reserve is outside items_at; its named components include the
      // release padding. Runtime/graph components in items must not be re-added.
      const names = arr(admitted?.reserve).filter(r => ['family reserve','release padding','GPU_RESERVE','SPARK_RESERVE'].includes(r.name));
      if (names.reduce((s,r) => s+r.bytes,0) === headroom) items.push(...names.map(r => owner('reserved',r.name,r.bytes)));
      else items.push(owner('reserved', host ? 'host admission headroom' : 'policy headroom', headroom));
    }
    return {id, cap: numeric(d.budget?.physical_bytes) && numeric(d.budget?.logical_bytes) ? Math.min(d.budget.physical_bytes,d.budget.logical_bytes) : null,
      items, reason: d.items == null ? 'accepted owner projection unavailable' : null, source: 'accepted serving-growth plan',
      baseline: d.budget?.baseline};
  });
  for (const [key,b] of Object.entries(inventory.admitted || {})) {
    const id = key.replace(/^device\./,'');
    if (!domains.some(d => d.id === id)) domains.push({id,cap:b.cap_bytes,items:breakdownItems(b),source:'admitted aggregate breakdown'});
  }
  for (const row of stages) if (row.owners && !domains.some(d => d.id === row.device)) {
    domains.push({id:row.device,cap:row.owners.cap_bytes,items:breakdownItems(row.owners),source:'stage admission breakdown'});
  }
  if (!domains.some(d => d.id === 'host')) {
    const h = inventory.host_admission;
    domains.unshift(h ? {id:'host',cap:h.total_bytes,source:'accepted host admission',items:[
      ...arr(h.items).map(i => owner(i.category,i.group,i.bytes)), owner('prefix','host prefix-cache quota',h.host_budget_bytes),
      owner('reserved','host admission headroom',h.reserve_bytes)].filter(i => numeric(i.bytes) && i.bytes > 0)} :
      {id:'host',cap:inventory.host?.MemTotal,items:[],reason:'accepted host plan not published by this build',source:'unavailable'});
  }
  domains.sort((a,b) => (a.id === 'host' ? -1 : b.id === 'host' ? 1 : a.id.localeCompare(b.id,undefined,{numeric:true})));
  return {stats,inventory,stages,domains,graphs:arr(stats.graph_cache)};
}
function segments(domain) {
  const big = [], small = [];
  for (const i of domain.items) (numeric(domain.cap) && i.bytes < domain.cap * .01 ? small : big).push(i);
  const by = new Map();
  for (const item of big) {
    if (!by.has(item.category)) by.set(item.category,[]);
    by.get(item.category).push(item);
  }
  const result = [...by].map(([category,members]) => ({category,members,bytes:members.reduce((s,i)=>s+i.bytes,0)}));
  if (small.length) result.push({category:'other-small',members:small,bytes:small.reduce((s,i)=>s+i.bytes,0)});
  return result.sort((a,b) => categoryIndex(a.category)-categoryIndex(b.category));
}
function rates(current, previous, elapsed) {
  const out = {};
  for (const [cycle,counts] of Object.entries(current || {})) {
    out[cycle] = {};
    for (const key of ['hits','misses','captures','eager_runs','evictions']) {
      const n = counts[key], before = previous?.[cycle]?.[key];
      out[cycle][key] = numeric(n) && numeric(before) && elapsed > 0 && n >= before ? (n-before)/elapsed : null;
    }
  }
  return out;
}
class Rolling {
  constructor() { this.samples = []; }
  push(time, values) {
    this.samples.push({time,values});
    this.samples = this.samples.filter(s => s.time >= time-300000).slice(-301);
  }
  series(key) { return this.samples.map(s => ({time:s.time,value:s.values[key]})); }
}
const model = {normalize,segments,rates,Rolling,bytes,exact,numeric};
if (typeof module !== 'undefined' && module.exports) module.exports = model;
if (!root.document) return;
const $ = id => document.getElementById(id), UI = root.CuteUI, esc = UI.fmt.esc;
const shell = UI.header($('header'),{page:'memory',subtitle:'Memory reservations & usage'});
const history = new Rolling();
let previous = null, previousTime = null, paused = false, busy = false, timer, reservationSignature;
const tooltip = $('tooltip');
function hide() { tooltip.hidden = true; }
function tip(text,event,target) {
  tooltip.textContent = text; tooltip.hidden = false;
  const r = target.getBoundingClientRect();
  const x = event?.clientX ?? r.left, y = event?.clientY ?? r.bottom;
  tooltip.style.left = `${Math.max(16,Math.min(x+12,innerWidth-tooltip.offsetWidth-16))}px`;
  tooltip.style.top = `${Math.max(16,Math.min(y+12,innerHeight-tooltip.offsetHeight-16))}px`;
}
function inspect(el,text,highlight) {
  const show = e => { tip(typeof text === 'function' ? text() : text,e,el); if (highlight) highlight(true); };
  const leave = () => { hide(); if (highlight) highlight(false); };
  el.addEventListener('pointerenter',show); el.addEventListener('pointermove',show);
  el.addEventListener('focus',e=>show(null)); el.addEventListener('pointerleave',leave); el.addEventListener('blur',leave);
  el.addEventListener('click',show); el.addEventListener('keydown',e=>{if(e.key === 'Escape')leave();});
}
function highlightCategory(category,on) {
  document.querySelectorAll('[data-category]').forEach(el => {
    el.classList.toggle('active',on && el.dataset.category === category);
    el.classList.toggle('dim',on && el.dataset.category !== category);
  });
}
function pattern(category,id) {
  const n = categoryIndex(category), color = ['var(--mem-blue)','var(--mem-orange)','var(--mem-aqua)'][n % 3];
  const texture = Math.floor(n/3);
  const paths = ['', '<path d="M-2 2L2-2M0 8L8 0M6 10L10 6"/>', '<path d="M0 4H8"/>', '<path d="M4 0V8M0 4H8"/>', '<circle cx="4" cy="4" r="1.5"/>'];
  return `<pattern id="${id}" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="${color}"/><g stroke="var(--panel)" stroke-width="1.5" fill="var(--panel)">${paths[texture]}</g></pattern>`;
}
function reservationText(domain,s) {
  return `${domain.id} / ${categoryName(s.category)}\n${exact(s.bytes)} (${bytes(s.bytes)})\n${s.members.map(i=>`${i.name}: ${exact(i.bytes)} [${i.category}]`).join('\n')}`;
}
function renderReservations(data) {
  const categoryMembers = new Map();
  const rows = [];
  $('reservations').innerHTML = data.domains.map((d,index) => {
    const parts = segments(d), total = d.items.reduce((s,i)=>s+i.bytes,0), scale = Math.max(d.cap || 0,total,1);
    let y = 218;
    const defs = parts.map(s=>pattern(s.category,`p${index}-${categoryIndex(s.category)}`)).join('');
    const marks = parts.map(s => {
      const h = s.bytes/scale*205; y -= h;
      const key = `s${index}-${categoryIndex(s.category)}`;
      if (!categoryMembers.has(s.category)) categoryMembers.set(s.category,[]);
      categoryMembers.get(s.category).push(reservationText(d,s));
      return `<rect class="segment" id="${key}" data-category="${esc(s.category)}" x="24" y="${y}" width="76" height="${h}" fill="url(#p${index}-${categoryIndex(s.category)})" tabindex="0" role="img" aria-label="${esc(reservationText(d,s))}"/>`;
    }).join('');
    rows.push(...d.items.map(i=>`<tr><td>${esc(d.id)} / ${esc(i.name)}</td><td>${esc(i.category)}</td><td>${exact(i.bytes)}</td></tr>`));
    return `<article class="domain"><h3>${esc(d.id)}</h3><span class="capacity">${bytes(d.cap)} capacity</span><svg viewBox="0 0 125 240" aria-label="${esc(d.id)} reservations"><defs>${defs}</defs><rect x="24" y="13" width="76" height="205" fill="var(--well)" stroke="var(--line-2)"/>${marks}<text x="62" y="236" text-anchor="middle" fill="currentColor" font-size="10">${esc(bytes(total))} planned</text></svg><p class="hint">${esc(d.reason ? `Unavailable: ${d.reason}` : d.source)}</p>${numeric(d.cap)&&total>d.cap?'<p>Owners exceed capacity</p>':''}</article>`;
  }).join('');
  $('owners').innerHTML = rows.join('');
  $('legend').innerHTML = [...categoryMembers].sort((a,b)=>categoryIndex(a[0])-categoryIndex(b[0])).map(([category]) =>
    `<button type="button" data-category="${esc(category)}"><svg class="swatch" viewBox="0 0 20 20" aria-hidden="true"><defs>${pattern(category,`l${categoryIndex(category)}`)}</defs><rect width="20" height="20" fill="url(#l${categoryIndex(category)})"/></svg>${esc(categoryName(category))}</button>`).join('');
  for (const [index,d] of data.domains.entries()) for (const s of segments(d)) {
    inspect($(`s${index}-${categoryIndex(s.category)}`),reservationText(d,s),on=>highlightCategory(s.category,on));
  }
  $('legend').querySelectorAll('button').forEach(el=>inspect(el,categoryMembers.get(el.dataset.category).join('\n\n'),on=>highlightCategory(el.dataset.category,on)));
}
function metric(label,value,limit,unit,key,reason) {
  return {label,value:numeric(value)?value:null,limit:numeric(limit)?limit:null,unit,key,reason};
}
function liveMetrics(data) {
  const p = data.stats.prefix_cache || {}, h = p.host || {}, inv = data.inventory;
  const pagesUsed = numeric(p.pages)&&numeric(p.pages_free)?p.pages-p.pages_free:null;
  const groups = [{id:'host',note:'Host cache is the prefix-cache host tier, not a second allocation.',metrics:[
    metric('Prefix host tier',h.bytes_used,h.quota_bytes,'bytes','host-cache','host prefix counters not published'),
    metric('Process RSS',data.stats.host_heap?.rss_bytes,null,'bytes','host-rss','host-heap reporter unavailable'),
    metric('System unavailable RAM',numeric(inv.host?.MemTotal)&&numeric(inv.host?.MemAvailable)?inv.host.MemTotal-inv.host.MemAvailable:null,inv.host?.MemTotal,'bytes','host-system','background host inventory unavailable'),
    metric('Pinned backings',inv.pinned?.tracked,null,'bytes','host-pinned','allocation inventory unavailable'),
    metric('Host registered backings',inv.host_registered?.tracked,null,'bytes','host-registered','allocation inventory unavailable')
  ]},{id:'shared KV / prompt cache',note:'Page and mark counters are logical/shared; aliases are not additive.',metrics:[
    metric('KV pages used',pagesUsed,p.pages,'pages','kv-pages','KV pool counters not published'),
    metric('Device marks',p.marks_in_use,p.mark_slots,'marks','marks','device mark counters not published'),
    metric('Marked pages',p.mark_pages,null,'pages','mark-pages','mark pages not published'),
    metric('Retained pages',p.pages_retained,null,'pages','retained-pages','retained pages not published')
  ]}];
  for (const d of data.domains.filter(d=>d.id !== 'host')) {
    const row = data.stages.find(r=>r.device === d.id), spark = d.id.startsWith('spark');
    const physical = row?.provenance?.device_sample?.device;
    const graph = data.graphs.find(g=>g.device === (numeric(physical)?physical:Number(d.id.replace('rtx',''))));
    const kvPages = p.pages_used_by_gpu?.[Number(d.id.replace('rtx',''))];
    groups.push({id:d.id,note:spark?'Launch-relative diagnostic only (#1568). Post-request sampling unavailable (#1631).':`Reconciliation stage: ${row?.sample_stage || 'unavailable'}. Stage values are not live.`,metrics:[
      ...(!spark?[metric('KV pages used (device)',kvPages,p.pages,'pages',`${d.id}-pages`,'per-GPU KV counters unavailable'),
        metric('Graph arena',graph?.charged,graph?.reserve,'bytes',`${d.id}-arena`,'graph controller unavailable')]:[]),
      metric('Measured occupancy (stage)',row?.measured_bytes,null,'bytes',`${d.id}-measured`,row?.unavailable_reason || 'reconciliation sample unavailable'),
      metric('delta_vs_measured (stage)',row?.delta_vs_measured,null,'bytes',`${d.id}-delta`,row?.unavailable_reason || 'reconciliation sample unavailable'),
      metric('graph_reserve_slack (stage)',row?.graph_reserve_slack,null,'bytes',`${d.id}-slack`,row?.graph_measurement_unavailable_reason || row?.unavailable_reason || 'reconciliation unavailable'),
      ...(spark&&row?.ready_sample?[metric('Launch use (ready diagnostic)',row.ready_sample.measured_bytes,null,'bytes',`${d.id}-ready`,'ready baseline unavailable')]:[])
    ]});
  }
  return groups;
}
function formatMetric(m,n) { return m.unit === 'bytes'?bytes(n):numeric(n)?`${number(n)} ${m.unit}`:'unavailable'; }
function sparkline(m) {
  const samples = history.series(m.key), values = samples.filter(s=>numeric(s.value));
  if (!values.length) return '';
  const min = Math.min(0,...values.map(s=>s.value)), max = Math.max(1,m.limit || 0,...values.map(s=>s.value));
  const end = history.samples.at(-1).time, start = end-300000;
  let path = '', gap = true;
  for (const s of samples) {
    if (!numeric(s.value)) { gap=true; continue; }
    path += `${gap?'M':'L'}${((s.time-start)/300000*300).toFixed(2)},${(40-(s.value-min)/(max-min)*34).toFixed(2)} `; gap=false;
  }
  return `<svg viewBox="0 0 300 48" preserveAspectRatio="none" tabindex="0" role="img" data-metric="${esc(m.key)}" aria-label="${esc(m.label)} over the last five minutes"><path d="M0 40H300" stroke="var(--line)"/><path class="sparkline" d="${path}"/><circle class="spark-dot" r="4" cx="300" cy="${40-(values.at(-1).value-min)/(max-min)*34}"/><line class="crosshair" y1="2" y2="44" stroke="var(--ink-2)" hidden/></svg>`;
}
function renderLive(data,time) {
  const groups = liveMetrics(data), values = {};
  groups.flatMap(g=>g.metrics).forEach(m=>{values[m.key]=m.value;}); history.push(time,values);
  $('live').innerHTML = groups.map(g=>`<article class="card"><h3>${esc(g.id)}</h3><p class="hint">${esc(g.note)}</p>${g.metrics.map(m=>`<div class="metric"><div class="metric-head"><span>${esc(m.label)}</span><strong>${esc(formatMetric(m,m.value))}${numeric(m.limit)?` / ${esc(formatMetric(m,m.limit))}`:''}</strong></div>${m.value === null?`<p class="hint">Unavailable: ${esc(m.reason)}</p>`:sparkline(m)}</div>`).join('')}</article>`).join('');
  const by = new Map(groups.flatMap(g=>g.metrics).map(m=>[m.key,m]));
  $('live').querySelectorAll('svg[data-metric]').forEach(svg=>{
    const m = by.get(svg.dataset.metric), samples = history.series(m.key);
    const describe = e=>{
      const rect = svg.getBoundingClientRect(), fraction = e?Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)):1;
      const selected = time-300000+fraction*300000;
      const sample = samples.reduce((best,s)=>Math.abs(s.time-selected)<Math.abs(best.time-selected)?s:best,samples[0]);
      const cross = svg.querySelector('.crosshair'); cross.removeAttribute('hidden');
      const x = Math.max(0,(sample.time-(time-300000))/300000*300); cross.setAttribute('x1',x);cross.setAttribute('x2',x);
      tip(`${m.label}\n${new Date(sample.time).toLocaleTimeString()}: ${formatMetric(m,sample.value)}${m.unit==='bytes'?' / '+exact(sample.value):''}\nLimit: ${formatMetric(m,m.limit)}`,e,svg);
    };
    svg.addEventListener('pointermove',describe);svg.addEventListener('focus',()=>describe(null));
    const leave = ()=>{svg.querySelector('.crosshair').setAttribute('hidden','');hide();};svg.addEventListener('pointerleave',leave);svg.addEventListener('blur',leave);
  });
}
function graphKeys(rows) {
  if (!Array.isArray(rows)) return '<p class="hint">Unavailable: per-key pin telemetry not published by this build.</p>';
  if (!rows.length) return '<p class="hint">No recorded keys in this top list.</p>';
  return `<div class="table-scroll"><table><thead><tr><th>Key</th><th>Hits</th><th>Churn (capture / evict / recapture)</th><th>Cost</th><th>Heat</th></tr></thead><tbody>${rows.slice(0,10).map(k=>{
    const counts = arr(k.hit_histogram?.counts), maximum = Math.max(1,...counts);
    return `<tr><td class="key">${esc(k.site)} / ${esc(k.shape)}<br><span class="hint">bank ${esc(k.bank)} / ${k.pinned?'pinned':'not pinned'}</span></td><td>${number(k.hits)}</td><td>${number(k.captures)} / ${number(k.evictions)} / ${number(k.recaptures)}</td><td>${bytes(k.permanent_bytes)}<br><span class="hint">capture ${bytes(k.capture_bytes)}; ${number(k.nodes)} nodes</span></td><td><div class="heat" role="img" aria-label="Hits by interval">${counts.map((n,i)=>`<i tabindex="0" title="${esc(`${number(n)} hits; ${(k.hit_histogram.interval_ms || 0)/1000}s interval ${i+1}`)}" style="height:${Math.max(1,n/maximum*30)}px"></i>`).join('')}</div><span class="hint">${counts.length?`${counts.length} x ${number(k.hit_histogram.interval_ms/1000)}s`:'unavailable histogram'}</span></td></tr>`;
  }).join('')}</tbody></table></div>`;
}
function renderGraphs(data,time) {
  if (!data.graphs.length) { $('graphs').innerHTML='<p>Unavailable: graph-cache controller not published by this family/build.</p>';return; }
  $('graphs').innerHTML=data.graphs.map(g=>{
    const before=previous?.graphs.find(p=>p.device===g.device), rate=rates(g.cycles,before?.cycles,(time-previousTime)/1000);
    return `<article class="card"><h3>CUDA device ${esc(g.device)}</h3><p class="hint">Reserve ${bytes(g.reserve)} / arena ${bytes(g.charged)} / pinned ${bytes(g.pinned)} / pending ${bytes(g.pending_capture)}<br>Node pool ${number(g.pools?.live_nodes)} live / ${number(g.pools?.node_high_water)} high-water. Executable pool ${number(g.pools?.live_execs)} live / ${number(g.pools?.exec_high_water)} high-water. Basis: ${esc(JSON.stringify(g.basis ?? 'unavailable'))}</p><div class="table-scroll"><table><caption>Per-second changes; unavailable until a second sample or after a counter reset.</caption><thead><tr><th>Cycle</th><th>Hits/s</th><th>Misses/s</th><th>Captures/s</th><th>Eager/s</th><th>Evictions/s</th></tr></thead><tbody>${Object.entries(rate).map(([cycle,c])=>`<tr><td>${esc(cycle)}</td>${['hits','misses','captures','eager_runs','evictions'].map(key=>`<td title="Lifetime: ${number(g.cycles[cycle][key])}">${number(c[key])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><details open><summary>Top 10 keys by hits</summary>${graphKeys(g.top_hits)}</details><details open><summary>Top 10 keys by churn</summary>${graphKeys(g.top_churn)}</details></article>`;
  }).join('');
  $('graphs').querySelectorAll('.heat i').forEach(el=>inspect(el,el.title));
}
function render(payload) {
  const time=Date.now(), data=normalize(payload);
  const signature=JSON.stringify(data.domains);
  if(signature!==reservationSignature) { renderReservations(data);reservationSignature=signature; }
  // Preserve a focused inspection while samples accumulate; update its table
  // on the next poll after focus leaves rather than removing the focused node.
  const inspecting=$('live').contains(document.activeElement)||$('graphs').contains(document.activeElement);
  if(!inspecting) { renderLive(data,time);renderGraphs(data,time); }
  else { const values={};liveMetrics(data).flatMap(g=>g.metrics).forEach(m=>{values[m.key]=m.value;});history.push(time,values); }
  previous=data;previousTime=time;
  const age=numeric(data.inventory.sampled_unix_s)?Math.max(0,time/1000-data.inventory.sampled_unix_s):null;
  $('notice').textContent=`Connected. Polling each second; family counters update at publication boundaries. Allocation inventory ${age===null?'unavailable':`${number(age)}s old (10s cadence)`}.`;
  $('window').textContent=`${history.samples.length} observations / five-minute rolling window`;
  shell.status.textContent='MEMORY'; document.body.dataset.memoryReady='true';
}
async function refresh() {
  if (busy||paused||document.hidden) return;
  busy=true;
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),8000);
  try {
    const response=await fetch('/console/memory',{credentials:'same-origin',cache:'no-store',signal:controller.signal});
    if(response.status===401) { paused=true;$('notice').textContent='Console locked. Open an admin console link and reload this page.';return; }
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch(e) { $('notice').textContent=`Unavailable: ${e.message}. Last successful observation retained; no zero samples inserted.`; }
  finally { clearTimeout(timeout);busy=false; }
}
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause';if(!paused)refresh();});
$('theme').addEventListener('change',()=>{document.documentElement.dataset.theme=$('theme').value;});
document.addEventListener('visibilitychange',()=>{if(!document.hidden){previous=null;refresh();}});
root.addEventListener('pagehide',()=>clearInterval(timer));
(async()=>{if(await UI.adminGate($('app'),'Memory dashboard','Inspect admitted memory owners, live caches and graph reserves.')){await refresh();timer=setInterval(refresh,1000);}})();
})(typeof window !== 'undefined'?window:globalThis);
