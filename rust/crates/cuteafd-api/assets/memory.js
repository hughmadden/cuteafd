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
const GROUPS = ['weights','kv','graph','workspace','runtime','transport','padding'];
const groupIndex = name => Math.max(0,GROUPS.indexOf(name));
function ownerGroup(i) {
  if (/^graph(?:\s|[.:_-])/i.test(i.name)) return 'graph';
  if (['weights','embedding','experts','tables'].includes(i.category)) return 'weights';
  if (['kv','prefix'].includes(i.category)) return 'kv';
  if (['workspace','drafter'].includes(i.category)) return 'workspace';
  if (['transport','staging'].includes(i.category)) return 'transport';
  if (i.category === 'reserved') return 'padding';
  return 'runtime';
}
const categoryIndex = name => Math.max(0, CATEGORIES.indexOf(name));
const categoryName = name => name === 'other-small' ? 'Other small' : name === 'unnamed' ? 'Unnamed owner' : name;
const owner = (category, name, value) => ({category, name, bytes: value});
// Provenance stays in owner identity/tooltips; only the visible caption is terse.
function ownerLabel(name) {
  if (/^glm\.(decode|prefill): Spark intake/.test(name)) return name.replace(/:.*/, ' Spark intake');
  return name.split(/: | \(/)[0].replace(/ ordinary RSS steady$/, ' ordinary RSS');
}
function graphIdentity(k) {
  const shape = String(k.shape ?? ''), field = name => shape.match(new RegExp(`\\b${name}: ([^,}]+)`))?.[1]?.trim();
  const rows = field('rows'), width = field('table_width'), stride = field('table_stride');
  const previous = field('previous')?.replace(/^First$/, 'first').replace(/^Delta$/, 'delta').replace(/^Full$/, 'full').replace(/^Planes\((\d+)\)$/, 'planes$1');
  return {
    label: rows ? `${field('layer') ? `L${field('layer')} ` : ''}r${rows}${width ? ` w${width}` : ''}${previous ? ` prev-${previous}` : ''}${field('head') === 'true' ? ' head' : ''}` : shape.length > 64 ? shape.slice(0,61)+'...' : shape,
    geometry: `${k.site} / ${rows ? `r${rows}${width ? ` w${width}` : ''}${stride ? ` s${stride}` : ''}${previous ? ` prev-${previous}` : ''}${field('head') ? ` head-${field('head')}` : ''}` : 'other geometry'} / bank ${k.bank}`,
    full: `${k.site} / ${shape}\nbank ${k.bank} / key ${k.key_id ?? 'unavailable'} / ${k.pinned ? 'pinned' : 'not pinned'}`
  };
}
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
    ...arr(b.reserve).map(r => owner(['release padding','family reserve','GPU_RESERVE','SPARK_RESERVE'].includes(r.name) ? 'reserved' : 'runtime',r.name,r.bytes))].filter(i => numeric(i.bytes) && i.bytes > 0);
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
  for (const row of stages) {
    const ledger = row.provenance?.owner_ledger;
    if (row.device.startsWith('spark') && ledger) {
      const allocations = arr(ledger.runtime?.detail?.allocations);
      const future = numeric(row.admitted_future_ring_bytes) && row.admitted_future_ring_bytes > 0;
      const items = [owner('experts','resident expert weights',ledger.resident_bytes),
        owner('workspace','device workspace',ledger.runtime?.workspace_bytes),
        owner('workspace','pinned workspace (UMA)',ledger.runtime?.pinned_workspace_bytes),
        owner('transport',future?'future endpoint mapped rings':'mapped endpoint rings',ledger.mapped_ring_bytes),
        owner('transport','host exchange (UMA)',ledger.host_exchange_bytes),
        owner('transport','row indices (UMA)',ledger.row_indices_bytes),
        ...allocations.filter(i=>i.space==='host').map(i=>owner('workspace',`host: ${i.scope}`,i.bytes))
      ].filter(i=>numeric(i.bytes)&&i.bytes>0);
      // Device/pinned detail allocations alias the aggregate fields above.
      const existing = domains.find(d=>d.id===row.device);
      const domain = {id:row.device,cap:existing?.cap || row.owners?.cap_bytes,items,
        source:'worker admitted owners (includes future rings; not ready occupancy)'};
      if(existing) domains.splice(domains.indexOf(existing),1,domain); else domains.push(domain);
    } else if (row.owners && !domains.some(d => d.id === row.device)) {
      domains.push({id:row.device,cap:row.owners.cap_bytes,items:breakdownItems(row.owners),source:'stage admission breakdown'});
    }
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
function segments(domain, foldSmall = true) {
  const named = new Map(), small = [];
  const threshold = numeric(domain.cap) ? Math.min(domain.cap * .002, 16 * 2 ** 20) : 0;
  for (const i of domain.items) {
    const required = /graph|draft|workspace|transport|intake|ring|exchange|indices|page table|BLAS|padding/i.test(i.name);
    if (foldSmall && !required && i.bytes < threshold) { small.push(i);continue; }
    const key = `${i.category}:${i.name}`;
    if (!named.has(key)) named.set(key,{key,name:i.name,category:i.category,group:ownerGroup(i),members:[],bytes:0});
    const s=named.get(key);s.members.push(i);s.bytes+=i.bytes;
  }
  const result=[...named.values()];
  if(small.length) result.push({key:'other-small',name:'Other small',category:'other-small',group:'runtime',members:small,bytes:small.reduce((s,i)=>s+i.bytes,0)});
  result.sort((a,b)=>groupIndex(a.group)-groupIndex(b.group)||a.name.localeCompare(b.name));
  // Stable name-derived shade; unrelated domains never repaint an owner.
  result.forEach((s,i)=>{s.variant=[...s.key].reduce((n,c)=>(n*31+c.charCodeAt(0))>>>0,0)%3;s.pattern=i>0&&result[i-1].group===s.group;});
  return result;
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
const model = {normalize,segments,rates,Rolling,bytes,exact,numeric,ownerLabel,graphIdentity};
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
function highlightOwner(key,on) {
  document.querySelectorAll('[data-owner]').forEach(el => {
    el.classList.toggle('active',on && el.dataset.owner === key);
    el.classList.toggle('dim',on && el.dataset.owner !== key);
  });
}
function ownerPaint(s,id,variant) {
  const color = `var(--mem-group-${groupIndex(s.group)})`;
  // Related shades keep owner identity stable. Texture is reserved for adjacent
  // siblings whose shades alone cannot reliably separate a nominal owner.
  const shade = variant % 3 === 1 ? '#fff' : '#000';
  const opacity = variant % 3 === 0 ? 0 : .10;
  return `<pattern id="${id}" width="10" height="10" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="${color}"/><rect width="10" height="10" fill="${shade}" opacity="${opacity}"/>${s.pattern ? '<path d="M-2 2L2-2M0 10L10 0M8 12L12 8" stroke="var(--panel)" stroke-opacity=".24" fill="none"/>' : ''}</pattern>`;
}
function reservationText(domain,s) {
  return `${domain.id} / ${s.name}\n${exact(s.bytes)} (${bytes(s.bytes)})\n${s.members.map(i=>`${i.name}: ${exact(i.bytes)} [${i.category}]`).join('\n')}`;
}
let selectedDomain = null, reservationData = null;
function renderDetail() {
  const d = reservationData.domains.find(d=>d.id === selectedDomain);
  if (!d) return;
  const parts = segments(d, false).filter(s=>!['weights','kv'].includes(s.group)).sort((a,b)=>b.bytes-a.bytes);
  const total = parts.reduce((n,s)=>n+s.bytes,0);
  $('detail-summary').textContent = d.reason ? `Unavailable: ${d.reason}` : `${d.id}: ${bytes(total)} outside target weights / KV. Each mini-bar is scaled to this subtotal, not device capacity. Drafter owners stay visible.`;
  $('owner-detail').innerHTML = parts.map((s,i)=>`<button class="owner-detail" data-owner="${esc(s.key)}" type="button"><span>${esc(ownerLabel(s.name))}</span><strong>${exact(s.bytes)}</strong><svg viewBox="0 0 600 12" preserveAspectRatio="none" aria-hidden="true"><defs>${ownerPaint(s,`detail${i}`,s.variant)}</defs><rect width="600" height="12" fill="var(--well)"/><rect width="${total?s.bytes/total*600:0}" height="12" fill="url(#detail${i})"/></svg></button>`).join('') || '<p class="hint">No non-weight/non-KV owners published.</p>';
  $('owner-detail').querySelectorAll('button').forEach((el,i)=>inspect(el,reservationText(d,parts[i]),on=>highlightOwner(parts[i].key,on)));
}
function renderReservations(data) {
  reservationData = data;
  if (!data.domains.some(d=>d.id === selectedDomain)) selectedDomain = data.domains.find(d=>d.id.startsWith('rtx'))?.id || data.domains[0]?.id;
  const owners = new Map(), rows = [];
  $('reservations').innerHTML = data.domains.map((d,index) => {
    const parts = segments(d), total = d.items.reduce((s,i)=>s+i.bytes,0), scale = Math.max(d.cap || 0,total,1);
    let y = 428;
    const defs = parts.map((s,n)=>ownerPaint(s,`p${index}-${n}`,s.variant)).join('');
    const marks = parts.map((s,n) => {
      const h = s.bytes/scale*410; y -= h;
      if (!owners.has(s.key)) owners.set(s.key,{...s,texts:[]});
      owners.get(s.key).texts.push(reservationText(d,s));
      return `<rect class="segment" id="s${index}-${n}" data-owner="${esc(s.key)}" x="24" y="${y}" width="76" height="${h}" fill="url(#p${index}-${n})" tabindex="0" role="img" aria-label="${esc(reservationText(d,s))}"/>`;
    }).join('');
    rows.push(...d.items.map(i=>`<tr><td>${esc(d.id)} / ${esc(i.name)}</td><td>${esc(i.category)}</td><td>${exact(i.bytes)}</td></tr>`));
    return `<article class="domain"><button class="domain-select" data-domain="${esc(d.id)}" type="button" aria-label="Inspect ${esc(d.id)} non-weight and non-KV owners">${esc(d.id)}</button><span class="capacity">${bytes(d.cap)} capacity</span><svg viewBox="0 0 125 455" aria-label="${esc(d.id)} reservations"><defs>${defs}</defs><rect x="24" y="18" width="76" height="410" fill="var(--well)"/>${marks}<text x="62" y="450" text-anchor="middle" fill="currentColor" font-size="9">${esc(d.reason?'unavailable':bytes(total)+' planned')}</text></svg><p class="hint">${esc(d.reason ? `Unavailable: ${d.reason}` : d.source)}</p>${numeric(d.cap)&&total>d.cap?'<p>Owners exceed capacity</p>':''}</article>`;
  }).join('');
  $('owners').innerHTML = rows.join('');
  $('legend').innerHTML = GROUPS.map(group=>{
    const entries=[...owners.values()].filter(s=>s.group===group);
    return entries.length?`<div class="legend-group"><h3>${esc(group)}</h3>${entries.map(s=>{
      const id=`l${[...owners.keys()].indexOf(s.key)}`;
      return `<button type="button" data-owner="${esc(s.key)}"><svg class="swatch" viewBox="0 0 20 20" aria-hidden="true"><defs>${ownerPaint(s,id,s.variant)}</defs><rect width="20" height="20" fill="url(#${id})"/></svg>${esc(ownerLabel(s.name))}</button>`;
    }).join('')}</div>`:'';
  }).join('');
  for (const [index,d] of data.domains.entries()) for (const [n,s] of segments(d).entries()) {
    inspect($(`s${index}-${n}`),reservationText(d,s),on=>highlightOwner(s.key,on));
    $(`s${index}-${n}`).addEventListener('click',()=>select(d.id));
  }
  $('legend').querySelectorAll('button').forEach(el=>inspect(el,owners.get(el.dataset.owner).texts.join('\n\n'),on=>highlightOwner(el.dataset.owner,on)));
  $('detail-domain').innerHTML=data.domains.map(d=>`<option value="${esc(d.id)}">${esc(d.id)}</option>`).join('');
  function select(id) { selectedDomain=id;$('detail-domain').value=id;renderDetail(); }
  $('reservations').querySelectorAll('.domain-select').forEach(el=>el.addEventListener('click',()=>select(el.dataset.domain)));
  $('detail-domain').onchange=()=>select($('detail-domain').value);
  select(selectedDomain);
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
    const graph = !spark && numeric(physical) ? data.graphs.find(g=>g.device === physical) : null;
    const kvPages = p.pages_used_by_gpu?.[Number(d.id.replace('rtx',''))];
    groups.push({id:d.id,spark,note:spark?'Launch-relative diagnostic only (#1568); ready values are not live.':`Reconciliation stage: ${row?.sample_stage || 'unavailable'}. Stage values are not live.`,metrics:[
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
  $('live').innerHTML = groups.map(g=>`<article class="card"><h3>${esc(g.id)}</h3><p class="hint">${esc(g.note)}</p>${g.metrics.filter(m=>!g.spark||m.value!==null).map(m=>`<div class="metric"><div class="metric-head"><span>${esc(m.label)}</span><strong>${esc(formatMetric(m,m.value))}${numeric(m.limit)?` / ${esc(formatMetric(m,m.limit))}`:''}</strong></div>${m.value === null?`<p class="hint">Unavailable: ${esc(m.reason)}</p>`:sparkline(m)}</div>`).join('')}${g.spark&&g.metrics.some(m=>m.value===null)?'<p class="hint">Unavailable: post-request telemetry, #1631.</p>':''}</article>`).join('');
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
  const groups = new Map();
  for (const k of rows.slice(0,10)) {
    const geometry = graphIdentity(k).geometry;
    if (!groups.has(geometry)) groups.set(geometry,[]);
    groups.get(geometry).push(k);
  }
  return `<div class="table-scroll"><table><thead><tr><th>Key</th><th>Hits</th><th>Churn (capture / evict / recapture)</th><th>Cost</th><th>Heat</th></tr></thead><tbody>${[...groups].map(([geometry, keys])=>`<tr><th colspan="5" style="text-align:left">${esc(geometry)}</th></tr>${keys.map(k=>{
    const counts = arr(k.hit_histogram?.counts), maximum = Math.max(1,...counts), identity=graphIdentity(k);
    return `<tr><td class="key" tabindex="0" data-key-identity="${esc(identity.full)}">${esc(identity.label)}<br><span class="hint">${k.pinned?'pinned':'not pinned'}</span></td><td>${number(k.hits)}</td><td>${number(k.captures)} / ${number(k.evictions)} / ${number(k.recaptures)}</td><td>${bytes(k.permanent_bytes)}<br><span class="hint">capture ${bytes(k.capture_bytes)}; ${number(k.nodes)} nodes</span></td><td><div class="heat" role="img" aria-label="Hits by interval">${counts.map((n,i)=>`<i tabindex="0" title="${esc(`${number(n)} hits; ${(k.hit_histogram.interval_ms || 0)/1000}s interval ${i+1}`)}" style="height:${Math.max(1,n/maximum*30)}px"></i>`).join('')}</div><span class="hint">${counts.length?`${counts.length} x ${number(k.hit_histogram.interval_ms/1000)}s`:'unavailable histogram'}</span></td></tr>`;
  }).join('')}`).join('')}</tbody></table></div>`;
}
function renderGraphs(data,time) {
  if (!data.graphs.length) { $('graphs').innerHTML='<p>Unavailable: graph-cache controller not published by this family/build.</p>';return; }
  $('graphs').innerHTML=data.graphs.map(g=>{
    const before=previous?.graphs.find(p=>p.device===g.device), rate=rates(g.cycles,before?.cycles,(time-previousTime)/1000);
    return `<article class="card"><h3>CUDA device ${esc(g.device)}</h3><p class="hint">Reserve ${bytes(g.reserve)} / arena ${bytes(g.charged)} / pinned ${bytes(g.pinned)} / pending ${bytes(g.pending_capture)}<br>${g.basis?.kind==='bytes'?'Estimator byte pool':'Node pool'} ${g.basis?.kind==='bytes'?bytes(g.pools?.live_nodes):number(g.pools?.live_nodes)} live / ${g.basis?.kind==='bytes'?bytes(g.pools?.node_high_water):number(g.pools?.node_high_water)} high-water. Executable pool ${number(g.pools?.live_execs)} live / ${number(g.pools?.exec_high_water)} high-water. Basis: ${esc(JSON.stringify(g.basis ?? 'unavailable'))}${g.basis?.kind==='bytes'?' (pool units are byte estimates, not CUDA node counts)':''}</p><div class="table-scroll"><table><caption>Per-second changes; unavailable until a second sample or after a counter reset.</caption><thead><tr><th>Cycle</th><th>Hits/s</th><th>Misses/s</th><th>Captures/s</th><th>Eager/s</th><th>Evictions/s</th></tr></thead><tbody>${Object.entries(rate).map(([cycle,c])=>`<tr><td>${esc(cycle)}</td>${['hits','misses','captures','eager_runs','evictions'].map(key=>`<td title="Lifetime: ${number(g.cycles[cycle][key])}">${number(c[key])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><details open><summary>Top 10 keys by hits</summary>${graphKeys(g.top_hits)}</details><details open><summary>Top 10 keys by churn</summary>${graphKeys(g.top_churn)}</details></article>`;
  }).join('');
  $('graphs').querySelectorAll('.heat i').forEach(el=>inspect(el,el.title));
  $('graphs').querySelectorAll('[data-key-identity]').forEach(el=>inspect(el,el.dataset.keyIdentity));
}
function render(payload) {
  const time=Date.now(), data=normalize(payload);
  if(payload.source?.includes('illustrative')) $('fixture-note').textContent='Illustrative accepted-plan / pin-stat extension; underlying GLM stats are a real final-pair sample.';
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
