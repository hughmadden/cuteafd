/* Fabric during a run: one small stacked-bar chart per used host/port.
 * Each bar is one observation (1 Hz while a run is live; one panel phase in a
 * saved report) in the busier direction, stacked engine / SparkNest / foreign /
 * other (headers, ACKs, skew). Idle and down ports collapse into one grey line;
 * exact values on hover; the per-port table sits behind the expander. */
(() => {
'use strict';
const NS = 'http://www.w3.org/2000/svg';
const SERIES = [
  { key: 'engine', name: 'engine', color: 'var(--fabric-engine)' },
  { key: 'sparknest', name: 'SparkNest', color: 'var(--fabric-storage)' },
  { key: 'foreign', name: 'foreign', color: 'var(--fabric-foreign)' },
  { key: 'other', name: 'other', color: 'var(--fabric-other)' },
];
const KEEP = 60;                       // live observations kept per port
const histories = new Map();           // port key -> [point]
let run = null, sequence = 0, names = new Map(), coordinatorName = 'coordinator', live = false;
const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const gb = (v) => !Number.isFinite(v) ? '–' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 0.1 ? v.toFixed(2) : v > 0 ? v.toFixed(3) : '0';
const address = (host) => host.replace(/^\[([^\]]+)\](?::\d+)?$/, '$1').replace(/:\d+$/, '');
function hostName(port) {
  if (port.hostname) return port.hostname;
  if (port.host === 'coordinator') return coordinatorName;
  return names.get(port.host) || names.get(address(port.host)) || 'Spark';
}
function mark(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.append(node);
  return node;
}

// One observation of one port, in its busier direction.
function point(port, label) {
  const s = port.seconds > 0 ? port.seconds : null;
  const rate = (bytes) => Number.isFinite(bytes) && s ? bytes * 8 / s / 1e9 : null;
  const tx = rate(port.sent_bytes) ?? port.xmit_gbps, rx = rate(port.received_bytes) ?? port.rcv_gbps;
  const sent = !(Number.isFinite(rx) && (!Number.isFinite(tx) || rx > tx));
  const pick = (a, b) => rate(sent ? port[a] : port[b]);
  const nic = sent ? tx : rx;
  const engine = pick('engine_sent_bytes', 'engine_received_bytes');
  const sparknest = pick('sparknest_sent_bytes', 'sparknest_received_bytes');
  const foreign = pick('foreign_sent_bytes', 'foreign_received_bytes');
  const known = [engine, sparknest, foreign].filter(Number.isFinite).reduce((a, b) => a + b, 0);
  const other = Number.isFinite(nic) ? Math.max(0, nic - known) : null;
  const h = port.health || {};
  return { label, port, nic: Number.isFinite(nic) ? nic : null, direction: sent ? 'transmit' : 'receive',
    engine, sparknest, foreign, other, loss: h.loss_events_per_gb, limit: limitOf(h),
    verdict: h.verdict || 'unknown', seconds: port.seconds };
}
const limitOf = (h) => h.loss_warning_threshold_per_gb > 0 ? h.loss_warning_threshold_per_gb : 100;

// What one port's history says, in a few words; problems only.
function judge(history) {
  const last = history.at(-1).port;
  const active = /ACTIVE/.test(last.state || '');
  const peak = Math.max(0, ...history.map((p) => p.nic || 0));
  const engine = history.some((p) => (p.engine || 0) > 0);
  const losses = history.map((p) => p.loss).filter(Number.isFinite);
  const worst = losses.length ? Math.max(...losses) : null;
  const limit = history.at(-1).limit;
  let issue = null;
  if (engine && !active) issue = 'down';
  else if (worst != null && worst > limit) issue = `loss ${worst.toFixed(0)}/GB`;
  else if (history.some((p) => p.verdict === 'warn')) issue = 'warning';
  const used = peak > 0 || engine;
  return { active, peak, engine, worst, limit, issue, used };
}

function chart(key, history, j) {
  const card = document.createElement('figure'); card.className = 'fx' + (j.issue ? ' issue' : '');
  const last = history.at(-1).port;
  const link = last.link_gbps;
  const port = `${last.device}/${last.port}`;
  const verdict = j.issue ? `<b class="bad">${esc(j.issue)}</b>` : j.worst != null ? `loss ${gb(j.worst)}/GB` : 'OK';
  card.innerHTML = `<figcaption><span class="host">${esc(hostName(last))}</span><span class="peak">peak ${gb(j.peak)} Gb/s</span>`
    + `<span class="sub" title="${esc(port)}">${esc(port.replace(/^roce/, ''))} · ${verdict}</span></figcaption>`;
  const W = 200, H = 96, top = 4, base = H - 2;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${hostName(last)} ${port}: stacked engine, SparkNest, foreign and other Gb/s per observation; peak ${gb(j.peak)} Gb/s${link ? `, link ${gb(link)} Gb/s` : ''}`);
  card.append(svg);
  const ceiling = Math.max(j.peak * 1.15, link && j.peak >= link * 0.25 ? link * 1.04 : 0, 1e-6);
  const y = (v) => base - Math.min(1, v / ceiling) * (base - top);
  mark('line', { x1: 0, x2: W, y1: base + 0.5, y2: base + 0.5, stroke: 'var(--line-2)', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke' }, svg);
  const slots = Math.max(history.length, history === histories.get(key) && live ? Math.min(KEEP, 30) : history.length);
  const step = W / slots, bar = Math.min(22, Math.max(1, step * 0.78)), gap = 0.6;
  history.forEach((p, i) => {
    const x = (slots - history.length + i) * step + (step - bar) / 2;
    let acc = 0;
    const g = mark('g', {}, svg);
    for (const s of SERIES) {
      const v = p[s.key];
      if (!(v > 0)) continue;
      const y0 = y(acc), y1 = y(acc + v);
      const h = Math.max(0, y0 - y1 - (acc > 0 ? gap : 0));
      if (h > 0) mark('rect', { x, y: y1, width: bar, height: h, fill: s.color, rx: Math.min(1.5, bar / 3) }, g);
      acc += v;
    }
    if (p.nic == null) mark('rect', { x, y: base - 3, width: bar, height: 3, fill: 'var(--line-2)' }, g);
    const hit = mark('rect', { x: x - (step - bar) / 2, y: 0, width: step, height: H, fill: 'transparent' }, g);
    hit.dataset.i = i;
  });
  if (link && link <= ceiling) {
    const ly = y(link);
    mark('line', { x1: 0, x2: W, y1: ly, y2: ly, stroke: 'var(--ink-2)', 'stroke-width': 1, 'stroke-dasharray': '3 3', 'vector-effect': 'non-scaling-stroke', opacity: 0.7 }, svg);
  }
  const foot = document.createElement('div'); foot.className = 'fx-foot';
  foot.innerHTML = `<span>${live ? `last ${history.length} s` : `${history.length} phase${history.length === 1 ? '' : 's'}`}</span>`
    + `<span>${link ? `${link <= ceiling ? '- - ' : ''}link ${gb(link)} Gb/s` : ''}</span>`;
  card.append(foot);
  const tip = document.createElement('div'); tip.className = 'fx-tip'; tip.hidden = true; card.append(tip);
  svg.addEventListener('pointermove', (event) => {
    const bounds = svg.getBoundingClientRect();
    const i = Math.floor((event.clientX - bounds.left) / bounds.width * slots) - (slots - history.length);
    const p = history[i];
    if (!p) { tip.hidden = true; return; }
    tip.innerHTML = `<b>${esc(p.label)}</b> · ${esc(p.direction)}${Number.isFinite(p.seconds) ? ` · ${p.seconds.toFixed(1)} s` : ''}`
      + SERIES.map((s) => `<div><i style="background:${s.color}"></i>${s.name}<span>${gb(p[s.key])} Gb/s</span></div>`).join('')
      + `<div class="total">NIC<span>${gb(p.nic)} Gb/s</span></div>`
      + (Number.isFinite(p.loss) ? `<div>loss<span>${gb(p.loss)}/GB</span></div>` : '');
    tip.hidden = false;
    const left = Math.min(Math.max(0, event.clientX - bounds.left + 10), card.clientWidth - tip.offsetWidth);
    tip.style.left = `${Math.max(0, left)}px`;
  });
  svg.addEventListener('pointerleave', () => { tip.hidden = true; });
  return card;
}

function table(rows) {
  const head = ['host', 'port', 'state', 'link', 'peak', 'engine', 'SparkNest', 'foreign', 'other', 'loss / GB', ''];
  const mean = (h, k) => { const v = h.map((p) => p[k]).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  return `<div class="fx-table"><table><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>`
    + rows.map(([, h, j]) => {
      const p = h.at(-1).port;
      return `<tr class="${j.issue ? 'issue' : j.used ? '' : 'spare'}"><td>${esc(hostName(p))}</td><td>${esc(`${p.device}/${p.port}`)}</td>`
        + `<td>${j.active ? 'up' : esc((p.state || 'unknown').replace(/^\d+:\s*/, '').toLowerCase())}</td><td>${gb(p.link_gbps)}</td><td>${gb(j.peak)}</td>`
        + ['engine', 'sparknest', 'foreign', 'other'].map((k) => `<td>${gb(mean(h, k))}</td>`).join('')
        + `<td>${j.worst == null ? '–' : gb(j.worst)}</td><td>${j.issue ? `<b>${esc(j.issue)}</b>` : j.used ? 'ok' : 'idle'}</td></tr>`;
    }).join('') + `</tbody></table><p class="note">Gb/s in the busier direction, mean over ${live ? 'live observations' : 'the run\'s phases'}; peak is the largest NIC rate. Loss signals per GB moved are overlapping counters, not lost bytes; warning above ${rows[0]?.[2].limit ?? 100}/GB. Other is NIC traffic not attributed to the engine, SparkNest or a foreign source (headers, ACKs, retries, timing skew).</p></div>`;
}

function render() {
  const root = document.getElementById('fabric'); if (!root) return;
  root.hidden = histories.size === 0;
  const plots = root.querySelector('.fabric-plots'); plots.replaceChildren();
  const rank = (p) => p.host === 'coordinator' ? -1 : [...names.keys()].indexOf(address(p.host));
  const rows = [...histories].map(([key, h]) => [key, h, judge(h)])
    .sort(([, a], [, b]) => rank(a.at(-1).port) - rank(b.at(-1).port)
      || `${a.at(-1).port.device}`.localeCompare(`${b.at(-1).port.device}`, undefined, { numeric: true }));
  const shown = rows.filter(([, , j]) => j.used || j.issue);
  const quiet = rows.filter(([, , j]) => !j.used && !j.issue);
  const legend = document.createElement('div'); legend.className = 'fx-legend';
  legend.innerHTML = SERIES.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')
    + `<span class="ref">- - link</span><span class="unit">Gb/s, busier direction, ${live ? '1 s per bar' : 'one bar per phase'}</span>`;
  plots.append(legend);
  const grid = document.createElement('div'); grid.className = 'fx-grid';
  for (const [key, h, j] of shown) grid.append(chart(key, h, j));
  plots.append(grid);
  if (quiet.length) {
    const counts = new Map();
    for (const [, h] of quiet) { const n = hostName(h.at(-1).port); counts.set(n, (counts.get(n) || 0) + 1); }
    const line = document.createElement('p'); line.className = 'fx-quiet';
    line.textContent = `Idle or spare (normal): ${[...counts].map(([n, c]) => c > 1 ? `${n} ×${c}` : n).join(' · ')}`;
    plots.append(line);
  }
  const details = document.createElement('details'); details.className = 'fx-details';
  details.innerHTML = `<summary>Per-port table</summary>${table(rows)}`;
  plots.append(details);
}

function remember(event) {
  names = new Map((event?.server?.hardware?.sparks || []).map((s) => [s.address, s.name || `Spark ${s.rank + 1}`]));
  coordinatorName = event?.server?.hardware?.host || 'coordinator';
}
window.CuteFabric = {
  interval(event) {
    if (event.run !== run) { run = event.run; sequence = 0; histories.clear(); }
    live = true; sequence++;
    for (const port of event.interval?.ports || []) {
      const key = `${port.host} ${port.device}/${port.port}`;
      const history = histories.get(key) || [];
      history.push(point(port, `+${sequence} s`));
      if (history.length > KEEP) history.shift();
      histories.set(key, history);
    }
    render();
  },
  report(report) {
    if (!report) return;
    remember(report);
    if (report.status === 'running' && report.id === run && live) return;
    histories.clear(); run = report.id; live = false;
    for (const phase of (report.fabric || []).filter((p) => p.timed !== false)) {
      for (const port of phase.ports || []) {
        const key = `${port.host} ${port.device}/${port.port}`;
        const history = histories.get(key) || []; history.push(point(port, phase.name)); histories.set(key, history);
      }
    }
    render();
  },
  name: hostName,
  observed: () => histories.size,
  // The problems in the shown history, one short phrase per port (empty when clean).
  problems() {
    return [...histories.values()].map((h) => [h, judge(h)]).filter(([, j]) => j.issue).map(([h, j]) => {
      const name = hostName(h.at(-1).port);
      return j.issue === 'down' ? `${name}: port down while used` : j.issue.startsWith('loss') ? `high loss on ${name} (${gb(j.worst)}/GB)` : `${name}: ${j.issue}`;
    });
  },
};
})();
