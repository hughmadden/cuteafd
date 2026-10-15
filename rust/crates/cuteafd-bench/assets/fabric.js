/* Live small multiples: one physical RDMA port per host, one full-duplex scale. */
(() => {
'use strict';
const NS = 'http://www.w3.org/2000/svg';
const series = [
  { name: 'NIC', key: 'nic', dash: '' },
  { name: 'Engine', key: 'engine', dash: '8 3' },
  { name: 'Foreign / retry residual', key: 'foreign', dash: '2 3' },
  { name: 'SparkNest', key: 'sparknest', dash: '10 3 2 3' },
];
const histories = new Map();
let run, sequence = 0;
let ranks = new Map();
const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const cnp = (c) => {
  const values = ['np_cnp_sent', 'rp_cnp_handled'].map(key => c?.[`hw_counters/${key}`] ?? c?.[key]).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
};
const fmt = (value) => Number.isFinite(value) ? value.toFixed(2) : 'unavailable';
function mark(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.append(node);
  return node;
}
function point(port, index) {
  const rate = (bytes) => Number.isFinite(bytes) && port.seconds > 0 ? bytes * 8 / port.seconds / 1e9 : null;
  const max = (a, b) => Number.isFinite(a) && Number.isFinite(b) ? Math.max(a, b) : null;
  return { index, port, nic: max(port.xmit_gbps, port.rcv_gbps),
    engine: max(rate(port.engine_sent_bytes), rate(port.engine_received_bytes)),
    sparknest: max(rate(port.sparknest_sent_bytes), rate(port.sparknest_received_bytes)),
    foreign: max(rate(port.foreign_sent_bytes), rate(port.foreign_received_bytes)) };
}
function rateHeaders(p) {
  return Number.isFinite(p.estimated_header_sent_bytes) && Number.isFinite(p.estimated_header_received_bytes) && p.seconds > 0
    ? Math.max(p.estimated_header_sent_bytes, p.estimated_header_received_bytes) * 8 / p.seconds / 1e9 : null;
}
function draw(container, key, history) {
  const last = history.at(-1), port = last.port;
  const section = document.createElement('section'); section.className = 'fabric-port';
  section.innerHTML = `<h4>${esc(key)} <small>${esc(port.state || '')} · ${fmt(port.link_gbps)} Gb/s link</small></h4>`;
  const legend = document.createElement('div'); legend.className = 'fabric-legend';
  series.forEach((s, i) => {
    const label = document.createElement('span');
    label.innerHTML = `<svg width="30" height="12" aria-hidden="true"><path d="M0 6 H30" stroke="var(--fabric-${i + 1})" stroke-width="2" stroke-dasharray="${s.dash}"/></svg>${s.name} ${fmt(last[s.key])} Gb/s`;
    legend.append(label);
  });
  const utilization = document.createElement('span');
  utilization.textContent = `Utilization ${fmt(Number.isFinite(port.utilization) ? port.utilization * 100 : null)}%`;
  legend.append(utilization);
  section.append(legend);
  const plot = document.createElement('div'); plot.className = 'fabric-plot';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 620 180'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${key}: NIC, engine, SparkNest and foreign residual traffic, in Gb/s. Autoscaled to observed traffic; link capacity is labeled separately.`);
  plot.append(svg);
  const observed = Math.max(...history.flatMap(p => series.map(s => p[s.key] || 0)));
  const ceiling = observed > 0 ? observed * 1.15 : 1;
  const x = (i) => 46 + i / Math.max(history.length - 1, 1) * 560;
  const y = (v) => 148 - v / ceiling * 128;
  [0, ceiling / 2, ceiling].forEach((v) => {
    mark('path', { d: `M46 ${y(v)} H606`, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
    const label = mark('text', { x: 40, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 10 }, svg);
    label.textContent = v >= 10 ? v.toFixed(0) : v.toFixed(v > 0 && v < 0.1 ? 3 : 2);
  });
  if (port.link_gbps) {
    const label = mark('text', { x: 604, y: 10, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 10 }, svg);
    label.textContent = `Link reference ${fmt(port.link_gbps)} Gb/s`;
  }
  series.forEach((s, slot) => {
    let path = '', previous = false;
    history.forEach((p, i) => {
      if (!Number.isFinite(p[s.key])) { previous = false; return; }
      path += `${previous ? 'L' : 'M'}${x(i)} ${y(p[s.key])} `; previous = true;
    });
    if (path) mark('path', { d: path, fill: 'none', stroke: `var(--fabric-${slot + 1})`, 'stroke-width': 2, 'stroke-dasharray': s.dash }, svg);
    if (Number.isFinite(last[s.key])) mark('circle', { cx: x(history.length - 1), cy: y(last[s.key]), r: 4,
      fill: `var(--fabric-${slot + 1})`, stroke: 'var(--panel)', 'stroke-width': 2 }, svg);
  });
  history.forEach((p, i) => {
    const c = p.port.counters || {};
    const wait = (c.port_xmit_wait || 0) > 0;
    const hasCnp = (cnp(c) || 0) > 0;
    if (wait || hasCnp) {
      const marker = mark('text', { x: x(i), y: 164, 'text-anchor': 'middle', fill: 'var(--ink-2)', 'font-size': 10 }, svg);
      marker.textContent = `${wait ? 'W' : ''}${hasCnp ? 'C' : ''}`;
      const title = mark('title', {}, marker); title.textContent = 'W: xmit_wait grew; C: CNP grew';
    }
  });
  const crosshair = mark('path', { stroke: 'var(--ink-2)', 'stroke-width': 1, visibility: 'hidden' }, svg);
  const tooltip = document.createElement('div'); tooltip.className = 'fabric-tooltip'; tooltip.hidden = true; plot.append(tooltip);
  const hit = mark('rect', { x: 46, y: 12, width: 560, height: 154, fill: 'transparent', tabindex: 0,
    'aria-label': 'Move pointer across observations; the data table contains all values.' }, svg);
  hit.addEventListener('pointermove', (event) => {
    const bounds = svg.getBoundingClientRect();
    const px = (event.clientX - bounds.left) / bounds.width * 620;
    const i = Math.max(0, Math.min(history.length - 1, Math.round((px - 46) / 560 * Math.max(history.length - 1, 1))));
    const p = history[i];
    crosshair.setAttribute('d', `M${x(i)} 12 V148`); crosshair.setAttribute('visibility', 'visible');
    tooltip.textContent = `Sample ${p.index}: NIC ${fmt(p.nic)}, engine ${fmt(p.engine)}, SparkNest ${fmt(p.sparknest)}, foreign ${fmt(p.foreign)} Gb/s; storage ${p.port.sparknest_status || "unknown"}; residual ${p.port.residual_kind || "unattributed"}; headers estimate ${fmt(rateHeaders(p.port))} Gb/s (engine: ceil(bytes/MTU)*62 lower bound; SparkNest: per-op packets*62); wait ${fmt(p.port.xmit_wait_ticks_per_s)} ticks/s`;
    tooltip.hidden = false;
  });
  hit.addEventListener('pointerleave', () => { tooltip.hidden = true; crosshair.setAttribute('visibility', 'hidden'); });
  section.append(plot);
  const details = document.createElement('details');
  details.innerHTML = `<summary>Data table · W: wait ticks · C: CNP · max(tx, rx), not duplex sum</summary><div class="fabric-table"><table><thead><tr><th>Sample</th><th>NIC Gb/s</th><th>Engine Gb/s</th><th>SparkNest Gb/s</th><th>Foreign Gb/s</th><th>Headers Gb/s (engine lower bound)</th><th>Storage status</th><th>Utilization</th><th>Wait ticks/s</th><th>Idle ticks/s</th><th>CNP</th></tr></thead><tbody>${history.map((p) => `<tr><td>${esc(p.index)}</td><td>${fmt(p.nic)}</td><td>${fmt(p.engine)}</td><td>${fmt(p.sparknest)}</td><td>${fmt(p.foreign)}</td><td>${fmt(rateHeaders(p.port))}</td><td>${esc(p.port.sparknest_status || "unknown")}</td><td>${fmt(Number.isFinite(p.port.utilization) ? p.port.utilization * 100 : null)}%</td><td>${fmt(p.port.xmit_wait_ticks_per_s)}</td><td>${fmt(p.port.idle_xmit_wait_ticks_per_s)}</td><td>${cnp(p.port.counters) ?? 'unavailable'}</td></tr>`).join('')}</tbody></table></div>`;
  section.append(details); container.append(section);
}
function render() {
  const root = document.getElementById('fabric'); if (!root) return;
  root.hidden = histories.size === 0;
  const plots = root.querySelector('.fabric-plots'); plots.replaceChildren();
  const hosts = new Map();
  for (const [key, history] of histories) {
    const host = history.at(-1).port.host;
    if (!hosts.has(host)) hosts.set(host, []);
    hosts.get(host).push([key, history]);
  }
  const address = host => host.replace(/^\[([^\]]+)\](?::\d+)?$/, '$1').replace(/:\d+$/, '');
  const rank = host => ranks.get(host) ?? ranks.get(address(host)) ?? Number.MAX_SAFE_INTEGER;
  const ordered = [...hosts].sort(([a], [b]) => a === 'coordinator' ? -1 : b === 'coordinator' ? 1
    : rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true }));
  for (const [host, ports] of ordered) {
    const group = document.createElement('section'); group.className = 'fabric-host';
    const heading = document.createElement('h4');
    heading.textContent = host === 'coordinator' ? 'Coordinator' : rank(host) !== Number.MAX_SAFE_INTEGER ? `Rank ${rank(host)} · ${host}` : host;
    group.append(heading);
    const grid = document.createElement('div'); grid.className = 'fabric-host-ports';
    let down = 0, idle = 0;
    ports.sort(([, a], [, b]) => a.at(-1).port.device.localeCompare(b.at(-1).port.device, undefined, { numeric: true })
      || a.at(-1).port.port - b.at(-1).port.port);
    for (const [key, history] of ports) {
      const state = history.at(-1).port.state || '';
      const changed = new Set(history.map(p => p.port.state).filter(Boolean)).size > 1;
      const traffic = history.some(p => series.some(s => Number.isFinite(p[s.key]) && p[s.key] > 0));
      if (!changed && !traffic && /\b(DOWN|ACTIVE)\b/.test(state)) {
        if (/\bDOWN\b/.test(state)) down++; else idle++;
        continue;
      }
      draw(grid, key, history);
    }
    if (down) { const line = document.createElement('p'); line.className = 'note fabric-down'; line.textContent = `${down} port${down === 1 ? '' : 's'} down`; group.append(line); }
    if (idle) { const line = document.createElement('p'); line.className = 'note fabric-idle'; line.textContent = `${idle} port${idle === 1 ? '' : 's'} idle`; group.append(line); }
    group.append(grid); plots.append(group);
  }
}
window.CuteFabric = {
  interval(event) {
    if (event.run !== run) { run = event.run; sequence = 0; histories.clear(); }
    sequence++;
    const seen = new Set();
    for (const port of event.interval?.ports || []) {
      const key = `${port.host} ${port.device}/${port.port}`;
      seen.add(key);
      const history = histories.get(key) || [];
      history.push(point(port, sequence));
      if (history.length > 60) history.shift();
      histories.set(key, history);
    }
    for (const [key, history] of histories) {
      if (!seen.has(key)) {
        history.push({ index: sequence, port: { ...history.at(-1).port, state: 'unavailable',
          utilization: null, counters: {}, xmit_wait_ticks_per_s: null, idle_xmit_wait_ticks_per_s: null },
          nic: null, engine: null, sparknest: null, foreign: null });
        if (history.length > 60) history.shift();
      }
    }
    render();
  },
  report(report) {
    if (!report) return;
    ranks = new Map((report.server?.hardware?.sparks || []).flatMap(s => [[s.address, s.rank], ...(s.name ? [[s.name, s.rank]] : [])]));
    const warnings = (report.fabric || []).flatMap((phase) => phase.assessments || []);
    const root = document.getElementById('fabric');
    if (root) root.querySelector('.fabric-assessments').textContent = [...new Set(warnings)].join(' · ');
    if (report.status === 'running' && report.id === run) return;
    histories.clear(); run = report.id;
    for (const phase of report.fabric || []) {
      for (const port of phase.ports || []) {
        const key = `${port.host} ${port.device}/${port.port}`;
        const history = histories.get(key) || []; history.push(point(port, phase.name)); histories.set(key, history);
      }
    }
    render();
  },
};
})();
