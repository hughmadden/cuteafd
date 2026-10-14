/* Live small multiples: one physical RDMA port per host, one full-duplex scale. */
(() => {
'use strict';
const NS = 'http://www.w3.org/2000/svg';
const series = [
  { name: 'NIC', key: 'nic', dash: '' },
  { name: 'Engine', key: 'engine', dash: '8 3' },
  { name: 'Foreign', key: 'foreign', dash: '2 3' },
];
const histories = new Map();
let run, sequence = 0;
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
    foreign: max(rate(port.foreign_sent_bytes), rate(port.foreign_received_bytes)) };
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
  section.append(legend);
  const plot = document.createElement('div'); plot.className = 'fabric-plot';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 620 180'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${key}: NIC, engine and foreign traffic, in Gb/s. Dashed rule is link capacity.`);
  plot.append(svg);
  const ceiling = Math.max(port.link_gbps || 0, ...history.map((p) => p.nic || 0), 1) * 1.06;
  const x = (i) => 46 + i / Math.max(history.length - 1, 1) * 560;
  const y = (v) => 148 - v / ceiling * 128;
  [0, ceiling / 2, ceiling].forEach((v) => {
    mark('path', { d: `M46 ${y(v)} H606`, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
    const label = mark('text', { x: 40, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 10 }, svg);
    label.textContent = v.toFixed(0);
  });
  if (port.link_gbps) {
    mark('path', { d: `M46 ${y(port.link_gbps)} H606`, stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '4 5' }, svg);
    const label = mark('text', { x: 604, y: y(port.link_gbps) - 4, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 10 }, svg);
    label.textContent = 'link';
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
    tooltip.textContent = `Sample ${p.index}: NIC ${fmt(p.nic)}, engine ${fmt(p.engine)}, foreign ${fmt(p.foreign)} Gb/s; wait ${fmt(p.port.xmit_wait_ticks_per_s)} ticks/s`;
    tooltip.hidden = false;
  });
  hit.addEventListener('pointerleave', () => { tooltip.hidden = true; crosshair.setAttribute('visibility', 'hidden'); });
  section.append(plot);
  const details = document.createElement('details');
  details.innerHTML = `<summary>Data table · W: wait ticks · C: CNP · max(tx, rx), not duplex sum</summary><div class="fabric-table"><table><thead><tr><th>Sample</th><th>NIC Gb/s</th><th>Engine Gb/s</th><th>Foreign Gb/s</th><th>Utilization</th><th>Wait ticks/s</th><th>Idle ticks/s</th><th>CNP</th></tr></thead><tbody>${history.map((p) => `<tr><td>${esc(p.index)}</td><td>${fmt(p.nic)}</td><td>${fmt(p.engine)}</td><td>${fmt(p.foreign)}</td><td>${fmt(Number.isFinite(p.port.utilization) ? p.port.utilization * 100 : null)}%</td><td>${fmt(p.port.xmit_wait_ticks_per_s)}</td><td>${fmt(p.port.idle_xmit_wait_ticks_per_s)}</td><td>${cnp(p.port.counters) ?? 'unavailable'}</td></tr>`).join('')}</tbody></table></div>`;
  section.append(details); container.append(section);
}
function render() {
  const root = document.getElementById('fabric'); if (!root) return;
  root.hidden = histories.size === 0;
  const plots = root.querySelector('.fabric-plots'); plots.replaceChildren();
  for (const [key, history] of histories) draw(plots, key, history);
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
          nic: null, engine: null, foreign: null });
        if (history.length > 60) history.shift();
      }
    }
    render();
  },
  report(report) {
    if (!report) return;
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
