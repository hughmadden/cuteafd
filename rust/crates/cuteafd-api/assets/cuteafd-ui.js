// CuteAFD UI: the page shell, number formatting and SVG chart primitives shared
// by the engine's built-in pages (live console at /, benchmarks at /bench).
// Served at /assets/cuteafd-ui.js next to /assets/cuteafd-ui.css; compiled into
// the binary, no external resources. Exposes `window.CuteUI`.
(() => {
'use strict';
const root = getComputedStyle(document.documentElement);
const token = (name) => root.getPropertyValue('--' + name).trim();
// Palette tokens by name (`accepted`, `target`, `spark`, ...); unknown names pass through as CSS colors.
const color = (name) => (name && /^[a-z][a-z0-9-]*$/.test(name) && token(name)) || name || token('ink-2');

// ---------------------------------------------------------------- formatting
const nf0 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const fmt = {
  n0: (v) => (Number.isFinite(v) ? nf0.format(v) : '–'),
  n1: (v) => (Number.isFinite(v) ? nf1.format(v) : '–'),
  compact(v) {
    if (!Number.isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
    return nf0.format(v);
  },
  bytes(v) {
    if (!Number.isFinite(v)) return '–';
    const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (i ? v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2) : v) + ' ' + u[i];
  },
  pct: (a, b) => (b > 0 ? (100 * a / b).toFixed(0) + '%' : '–'),
  // Microseconds as HTML with a small unit.
  us(v) {
    if (!Number.isFinite(v)) return '–';
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)} <small>s</small>`;
    if (v >= 1000) return `${(v / 1000).toFixed(v >= 1e4 ? 1 : 2)} <small>ms</small>`;
    return `${v.toFixed(0)} <small>µs</small>`;
  },
  ms: (v) => (!Number.isFinite(v) ? '–' : v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(0) + ' ms'),
  clock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  },
  esc: (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])),
};

// ---------------------------------------------------------------- page shell
// Fills `el` (a <header>) with the brand, a slot for page status, the header
// facts and the page navigation. Returns {status, meta, extra} slot elements.
// opts: {page: 'console' | 'bench', subtitle}.
const PAGES = [
  { id: 'console', href: '/', label: 'LIVE CONSOLE' },
  { id: 'bench', href: '/bench', label: 'BENCHMARK', primary: true },
];
function header(el, opts = {}) {
  const page = opts.page || 'console';
  el.innerHTML = `<div class="brand"><b>CUTEAFD</b><span>${fmt.esc(opts.subtitle || 'Engine console')}</span></div>
    <span class="slot-status"></span><div class="meta"></div><div class="spacer"></div><span class="slot-extra"></span>
    <nav class="nav" aria-label="Pages">${PAGES.map((p) => `<a href="${p.href}"${p.primary ? ' class="primary"' : ''}${p.id === page ? ' aria-current="page"' : ''}>${p.label}</a>`).join('')}</nav>`;
  return { status: el.querySelector('.slot-status'), meta: el.querySelector('.meta'), extra: el.querySelector('.slot-extra') };
}
// Header facts: [[key, valueHtml, title?]] -> `key <b>value</b>` spans.
function facts(el, items) {
  el.innerHTML = items.filter(([, v]) => v != null && v !== '').map(([k, v, title]) =>
    `<span${title ? ` title="${fmt.esc(title)}"` : ''}>${fmt.esc(k)} <b>${v}</b></span>`).join('');
}
// The running build from a `{release, commit, dirty}` object, as HTML.
function build(b) {
  if (!b) return 'unknown';
  const commit = b.commit ? fmt.esc(String(b.commit).slice(0, 12)) : 'unknown';
  const dirty = b.dirty ? '<span class="dirty">+dirty</span>' : '';
  return b.release ? `${fmt.esc(b.release)} · ${commit}${dirty}` : `dev · ${commit}${dirty}`;
}

// ---------------------------------------------------------------- SVG charts
let ids = 0;
const NS = 'http://www.w3.org/2000/svg';
function svgIn(el, cls, h) {
  const w = el.clientWidth || el.getBoundingClientRect().width;
  let svg = el.tagName === 'svg' ? el : el.querySelector(`svg.${cls}`);
  if (!svg) { svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', cls); el.appendChild(svg); }
  const height = h ?? (svg.clientHeight || 30);
  svg.setAttribute('viewBox', `0 0 ${Math.max(1, w)} ${height}`);
  svg.setAttribute('height', height);
  return { svg, w, h: height };
}
// A rate sparkline: newest value at the right edge, `points` slots wide, area
// fill fading to transparent, a dot on the newest value. `el` is an <svg> or
// its container. opts: {color, points = 60, height = 30, max}.
function sparkline(el, data, opts = {}) {
  const { svg, w, h } = svgIn(el, 'spark', opts.height ?? 30);
  if (!w) return;
  const c = color(opts.color);
  const finite = data.filter(Number.isFinite);
  if (finite.length < 2) { svg.innerHTML = ''; return; }
  const n = opts.points ?? 60, step = w / (n - 1);
  const max = (opts.max ?? Math.max(1e-9, ...finite)) * 1.1;
  const x = (i) => w - (data.length - 1 - i) * step, y = (v) => h - 1.5 - (v / max) * (h - 4);
  let line = '', first = null;
  data.forEach((v, i) => { if (!Number.isFinite(v)) return; line += `${first == null ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`; first ??= i; });
  const id = `cg${++ids}`, last = data[data.length - 1];
  svg.innerHTML = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${c}" stop-opacity=".25"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></linearGradient></defs>
    <path d="${line}L${w},${h}L${x(first).toFixed(1)},${h}Z" fill="url(#${id})"/>
    <path d="${line}" fill="none" stroke="${c}" stroke-width="1.5" stroke-linejoin="round" style="filter:drop-shadow(0 0 3px ${c})"/>
    ${Number.isFinite(last) ? `<circle cx="${w - 2}" cy="${y(last).toFixed(1)}" r="2.5" fill="${c}"/>` : ''}`;
}
// Vertical columns (a per-layer profile): one slot per value, a wide muted bar
// for `opts.median[i]` behind a narrow colored bar for the value; labels under
// slots `opts.label(i)` returns text for. opts: {colors: [i] -> color, median,
// height = 64, note}.
function columns(el, values, opts = {}) {
  const { svg, w, h } = svgIn(el, 'columns', opts.height ?? 64);
  if (!w) return;
  if (!values.length) {
    svg.innerHTML = `<text x="4" y="${h / 2}" fill="${token('muted')}" font-size="11" font-family="${fmt.esc(token('mono'))}">${fmt.esc(opts.empty || 'No data yet')}</text>`;
    return;
  }
  const med = opts.median || [];
  const max = Math.max(1, ...values.filter(Number.isFinite), ...med.filter(Number.isFinite));
  const bw = w / values.length, top = 12, base = h - 13;
  const muted = token('muted'), line2 = token('line-2'), mono = fmt.esc(token('mono'));
  let out = '';
  values.forEach((v, i) => {
    const x0 = i * bw, m = med[i];
    if (Number.isFinite(m)) { const mh = (base - top) * m / max; out += `<rect x="${(x0 + 1).toFixed(1)}" y="${(base - mh).toFixed(1)}" width="${Math.max(0, bw - 2).toFixed(1)}" height="${mh.toFixed(1)}" fill="${line2}"/>`; }
    if (Number.isFinite(v)) { const vh = Math.max(1, (base - top) * v / max); out += `<rect x="${(x0 + bw * .22).toFixed(1)}" y="${(base - vh).toFixed(1)}" width="${(bw * .56).toFixed(1)}" height="${vh.toFixed(1)}" fill="${color(opts.colors ? opts.colors(i) : 'target')}" opacity=".9"/>`; }
    const label = opts.label ? opts.label(i) : null;
    if (label != null) out += `<text x="${(x0 + bw / 2).toFixed(1)}" y="${h - 2}" fill="${muted}" font-size="9.5" text-anchor="middle" font-family="${mono}">${fmt.esc(label)}</text>`;
  });
  if (opts.note) out += `<text x="2" y="9" fill="${muted}" font-size="9.5" font-family="${mono}">${fmt.esc(opts.note(max))}</text>`;
  svg.innerHTML = out;
}
// Horizontal timeline rows: rows = [{label, segments: [{t0, t1, color, title}]}]
// over [opts.t0, opts.t1]. opts: {t0, t1, row = 18, gutter = 120}.
function timeline(el, rows, opts = {}) {
  const rowH = opts.row ?? 18, gutter = opts.gutter ?? 120;
  const { svg, w } = svgIn(el, 'timeline', rows.length * rowH + 4);
  if (!w) return;
  const t0 = opts.t0 ?? Math.min(...rows.flatMap((r) => r.segments.map((s) => s.t0))), t1 = opts.t1 ?? Math.max(...rows.flatMap((r) => r.segments.map((s) => s.t1)));
  const span = Math.max(1e-9, t1 - t0), X = (t) => gutter + (w - gutter - 2) * (t - t0) / span;
  const mono = fmt.esc(token('mono')), ink2 = token('ink-2');
  svg.innerHTML = rows.map((r, i) => {
    const y = 2 + i * rowH;
    return `<text x="0" y="${y + rowH / 2 + 3.5}" fill="${ink2}" font-size="11" font-family="${mono}">${fmt.esc(r.label)}</text>` +
      r.segments.map((s) => `<rect x="${X(s.t0).toFixed(1)}" y="${y + 2}" width="${Math.max(1, X(s.t1) - X(s.t0) - .5).toFixed(1)}" height="${rowH - 4}" rx="1.5" fill="${color(s.color)}">${s.title ? `<title>${fmt.esc(s.title)}</title>` : ''}</rect>`).join('');
  }).join('');
}
// A horizontal stacked meter as HTML: segments = [{value, color}] of `total`.
function meter(segments, total) {
  return `<div class="track">${segments.map((s) => `<span style="background:${color(s.color)};width:${(100 * Math.max(0, s.value) / Math.max(1e-9, total)).toFixed(2)}%"></span>`).join('')}</div>`;
}

window.CuteUI = { token, color, fmt, header, facts, build, svg: { sparkline, columns, timeline, meter } };
})();
