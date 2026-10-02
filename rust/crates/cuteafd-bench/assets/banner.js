// BENCHMARKING banner for any page served by cuteafd (include with
// <script src="/bench/banner.js" defer></script>). While a benchmark holds the
// server it shows the panel, ETA and a link to /bench; other clients' requests
// get 503 + Retry-After until it finishes.
(() => {
  if (window.__cuteafdBenchBanner) return;
  window.__cuteafdBenchBanner = true;
  const bar = document.createElement('div');
  bar.setAttribute('role', 'status');
  bar.style.cssText = 'position:fixed;left:0;right:0;top:0;z-index:2147483000;display:none;align-items:center;gap:14px;' +
    'padding:7px 16px;font:600 12px/1.3 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;color:#05070b;' +
    'background:linear-gradient(90deg,#4cc3ff,#c08bff);box-shadow:0 4px 18px rgba(0,0,0,.45)';
  const text = document.createElement('span');
  const link = document.createElement('a');
  link.href = '/bench';
  link.textContent = 'OPEN BENCH →';
  link.style.cssText = 'margin-left:auto;color:#05070b;text-decoration:none;border:1px solid rgba(5,7,11,.4);' +
    'border-radius:6px;padding:2px 8px';
  bar.append(text, link);
  const fmt = s => s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, '0')}s` : `${Math.round(s)}s`;
  let shown = false;
  async function poll() {
    try {
      const status = await (await fetch('/v1/bench/status', {cache: 'no-store'})).json();
      const active = status.active;
      if (active) {
        if (!bar.isConnected) document.body.prepend(bar);
        text.textContent = `● BENCHMARKING · ${active.panel} · ${Math.round(100 * (active.fraction || 0))}% · ETA ${fmt(active.eta_s || 0)}` +
          ' · other requests get 503 until it finishes';
        if (status.quality_failed) bar.style.background = 'repeating-linear-gradient(45deg,#ffb020 0 12px,#ff3b4e 12px 24px)';
        bar.style.display = 'flex';
        shown = true;
      } else if (shown) {
        bar.style.display = 'none';
        shown = false;
      }
    } catch (_) { /* the server is restarting */ }
    setTimeout(poll, shown ? 2000 : 4000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', poll); else poll();
})();
