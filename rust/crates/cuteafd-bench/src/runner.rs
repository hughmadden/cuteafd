//! The benchmark runner inside the server: one run at a time on its own
//! thread, driving the server's own API over loopback. While a run is active
//! every other inference request is refused (`lockout`), progress and partial
//! results stream to `/v1/bench/events`, and the report lands in SQLite.
use crate::client::Client;
use crate::panels::{self, Ctx, Progress, Rates};
use crate::profiles::{self, Profile};
use crate::report::{
    now_rfc3339, Baseline, PanelResult, PanelStatus, PlannedPanel, Report, RunStatus, ServerInfo, SCHEMA,
};
use crate::store::{self, Store};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// What a client asks to run.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RunRequest {
    /// A built-in or saved profile name.
    #[serde(default)]
    pub profile: Option<String>,
    /// Explicit panels (a custom run); overrides the profile's list.
    #[serde(default)]
    pub panels: Option<Vec<String>>,
    /// `panel -> passes` overrides.
    #[serde(default)]
    pub passes: HashMap<String, u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("a benchmark is already running ({0})")]
    Busy(String),
    #[error("unknown profile {0}")]
    UnknownProfile(String),
    #[error("the server is not ready")]
    NotReady,
}

/// The run holding the lock.
#[derive(Debug, Clone)]
pub struct ActiveRun {
    pub id: String,
    pub token: String,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    pub eta_s: f64,
    pub panel: String,
    pub fraction: f64,
}

pub struct Bench {
    store: Mutex<Store>,
    active: Mutex<Option<ActiveRun>>,
    live: Mutex<HashMap<String, Arc<Mutex<Report>>>>,
    baselines: Mutex<HashMap<String, Baseline>>,
    info: Mutex<Option<ServerInfo>>,
    events: broadcast::Sender<Arc<str>>,
    /// `CUTEAFD_API_KEY`: when set, bench controls from outside the local
    /// network need it as a bearer token.
    pub api_key: Option<String>,
}

impl Bench {
    pub fn new(store: Store) -> Arc<Self> {
        let (events, _) = broadcast::channel(512);
        Arc::new(Self {
            store: Mutex::new(store),
            active: Mutex::new(None),
            live: Mutex::new(HashMap::new()),
            baselines: Mutex::new(HashMap::new()),
            info: Mutex::new(None),
            events,
            api_key: std::env::var("CUTEAFD_API_KEY").ok().filter(|k| !k.is_empty()),
        })
    }

    /// The process-wide instance (SQLite under `store::default_dir()`, in memory if unwritable).
    pub fn global() -> Arc<Self> {
        static BENCH: OnceLock<Arc<Bench>> = OnceLock::new();
        BENCH.get_or_init(|| {
            let dir = store::default_dir();
            let store = Store::open(&dir).unwrap_or_else(|error| {
                tracing::warn!(%error, dir = %dir.display(), "benchmark history is not persisted");
                Store::memory().expect("in-memory SQLite")
            });
            Self::new(store)
        }).clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<str>> {
        self.events.subscribe()
    }

    fn emit(&self, event: Value) {
        let _ = self.events.send(Arc::from(event.to_string()));
    }

    pub fn active(&self) -> Option<ActiveRun> {
        self.active.lock().ok()?.clone()
    }

    /// For the lockout: Some(retry-after seconds) when a run holds the server
    /// and `token` is not its token.
    pub fn locked(&self, token: Option<&str>) -> Option<u64> {
        let active = self.active()?;
        if token == Some(active.token.as_str()) {
            return None;
        }
        Some((active.eta_s.ceil() as u64).clamp(5, 3600))
    }

    pub fn store<T>(&self, f: impl FnOnce(&Store) -> anyhow::Result<T>) -> anyhow::Result<T> {
        let store = self.store.lock().map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        f(&store)
    }

    pub fn profiles(&self) -> Vec<Profile> {
        let mut all = profiles::builtin();
        all.extend(self.store(|s| s.profiles()).unwrap_or_default());
        all
    }

    /// A live or stored report.
    pub fn report(&self, id: &str) -> Option<Report> {
        if let Some(live) = self.live.lock().ok()?.get(id) {
            return live.lock().ok().map(|r| r.clone());
        }
        self.store(|s| s.load(id)).ok().flatten()
    }

    /// The newest report (live first).
    pub fn latest(&self) -> Option<Report> {
        if let Some(active) = self.active() {
            return self.report(&active.id);
        }
        let id = self.store(|s| s.list(1)).ok()?.into_iter().next()?.id;
        self.report(&id)
    }

    /// Stores an imported `report.json`; returns its id.
    pub fn import(&self, mut report: Report) -> anyhow::Result<String> {
        anyhow::ensure!(report.schema.starts_with("cuteafd.bench.report/"), "not a cuteafd bench report");
        if self.report(&report.id).is_some() {
            report.id = format!("{}-import-{}", report.id, &uuid::Uuid::new_v4().simple().to_string()[..6]);
        }
        report.status = RunStatus::Imported;
        self.store(|s| s.save(&report))?;
        Ok(report.id)
    }

    fn server_info(&self, model: &str) -> ServerInfo {
        let mut slot = self.info.lock().expect("info lock");
        if let Some(info) = slot.as_ref().filter(|i| i.model == model) {
            return info.clone();
        }
        let info = crate::server::server_info(model);
        *slot = Some(info.clone());
        info
    }

    /// Rates for estimates: this lifetime's baseline, else a stored one, else defaults.
    fn rates(&self, fingerprint: Option<&str>) -> Rates {
        let Some(fingerprint) = fingerprint else { return Rates::default() };
        if let Some(b) = self.baselines.lock().ok().and_then(|b| b.get(fingerprint).cloned()) {
            return Rates::from_baseline(&b);
        }
        self.store(|s| s.reports_for(fingerprint, 5)).ok().into_iter().flatten()
            .find_map(|r| r.baseline).map(|b| Rates::from_baseline(&b)).unwrap_or_default()
    }

    /// The panel catalog with per-pass estimates on this server.
    pub fn catalog(&self) -> Value {
        let info = self.info.lock().ok().and_then(|i| i.clone()).unwrap_or_default();
        let fingerprint = (!info.model.is_empty()).then(|| crate::server::fingerprint(&info));
        let rates = self.rates(fingerprint.as_deref());
        let has_baseline = fingerprint.as_ref().is_some_and(|f| self.baselines.lock().is_ok_and(|b| b.contains_key(f)));
        let mut panels = vec![json!({"id": "baseline", "title": "Basic card + quick quality", "mandatory": true,
            "description": "C1 decode on code, prose and JSON (thinking off), 8K prefill and TTFT; logit fidelity, \
                prefix-cache restore exactness, lossless speculation, template round trip, C1 vs C4.",
            "estimate_s": if has_baseline { 0.0 } else { crate::baseline::estimate_s(&rates) },
            "done": has_baseline})];
        for panel in panels::catalog() {
            panels.push(json!({"id": panel.id(), "title": panel.title(), "description": panel.description(),
                "estimate_s": panel.estimate_s(&rates, &info), "unavailable": panel.unavailable(&info),
                "always": panels::ALWAYS.contains(&panel.id())}));
        }
        json!({"panels": panels, "rates": rates})
    }

    pub fn status(&self) -> Value {
        let active = self.active();
        let info = self.info.lock().ok().and_then(|i| i.clone());
        let fingerprint = info.as_ref().map(crate::server::fingerprint);
        let baseline = fingerprint.as_ref().and_then(|f| self.baselines.lock().ok()?.get(f).cloned());
        json!({
            "active": active.as_ref().map(|a| json!({"id": a.id, "panel": a.panel, "fraction": a.fraction,
                "eta_s": a.eta_s, "elapsed_s": a.started.elapsed().as_secs_f64()})),
            "readiness_s": crate::context::readiness_s(),
            "fingerprint": fingerprint,
            "model": info.as_ref().map(|i| i.model.clone()),
            "baseline": baseline.as_ref().map(|b| json!({"run": b.run_id, "quality": b.quality.status,
                "badge": b.quality.badge()})),
            "quality_failed": baseline.as_ref().is_some_and(|b| b.quality.status == crate::report::CheckStatus::Fail),
            "auth": if self.api_key.is_some() { "local network or API key" } else { "local network" },
        })
    }

    pub fn cancel(&self, id: &str) -> bool {
        match self.active() {
            Some(active) if active.id == id => {
                active.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Starts a run; returns its id.
    pub fn start(self: &Arc<Self>, request: RunRequest) -> Result<String, StartError> {
        let base = crate::context::loopback().ok_or(StartError::NotReady)?;
        let (name, planned) = match (&request.panels, &request.profile) {
            (Some(panels), _) => ("custom".to_string(),
                panels.iter().map(|id| PlannedPanel { id: id.clone(), passes: 1 }).collect::<Vec<_>>()),
            (None, Some(name)) => {
                let profile = self.profiles().into_iter().find(|p| &p.name == name)
                    .ok_or_else(|| StartError::UnknownProfile(name.clone()))?;
                (profile.name, profile.panels)
            }
            (None, None) => ("share".to_string(), Vec::new()),
        };
        let passes: Vec<(String, u32)> = request.passes.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let (plan, dropped) = profiles::resolve(&planned, &passes);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let active = ActiveRun { id: id.clone(), token: uuid::Uuid::new_v4().simple().to_string(),
            cancel: Arc::new(AtomicBool::new(false)), started: Instant::now(), eta_s: 0.0, panel: "starting".into(),
            fraction: 0.0 };
        {
            let mut slot = self.active.lock().expect("active lock");
            if let Some(current) = slot.as_ref() {
                return Err(StartError::Busy(current.id.clone()));
            }
            *slot = Some(active.clone());
        }
        let bench = self.clone();
        std::thread::Builder::new().name("cuteafd-bench".into()).spawn(move || {
            bench.execute(active, base, name, plan, dropped);
        }).expect("spawn the benchmark thread");
        Ok(id)
    }

    fn update_active(&self, f: impl FnOnce(&mut ActiveRun)) {
        if let Ok(mut slot) = self.active.lock() {
            if let Some(active) = slot.as_mut() {
                f(active);
            }
        }
    }

    fn execute(self: Arc<Self>, active: ActiveRun, base: String, profile: String, plan: Vec<PlannedPanel>,
        dropped: Vec<String>) {
        let id = active.id.clone();
        let report = Arc::new(Mutex::new(Report {
            schema: SCHEMA.into(), id: id.clone(), created: now_rfc3339(), finished: None, status: RunStatus::Running,
            profile, plan: plan.clone(), server: ServerInfo::default(), fingerprint: String::new(), baseline: None,
            panels: plan.iter().map(|p| PanelResult { id: p.id.clone(),
                title: panels::find(&p.id).map_or(p.id.clone(), |panel| panel.title().to_string()),
                ..PanelResult::default() }).collect(),
            error: (!dropped.is_empty()).then(|| format!("panels not in this build: {}", dropped.join(", "))),
        }));
        self.live.lock().expect("live lock").insert(id.clone(), report.clone());
        let progress = Progress::default();
        let done = Arc::new(AtomicBool::new(false));
        let ticker = {
            let (bench, progress, done, report, base) = (self.clone(), progress.clone(), done.clone(), report.clone(),
                base.clone());
            std::thread::Builder::new().name("cuteafd-bench-tick".into())
                .spawn(move || bench.tick(&report, &progress, &done, &base)).ok()
        };
        let outcome = self.run_plan(&active, &base, &report, &progress, &plan);
        {
            let mut r = report.lock().expect("report lock");
            r.finished = Some(now_rfc3339());
            match outcome {
                Ok(()) => r.status = RunStatus::Done,
                Err(error) if active.cancel.load(Ordering::Relaxed)
                    || error.downcast_ref::<crate::client::Cancelled>().is_some() => r.status = RunStatus::Cancelled,
                Err(error) => {
                    r.status = RunStatus::Failed;
                    r.error = Some(format!("{error:#}"));
                }
            }
            for panel in r.panels.iter_mut().filter(|p| matches!(p.status, PanelStatus::Pending | PanelStatus::Running)) {
                panel.status = PanelStatus::Cancelled;
            }
            if let Err(error) = self.store(|s| s.save(&r)) {
                tracing::warn!(%error, "benchmark report not stored");
            }
            tracing::info!(run = %r.id, status = ?r.status, "benchmark finished");
        }
        done.store(true, Ordering::Relaxed);
        if let Some(ticker) = ticker {
            let _ = ticker.join();
        }
        if let Ok(mut plan) = self.plan_state().lock() {
            *plan = (String::new(), String::new(), 0.0, 0.0, 1.0);
        }
        *self.active.lock().expect("active lock") = None;
        let snapshot = report.lock().expect("report lock").clone();
        self.emit(json!({"type": "report", "run": id, "report": snapshot}));
        self.emit(json!({"type": "status", "status": self.status()}));
        // Finished runs are served from the store from now on.
        self.live.lock().expect("live lock").remove(&id);
    }

    fn run_plan(&self, active: &ActiveRun, base: &str, report: &Arc<Mutex<Report>>, progress: &Progress,
        plan: &[PlannedPanel]) -> anyhow::Result<()> {
        let mut client = Client::new(base, Some(active.token.clone()), active.cancel.clone());
        let record = client.discover()?;
        let max_context = record["max_context_tokens"].as_u64().unwrap_or(8192);
        let max_output = record["max_output_tokens"].as_u64().unwrap_or(4096);
        let info = self.server_info(&client.model);
        let fingerprint = crate::server::fingerprint(&info);
        {
            let mut r = report.lock().expect("report lock");
            r.server = info.clone();
            r.fingerprint = fingerprint.clone();
        }
        self.emit(json!({"type": "status", "status": self.status()}));
        let baseline = self.baselines.lock().ok().and_then(|b| b.get(&fingerprint).cloned());
        let mut rates = self.rates(Some(&fingerprint));
        // Estimates for the ETA: (panel, seconds per pass, passes).
        let estimates: Vec<(String, f64, u32)> = plan.iter().filter_map(|p| panels::find(&p.id)
            .map(|panel| (p.id.clone(), panel.estimate_s(&rates, &info), p.passes))).collect();
        let mut remaining: f64 = estimates.iter().map(|(_, s, n)| s * f64::from(*n)).sum::<f64>()
            + if baseline.is_none() { crate::baseline::estimate_s(&rates) } else { 0.0 };
        let total = remaining.max(1.0);
        let baseline = match baseline {
            Some(b) => b,
            None => {
                let estimate = crate::baseline::estimate_s(&rates);
                self.begin(active, "baseline", estimate, remaining, total, progress);
                let b = crate::baseline::run(&client, &info, progress, &active.id, &fingerprint, max_context)?;
                remaining -= estimate;
                if b.quality.status != crate::report::CheckStatus::Pending {
                    self.baselines.lock().expect("baselines lock").insert(fingerprint.clone(), b.clone());
                }
                rates = Rates::from_baseline(&b);
                b
            }
        };
        report.lock().expect("report lock").baseline = Some(baseline.clone());
        self.emit(json!({"type": "report", "run": active.id, "report": report.lock().expect("report lock").clone()}));
        let history = self.store(|s| s.reports_for(&fingerprint, 50)).unwrap_or_default();
        for (index, planned) in plan.iter().enumerate() {
            let Some(panel) = panels::find(&planned.id) else { continue };
            let estimate = panel.estimate_s(&rates, &info);
            if let Some(reason) = panel.unavailable(&info) {
                let mut r = report.lock().expect("report lock");
                r.panels[index].status = PanelStatus::Unsupported;
                r.panels[index].error = Some(reason);
                remaining -= estimate * f64::from(planned.passes);
                continue;
            }
            let earlier: Vec<Value> = history.iter().filter_map(|r| r.panel(panel.id()))
                .flat_map(|p| p.passes.iter().cloned()).collect();
            {
                let mut r = report.lock().expect("report lock");
                r.panels[index].status = PanelStatus::Running;
                r.panels[index].started = Some(now_rfc3339());
                r.panels[index].history = earlier.clone();
            }
            let started = Instant::now();
            for pass in 1..=planned.passes {
                client.check()?;
                self.begin(active, panel.id(), estimate, remaining, total, progress);
                let ctx = Ctx { client: &client, info: &info, baseline: Some(&baseline), rates, progress, pass,
                    history: &earlier, max_context, max_output };
                let outcome = panel.run(&ctx);
                remaining -= estimate;
                let mut r = report.lock().expect("report lock");
                match outcome {
                    Ok(value) => r.panels[index].passes.push(value),
                    Err(error) => {
                        if error.downcast_ref::<crate::client::Cancelled>().is_some() {
                            return Err(error);
                        }
                        r.panels[index].status = PanelStatus::Failed;
                        r.panels[index].error = Some(format!("{error:#}"));
                        break;
                    }
                }
                drop(r);
                self.emit(json!({"type": "report", "run": active.id, "report": report.lock().expect("report lock").clone()}));
            }
            let mut r = report.lock().expect("report lock");
            if r.panels[index].status == PanelStatus::Running {
                r.panels[index].status = PanelStatus::Done;
            }
            r.panels[index].finished = Some(now_rfc3339());
            r.panels[index].seconds = started.elapsed().as_secs_f64();
            if let Err(error) = self.store(|s| s.save(&r)) {
                tracing::warn!(%error, "benchmark report not stored");
            }
        }
        Ok(())
    }

    /// A panel (or the baseline) starts: reset its progress and the ETA base.
    fn begin(&self, active: &ActiveRun, panel: &str, estimate: f64, remaining: f64, total: f64, progress: &Progress) {
        progress.reset();
        if let Ok(mut plan) = self.plan_state().lock() {
            *plan = (active.id.clone(), panel.to_string(), estimate, remaining, total);
        }
        self.update_active(|a| a.panel = panel.to_string());
    }

    fn plan_state(&self) -> &'static Mutex<(String, String, f64, f64, f64)> {
        static STATE: OnceLock<Mutex<(String, String, f64, f64, f64)>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new((String::new(), String::new(), 0.0, 0.0, 1.0)))
    }

    /// Once a second while a run is active: progress, ETA, the live strip, partial results.
    fn tick(&self, report: &Arc<Mutex<Report>>, progress: &Progress, done: &AtomicBool, base: &str) {
        let stats_client = Client::new(base, None, Arc::new(AtomicBool::new(false)));
        let mut last: Option<(Instant, f64)> = None;
        let mut revision = u64::MAX;
        while !done.load(Ordering::Relaxed) {
            let state = progress.get();
            let (run, panel, estimate, remaining, total) = self.plan_state().lock().map(|s| s.clone())
                .unwrap_or_default();
            let eta = (remaining - estimate * state.fraction).max(0.0);
            let fraction = (1.0 - eta / total.max(1.0)).clamp(0.0, 1.0);
            self.update_active(|a| {
                a.eta_s = eta;
                a.fraction = fraction;
            });
            // Output tokens per second and active requests from the server's own counters.
            let (mut tok_s, mut active_requests) = (None, None);
            if let Ok(stats) = stats_client.stats() {
                let generated = stats["generated_tokens"].as_f64().or_else(|| stats["totals"]["output_tokens"].as_f64());
                if let Some(generated) = generated {
                    let now = Instant::now();
                    if let Some((then, before)) = last {
                        let dt = now.duration_since(then).as_secs_f64();
                        if dt > 0.0 {
                            tok_s = Some(((generated - before) / dt).max(0.0));
                        }
                    }
                    last = Some((now, generated));
                }
                active_requests = stats["active"].as_u64().or_else(|| {
                    let t = &stats["totals"];
                    Some(t["requests_admitted"].as_u64()?.saturating_sub(t["requests_retired"].as_u64()?))
                });
            }
            let title = if panel == "baseline" { "Basic card + quick quality".to_string() }
                else { panels::find(&panel).map_or(panel.clone(), |p| p.title().to_string()) };
            if run.is_empty() {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            self.emit(json!({"type": "progress", "run": run, "panel": panel, "title": title, "label": state.label,
                "panel_fraction": state.fraction, "fraction": fraction, "eta_s": eta,
                "live": {"tok_s": tok_s, "active": active_requests}}));
            if state.revision != revision {
                revision = state.revision;
                if let Some(partial) = state.partial {
                    if panel == "baseline" {
                        if let Ok(b) = serde_json::from_value::<Baseline>(partial.clone()) {
                            if let Ok(mut r) = report.lock() {
                                r.baseline = Some(b);
                            }
                        }
                    }
                    self.emit(json!({"type": "partial", "run": run, "panel": panel, "value": partial}));
                }
            }
            std::thread::sleep(Duration::from_millis(1000));
        }
    }
}
