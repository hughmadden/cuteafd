//! Per-engine runtime prefill depth (#762). An engine whose prefill runs on
//! the shared executor holds one [`PrefillDepth`]: the planner's admitted
//! slots and unit extent, and the shared [`WaveControl`]. Before each
//! prefill call (one window cycle) the engine reports every finished call's
//! pace from its Spark link ([`PrefillDepth::report`]) and takes this call's
//! depth ([`PrefillDepth::start`]): fewer, wider units only at a prefill's
//! opening call, more units at any call. Storage never changes: every
//! depth's slots are views at `slot * unit` inside per-slot planes sized for
//! [`reserve_rows`], and the Spark prefill owner accepts any unit up to the
//! admitted one.
//!
//! Before readiness the engine calibrates the controller (one cold
//! full-call prefill per candidate depth, [`PrefillDepth::calibration`]) or
//! loads its persisted evidence ([`CalibrationStore`]).

use anyhow::Context;
use cuteafd_loader::serving_capacity::wave_control::{call_rows_in, reserve_rows_in, Boundary, Decision, Evidence,
    PrefillPace, WaveControl, REFERENCE_SHARE};
use cuteafd_loader::serving_capacity::prefill_waves::RESERVE_ROWS;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The shape of one prefill call: `waves` units of at most `unit_rows` in
/// flight, and its id on the controller's clock (passed to the link's
/// `begin_prefill`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Depth { pub waves: usize, pub unit_rows: usize, pub call: u64 }

/// Calls whose pace may still arrive (older ones are dropped unreported).
const PENDING: usize = 16;
/// Persisted evidence is rewritten at most this often while it changes.
const SAVE_EVERY: Duration = Duration::from_secs(600);

pub(crate) struct PrefillDepth {
    slots: usize,
    unit_cap: usize,
    /// The reserve every depth's units are cut from (the standard one, or
    /// the largest that fits a smaller card).
    reserve: usize,
    control: RefCell<WaveControl>,
    /// Started calls not yet reported: the decision and the rows the call ran.
    pending: RefCell<VecDeque<(Decision, u64)>>,
    /// The last call's sequence key and end row: a call starting there runs on.
    last: Cell<Option<(u64, usize)>>,
    /// Calibration calls not yet started, and not yet reported (the
    /// calibration finishes when every call reported, or at the first
    /// serving call after them; a late pace still records its ratio).
    calibration: Cell<(usize, usize)>,
    /// Ids of the calibration calls started (only their paces count down).
    calibration_calls: RefCell<Vec<u64>>,
    /// Where the evidence persists, and when it was last written.
    store: RefCell<Option<(CalibrationStore, Instant)>>,
}

impl PrefillDepth {
    /// `initial` waves (the planner's estimate) of at most `unit_cap` rows,
    /// within `slots` admitted slots. `pinned` (an explicit wave count)
    /// keeps the initial depth.
    pub(crate) fn new(initial: usize, slots: usize, unit_cap: usize, pinned: bool) -> anyhow::Result<Self> {
        Self::in_reserve(initial, slots, unit_cap, RESERVE_ROWS as usize, pinned)
    }
    /// [`Self::new`] for depths cut from a `reserve` of rows (TJ's reserve
    /// rule: the planner's fit cap; `wave_control::unit_for_in`).
    pub(crate) fn in_reserve(initial: usize, slots: usize, unit_cap: usize, reserve: usize, pinned: bool)
        -> anyhow::Result<Self> {
        anyhow::ensure!((1..=4).contains(&slots) && (1..=slots).contains(&initial)
            && unit_cap >= 64 && unit_cap % 64 == 0 && unit_cap <= 4096
            && reserve % 64 == 0 && reserve <= RESERVE_ROWS as usize && unit_cap <= reserve,
            "prefill depth {initial} of {slots} slots x {unit_cap} rows outside the {reserve}-row reserve");
        let control = WaveControl::in_reserve(initial as u64, slots as u64, unit_cap as u64, reserve as u64, pinned);
        tracing::info!(slots, unit_cap, reserve, initial = control.waves(), pinned, candidates = ?control.candidates(),
            reserve_rows = reserve_rows_in(slots as u64, unit_cap as u64, reserve as u64), "prefill depth control");
        Ok(Self { slots, unit_cap, reserve, control: RefCell::new(control), pending: RefCell::new(VecDeque::new()),
            last: Cell::new(None), calibration: Cell::new((0, 0)),
            calibration_calls: RefCell::new(Vec::new()), store: RefCell::new(None) })
    }

    /// Admitted in-flight slots (fixed at startup).
    pub(crate) fn slots(&self) -> usize { self.slots }
    /// The widest unit admitted (fixed at startup).
    pub(crate) fn unit_cap(&self) -> usize { self.unit_cap }
    /// Rows every per-slot plane spans so any depth's slots are views in it.
    pub(crate) fn reserve_rows(&self) -> usize {
        reserve_rows_in(self.slots as u64, self.unit_cap as u64, self.reserve as u64) as usize
    }
    /// Longest prefill call: whole units at every depth the controller may pick.
    pub(crate) fn call_rows(&self) -> usize {
        if self.control.borrow().pinned() { return self.reserve_rows(); }
        call_rows_in(self.slots as u64, self.unit_cap as u64, self.reserve as u64) as usize
    }
    /// The candidate depths (ascending).
    pub(crate) fn candidates(&self) -> Vec<u64> { self.control.borrow().candidates().to_vec() }

    /// The current depth (a pinned controller's only one), without starting a call.
    pub(crate) fn current(&self) -> (usize, usize) {
        let control = self.control.borrow();
        (control.waves() as usize, control.unit_rows() as usize)
    }

    /// Finished calls' paces from the link ([`crate::shared::spark_waves::Waves::take_prefill`]).
    pub(crate) fn report(&self, paces: Vec<(u64, PrefillPace)>) {
        let full = self.call_rows() as u64;
        for (call, pace) in paces {
            let Some(at) = self.pending.borrow().iter().position(|(decision, _)| decision.call == call) else { continue };
            let (decision, rows) = self.pending.borrow_mut().remove(at).expect("pending call");
            let sample = self.control.borrow_mut().record(&decision, rows, full, &pace);
            // One line per finished call: its depth, period, ratio and both waits.
            tracing::info!(call, ran = sample.waves, rows, boundary = ?sample.boundary, reference = decision.reference,
                transition = decision.transition, counted = sample.counted, wall_us = pace.wall_us,
                period_us = ?sample.period_us, ratio = ?sample.ratio, idle_share = ?sample.idle_share,
                reply_wait_share = ?sample.reply_wait_share, reason = decision.reason, "prefill call");
            let (unstarted, unreported) = self.calibration.get();
            let calibration_call = self.calibration_calls.borrow().contains(&call);
            if unreported > 0 && calibration_call {
                self.calibration.set((unstarted, unreported - 1));
                if unreported == 1 { self.finish_calibration(); }
            }
        }
        self.save(false);
    }

    /// A prefill call of sequence `key` starts at row `start` with `rows`
    /// rows: it runs on when the previous call was the same sequence's and
    /// ended there; otherwise it opens a prefill. An opening at row 0 of at
    /// least 90% of a full call is a reference call (evidence). Returns its depth.
    pub(crate) fn start(&self, key: u64, start: usize, rows: usize) -> Depth {
        let calibration_call = match self.calibration.get() {
            (0, unreported) if unreported > 0 => {
                // Serving begins with calibration paces outstanding: choose from what arrived.
                self.calibration.set((0, 0));
                self.finish_calibration();
                false
            }
            (unstarted, unreported) if unstarted > 0 => { self.calibration.set((unstarted - 1, unreported)); true }
            _ => false,
        };
        let running = start > 0 && self.last.get() == Some((key, start));
        let boundary = if running { Boundary::Running } else { Boundary::Opening };
        let reference = start == 0 && rows as f64 >= REFERENCE_SHARE * self.call_rows() as f64;
        let decision = self.control.borrow_mut().next(boundary, reference);
        let mut pending = self.pending.borrow_mut();
        if pending.len() == PENDING { pending.pop_front(); }
        pending.push_back((decision, 0));
        if calibration_call { self.calibration_calls.borrow_mut().push(decision.call); }
        self.last.set(Some((key, start)));
        tracing::debug!(call = decision.call, waves = decision.waves, unit_rows = decision.unit_rows,
            boundary = ?boundary, reference, transition = decision.transition, reason = decision.reason,
            "prefill depth");
        Depth { waves: decision.waves as usize, unit_rows: decision.unit_rows as usize, call: decision.call }
    }

    /// Call `depth.call` cut `units` units of `rows` prompt rows in all
    /// (fewer units than its depth when short) and ends at row `end`.
    pub(crate) fn ran(&self, depth: Depth, units: usize, rows: usize) {
        if let Some((decision, ran)) = self.pending.borrow_mut().iter_mut().find(|(d, _)| d.call == depth.call) {
            decision.waves = decision.waves.min(units as u64);
            *ran = rows as u64;
        }
        if let Some((key, start)) = self.last.get() { self.last.set(Some((key, start + rows))); }
    }

    /// Where the evidence persists (a release build; `None` always calibrates).
    pub(crate) fn set_store(&self, store: Option<CalibrationStore>) {
        *self.store.borrow_mut() = store.map(|store| (store, Instant::now()));
    }

    /// Startup calibration (#762): loads the persisted evidence when the
    /// store's key matches; otherwise returns how many cold full-call
    /// prefills (one warm, then one per candidate) the engine must run now,
    /// before readiness, each at row 0 of a fresh sequence. The calibration
    /// finishes as their paces arrive ([`Self::report`]), or at the first
    /// serving call after them.
    pub(crate) fn calibration(&self) -> usize {
        let store = self.store.borrow().as_ref().map(|(store, _)| store.clone());
        let mut control = self.control.borrow_mut();
        if !control.adaptive() {
            tracing::info!(pinned = control.pinned(), candidates = ?control.candidates(),
                "prefill depth calibration skipped: the depth cannot move");
            return 0;
        }
        if let Some(store) = &store {
            match store.load() {
                Ok(evidence) => match control.restore(&evidence) {
                    Ok(()) => {
                        tracing::info!(path = %store.path.display(), waves = control.waves(), ratios = ?control.ratios(),
                            "prefill depth evidence loaded; calibration skipped");
                        return 0;
                    }
                    Err(reason) => tracing::info!(path = %store.path.display(), reason,
                        "prefill depth evidence refused; calibrating"),
                },
                Err(reason) => tracing::info!(path = %store.path.display(), reason = %reason,
                    "prefill depth evidence unavailable; calibrating"),
            }
        } else {
            tracing::info!("prefill depth evidence not persisted (WIP/bench build); calibrating");
        }
        let calls = control.calibrate();
        self.calibration.set((calls, calls));
        calls
    }

    fn finish_calibration(&self) {
        let reason = self.control.borrow_mut().finish_calibration();
        let control = self.control.borrow();
        tracing::info!(waves = control.waves(), unit_rows = control.unit_rows(), ratios = ?control.ratios(), reason,
            "prefill depth calibrated");
        drop(control);
        self.save(true);
    }

    /// Whether calibration calls are still unreported.
    pub(crate) fn calibrating(&self) -> bool { self.calibration.get().1 > 0 }

    /// Persists the evidence: `now`, or when it changed and the last write is
    /// older than ten minutes. Failures are logged, never fatal.
    pub(crate) fn save(&self, now: bool) {
        let mut store = self.store.borrow_mut();
        let Some((store, written)) = store.as_mut() else { return };
        if self.calibrating() { return; }
        if !now && written.elapsed() < SAVE_EVERY { return; }
        let (evidence, dirty) = self.control.borrow_mut().evidence();
        if !(now || dirty) { return; }
        match store.save(&evidence) {
            Ok(()) => { *written = Instant::now(); tracing::debug!(path = %store.path.display(), "prefill depth evidence saved"); }
            Err(error) => tracing::warn!(path = %store.path.display(), %error, "saving prefill depth evidence"),
        }
    }
}

impl Drop for PrefillDepth {
    /// Clean shutdown writes the latest evidence.
    fn drop(&mut self) { self.save(true); }
}

/// Startup calibration prompt: about `rows` tokens of real text (the bench
/// corpus, encoded by the model's own tokenizer), so expert routing is
/// representative. Deterministic per `seed`.
pub(crate) fn calibration_tokens(tokenizer: &cuteafd_loader::LoadedTokenizer, seed: u64, rows: usize)
    -> anyhow::Result<Vec<u32>> {
    let mut tokens = tokenizer.encode_ids(&cuteafd_bench::text::corpus(seed, rows + rows / 4))?;
    anyhow::ensure!(tokens.len() >= rows, "calibration corpus encoded to {} of {rows} tokens", tokens.len());
    tokens.truncate(rows);
    Ok(tokens)
}

/// The checkpoint's quantization for the calibration key: its config's
/// `quantization_config` (method, format and bits), or "none".
pub(crate) fn checkpoint_quant(snapshot: &Path) -> String {
    let config = std::fs::read(snapshot.join("config.json")).ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let quant = config.as_ref().map(|config| &config["quantization_config"]).filter(|quant| !quant.is_null());
    quant.map_or_else(|| "none".into(), |quant| ["quant_method", "format", "bits", "fmt"].iter()
        .filter_map(|field| quant.get(*field).filter(|value| !value.is_null()).map(|value| value.to_string()))
        .collect::<Vec<_>>().join("/"))
}

/// Attaches the persisted-evidence store (release builds only) and runs the
/// startup calibration: `prefill(seed, tokens)` runs one cold full-call
/// prefill of a fresh sequence at row 0 and releases it. Logs the added
/// readiness time. Calibration failures are fatal (they are real prefills).
pub(crate) fn calibrate(depth: &PrefillDepth, key: CalibrationKey<'_>, tokenizer: &cuteafd_loader::LoadedTokenizer,
    mut prefill: impl FnMut(u64, &[u32]) -> anyhow::Result<()>) -> anyhow::Result<()> {
    let started = Instant::now();
    let family = key.family.to_string();
    depth.set_store(CalibrationStore::for_release(key)?);
    let calls = depth.calibration();
    let rows = depth.call_rows();
    for call in 0..calls {
        let tokens = calibration_tokens(tokenizer, 0x762 + call as u64, rows)?;
        prefill(call as u64, &tokens).with_context(|| format!("prefill depth calibration call {call}"))?;
    }
    tracing::info!(family, calls, rows, elapsed_ms = started.elapsed().as_millis() as u64,
        "prefill depth calibration readiness cost");
    Ok(())
}

/// Schema version of the persisted evidence.
const VERSION: u32 = 1;

/// Where one deployment's controller evidence persists (#762). Release
/// images only: a WIP or dev build (no `CUTEAFD_RELEASE_VERSION`) always
/// calibrates, because its code changes between runs. The file lives in the
/// bench store every coordinator already mounts (`CUTEAFD_BENCH_DIR`,
/// `~/.cache/cuteafd/bench` on the host) and is keyed by everything that
/// moves the ratios: checkpoint, family, build, GPUs, Sparks, topology,
/// reserve and candidates.
#[derive(Debug, Clone)]
pub(crate) struct CalibrationStore {
    path: PathBuf,
    key: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Persisted {
    version: u32,
    key: serde_json::Value,
    evidence: Evidence,
}

/// A deployment's hardware identity for the calibration key (each family
/// supplies it; the rest comes from the checkpoint and the depth).
#[derive(Debug, Clone, Default)]
pub(crate) struct DeploymentIdentity {
    /// Coordinator GPUs: (UUID, SM count).
    pub gpus: Vec<(String, u32)>,
    /// Spark endpoints in rank order.
    pub sparks: Vec<String>,
    /// TP/EP topology, as the family describes it.
    pub topology: String,
}

impl DeploymentIdentity {
    /// The visible coordinator GPUs, `sparks` and `topology`.
    pub(crate) fn new(library: &cuteafd_ffi::NativeLibrary, sparks: impl IntoIterator<Item = impl ToString>,
        topology: impl Into<String>) -> Self {
        Self { gpus: coordinator_gpus(library), sparks: sparks.into_iter().map(|s| s.to_string()).collect(),
            topology: topology.into() }
    }
}

impl<'a> CalibrationKey<'a> {
    /// The key of `depth` serving `snapshot` as `family` on `identity`.
    pub(crate) fn new(family: &'a str, snapshot: &'a Path, depth: &PrefillDepth, identity: DeploymentIdentity) -> Self {
        Self { family, snapshot, quant: checkpoint_quant(snapshot), gpus: identity.gpus, sparks: identity.sparks,
            topology: identity.topology, slots: depth.slots(), unit_cap: depth.unit_cap(), reserve: depth.reserve,
            candidates: depth.candidates() }
    }
}

/// Everything the persisted ratios depend on.
pub(crate) struct CalibrationKey<'a> {
    pub family: &'a str,
    pub snapshot: &'a Path,
    /// The checkpoint's quantization (from its config), or the family's format.
    pub quant: String,
    /// Coordinator GPUs: (UUID, SM count).
    pub gpus: Vec<(String, u32)>,
    /// Spark endpoints in rank order.
    pub sparks: Vec<String>,
    /// TP/EP topology, as the family describes it.
    pub topology: String,
    pub slots: usize,
    pub unit_cap: usize,
    /// The reserve depths are cut from.
    pub reserve: usize,
    pub candidates: Vec<u64>,
}

impl CalibrationStore {
    /// The store for a release build, or `None` for WIP/dev builds (always calibrate).
    pub(crate) fn for_release(key: CalibrationKey<'_>) -> anyhow::Result<Option<Self>> {
        let release = std::env::var("CUTEAFD_RELEASE_VERSION").ok()
            .filter(|version| !version.is_empty() && version != "unknown");
        let Some(release) = release else { return Ok(None) };
        Self::new(&cuteafd_bench::store::default_dir(), &release, key).map(Some)
    }

    /// A store under `dir` for `build` (the release version plus the binary's identity).
    pub(crate) fn new(dir: &Path, build: &str, key: CalibrationKey<'_>) -> anyhow::Result<Self> {
        use sha2::Digest;
        let key_reserve = key.reserve;
        let snapshot = std::fs::canonicalize(key.snapshot).unwrap_or_else(|_| key.snapshot.to_path_buf());
        let revision = snapshot.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let model = snapshot.parent().and_then(Path::parent).and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let key = serde_json::json!({
            "family": key.family, "model": model, "revision": revision, "quant": key.quant,
            "build": { "release": build, "commit": option_env!("CUTEAFD_BUILD_COMMIT"),
                "dirty": option_env!("CUTEAFD_BUILD_DIRTY"), "binary": binary_identity() },
            "gpus": key.gpus, "sparks": key.sparks, "topology": key.topology,
            "reserve": { "slots": key.slots, "unit_cap": key.unit_cap }, "candidates": key.candidates,
        });
        // A smaller reserve cuts different units (the standard one keeps its key).
        let mut key = key;
        if key_reserve != RESERVE_ROWS as usize { key["reserve"]["rows"] = key_reserve.into(); }
        let digest = sha2::Sha256::digest(serde_json::to_vec(&key)?);
        let name: String = digest[..8].iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Self { path: dir.join("prefill-depth").join(format!("{}-{name}.json", key["family"].as_str()
            .unwrap_or("family"))), key })
    }

    /// The evidence when the file exists and its key matches exactly.
    fn load(&self) -> anyhow::Result<Evidence> {
        let text = std::fs::read(&self.path).with_context(|| format!("reading {}", self.path.display()))?;
        let persisted: Persisted = serde_json::from_slice(&text).context("parsing the persisted evidence")?;
        anyhow::ensure!(persisted.version == VERSION, "schema version {} (want {VERSION})", persisted.version);
        anyhow::ensure!(persisted.key == self.key, "cache key differs");
        Ok(persisted.evidence)
    }

    /// Writes the evidence atomically (a temporary file renamed into place).
    fn save(&self, evidence: &Evidence) -> anyhow::Result<()> {
        let dir = self.path.parent().context("evidence directory")?;
        std::fs::create_dir_all(dir)?;
        let temporary = dir.join(format!(".{}.{}.tmp", self.path.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()));
        let persisted = Persisted { version: VERSION, key: self.key.clone(), evidence: evidence.clone() };
        std::fs::write(&temporary, serde_json::to_vec_pretty(&persisted)?)?;
        std::fs::rename(&temporary, &self.path).inspect_err(|_| { let _ = std::fs::remove_file(&temporary); })?;
        Ok(())
    }
}

/// The running binary's identity: its size and modification time (cheap,
/// and changes with every rebuild; the release version and commit cover the rest).
fn binary_identity() -> Option<String> {
    let metadata = std::env::current_exe().ok().and_then(|exe| std::fs::metadata(exe).ok())?;
    let modified = metadata.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    Some(format!("{}:{modified}", metadata.len()))
}

/// The coordinator GPUs' (UUID, SM count) for the calibration key: the
/// devices `CUDA_VISIBLE_DEVICES` names, in NVML order (empty when NVML is unavailable).
pub(crate) fn coordinator_gpus(library: &cuteafd_ffi::NativeLibrary) -> Vec<(String, u32)> {
    let sms = library.sm_count().unwrap_or(0) as u32;
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let Ok(nvml) = cuteafd_ffi::nvml::Nvml::open() else { return Vec::new() };
    let Ok(devices) = nvml.devices() else { return Vec::new() };
    devices.iter().filter(|device| visible.as_deref().map(str::trim).filter(|v| !v.is_empty()).is_none_or(|visible|
        visible.split(',').map(str::trim).any(|entry| if entry.starts_with("GPU-") {
            device.identity.uuid.starts_with(entry) } else { entry.parse::<u32>().is_ok_and(|i| i == device.identity.index) })))
        .map(|device| (device.identity.uuid.clone(), sms)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pace(rows: u64, wall_us: u64, busy: u64, idle: u64) -> PrefillPace {
        PrefillPace { rows, ranks: vec![(busy, idle); 6], wall_us, reply_wait_us: Some(0) }
    }

    /// Wall µs of a full call per depth: 3 x 2752 best (cards E figures).
    fn wall(waves: usize) -> u64 { match waves { 2 => 2_460_000, 3 => 2_318_000, _ => 2_502_000 } }

    /// Runs `calls` calibration prefills as an engine would: one fresh
    /// sequence each at row 0, every pace reported at the next boundary.
    fn calibrate(depth: &PrefillDepth, calls: usize) -> Vec<usize> {
        let mut ran = Vec::new();
        for call in 0..calls {
            let d = depth.start(1000 + call as u64, 0, 8192);
            depth.ran(d, d.waves, 8192);
            depth.report(vec![(d.call, pace(8192, wall(d.waves), 1_000_000, 100_000))]);
            ran.push(d.waves);
        }
        ran
    }

    fn key(dir: &Path) -> CalibrationKey<'_> {
        CalibrationKey { family: "glm5", snapshot: dir, quant: "exl3".into(), gpus: vec![("GPU-a".into(), 188)],
            sparks: vec!["10.55.0.1:9000".into()], topology: "tp1".into(), slots: 4, unit_cap: 4096, reserve: 8192,
            candidates: vec![2, 3, 4] }
    }

    #[test]
    fn depths_stay_inside_the_admitted_slots_and_reserve() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 4, 2752, false)?;
        assert_eq!((depth.current(), depth.reserve_rows()), ((3, 2752), 8256));
        assert_eq!(depth.call_rows(), 8192);
        assert_eq!(PrefillDepth::new(3, 3, 2752, true)?.call_rows(), 8256);
        let pinned = PrefillDepth::new(2, 4, 4096, true)?;
        assert_eq!(pinned.calibration(), 0);
        for call in 0..4 {
            let d = pinned.start(1, call * 8192, 8192);
            pinned.ran(d, 2, 8192);
            pinned.report(vec![(d.call, pace(8192, 3_000_000, 1_000_000, 500_000))]);
        }
        assert_eq!(pinned.current(), (2, 4096));
        assert!(PrefillDepth::new(5, 4, 2048, false).is_err());
        assert!(PrefillDepth::new(1, 4, 8192, false).is_err());
        Ok(())
    }

    #[test]
    fn calibration_populates_every_depth_then_holds_the_best() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(4, 4, 4096, false)?;
        let calls = depth.calibration();
        assert_eq!(calls, 4);
        assert_eq!(calibrate(&depth, calls), [4, 4, 2, 3]);
        assert!(!depth.calibrating());
        assert!(depth.control.borrow().calibrated());
        assert_eq!(depth.current(), (3, 2752));
        // Serving: Sparks idle and the coordinator waits (both signals), but
        // the evidence holds 3; running calls and short prompts never move it.
        for prompt in 0..10u64 {
            for call in 0..4usize {
                let d = depth.start(prompt, call * 8192, 8192);
                assert_eq!(d.waves, 3, "prompt {prompt} call {call}");
                depth.ran(d, 3, 8192);
                depth.report(vec![(d.call, PrefillPace { rows: 8192, ranks: vec![(2_000_000, 300_000); 6],
                    wall_us: wall(3), reply_wait_us: Some(900_000) })]);
            }
        }
        Ok(())
    }

    #[test]
    fn calibration_finishes_at_the_first_serving_call_when_a_pace_is_late() -> anyhow::Result<()> {
        // Kimi and Step may report a call's pace one call later: the last
        // calibration call is still in flight when serving starts.
        let depth = PrefillDepth::new(4, 4, 4096, false)?;
        let calls = depth.calibration();
        let mut last = None;
        for call in 0..calls {
            let d = depth.start(2000 + call as u64, 0, 8192);
            depth.ran(d, d.waves, 8192);
            if let Some(previous) = last.replace(d) {
                let previous: Depth = previous;
                depth.report(vec![(previous.call, pace(8192, wall(previous.waves), 1_000_000, 100_000))]);
            }
        }
        assert!(depth.calibrating());
        let serving = depth.start(1, 0, 8192);
        assert!(!depth.calibrating());
        // Its late pace (3 x 2752) still records: the next opening runs 3.
        let late = last.unwrap();
        depth.ran(serving, serving.waves, 8192);
        depth.report(vec![(late.call, pace(8192, wall(late.waves), 1_000_000, 100_000)),
            (serving.call, pace(8192, wall(serving.waves), 1_000_000, 100_000))]);
        assert_eq!(depth.start(2, 0, 8192).waves, 3);
        Ok(())
    }

    #[test]
    fn an_earlier_calls_pace_during_calibration_does_not_finish_it_early() -> anyhow::Result<()> {
        // GLM cards F: a 512-row request before calibration reported during
        // it; only calibration calls count down, so all depths get measured.
        let depth = PrefillDepth::new(3, 4, 4096, false)?;
        let early = depth.start(9, 0, 512);
        depth.ran(early, 1, 512);
        let calls = depth.calibration();
        let mut started = Vec::new();
        for call in 0..calls {
            let d = depth.start(3000 + call as u64, 0, 8192);
            depth.ran(d, d.waves, 8192);
            started.push(d);
            if call == 0 { depth.report(vec![(early.call, pace(512, 400_000, 200_000, 100_000))]); }
        }
        for d in &started[..calls - 1] { depth.report(vec![(d.call, pace(8192, wall(d.waves), 1_000_000, 100_000))]); }
        assert!(depth.calibrating(), "the 512-row pace must not count as a calibration report");
        let last = started[calls - 1];
        depth.report(vec![(last.call, pace(8192, wall(last.waves), 1_000_000, 100_000))]);
        assert!(!depth.calibrating() && depth.control.borrow().calibrated());
        Ok(())
    }

    #[test]
    fn calls_open_a_prefill_or_run_on_where_the_last_one_ended() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 4, 4096, false)?;
        let first = depth.start(7, 0, 8192);
        assert_eq!(first.waves, 3);
        depth.ran(first, 3, 8192);
        // Sparks waiting on the coordinator point at fewer, wider units, but a
        // running call never widens: only the next reference opening does.
        depth.report(vec![(first.call, pace(8192, 2_400_000, 1_600_000, 400_000))]);
        let second = depth.start(7, 8192, 8192);
        assert_eq!(second.waves, 3, "a running call keeps or deepens its depth");
        depth.ran(second, 3, 8192);
        // A short opening is not a reference call: no trial there.
        let short = depth.start(8, 0, 1000);
        assert_eq!(short.waves, 3);
        depth.ran(short, 1, 1000);
        depth.report(vec![(second.call, pace(8192, 2_400_000, 1_600_000, 400_000)),
            (short.call, pace(1000, 1_000_000, 900_000, 100_000))]);
        // The next full opening measures the unmeasured wider neighbour.
        assert_eq!(depth.start(9, 0, 8192).waves, 2);
        Ok(())
    }

    #[test]
    fn evidence_persists_and_a_matching_key_skips_calibration() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = || CalibrationStore::new(dir.path(), "v9.9.9", key(dir.path()));
        let depth = PrefillDepth::new(4, 4, 4096, false)?;
        depth.set_store(Some(store()?));
        let calls = depth.calibration();
        assert_eq!(calls, 4);
        calibrate(&depth, calls);
        // Calibration wrote the file atomically: no temporary left behind.
        let files: Vec<_> = std::fs::read_dir(dir.path().join("prefill-depth"))?.map(|e| e.unwrap().file_name()).collect();
        assert_eq!(files.len(), 1, "{files:?}");
        assert!(files[0].to_string_lossy().ends_with(".json"));
        drop(depth);
        // A matching key loads: no calibration, the same depth and ratios.
        let loaded = PrefillDepth::new(4, 4, 4096, false)?;
        loaded.set_store(Some(store()?));
        assert_eq!(loaded.calibration(), 0);
        assert_eq!(loaded.current(), (3, 2752));
        assert!(loaded.control.borrow().calibrated());
        Ok(())
    }

    fn calibrations(store: CalibrationStore) -> anyhow::Result<usize> {
        let depth = PrefillDepth::new(3, 4, 4096, false)?;
        depth.set_store(Some(store));
        Ok(depth.calibration())
    }

    #[test]
    fn every_kind_of_key_mismatch_or_a_corrupt_file_recalibrates() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let depth = PrefillDepth::new(3, 4, 4096, false)?;
        depth.set_store(Some(CalibrationStore::new(dir.path(), "v1", key(dir.path()))?));
        let calls = depth.calibration();
        calibrate(&depth, calls);
        drop(depth);
        assert_eq!(calibrations(CalibrationStore::new(dir.path(), "v1", key(dir.path()))?)?, 0);
        let edits: [fn(&mut CalibrationKey<'_>); 8] = [
            |k| k.family = "kimi_k3", |k| k.quant = "nvfp4".into(), |k| k.gpus[0].0 = "GPU-b".into(),
            |k| k.gpus[0].1 = 170, |k| k.sparks.push("10.55.0.2:9000".into()), |k| k.topology = "tp2".into(),
            |k| k.unit_cap = 2752, |k| k.candidates = vec![3, 4],
        ];
        for (index, edit) in edits.iter().enumerate() {
            let mut changed = key(dir.path());
            edit(&mut changed);
            assert_eq!(calibrations(CalibrationStore::new(dir.path(), "v1", changed)?)?, 4, "edit {index}");
        }
        // A different build (release version) recalibrates too.
        assert_eq!(calibrations(CalibrationStore::new(dir.path(), "v2", key(dir.path()))?)?, 4);
        // The same key with the file in the same place, but corrupt or with
        // a mismatched key inside: recalibrates, and the next save repairs it.
        let store = CalibrationStore::new(dir.path(), "v1", key(dir.path()))?;
        std::fs::write(&store.path, b"{not json")?;
        let fresh = PrefillDepth::new(3, 4, 4096, false)?;
        fresh.set_store(Some(store.clone()));
        assert_eq!(fresh.calibration(), 4);
        calibrate(&fresh, 4);
        assert!(store.load().is_ok());
        let mut persisted: serde_json::Value = serde_json::from_slice(&std::fs::read(&store.path)?)?;
        persisted["key"]["gpus"] = serde_json::json!([]);
        std::fs::write(&store.path, serde_json::to_vec(&persisted)?)?;
        assert_eq!(calibrations(store.clone())?, 4);
        persisted["key"] = store.key.clone();
        persisted["evidence"]["waves"] = serde_json::json!(1);
        std::fs::write(&store.path, serde_json::to_vec(&persisted)?)?;
        assert_eq!(calibrations(store)?, 4);
        Ok(())
    }

    #[test]
    fn wip_and_dev_builds_always_calibrate() -> anyhow::Result<()> {
        // No release version (WIP slots, dev images, benches on them): no store.
        if std::env::var("CUTEAFD_RELEASE_VERSION").is_err() {
            let dir = tempfile::tempdir()?;
            assert!(CalibrationStore::for_release(key(dir.path()))?.is_none());
            assert_eq!(PrefillDepth::new(3, 4, 4096, false)?.calibration(), 4);
        }
        Ok(())
    }
}
