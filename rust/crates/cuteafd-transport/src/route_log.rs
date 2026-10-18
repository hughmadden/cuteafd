//! Opt-in expert route log (`PROFILE=expert-routes`, #408): one JSONL record
//! per Spark wave the coordinator posts, i.e. per (step, MoE layer), with the
//! top-k expert ids of every row.
//!
//! - **Where:** [`crate::expert::SparkExperts`] posts every Spark family's
//!   expert requests (serial waves, prefill lanes and the device lane), and
//!   the request already carries the routes in host memory. The log reads them
//!   there: no device copy, no sync, no extra host hop.
//! - **Off by default:** `CUTEAFD_EXPERT_ROUTES=1` and `CUTEAFD_PROFILE_DIR`
//!   turn it on. Off, a transport holds no [`Tap`] and its post path pays one
//!   branch on `None`; the family hooks ([`step`], [`input_sample`]) pay one
//!   load of an initialized `OnceLock`.
//! - **Serving only:** records start at readiness ([`mark_serving`], called by
//!   the shared readiness hook), so start-up bootstraps, warm-up waves and
//!   graph capture (synthetic routes) never enter the log.
//! - **Off the hot path:** the tap copies the ids into a message for a writer
//!   thread, which serializes and appends to
//!   `PROFILE_DIR/expert-routes-HOST.jsonl` and flushes whenever its queue
//!   drains (a stopped container keeps what it logged).
//!
//! Record (`"event": "expert_routes"`):
//! - `transport`, `pass`: the posting transport (a process-unique number) and
//!   its layer pass (a new pass starts when a wave's layer is not above the
//!   previous wave's on that transport).
//! - `step`, `requests`: when the family declared the serving step
//!   ([`step`]): a process-wide step number and `[[request_id, rows], ...]`.
//! - `kind` (`decode`, `prefill`, `mtp_verify`, `benchmark`), `layer` (the
//!   wire's layer id; the model layer index for GLM), `rows`, `ranks`.
//! - Waves of at most [`ROW_DETAIL_LIMIT`] rows: `topk` (`[[expert, ...] per
//!   row]`) and `weights` (the gate weights, same shape). Wider (prefill)
//!   waves: `expert_rows` (`[[expert, rows], ...]`, ascending expert id),
//!   which bounds a 4096-row record to the expert count.
//!
//! `CUTEAFD_EXPERT_INPUT_SAMPLE=N` (default 64 with the log on, 0 disables)
//! also lets a family save the MoE input rows of the first N serving
//! layer-steps ([`input_sample`]) as `expert-input-HOST-NNNNN.safetensors`.
use crate::{ExpertProtocolV2Request, ExpertV2SourceKind};
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};

/// Waves of up to this many rows keep every row's top-k (decode and verify);
/// wider waves keep the rows per expert.
pub const ROW_DETAIL_LIMIT: usize = 64;
/// Rows of one input sample (a wider wave contributes its first rows).
pub const SAMPLE_ROWS: usize = 64;
const DEFAULT_SAMPLE_STEPS: u64 = 64;
const ID_MASK: u32 = 511;

/// The route log's settings, from the environment once per process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub directory: PathBuf,
    pub host: String,
    /// Layer-steps whose MoE inputs are sampled (0: none).
    pub sample_steps: u64,
}

impl Config {
    /// `None` unless `CUTEAFD_EXPERT_ROUTES=1` (exactly). Fails when it is on
    /// without a usable profile directory or with a bad sample count.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        if get("CUTEAFD_EXPERT_ROUTES").as_deref() != Some("1") {
            return Ok(None);
        }
        let directory = get("CUTEAFD_PROFILE_DIR").filter(|d| !d.is_empty())
            .context("CUTEAFD_EXPERT_ROUTES=1 needs CUTEAFD_PROFILE_DIR (PROFILE_DIR)")?;
        let host = get("CUTEAFD_PROFILE_HOST").or_else(|| get("HOSTNAME")).unwrap_or_default();
        let sample_steps = match get("CUTEAFD_EXPERT_INPUT_SAMPLE").filter(|v| !v.is_empty()) {
            None => DEFAULT_SAMPLE_STEPS,
            Some(value) => value.parse().ok().filter(|&n: &u64| n <= 1_000_000)
                .with_context(|| format!("CUTEAFD_EXPERT_INPUT_SAMPLE must be 0..1000000 layer-steps, not {value:?}"))?,
        };
        Ok(Some(Self { directory: directory.into(), host: safe_name(&host), sample_steps }))
    }
}

/// A host name that cannot leave the profile directory.
fn safe_name(host: &str) -> String {
    let name: String = host.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if name.is_empty() { "unknown".into() } else { name }
}

/// One wave, as the tap hands it to the writer.
#[derive(Debug, Clone, PartialEq)]
pub struct Wave {
    pub unix_ns: u64,
    pub transport: u64,
    pub pass: u64,
    pub step: Option<StepContext>,
    pub kind: ExpertV2SourceKind,
    pub layer: u32,
    pub rows: u32,
    pub ranks: u32,
    pub routes: Routes,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Routes {
    /// Every row's expert ids and gate weights (`[rows][topk]`, row-major).
    Rows { topk: Vec<Vec<u32>>, weights: Vec<Vec<f32>> },
    /// `(expert, rows)` in ascending expert order.
    PerExpert(Vec<(u32, u32)>),
}

/// The serving step a family declared, with its rows' requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepContext {
    pub step: u64,
    pub requests: Vec<(u64, u32)>,
}

impl Wave {
    /// The wave's routes, read from `request` (owner bits of native group
    /// requests are masked off).
    pub fn from_request(request: &ExpertProtocolV2Request, transport: u64, pass: u64, ranks: usize,
        step: Option<StepContext>) -> Self {
        let ids = |slot: &crate::ExpertProtocolV2RowDescriptor| {
            let span = slot.route_offset as usize..(slot.route_offset + slot.route_count) as usize;
            request.routes.get(span).unwrap_or(&[])
        };
        let routes = if request.rows.len() <= ROW_DETAIL_LIMIT {
            Routes::Rows {
                topk: request.rows.iter().map(|row| ids(row).iter().map(|r| r.expert_id & ID_MASK).collect()).collect(),
                weights: request.rows.iter().map(|row| ids(row).iter().map(|r| r.gate_weight).collect()).collect(),
            }
        } else {
            let mut counts = std::collections::BTreeMap::<u32, u32>::new();
            for route in &request.routes {
                *counts.entry(route.expert_id & ID_MASK).or_default() += 1;
            }
            Routes::PerExpert(counts.into_iter().collect())
        };
        let unix_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64);
        Self { unix_ns, transport, pass, step, kind: request.rows.first().map_or(ExpertV2SourceKind::Decode, |r| r.source_kind),
            layer: request.header.layer_id, rows: request.header.row_count, ranks: ranks as u32, routes }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut record = serde_json::json!({
            "event": "expert_routes", "unix_ns": self.unix_ns, "transport": self.transport, "pass": self.pass,
            "kind": kind_name(self.kind), "layer": self.layer, "rows": self.rows, "ranks": self.ranks,
        });
        let object = record.as_object_mut().expect("record is an object");
        if let Some(step) = &self.step {
            object.insert("step".into(), step.step.into());
            object.insert("requests".into(), step.requests.iter().map(|&(id, rows)| serde_json::json!([id, rows]))
                .collect());
        }
        match &self.routes {
            Routes::Rows { topk, weights } => {
                object.insert("topk".into(), serde_json::json!(topk));
                object.insert("weights".into(), serde_json::json!(weights));
            }
            Routes::PerExpert(counts) => {
                object.insert("expert_rows".into(), counts.iter().map(|&(e, n)| serde_json::json!([e, n])).collect());
            }
        }
        record
    }
}

pub fn kind_name(kind: ExpertV2SourceKind) -> &'static str {
    match kind {
        ExpertV2SourceKind::Decode => "decode",
        ExpertV2SourceKind::Prefill => "prefill",
        ExpertV2SourceKind::MtpVerify => "mtp_verify",
        ExpertV2SourceKind::Benchmark => "benchmark",
    }
}

/// One sampled layer-step's MoE input: `rows` BF16 rows of `hidden`, with
/// their top-k ids and gate weights.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSample {
    pub ordinal: u64,
    pub layer: u32,
    pub kind: ExpertV2SourceKind,
    /// Rows of the wave (the sample keeps at most [`SAMPLE_ROWS`]).
    pub wave_rows: u32,
    pub hidden: u32,
    pub topk: u32,
    pub x_bf16: Vec<u8>,
    pub ids: Vec<i32>,
    pub weights: Vec<f32>,
    pub step: Option<StepContext>,
}

impl InputSample {
    pub fn rows(&self) -> usize { self.ids.len() / self.topk.max(1) as usize }

    /// The sample as a safetensors file: `x` BF16 `[rows, hidden]`,
    /// `topk_ids` I32 and `topk_weights` F32 `[rows, topk]`, and string
    /// metadata (layer, kind, wave_rows, ordinal, step, requests).
    pub fn to_safetensors(&self) -> Result<Vec<u8>> {
        let rows = self.rows();
        anyhow::ensure!(self.x_bf16.len() == rows * self.hidden as usize * 2 && self.weights.len() == self.ids.len(),
            "input sample shapes disagree");
        let mut metadata = serde_json::Map::new();
        for (key, value) in [("layer", self.layer.to_string()), ("kind", kind_name(self.kind).to_string()),
            ("wave_rows", self.wave_rows.to_string()), ("ordinal", self.ordinal.to_string()),
            ("format", "cuteafd-expert-input-v1".to_string())] {
            metadata.insert(key.into(), value.into());
        }
        if let Some(step) = &self.step {
            metadata.insert("step".into(), step.step.to_string().into());
            metadata.insert("requests".into(), serde_json::to_string(&step.requests)?.into());
        }
        let ids: Vec<u8> = self.ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weights: Vec<u8> = self.weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors: [(&str, &str, Vec<usize>, &[u8]); 3] = [
            ("x", "BF16", vec![rows, self.hidden as usize], &self.x_bf16),
            ("topk_ids", "I32", vec![rows, self.topk as usize], &ids),
            ("topk_weights", "F32", vec![rows, self.topk as usize], &weights),
        ];
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata.into());
        let mut offset = 0usize;
        for (name, dtype, shape, data) in &tensors {
            header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape,
                "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let mut text = serde_json::to_vec(&serde_json::Value::Object(header))?;
        text.resize(text.len().next_multiple_of(8), b' ');
        let mut out = Vec::with_capacity(8 + text.len() + offset);
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(&text);
        for (_, _, _, data) in &tensors {
            out.extend_from_slice(data);
        }
        Ok(out)
    }
}

enum Message {
    Wave(Box<Wave>),
    Sample(Box<InputSample>),
}

/// The process's route log (present only when configured).
pub struct Log {
    config: Config,
    sender: Mutex<mpsc::Sender<Message>>,
    transports: AtomicU64,
    samples: AtomicU64,
    step: Mutex<Option<StepContext>>,
    steps: AtomicU64,
}

static SERVING: AtomicBool = AtomicBool::new(false);

/// The process's log, from the environment on first use.
pub fn global() -> Option<&'static Log> {
    static LOG: OnceLock<Option<Log>> = OnceLock::new();
    LOG.get_or_init(|| match Config::from_vars(|name| std::env::var(name).ok()) {
        Ok(None) => None,
        Ok(Some(config)) => match Log::start(config) {
            Ok(log) => Some(log),
            Err(error) => { tracing::error!("expert route log disabled: {error:#}"); None }
        },
        Err(error) => { tracing::error!("expert route log disabled: {error:#}"); None }
    }).as_ref()
}

/// Readiness: records start now (start-up waves carry synthetic routes).
pub fn mark_serving() {
    if global().is_some() {
        SERVING.store(true, Ordering::Release);
        tracing::info!("expert route log: recording serving waves");
    }
}

/// A family declares the serving step its next waves belong to; `requests`
/// (`(request id, rows)` in row order) is evaluated only while logging.
pub fn step(requests: impl FnOnce() -> Vec<(u64, u32)>) {
    if let Some(log) = global().filter(|_| SERVING.load(Ordering::Acquire)) {
        let step = log.steps.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut current) = log.step.lock() {
            *current = Some(StepContext { step, requests: requests() });
        }
    }
}

/// The next input sample's ordinal when this layer-step should be sampled,
/// else `None` (log off, not serving yet, or the cap reached).
pub fn input_sample() -> Option<u64> {
    let log = global().filter(|_| SERVING.load(Ordering::Acquire))?;
    let cap = log.config.sample_steps;
    if cap == 0 || log.samples.load(Ordering::Relaxed) >= cap { return None; }
    let ordinal = log.samples.fetch_add(1, Ordering::Relaxed);
    (ordinal < cap).then_some(ordinal)
}

/// Hands a sampled input to the writer (its `step` is filled in here).
pub fn save_sample(mut sample: InputSample) {
    if let Some(log) = global() {
        sample.step = log.current_step();
        log.send(Message::Sample(Box::new(sample)));
    }
}

impl Log {
    fn start(config: Config) -> Result<Self> {
        std::fs::create_dir_all(&config.directory)
            .with_context(|| format!("creating {}", config.directory.display()))?;
        let path = config.directory.join(format!("expert-routes-{}.jsonl", config.host));
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let (sender, receiver) = mpsc::channel();
        let (directory, host) = (config.directory.clone(), config.host.clone());
        std::thread::Builder::new().name("expert-route-log".into())
            .spawn(move || write_loop(receiver, std::io::BufWriter::with_capacity(1 << 20, file), &directory, &host))?;
        tracing::info!(path = %path.display(), sample_steps = config.sample_steps, "expert route log on");
        Ok(Self { config, sender: Mutex::new(sender), transports: AtomicU64::new(0), samples: AtomicU64::new(0),
            step: Mutex::new(None), steps: AtomicU64::new(0) })
    }

    fn current_step(&self) -> Option<StepContext> {
        self.step.lock().ok().and_then(|s| s.clone())
    }

    fn send(&self, message: Message) {
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(message);
        }
    }

    /// A tap for a new transport.
    pub fn tap(&'static self) -> Tap {
        Tap { log: self, transport: self.transports.fetch_add(1, Ordering::Relaxed), pass: 0, last_layer: None }
    }
}

/// One transport's view of the log: it numbers the transport's layer passes.
pub struct Tap {
    log: &'static Log,
    transport: u64,
    pass: u64,
    last_layer: Option<u32>,
}

impl Tap {
    /// A tap when the log is on, else `None` (the transport then never calls it).
    pub fn from_global() -> Option<Self> {
        global().map(Log::tap)
    }

    /// Records a posted wave of `ranks` ranks (serving waves only).
    pub fn record(&mut self, request: &ExpertProtocolV2Request, ranks: usize) {
        if !SERVING.load(Ordering::Acquire) { return; }
        let layer = request.header.layer_id;
        if self.last_layer.is_some_and(|last| layer <= last) { self.pass += 1; }
        self.last_layer = Some(layer);
        let wave = Wave::from_request(request, self.transport, self.pass, ranks, self.log.current_step());
        self.log.send(Message::Wave(Box::new(wave)));
    }
}

fn write_loop(receiver: mpsc::Receiver<Message>, mut out: std::io::BufWriter<std::fs::File>, directory: &Path,
    host: &str) {
    let mut failed = false;
    let handle = |message: Message, out: &mut std::io::BufWriter<std::fs::File>| -> Result<()> {
        match message {
            Message::Wave(wave) => {
                serde_json::to_writer(&mut *out, &wave.to_json())?;
                out.write_all(b"\n")?;
            }
            Message::Sample(sample) => {
                let path = directory.join(format!("expert-input-{host}-{:05}.safetensors", sample.ordinal));
                std::fs::write(&path, sample.to_safetensors()?).with_context(|| format!("writing {}", path.display()))?;
            }
        }
        Ok(())
    };
    while let Ok(message) = receiver.recv() {
        let mut next = Some(message);
        while let Some(message) = next.take() {
            if let Err(error) = handle(message, &mut out) {
                if !failed { tracing::error!("expert route log: {error:#}"); }
                failed = true;
            }
            next = receiver.try_recv().ok();
        }
        if let Err(error) = out.flush() {
            if !failed { tracing::error!("expert route log flush: {error:#}"); }
            failed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype};

    fn request(rows: usize, topk: usize, layer: u32, kind: ExpertV2SourceKind) -> ExpertProtocolV2Request {
        let routes = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: ((i * 7) % 256) as u32, gate_weight: 0.125 * (i % topk) as f32,
        }).collect();
        ExpertProtocolV2Request::new(9, 17, layer, 32, ExpertV2Dtype::Bf16,
            (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor { row_id: u64::from(row), source_kind: kind,
                source_request_id: 1, token_position: u64::from(row), route_offset: row * topk as u32,
                route_count: topk as u32 }).collect(),
            routes, vec![0; rows * 64]).unwrap()
    }

    fn vars<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
    }

    #[test]
    fn off_unless_exactly_one() {
        for value in [None, Some(""), Some("0"), Some("true"), Some("on")] {
            let pairs: Vec<(&str, &str)> = value.map(|v| ("CUTEAFD_EXPERT_ROUTES", v)).into_iter()
                .chain([("CUTEAFD_PROFILE_DIR", "/p")]).collect();
            assert_eq!(Config::from_vars(vars(&pairs)).unwrap(), None, "{value:?}");
        }
    }

    #[test]
    fn on_needs_a_profile_directory_and_parses_the_sample_cap() {
        assert!(Config::from_vars(vars(&[("CUTEAFD_EXPERT_ROUTES", "1")])).is_err());
        let config = Config::from_vars(vars(&[("CUTEAFD_EXPERT_ROUTES", "1"), ("CUTEAFD_PROFILE_DIR", "/p"),
            ("CUTEAFD_PROFILE_HOST", "../raptor")])).unwrap().unwrap();
        assert_eq!(config, Config { directory: "/p".into(), host: "___raptor".into(), sample_steps: 64 });
        let zero = Config::from_vars(vars(&[("CUTEAFD_EXPERT_ROUTES", "1"), ("CUTEAFD_PROFILE_DIR", "/p"),
            ("CUTEAFD_EXPERT_INPUT_SAMPLE", "0")])).unwrap().unwrap();
        assert_eq!(zero.sample_steps, 0);
        for bad in ["-1", "x", "2000000"] {
            assert!(Config::from_vars(vars(&[("CUTEAFD_EXPERT_ROUTES", "1"), ("CUTEAFD_PROFILE_DIR", "/p"),
                ("CUTEAFD_EXPERT_INPUT_SAMPLE", bad)])).is_err(), "{bad}");
        }
    }

    #[test]
    fn narrow_waves_keep_every_rows_topk_and_weights() {
        let wave = Wave::from_request(&request(3, 8, 40, ExpertV2SourceKind::Decode), 2, 5, 6,
            Some(StepContext { step: 11, requests: vec![(4, 2), (7, 1)] }));
        let json = wave.to_json();
        assert_eq!(json["event"], "expert_routes");
        assert_eq!((json["layer"].as_u64(), json["rows"].as_u64(), json["ranks"].as_u64()), (Some(40), Some(3), Some(6)));
        assert_eq!((json["transport"].as_u64(), json["pass"].as_u64(), json["step"].as_u64()), (Some(2), Some(5), Some(11)));
        assert_eq!(json["kind"], "decode");
        assert_eq!(json["requests"], serde_json::json!([[4, 2], [7, 1]]));
        let topk: Vec<Vec<u32>> = serde_json::from_value(json["topk"].clone()).unwrap();
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[1], (8..16).map(|i| ((i * 7) % 256) as u32).collect::<Vec<_>>());
        let weights: Vec<Vec<f32>> = serde_json::from_value(json["weights"].clone()).unwrap();
        assert_eq!(weights[2][3], 0.375);
        assert!(json.get("expert_rows").is_none());
    }

    #[test]
    fn wide_waves_keep_rows_per_expert() {
        let wave = Wave::from_request(&request(ROW_DETAIL_LIMIT + 1, 8, 3, ExpertV2SourceKind::Prefill), 0, 0, 4, None);
        let json = wave.to_json();
        assert_eq!(json["kind"], "prefill");
        assert!(json.get("topk").is_none() && json.get("step").is_none() && json.get("requests").is_none());
        let counts: Vec<(u32, u32)> = serde_json::from_value(json["expert_rows"].clone()).unwrap();
        assert_eq!(counts.iter().map(|c| c.1).sum::<u32>(), (ROW_DETAIL_LIMIT as u32 + 1) * 8);
        assert!(counts.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn owner_bits_are_masked_off() {
        let mut request = request(1, 2, 4, ExpertV2SourceKind::Decode);
        request.routes[0].expert_id = 383 | (1 << 10);
        let Routes::Rows { topk, .. } = Wave::from_request(&request, 0, 0, 4, None).routes else { panic!() };
        assert_eq!(topk[0][0], 383);
    }

    #[test]
    fn input_sample_is_a_valid_safetensors_file() {
        let sample = InputSample { ordinal: 3, layer: 40, kind: ExpertV2SourceKind::Decode, wave_rows: 5, hidden: 4,
            topk: 2, x_bf16: (0..2 * 4 * 2).map(|b| b as u8).collect(), ids: vec![1, 2, 3, 4],
            weights: vec![0.5, 0.25, 1.0, 2.0], step: Some(StepContext { step: 9, requests: vec![(1, 2)] }) };
        let bytes = sample.to_safetensors().unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        assert_eq!(header["__metadata__"]["layer"], "40");
        assert_eq!(header["__metadata__"]["step"], "9");
        assert_eq!(header["__metadata__"]["requests"], "[[1,2]]");
        assert_eq!(header["x"]["shape"], serde_json::json!([2, 4]));
        assert_eq!(header["topk_ids"]["dtype"], "I32");
        let data = &bytes[8 + header_len..];
        assert_eq!(data.len(), 16 + 16 + 16);
        let ids = header["topk_ids"]["data_offsets"].as_array().unwrap();
        let start = ids[0].as_u64().unwrap() as usize;
        assert_eq!(i32::from_le_bytes(data[start + 12..start + 16].try_into().unwrap()), 4);
        let bad = InputSample { ids: vec![1, 2, 3], ..sample };
        assert!(bad.to_safetensors().is_err());
    }

    #[test]
    fn writer_appends_records_and_samples_then_flushes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("expert-routes-h.jsonl");
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let (sender, receiver) = mpsc::channel();
        let directory = dir.path().to_path_buf();
        let writer = std::thread::spawn(move || write_loop(receiver, std::io::BufWriter::new(file), &directory, "h"));
        for layer in [3, 4, 3] {
            sender.send(Message::Wave(Box::new(Wave::from_request(&request(2, 8, layer, ExpertV2SourceKind::Decode),
                0, 0, 4, None)))).unwrap();
        }
        sender.send(Message::Sample(Box::new(InputSample { ordinal: 0, layer: 3, kind: ExpertV2SourceKind::Decode,
            wave_rows: 1, hidden: 2, topk: 1, x_bf16: vec![0; 4], ids: vec![5], weights: vec![1.0], step: None })))
            .unwrap();
        drop(sender);
        writer.join().unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path).unwrap().lines()
            .map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.iter().map(|l| l["layer"].as_u64().unwrap()).collect::<Vec<_>>(), [3, 4, 3]);
        assert!(dir.path().join("expert-input-h-00000.safetensors").is_file());
    }

    #[test]
    fn tap_numbers_layer_passes_and_records_only_while_serving() {
        let dir = tempfile::tempdir().unwrap();
        let log: &'static Log = Box::leak(Box::new(Log::start(Config { directory: dir.path().into(), host: "t".into(),
            sample_steps: 0 }).unwrap()));
        let mut tap = log.tap();
        assert_eq!(tap.transport, 0);
        assert_eq!(log.tap().transport, 1);
        // SERVING is process-wide; this is the only test that sets it.
        tap.record(&request(1, 8, 3, ExpertV2SourceKind::Decode), 4);
        assert_eq!((tap.pass, tap.last_layer), (0, None), "start-up waves are not recorded");
        SERVING.store(true, Ordering::Release);
        for (layer, pass) in [(3, 0), (4, 0), (77, 0), (3, 1), (3, 2), (5, 2)] {
            tap.record(&request(1, 8, layer, ExpertV2SourceKind::Decode), 4);
            assert_eq!(tap.pass, pass, "layer {layer}");
        }
    }

    #[test]
    fn transports_have_no_tap_when_the_log_is_off() {
        // The test process never sets CUTEAFD_EXPERT_ROUTES.
        assert!(std::env::var_os("CUTEAFD_EXPERT_ROUTES").is_none());
        assert!(global().is_none() && Tap::from_global().is_none() && input_sample().is_none());
        let mut called = false;
        step(|| { called = true; Vec::new() });
        assert!(!called, "the step closure runs only while logging");
    }
}
