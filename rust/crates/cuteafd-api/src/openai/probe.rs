//! Benchmark probes: per-request diagnostics the in-server benchmark attaches
//! to its own loopback requests.
//!
//! The benchmark registers a [`ProbeSpec`] in the process-wide [`registry`] and
//! sends the request with the `x-cuteafd-probe: <id>` header. The chat handler
//! claims the probe (an id is good for one request) and hands it to the engine
//! in [`super::NativeRequest::probe`]; the engine honours the switches and
//! records what was asked for into the shared [`ProbeRecord`], which the
//! benchmark reads once the response is complete. Unknown ids are ignored, so
//! the header does nothing for an ordinary client.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The request header naming a registered probe.
pub const HEADER: &str = "x-cuteafd-probe";

/// What the engine should do differently for one request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeSpec {
    /// No prefix-cache lookup and nothing retained: a cold prefill.
    #[serde(default)]
    pub cold: bool,
    /// Decode one token per step (no drafts of any kind).
    #[serde(default)]
    pub no_speculation: bool,
    /// Token ids to run instead of tokenizing the rendered prompt.
    #[serde(default)]
    pub prompt_ids: Option<Vec<u32>>,
    /// Already-expanded native image spans, verified against prepared sources by the engine.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ProbeMedia>,
    /// Teacher-forced scoring: run the prompt and record the logits row
    /// predicting every prompt token from this index on; the request then
    /// ends without generating.
    #[serde(default)]
    pub score_from: Option<usize>,
    /// Record the row the first generated token is selected from.
    #[serde(default)]
    pub record_first: bool,
    /// Record the rows the first N generated tokens are selected from (the
    /// first from the prefill or a retained row, the rest from decode steps).
    #[serde(default)]
    pub record_rows: usize,
    /// Top entries kept per recorded row.
    #[serde(default)]
    pub top_k: usize,
    /// Per predicted position: token ids whose log-probability to report
    /// (a reference's top-k, so KL can be estimated against it).
    #[serde(default)]
    pub want: HashMap<usize, Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeFixture {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProbeMedia {
    pub start: usize,
    pub len: usize,
    pub kind: String,
    pub key: String,
    pub grid: [u32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture: Option<ProbeFixture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ProbeImageUrl>,
}

pub fn sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl ProbeSpec {
    pub fn validate_media(&self) -> anyhow::Result<()> {
        if self.media.is_empty() { return Ok(()); }
        let tokens = self.prompt_ids.as_ref().ok_or_else(|| anyhow::anyhow!("probe media requires prompt_ids"))?;
        anyhow::ensure!(self.media.len() <= 128, "probe media exceeds history limit");
        let mut previous = 0;
        for span in &self.media {
            let end = span.start.checked_add(span.len).ok_or_else(|| anyhow::anyhow!("probe media extent overflow"))?;
            let [t, h, w] = span.grid;
            anyhow::ensure!(span.kind == "image" && span.len > 0 && span.start >= previous && end <= tokens.len(),
                "probe media spans must be sorted, disjoint and inside prompt_ids");
            anyhow::ensure!(sha256_hex(&span.key) && t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0
                && u64::from(h) * u64::from(w) / 4 == span.len as u64, "invalid probe media identity/grid");
            let source = span.image_url.as_ref().ok_or_else(|| anyhow::anyhow!("probe media image_url required"))?;
            anyhow::ensure!(!source.url.is_empty() && source.detail.as_deref().is_none_or(|v| matches!(v, "auto" | "high" | "low")),
                "invalid probe image source/detail");
            if let Some(fixture) = &span.fixture {
                anyhow::ensure!(sha256_hex(&fixture.sha256) && !fixture.path.is_empty() && !fixture.path.contains('\\')
                    && !std::path::Path::new(&fixture.path).is_absolute()
                    && fixture.path.split('/').all(|v| !v.is_empty() && v != "." && v != ".."), "invalid probe fixture identity");
            }
            previous = end;
        }
        Ok(())
    }
}

/// One recorded logits row (log-softmax over the full vocabulary).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProbeRow {
    /// Index of the token this row predicts (the prompt length for the first
    /// generated token).
    pub position: usize,
    /// FNV-1a 64 of the raw f32 row bytes: equal rows are byte-identical.
    pub hash: String,
    pub argmax: u32,
    /// The row's `top_k` entries, most likely first.
    pub top: Vec<(u32, f32)>,
    /// Log-probabilities of the requested ids at this position.
    pub wanted: Vec<(u32, f32)>,
    /// Whether every logit was finite.
    pub finite: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeRecord {
    /// Whether an engine honoured the probe at all.
    pub engine: Option<String>,
    pub prompt_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<ProbeMedia>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
    pub cached_tokens: usize,
    pub rows: Vec<ProbeRow>,
    pub generated: Vec<u32>,
    /// Speculation actually skipped / cache actually bypassed, as the engine saw it.
    pub cold: bool,
    pub no_speculation: bool,
    pub scored: usize,
    pub error: Option<String>,
}

/// A probe shared by the benchmark and the engine serving its request.
#[derive(Debug)]
pub struct Probe {
    pub spec: ProbeSpec,
    record: Mutex<ProbeRecord>,
}

impl Probe {
    pub fn new(spec: ProbeSpec) -> Arc<Self> {
        Arc::new(Self { spec, record: Mutex::new(ProbeRecord::default()) })
    }

    fn with(&self, f: impl FnOnce(&mut ProbeRecord)) {
        if let Ok(mut record) = self.record.lock() {
            f(&mut record);
        }
    }

    /// The engine took the request: its name, the prompt ids it runs and the
    /// rows restored from the prefix cache.
    pub fn admitted(&self, engine: &str, prompt_ids: &[u32], cached_tokens: usize) {
        self.with(|r| {
            r.engine = Some(engine.to_owned());
            r.prompt_ids = prompt_ids.to_vec();
            r.cached_tokens = cached_tokens;
            r.cold = self.spec.cold;
            r.no_speculation = self.spec.no_speculation;
        });
    }

    /// Echo verified descriptors only, never the image's data URL.
    pub fn media(&self, mut media: Vec<ProbeMedia>) {
        for span in &mut media { span.image_url = None; }
        self.with(|r| r.media = media);
    }

    pub fn provenance(&self, value: serde_json::Value) {
        self.with(|r| r.provenance = Some(value));
    }

    /// Records one host logits row predicting token `position`.
    pub fn row(&self, position: usize, logits: &[f32]) {
        let want = self.spec.want.get(&position).map(Vec::as_slice).unwrap_or(&[]);
        let row = summarize(position, logits, self.spec.top_k.max(1), want);
        self.with(|r| {
            r.rows.push(row);
            if self.spec.score_from.is_some_and(|from| position >= from) {
                r.scored += 1;
            }
        });
    }

    /// Records a generated token.
    pub fn token(&self, token: u32) {
        self.with(|r| r.generated.push(token));
    }

    pub fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        self.with(|r| { r.error.get_or_insert(message); });
    }

    pub fn record(&self) -> ProbeRecord {
        self.record.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// The scoring request's rows: positions `from..len` of the prompt.
    pub fn scoring(&self) -> Option<usize> {
        self.spec.score_from
    }
}

/// FNV-1a 64 over bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Log-softmax summary of one logits row.
pub fn summarize(position: usize, logits: &[f32], top_k: usize, want: &[u32]) -> ProbeRow {
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let hash = format!("{:016x}", fnv1a(&bytes));
    let finite = logits.iter().all(|v| v.is_finite());
    let max = logits.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
    let lse = f64::from(max) + sum.ln();
    let lp = |v: f32| if v.is_finite() { (f64::from(v) - lse) as f32 } else { f32::NEG_INFINITY };
    // Partial selection of the top entries (ties to the lower id).
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    let k = top_k.min(order.len());
    let key = |&i: &u32| (std::cmp::Reverse(OrderedF32(logits[i as usize])), i);
    if k > 0 && k < order.len() {
        order.select_nth_unstable_by_key(k - 1, key);
        order.truncate(k);
    }
    order.sort_by_key(key);
    let top: Vec<(u32, f32)> = order.iter().map(|&i| (i, lp(logits[i as usize]))).collect();
    let argmax = top.first().map_or(0, |&(i, _)| i);
    let wanted = want.iter().filter(|&&i| (i as usize) < logits.len()).map(|&i| (i, lp(logits[i as usize]))).collect();
    ProbeRow { position, hash, argmax, top, wanted, finite }
}

/// Total order on f32 for selection (NaN lowest).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
        key(self.0).total_cmp(&key(other.0))
    }
}

/// Probes registered by the benchmark and not yet claimed by a request.
#[derive(Default)]
pub struct ProbeRegistry {
    pending: Mutex<HashMap<String, Arc<Probe>>>,
}

impl ProbeRegistry {
    /// Registers `spec`; the returned id goes in the [`HEADER`] of one request.
    pub fn register(&self, spec: ProbeSpec) -> (String, Arc<Probe>) {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let probe = Probe::new(spec);
        if let Ok(mut pending) = self.pending.lock() {
            // A probe whose request never arrived is dropped with its owner.
            pending.retain(|_, p| Arc::strong_count(p) > 1);
            pending.insert(id.clone(), probe.clone());
        }
        (id, probe)
    }

    /// Takes the probe registered under `id`, if any.
    pub fn claim(&self, id: &str) -> Option<Arc<Probe>> {
        self.pending.lock().ok()?.remove(id)
    }
}

pub fn registry() -> &'static ProbeRegistry {
    static REGISTRY: OnceLock<ProbeRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ProbeRegistry::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_log_softmax_with_ordered_top() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        let row = summarize(7, &logits, 3, &[4, 9]);
        assert_eq!(row.position, 7);
        assert_eq!(row.argmax, 1, "ties go to the lower id");
        assert_eq!(row.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 2]);
        let total: f64 = logits.iter().map(|&v| f64::from(v).exp()).sum();
        let expect = (3.0 - total.ln()) as f32;
        assert!((row.top[0].1 - expect).abs() < 1e-6);
        assert_eq!(row.wanted.len(), 1, "out-of-vocabulary ids are skipped");
        assert!(row.finite);
        assert_eq!(row.hash, summarize(0, &logits, 1, &[]).hash);
        assert_ne!(row.hash, summarize(0, &[1.0, 3.0, 2.0, 3.0, -1.5], 1, &[]).hash);
    }

    #[test]
    fn registry_hands_a_probe_out_once() {
        let (id, probe) = registry().register(ProbeSpec { cold: true, ..ProbeSpec::default() });
        let claimed = registry().claim(&id).expect("registered");
        assert!(Arc::ptr_eq(&probe, &claimed));
        assert!(registry().claim(&id).is_none());
        claimed.admitted("test", &[1, 2, 3], 2);
        claimed.token(5);
        let record = probe.record();
        assert_eq!((record.prompt_ids.len(), record.cached_tokens, record.generated.as_slice(), record.cold),
            (3, 2, &[5u32][..], true));
    }
}
