//! Allocation recipe shared by native serving and manifest-backed planning.
use std::collections::BTreeMap;
use crate::formats::native_dense::RankWeightLayout;
use crate::serving_capacity::{prefill_return_bytes, prefill_waves, PrefillShape};
use crate::families::minimax_m3::{cache::{MiniMaxCacheLayout, PAGE_ROWS}, config::MiniMaxM3Config};
use crate::families::qwen4::Qwen4KvCache;
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("MiniMax admission does not support {0}")]
    Unsupported(&'static str),
    #[error("MiniMax native manifest: {0}")]
    Manifest(String),
    #[error("MiniMax native expert headers: {0}")]
    Checkpoint(String),
    #[error("MiniMax allocation overflow: {0}")]
    Overflow(&'static str),
    #[error("MiniMax allocation needs {needed} bytes, available {available}")]
    Budget { needed: u64, available: u64 },
    #[error(transparent)]
    Cache(#[from] crate::serving_capacity::CacheGeometryError),
}
type Result<T> = std::result::Result<T, AdmissionError>;
fn manifest(message: impl Into<String>) -> AdmissionError { AdmissionError::Manifest(message.into()) }
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |a, b| a.checked_add(b).ok_or(AdmissionError::Overflow("sum")))
}
fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b).ok_or(AdmissionError::Overflow("product")))
}

/// The prefill shape and Spark exchange one coordinator admits (PLAN "Prefill contract"): `lanes`
/// waves of `rows` rows each in flight on one Spark link, which decode shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeSizing {
    /// Rows of one prefill unit (`--prefill-rows`), one wave on the Spark link: the most rows any
    /// program launch takes.
    pub rows: usize,
    pub decode_rows: usize,
    pub sequences: usize,
    pub context: u64,
    pub spark_ranks: usize,
    /// Rows one head launch projects: decode and draft rows, or every prefill
    /// row when full prefill logits are served. Prefill otherwise projects
    /// only the rows whose logits a request needs.
    pub logit_rows: usize,
    /// Prefill waves in flight on the one Spark link (`--prefill-waves`): a call spans up to
    /// `lanes` units, each wave with its own return slot on every rank. The runtime's lane loop
    /// runs them as that many lanes until its prefill moves onto the shared executor.
    pub lanes: usize,
    /// No admission effect: MiniMax serves coordinators whose memory the NIC lands Spark replies
    /// in (RTX PRO 6000), so every wave's return slots are device planes. Kept until the Spark link
    /// stops taking it.
    pub host_landing: bool,
}
impl RuntimeSizing {
    /// The prefill unit shape this sizing admits.
    pub fn shape(self) -> PrefillShape { PrefillShape { unit_rows: self.rows as u64, waves: self.lanes as u64 } }
    /// Rows of the longest prefill call, which the per-row buffers hold: one unit per wave in flight.
    pub fn slot_rows(self) -> usize { self.rows * self.lanes }
    /// [`Self::slot_rows`], the name the runtime's lane loop reads until it moves onto prefill units.
    pub fn chunk_rows(self) -> usize { self.slot_rows() }
    /// Head rows for a server: every prefill row with full prefill logits,
    /// otherwise decode rows or the drafter's block rows, whichever is wider.
    pub fn head_rows(rows: usize, full_prefill_logits: bool, draft_rows: usize) -> usize {
        if full_prefill_logits { rows } else { rows.min(64).max(draft_rows).min(rows) }
    }
    pub fn validate(self, cfg: &MiniMaxM3Config) -> Result<()> {
        // One wave may take any rows a launch holds (golden's windows); several cut the call on
        // whole KV pages, as the shared unit shape requires.
        let shape = if self.lanes == 1 { (1..=4096).contains(&self.rows) } else {
            self.shape().validate().is_ok() && self.rows % PAGE_ROWS as usize == 0 };
        if !shape || !(1..=64).contains(&self.decode_rows)
            || self.sequences == 0 || self.sequences > self.decode_rows || self.sequences > self.rows
            || self.logit_rows < self.decode_rows || self.logit_rows > self.slot_rows() {
            return Err(AdmissionError::Unsupported("row capacity or sequence count"));
        }
        if self.context == 0 || self.context > cfg.max_position as u64 {
            return Err(AdmissionError::Unsupported("context outside the checkpoint extent"));
        }
        if !matches!(self.spark_ranks, 0 | 2 | 3 | 4 | 6) {
            return Err(AdmissionError::Unsupported("expert world outside local, TP2, TP3, TP4 or TP6"));
        }
        cfg.release_geometry().map_err(|e| manifest(e.to_string()))
    }
    /// Execution slots every Spark rank runs (the shared rule, `plan::layout::spark_execution_slots`):
    /// one per prefill wave in flight; decode keeps one request per rank.
    pub fn execution_slots(self) -> u8 {
        crate::plan::layout::spark_execution_slots("minimax_m3", self.lanes as u64, 1)
    }
    /// The one Spark endpoint decode and prefill share: a wave of up to a unit's rows, BF16 in.
    pub fn spark_endpoint(self) -> std::result::Result<cuteafd_core::SparkEndpoint, std::num::TryFromIntError> {
        Ok(cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(self.rows)?,
            ingress: cuteafd_core::SparkIngress::Bf16 })
    }
}

/// The prefill shape a plan or server admits when it names none, after #762: the 8192-row reserve
/// split into `planned_waves(spark_ranks)` units of `unit_rows(k)` (2 x 4096 below six Sparks,
/// 3 x 2752 at six). Explicit waves pin k, and their unit follows k unless the rows are pinned too.
/// A 2752-row wave runs on the expert package's 4096-row programs, which take any rows up to their
/// capacity, so six Sparks need no other export.
pub fn prefill_shape(rows: u64, waves: u64, spark_ranks: usize) -> anyhow::Result<PrefillShape> {
    let waves = if waves > 0 { waves } else { prefill_waves::planned_waves(spark_ranks) };
    let unit_rows = if rows > 0 { rows } else { prefill_waves::unit_rows(waves) };
    PrefillShape { unit_rows, waves }.validate()
}

pub const PROGRAM_NAMES: &[&str] = &[
    "norm", "q_norm_rope", "k_norm_rope", "index_q_norm_rope", "index_k_norm_rope",
    "rope", "scale_prep", "kv_split", "swiglu_i3072", "swiglu_i12288", "router_scores",
    "route", "residual", "index_q_quant", "index_k_pack", "kv_pack",
    "mxfp8_q", "mxfp8_kv", "mxfp8_o", "mxfp8_index_q", "mxfp8_index_k",
    "mxfp8_shared_gate_up", "mxfp8_shared_down", "mxfp8_dense_gate_up", "mxfp8_dense_down",
    "mxfp8_q_decode", "mxfp8_kv_decode", "mxfp8_index_q_decode", "mxfp8_index_k_decode",
    "index_decode", "index_prefill", "attention_decode", "attention_extend",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramSize {
    pub capacity_rows: u64,
    pub scratch_bytes: u64,
}
#[derive(Debug, Clone)]
pub struct ProgramSizes {
    pub programs: BTreeMap<String, ProgramSize>,
    decode_capacity: u64,
    extend_capacity: u64,
    workspace_bytes: u64,
}
impl ProgramSizes {
    pub fn from_manifest(value: &Value, sizing: RuntimeSizing) -> Result<Self> {
        let entries = value["programs"].as_array().ok_or_else(|| manifest("missing programs"))?;
        let mut programs = BTreeMap::new();
        for suffix in PROGRAM_NAMES {
            let name = format!("minimax_{suffix}");
            let matches = entries.iter().filter(|p| p["name"] == name || p["label"] == name).collect::<Vec<_>>();
            if matches.len() != 1 { return Err(manifest(format!("expected one export {name}, found {}", matches.len()))); }
            let entry = matches[0];
            let capacity_rows = entry["capacity_rows"].as_u64().filter(|c| *c > 0)
                .ok_or_else(|| manifest(format!("{name}: missing row capacity")))?;
            // Prefill tiles any chunk through the scorer/extend tile, whatever its rows.
            let required = if suffix.ends_with("_decode") { sizing.decode_rows }
                else if matches!(*suffix, "index_prefill" | "attention_extend") { 1 } else { sizing.rows } as u64;
            if capacity_rows < required { return Err(manifest(format!("{name}: capacity {capacity_rows} is below {required}"))); }
            let scratch = entry["scratch_bytes_at_capacity"].as_object()
                .ok_or_else(|| manifest(format!("{name}: missing scratch inventory")))?;
            let scratch_bytes = sum(scratch.values().map(|v| v.as_u64().ok_or_else(|| manifest(format!("{name}: invalid scratch bytes"))))
                .collect::<Result<Vec<_>>>()?)?;
            if matches!(*suffix, "index_decode" | "index_prefill") {
                let pages = entry["geometry"]["max_pages"].as_u64().ok_or_else(|| manifest(format!("{name}: missing page extent")))?;
                if pages < sizing.context.div_ceil(PAGE_ROWS) { return Err(manifest(format!("{name}: insufficient context extent"))); }
            }
            if matches!(*suffix, "attention_decode" | "attention_extend") {
                let batch = entry["geometry"]["max_batch"].as_u64().ok_or_else(|| manifest(format!("{name}: missing batch capacity")))?;
                if batch < sizing.decode_rows as u64 { return Err(manifest(format!("{name}: insufficient batch capacity"))); }
            }
            programs.insert(name, ProgramSize { capacity_rows, scratch_bytes });
        }
        let decode_capacity = programs["minimax_index_decode"].capacity_rows;
        if decode_capacity > 64 || decode_capacity != programs["minimax_attention_decode"].capacity_rows {
            return Err(manifest("index/attention decode selection strides disagree"));
        }
        let extend_capacity = programs["minimax_index_prefill"].capacity_rows;
        if extend_capacity != programs["minimax_attention_extend"].capacity_rows {
            return Err(manifest("index/attention prefill selection strides disagree"));
        }
        // One stream runs every program, so they share one scratch arena.
        let workspace_bytes = programs.values().map(|p| p.scratch_bytes).max().unwrap_or(0).max(256);
        Ok(Self { programs, decode_capacity, extend_capacity, workspace_bytes })
    }
    pub fn decode_capacity(&self) -> u64 { self.decode_capacity }
    /// Rows one prefill scorer/extend launch takes; prefill tiles larger chunks.
    pub fn extend_capacity(&self) -> u64 { self.extend_capacity }
    pub fn workspace_bytes(&self) -> u64 { self.workspace_bytes }
}

/// Buffers with one row per batch row: (name, elements per row, bytes per
/// element). A prefill unit's view of one starts at its first row.
pub const ROW_BUFFERS: &[(&str, u64, u64)] = &[
    ("residual", 6144, 4), ("embedded", 6144, 2), ("norm", 6144, 2), ("q", 8192, 2), ("q_norm", 8192, 2),
    ("kv", 1024, 2), ("k", 512, 2), ("k_norm", 512, 2), ("v", 512, 2),
    ("index_q", 512, 2), ("index_q_norm", 512, 2), ("index_k", 128, 2), ("index_k_norm", 128, 2),
    ("index_q_fp8", 512, 1), ("index_q_scales", 4, 4), ("attention", 8192, 2),
    ("delta", 6144, 2), ("gate_up", 24576, 2), ("activation", 12288, 2),
    ("shared", 6144, 2), ("routed", 6144, 2), ("router_logits", 128, 4), ("route_ids", 4, 4),
    ("route_weights", 4, 4), ("token_ids", 1, 4), ("positions", 1, 8), ("slots", 1, 8), ("batch_ids", 1, 4),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub name: &'static str,
    pub bytes: u64,
}
#[derive(Debug, Clone)]
pub struct RuntimeLedger {
    pub allocations: Vec<Allocation>,
    pub fixed_bytes: u64,
}
impl RuntimeLedger {
    pub fn new(cfg: &MiniMaxM3Config, sizing: RuntimeSizing, programs: &ProgramSizes) -> Result<Self> {
        sizing.validate(cfg)?;
        let (r, d, h) = (sizing.slot_rows() as u64, programs.decode_capacity(), cfg.hidden as u64);
        if d < sizing.decode_rows as u64 { return Err(manifest("decode capacity changed after validation")); }
        let mut allocations = Vec::new();
        let mut add = |name, dimensions: &[u64]| -> Result<()> {
            allocations.push(Allocation { name, bytes: product(dimensions)?.max(256) }); Ok(())
        };
        for &(name, width, bytes) in ROW_BUFFERS { add(name, &[r, width, bytes])?; }
        add("head_input", &[sizing.logit_rows as u64, h, 2])?;
        add("head_workspace", &[4 << 20])?;
        add("logits", &[sizing.logit_rows as u64, cfg.vocab_size as u64, 4])?;
        // Decode steps (at most `d` rows) give each row its own table; prefill
        // launches one extend tile of at most `t` rows per request piece, each
        // with a 16-byte aligned length, offset pair and causal key ends.
        let (t, stride) = (programs.extend_capacity(), extend_table_stride(sizing.context));
        let launches = extend_launches(sizing, t);
        add("q2k", &[4, d, 16, 4])?;
        add("q2k_extend", &[4, t, 16, 4])?;
        add("lse", &[d.max(t), 64, 4])?;
        add("page_table", &[d.max(launches), stride, 4])?;
        add("cache_lengths", &[d.max(4 * launches), 4])?;
        add("cu_seqlens_q", &[(d + 1).max(4 * launches), 4])?;
        add("index_bounds", &[t + r + 4 * launches, 4])?;
        add("index_keys", &[sizing.context.next_multiple_of(256), 128])?;
        add("index_scales", &[sizing.context.next_multiple_of(256), 4])?;
        add("k_descale", &[d, 4, 4])?;
        add("v_descale", &[d, 4, 4])?;
        add("prefix_tail", &[1])?;
        add("cos", &[sizing.context, 32, 2])?;
        add("sin", &[sizing.context, 32, 2])?;
        add("scratch", &[programs.workspace_bytes()])?;
        add("sampler", &[cuteafd_core::target_sampler_device_bytes(sizing.decode_rows, cfg.vocab_size) as u64])?;
        add("selector", &[sizing.decode_rows as u64, 12])?;
        // The Spark link's return slots: one BF16 device plane of a unit's rows per wave and rank,
        // where the NIC lands every reply.
        add("spark_intake", &[prefill_return_bytes(sizing.lanes as u64, sizing.rows as u64, sizing.spark_ranks as u64, h)])?;
        let fixed_bytes = sum(allocations.iter().map(|a| a.bytes))?;
        Ok(Self { allocations, fixed_bytes })
    }
    pub fn bytes(&self, name: &str) -> Option<u64> { self.allocations.iter().find(|a| a.name == name).map(|a| a.bytes) }
}

/// Page-table entries per extend launch: every page of the admitted context,
/// padded so each launch's row starts 16-byte aligned.
pub fn extend_table_stride(context: u64) -> u64 { context.div_ceil(PAGE_ROWS).next_multiple_of(4) }

/// Device bytes one recorded decode step is charged: a whole step (60 layers
/// with their Spark exchanges and the head, ~3,100 nodes) measured 6-8 MiB on
/// an RTX 5090 (the driver allocates in 2 MiB chunks). Estimated until the
/// startup census replaces it.
pub const DECODE_GRAPH_BYTES: u64 = 8 << 20;

/// The decode steps the coordinator records at startup and replays: one per
/// row count a verify step can have. A drafting sequence verifies its anchor
/// and up to seven drafts, the others one row each, all within the decode tile.
/// Keys are exact row counts, not padded buckets: a padded row would need its
/// K/V and index writes masked and its expert routes withheld from the Sparks,
/// and the recordings cost readiness and memory only once.
pub fn decode_graphs(sizing: RuntimeSizing, draft_rings: usize) -> Result<crate::placement::inventory::StartupGraphs<usize>> {
    if draft_rings > sizing.sequences { return Err(AdmissionError::Unsupported("draft rings beyond the sequences")); }
    let rows = (sizing.sequences + 7 * draft_rings).min(sizing.decode_rows);
    let keys: Vec<usize> = (1..=rows).collect();
    use crate::placement::inventory::{GraphSet, Lifetime};
    GraphSet::new(&[keys.len() as u64], DECODE_GRAPH_BYTES, 10, 16 << 20, keys.len() as u64, Lifetime::Startup)
        .startup(keys).map_err(manifest)
}

/// Most extend launches one prefill step makes: each request's rows split
/// into tiles of at most `tile` rows, and at each unit cut.
pub fn extend_launches(sizing: RuntimeSizing, tile: u64) -> u64 {
    sizing.sequences as u64 + (sizing.slot_rows() as u64).div_ceil(tile) + sizing.lanes as u64 - 1
}

/// Source scales are released after preparation completes; the MMA arrangement
/// replaces them byte-for-byte. Only one source matrix is transient at load.
pub fn scale_prep_staging_bytes(weights: &RankWeightLayout) -> u64 {
    weights.operands.iter().filter(|o| o.name.ends_with(".weight_scale_inv"))
        .map(|o| o.allocation_bytes).max().unwrap_or(0)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Reservations {
    pub weights: u64,
    pub local_experts: u64,
    pub expert_workspace: u64,
    pub media: u64,
    pub pending_code: u64,
    pub runtime_baseline: u64,
    pub headroom: u64,
    /// The startup decode graphs ([`decode_graphs`]), recorded before ready.
    pub graphs: u64,
    /// Additional absolute peak above the same pre-weight sample.
    pub startup: u64,
}
impl Reservations {
    pub fn fixed_bytes(self, ledger: &RuntimeLedger) -> Result<u64> {
        sum([ledger.fixed_bytes, self.weights, self.local_experts,
            self.expert_workspace, self.media, self.pending_code, self.runtime_baseline, self.headroom, self.graphs])
    }
}

#[derive(Debug, Clone)]
pub struct Admission {
    pub cache: MiniMaxCacheLayout,
    pub pool_pages: u64,
    pub pool_tokens: u64,
    pub cache_bytes: u64,
    pub fixed_bytes: u64,
    pub total_bytes: u64,
}
pub fn admit(cfg: &MiniMaxM3Config, sizing: RuntimeSizing, ledger: &RuntimeLedger, reservations: Reservations,
    available: u64, requested_pool: Option<u64>, kv_format: Qwen4KvCache) -> Result<Admission> {
    sizing.validate(cfg)?;
    let cache = MiniMaxCacheLayout::new(cfg, sizing.context, kv_format)?;
    let fixed_bytes = reservations.fixed_bytes(ledger)?;
    let needed = fixed_bytes.max(reservations.startup);
    if needed > available { return Err(AdmissionError::Budget { needed, available }); }
    let page_bytes = cache.page_bytes;
    let fitting = (available - fixed_bytes) / page_bytes;
    let pool_pages = match requested_pool.filter(|n| *n > 0) {
        Some(tokens) if tokens % PAGE_ROWS != 0 => return Err(AdmissionError::Unsupported("pool not aligned to 64-token pages")),
        Some(tokens) => tokens / PAGE_ROWS,
        None => fitting.min(i32::MAX as u64),
    };
    if pool_pages == 0 || pool_pages > i32::MAX as u64 { return Err(AdmissionError::Unsupported("physical page count outside Int32")); }
    let cache_bytes = product(&[pool_pages, page_bytes])?;
    let total_bytes = sum([fixed_bytes, cache_bytes])?;
    if total_bytes > available { return Err(AdmissionError::Budget { needed: total_bytes, available }); }
    Ok(Admission { cache, pool_pages, pool_tokens: pool_pages * PAGE_ROWS, cache_bytes, fixed_bytes, total_bytes })
}

/// The package manifest is selected explicitly; a missing capacity is refused.
pub fn expert_workspace(manifest_value: &Value, world: usize, rows: usize) -> Result<u64> {
    if !matches!(world, 1..=4 | 6) { return Err(AdmissionError::Unsupported("expert package TP degree")); }
    let layout = format!("tp{world}");
    let info = &manifest_value["layouts"][&layout];
    if info["hidden"] != 6144 || info["intermediate"] != 3072 || info["experts"] != 128
        || info["top_k"] != 4 || info["tp"] != world || info["slice"] != 3072 / world
        || info["input"] != "bf16" || !matches!(info["weights"].as_str(), Some("nvfp4" | "nvfp4a4"))
        || info["activation"] != "swigluoai_uninterleave" || info["projection"] != "bf16"
        || info["swiglu_limit"] != 7.0 || info["abi_version"] != 5 {
        return Err(manifest(format!("{layout}: requires native MiniMax H6144/E128/K4/I3072 NVFP4 W4A4")));
    }
    crate::placement::inventory::fp8moe_scratch_bytes(manifest_value, &layout, rows as u64)
        .ok_or_else(|| manifest(format!("{layout}: no expert capacity for {rows} rows")))
}

pub fn expert_resident_bytes(catalog: &crate::OfficialV41Catalog, layers: &[usize], world: usize, rank: usize) -> Result<u64> {
    let tensors = catalog.fp8().ok_or_else(|| AdmissionError::Checkpoint("native NVFP4 catalog is absent".into()))?;
    let bytes = layers.iter().map(|&layer| {
        let native = tensors.for_layer(layer).map_err(|e| AdmissionError::Checkpoint(e.to_string()))?;
        if native.format() != crate::formats::fp8_experts::ExpertFormat::Nvfp4 {
            return Err(AdmissionError::Checkpoint(format!("layer {layer} requires NVFP4")));
        }
        let read = native.weight_read_bytes_with(world, rank, native.exact_slicing(world))
            .map_err(|e| AdmissionError::Checkpoint(e.to_string()))? as u64;
        product(&[read, native.shape().experts as u64])
    }).collect::<Result<Vec<_>>>()?;
    sum(bytes)
}

/// The routed layers the coordinator holds; the Sparks hold the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalLayers {
    /// The fewest bottom routed layers that leave every Spark rank within its budget.
    Fewest,
    /// This many bottom routed layers.
    Bottom(usize),
    /// Exactly these beside these Spark layers (golden's split of the layers it executes).
    Exactly { local: Vec<usize>, spark: Vec<usize> },
}

#[derive(Debug, Clone)]
pub struct Onboarding {
    pub local_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    pub local_bytes: u64,
    pub spark_bytes: Vec<u64>,
}

/// What a Spark rank's worker holds beside its expert residents, at each of the two peaks it admits
/// (`spark_admission_budget`): serving adds transport, rings, package scratch, runtime residual and
/// any remote media; loading adds package scratch, runtime residual and remote media, plus the
/// largest pinned upload and read scratch of one sequential layer load ([`expert_load_transient_bytes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteReserve {
    pub serve: u64,
    pub load: u64,
}

/// The worker's load-only owners on `rank` for `layer`: its pinned upload plus its read scratch
/// (`Fp8LayerLoad`, at the residents' exact slicing). A rank's load peak holds the largest of them.
pub fn expert_load_transient_bytes(catalog: &crate::OfficialV41Catalog, layer: usize, world: usize, rank: usize) -> Result<u64> {
    let tensors = catalog.fp8().ok_or_else(|| AdmissionError::Checkpoint("native NVFP4 catalog is absent".into()))?;
    let native = tensors.for_layer(layer).map_err(|e| AdmissionError::Checkpoint(e.to_string()))?;
    let load = native.layer_load(layer, world, rank, native.exact_slicing(world))
        .map_err(|e| AdmissionError::Checkpoint(e.to_string()))?;
    sum([load.pinned as u64, load.host as u64])
}

/// The coordinator's routed layers `local`, meeting every remote owner's allocation ceiling at
/// both its serve and its load peak.
pub fn onboard_layers(catalog: &crate::OfficialV41Catalog, cfg: &MiniMaxM3Config, sizing: RuntimeSizing,
    budget: u64, remote: RemoteReserve, local: &LocalLayers) -> Result<Onboarding> {
    sizing.validate(cfg)?;
    let layers = cfg.moe.iter().enumerate().filter_map(|(layer, &moe)| moe.then_some(layer)).collect::<Vec<_>>();
    let bottom = |count: usize| (layers[..count].to_vec(), layers[count..].to_vec());
    let candidates: Vec<(Vec<usize>, Vec<usize>)> = match local {
        LocalLayers::Fewest => (0..layers.len()).map(bottom).collect(),
        LocalLayers::Bottom(count) if *count < layers.len() => vec![bottom(*count)],
        LocalLayers::Bottom(_) => return Err(AdmissionError::Unsupported("all-local or excessive routed layer count")),
        LocalLayers::Exactly { local, spark } => vec![(local.clone(), spark.clone())],
    };
    // Per rank, each routed layer's load-only owners; a candidate's load peak takes its largest.
    let transients = (0..sizing.spark_ranks).map(|rank| layers.iter()
        .map(|&layer| Ok((layer, expert_load_transient_bytes(catalog, layer, sizing.spark_ranks, rank)?)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()).collect::<Result<Vec<_>>>()?;
    let mut shortfall = remote.serve.max(remote.load);
    for (local_layers, spark_layers) in candidates {
        let spark_bytes = (0..sizing.spark_ranks).map(|rank| expert_resident_bytes(catalog, &spark_layers, sizing.spark_ranks, rank))
            .collect::<Result<Vec<_>>>()?;
        let needed = spark_bytes.iter().zip(&transients).map(|(&bytes, transient)| {
            let largest = spark_layers.iter().filter_map(|layer| transient.get(layer).copied()).max().unwrap_or(0);
            Ok(sum([bytes, remote.serve])?.max(sum([bytes, remote.load, largest])?))
        }).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(shortfall);
        shortfall = needed;
        if needed <= budget {
            return Ok(Onboarding { local_bytes: expert_resident_bytes(catalog, &local_layers, 1, 0)?,
                local_layers, spark_layers, spark_bytes });
        }
    }
    Err(AdmissionError::Budget { needed: shortfall, available: budget })
}

/// Per-rank Spark allocation ceiling without `--spark-budget-gib`; the planner and serve-minimax share it.
/// Measured on headless DGX Sparks (OTA 26.09.2, multi-user.target, desktop services off): 117.77 GiB
/// MemAvailable idle, and every TP2 load and serve peak since stayed within this ceiling. It leaves 3 GiB
/// beside the worker's peaks, the margin below which a GB10 swaps and stops answering. A Spark that keeps
/// a desktop session or other services needs a lower `--spark-budget-gib`.
pub const SPARK_BUDGET_BYTES: u64 = 114 << 30;

/// Rows one decode step verifies at most (of a lane's rows): its startup recordings cover 1..=64.
pub const DECODE_ROWS: usize = 64;

/// Drafter context rings when none are requested. A ring is a request's for its life (a request
/// without one decodes target-only), and a decode step verifies at most `decode_rows` rows: with
/// k requests running, their drafts share `decode_rows - k` rows, so at most `decode_rows / 2`
/// of them verify a draft each at once. Every running request up to that many holds a ring;
/// more rings could never all draft. The drafter's bytes, rings included, come off the pool.
pub fn default_draft_rings(sequences: usize, decode_rows: usize) -> usize {
    sequences.min(decode_rows / 2).max(1)
}

/// Expert package scratch per Spark rank when the coordinator's package manifest has no layout
/// for the Spark world (remote packages are architecture-specific and need not be installed on
/// the coordinator): an estimate, until measured worker admission replaces it.
pub const REMOTE_SCRATCH_ESTIMATE_BYTES: u64 = 512 << 20;

/// What one coordinator admits, beside its checkpoint and manifests.
#[derive(Debug, Clone)]
pub struct Request {
    pub sizing: RuntimeSizing,
    pub kv_format: Qwen4KvCache,
    pub local_layers: LocalLayers,
    /// Every Spark rank's allocation ceiling.
    pub spark_budget: u64,
    /// What a Spark-hosted vision tower holds on every rank beside the experts (0 when local).
    pub spark_vision_bytes: u64,
    pub headroom: u64,
    /// The drafter's bytes (0 without it): the bundled DSpark's, or a DFlash2 checkpoint's on the
    /// shared drafter (its weights, rings, taps and step workspace).
    pub draft_bytes: u64,
    /// Drafter context rings (0 without the drafter): bounds the verify rows.
    pub draft_rings: usize,
    /// The largest transient the drafter holds while it loads (a DFlash2 FP8 packing's BF16
    /// source matrix; 0 otherwise): the load peak takes it after the target's weights.
    pub draft_staging: u64,
    pub host_embedding: bool,
    /// Startup records the decode graphs (golden runs every step eagerly and records none).
    pub startup_graphs: bool,
    /// The ring geometry every Spark link registers.
    pub rings: cuteafd_core::SparkRings,
}

/// One coordinator's allocations before its KV pool, the one recipe the server admits into
/// measured free memory and the planner into the card's budget (adding the CUDA context, code
/// and an RTX-hosted tower that the server's sample already holds).
#[derive(Debug, Clone)]
pub struct Recipe {
    pub request: Request,
    pub cache: MiniMaxCacheLayout,
    pub ledger: RuntimeLedger,
    pub programs: ProgramSizes,
    pub weights: RankWeightLayout,
    pub onboarding: Onboarding,
    /// Decode steps the warm-up records before ready; their bytes are reserved.
    pub graphs: crate::placement::inventory::StartupGraphs<usize>,
    /// The one Spark endpoint decode and every prefill wave share ([`RuntimeSizing::spark_endpoint`]).
    pub endpoints: Vec<cuteafd_core::SparkEndpoint>,
    /// The package scratch on each Spark rank, when the package exports the world's layout.
    pub remote_scratch: Option<u64>,
    /// Execution slots every Spark rank runs ([`RuntimeSizing::execution_slots`]).
    pub execution_slots: u8,
    /// One execution slot's workspace on a Spark rank, as the worker admits it: the slot's BF16
    /// inputs, routes and outputs at a unit's rows plus the package scratch (exported, or the estimate).
    pub slot_workspace: u64,
    /// Every Spark rank's bytes beside its experts: transport, every execution slot's workspace and
    /// the runtime residual.
    pub remote_reserved: u64,
    pub reservations: Reservations,
}

impl Recipe {
    pub fn new(cfg: &MiniMaxM3Config, checkpoint: &crate::plan::checkpoint::Checkpoint, catalog: &crate::OfficialV41Catalog,
        programs: &Value, package: Option<&Value>, request: Request) -> anyhow::Result<Self> {
        use anyhow::Context;
        let sizing = request.sizing;
        sizing.validate(cfg)?;
        let cache = MiniMaxCacheLayout::new(cfg, sizing.context, request.kv_format)?;
        let programs = ProgramSizes::from_manifest(programs, sizing)?;
        let ledger = RuntimeLedger::new(cfg, sizing, &programs)?;
        let weights = super::coordinator::resident_weights(checkpoint, request.host_embedding)?;
        let endpoints = vec![sizing.spark_endpoint()?];
        let all_remote = crate::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: cfg.moe.iter().enumerate()
            .filter_map(|(layer, &moe)| moe.then_some(layer)).collect(), spark_ranks: sizing.spark_ranks, prefill_lanes: 1 };
        let io = all_remote.rank_endpoints_io(0, catalog.routed_experts().geometry()?, &endpoints, request.rings)?;
        let world = format!("tp{}", sizing.spark_ranks);
        let remote_scratch = package.filter(|p| p["layouts"][&world].is_object())
            .map(|p| expert_workspace(p, sizing.spark_ranks, sizing.rows)).transpose()?;
        let scratch = remote_scratch.unwrap_or(REMOTE_SCRATCH_ESTIMATE_BYTES);
        let residual = cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES;
        // The worker sizes every execution slot alike: its I/O (the rank's intake) and the scratch.
        let (execution_slots, slot_workspace) = (sizing.execution_slots(), sum([io.intake, scratch])?);
        let remote_reserved = sum([product(&[slot_workspace, u64::from(execution_slots)])?, io.host_exchange,
            io.row_indices, io.endpoint_staging, io.mapped_rings, residual])?;
        let remote = RemoteReserve { serve: remote_reserved, load: sum([scratch, residual])? };
        let with_vision = RemoteReserve { serve: sum([remote.serve, request.spark_vision_bytes])?,
            load: sum([remote.load, request.spark_vision_bytes])? };
        let onboarding = onboard_layers(catalog, cfg, sizing, request.spark_budget, with_vision, &request.local_layers);
        if request.spark_vision_bytes > 0 && request.local_layers == LocalLayers::Fewest
            && onboarding.as_ref().map_or(true, |o| !o.local_layers.is_empty()) {
            // Every routed layer fits the Sparks without the tower: the tower is the cause, not onboarding.
            if let Ok(all) = onboard_layers(catalog, cfg, sizing, request.spark_budget, remote, &LocalLayers::Bottom(0)) {
                let experts = all.spark_bytes.iter().copied().max().unwrap_or(0).saturating_add(remote_reserved);
                let gb = |bytes: u64| bytes as f64 / 1e9;
                anyhow::bail!("Spark-hosted vision needs {:.2} GB on every Spark rank, but each TP{} rank has {:.2} GB left \
                    after its experts ({:.2} of {:.2} GB): use the local tower (--vision auto or rtx, without --vision-peers) \
                    or more Spark ranks", gb(request.spark_vision_bytes), sizing.spark_ranks,
                    gb(request.spark_budget.saturating_sub(experts)), gb(experts), gb(request.spark_budget));
            }
        }
        let onboarding = onboarding?;
        let expert_workspace = if onboarding.local_layers.is_empty() { 0 } else {
            expert_workspace(package.context("MiniMax onboarded layers require the native expert package manifest \
                (fp8-minimax-nvfp4a4/manifest.json)")?, 1, sizing.rows)?
        };
        let graphs = if request.startup_graphs { decode_graphs(sizing, request.draft_rings)? } else {
            use crate::placement::inventory::{GraphSet, Lifetime, StartupGraphs};
            StartupGraphs { inventory: GraphSet { ranks: Vec::new(), lifetime: Lifetime::Startup, shapes: 0 }, keys: Vec::new() }
        };
        let reservations = Reservations { weights: weights.resident_bytes.checked_add(request.draft_bytes)
                .context("MiniMax drafter reservation overflow")?, local_experts: onboarding.local_bytes,
            expert_workspace, headroom: request.headroom, graphs: graphs.inventory.bytes(0), ..Default::default() };
        Ok(Self { request, cache, ledger, programs, weights, onboarding, graphs, endpoints, remote_scratch,
            execution_slots, slot_workspace, remote_reserved, reservations })
    }

    /// The Spark working set once onboarding chose the coordinator's routed layers: one transport.
    pub fn working_set(&self) -> crate::placement::SparkWorkingSet {
        crate::placement::SparkWorkingSet { rtx_layers: self.onboarding.local_layers.clone(),
            spark_layers: self.onboarding.spark_layers.clone(), spark_ranks: self.request.sizing.spark_ranks,
            prefill_lanes: 1 }
    }

    /// Admit the KV pool into `available` bytes beside the reservations. The load peak adds the
    /// one source scale matrix staged at a time while the weights load, or the drafter's load
    /// transient after them, whichever is larger; the returned reservations admit the same recipe again.
    pub fn admit(&self, cfg: &MiniMaxM3Config, available: u64, requested_pool: Option<u64>) -> Result<(Admission, Reservations)> {
        let mut reservations = self.reservations;
        let staging = scale_prep_staging_bytes(&self.weights).max(self.request.draft_staging);
        reservations.startup = reservations.fixed_bytes(&self.ledger)?.checked_add(staging)
            .ok_or(AdmissionError::Overflow("load peak"))?;
        let admitted = admit(cfg, self.request.sizing, &self.ledger, reservations, available, requested_pool, self.request.kv_format)?;
        Ok((admitted, reservations))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::families::minimax_m3::config::tests::release_config;

    /// Every running request up to half the decode rows holds a ring: the shipped 64 sequences
    /// get 32, fewer sequences one each, short decode steps fewer.
    #[test]
    fn draft_rings_default_to_the_requests_that_can_draft_together() {
        assert_eq!(default_draft_rings(64, DECODE_ROWS), 32);
        assert_eq!(default_draft_rings(16, DECODE_ROWS), 16);
        assert_eq!(default_draft_rings(1, DECODE_ROWS), 1);
        assert_eq!(default_draft_rings(64, 16), 8);
        assert_eq!(default_draft_rings(4, 1), 1);
    }
    use crate::plan::{checkpoint::Checkpoint, testing};

    pub(crate) fn programs_manifest() -> Value {
        serde_json::json!({"programs": PROGRAM_NAMES.iter().map(|suffix| {
            let rows = if suffix.ends_with("_decode") { 64 } else if matches!(*suffix, "index_prefill" | "attention_extend") { 1024 } else { 4096 };
            let scratch = match *suffix {
                "mxfp8_dense_down" => 53_478_400,
                "index_decode" => 25_202_688,
                "index_prefill" => 134_799_872,
                "attention_decode" => 1_280,
                "attention_extend" => 792_576,
                _ => 0,
            };
            serde_json::json!({"name":format!("minimax_{suffix}"),"capacity_rows":rows,
                "scratch_bytes_at_capacity":{"scratch":scratch},"geometry":{"max_rows":rows,"max_pages":16384,"max_batch":64}})
        }).collect::<Vec<_>>()})
    }
    pub(crate) fn expert_manifest() -> Value {
        let layouts = [1, 2, 3, 4, 6].into_iter().map(|world| (format!("tp{world}"), serde_json::json!({
            "hidden":6144,"intermediate":3072,"experts":128,"top_k":4,"tp":world,"slice":3072/world,
            "input":"bf16","weights":"nvfp4","activation":"swigluoai_uninterleave","projection":"bf16",
            "swiglu_limit":7.0,"abi_version":5,"capacities":[
                {"capacity":256,"scratch_bytes":102837248},{"capacity":4096,"scratch_bytes":987654321}]})))
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({"layouts":layouts})
    }
    fn sizing() -> RuntimeSizing { RuntimeSizing { rows: 256, decode_rows: 64, sequences: 16, context: 1 << 20, spark_ranks: 3,
        logit_rows: 64, lanes: 1, host_landing: false } }
    fn config() -> MiniMaxM3Config { MiniMaxM3Config::from_hf(&release_config()).unwrap() }

    #[test]
    fn runtime_ledger_uses_exported_capacity_and_fp32_residual() {
        let sizing = sizing();
        let programs = ProgramSizes::from_manifest(&programs_manifest(), sizing).unwrap();
        let ledger = RuntimeLedger::new(&config(), sizing, &programs).unwrap();
        assert_eq!(ledger.bytes("residual"), Some(256 * 6144 * 4));
        assert_eq!(ledger.bytes("logits"), Some(64 * config().vocab_size as u64 * 4));
        assert_eq!(ledger.bytes("head_input"), Some(64 * 6144 * 2));
        let full = RuntimeSizing { logit_rows: 256, ..sizing };
        assert_eq!(RuntimeLedger::new(&config(), full, &programs).unwrap().bytes("logits"),
            Some(256 * config().vocab_size as u64 * 4));
        assert_eq!(ledger.bytes("q2k"), Some(4 * 64 * 16 * 4));
        assert_eq!(ledger.bytes("q2k_extend"), Some(4 * 1024 * 16 * 4));
        assert_eq!(ledger.bytes("lse"), Some(1024 * 64 * 4));
        // 16 sequences + one 1024-row tile of a 256-row step.
        assert_eq!(ledger.bytes("page_table"), Some(64 * 16384 * 4));
        assert_eq!(ledger.bytes("cache_lengths"), Some(4 * 17 * 4));
        assert_eq!(ledger.bytes("index_bounds"), Some((1024 + 256 + 4 * 17) * 4));
        assert_eq!(ledger.bytes("index_keys"), Some((1 << 20) * 128));
        assert_eq!(ledger.bytes("index_scales"), Some((1 << 20) * 4));
        assert_eq!(ledger.bytes("scratch"), Some(134_799_872));
        assert_eq!(ledger.bytes("head_workspace"), Some(4 << 20));
        assert_eq!(ledger.bytes("prefix_tail"), Some(256));
        assert_eq!(ledger.bytes("spark_intake"), Some(3 * 256 * 6144 * 2));
        assert_eq!(ledger.bytes("selector"), Some(64 * 12));
        assert_eq!(ledger.bytes("sampler"), Some(cuteafd_core::target_sampler_device_bytes(64, config().vocab_size) as u64));
        assert_eq!(ledger.fixed_bytes, ledger.allocations.iter().map(|a| a.bytes).sum::<u64>());
        assert!(ledger.allocations.iter().all(|a| a.bytes >= 256));

    }

    #[test]
    fn minimax_row_buffers_hold_one_row_per_chunk_row() {
        for lanes in [1, 2, 3, 4] {
            let sizing = RuntimeSizing { lanes, ..sizing() };
            let programs = ProgramSizes::from_manifest(&programs_manifest(), sizing).unwrap();
            let ledger = RuntimeLedger::new(&config(), sizing, &programs).unwrap();
            // Unit views stride these by row: each holds exactly waves x unit rows x its row bytes.
            assert_eq!((sizing.slot_rows(), sizing.chunk_rows()), (lanes * 256, lanes * 256));
            assert_eq!(sizing.shape().slot_rows(), sizing.slot_rows() as u64);
            for &(name, width, bytes) in ROW_BUFFERS {
                assert_eq!(ledger.bytes(name), Some((sizing.slot_rows() as u64 * width * bytes).max(256)), "{name}");
            }
            // The one link's return slots: a unit's rows per wave and rank, as the shared contract charges them.
            assert_eq!(ledger.bytes("spark_intake"), Some(lanes as u64 * 3 * 256 * 6144 * 2));
            assert_eq!(ledger.bytes("spark_intake"), Some(crate::serving_capacity::prefill_return_bytes(lanes as u64, 256, 3, 6144)));
            assert_eq!(extend_launches(sizing, 1024), 16 + 1 + lanes as u64 - 1);
            // One execution slot per wave in flight on every rank.
            assert_eq!(sizing.execution_slots(), lanes as u8);
        }
        assert!(RuntimeSizing { lanes: 0, ..sizing() }.validate(&config()).is_err());
        assert!(RuntimeSizing { lanes: 5, ..sizing() }.validate(&config()).is_err());
        // Several waves cut on whole KV pages; one wave takes any rows a launch holds.
        let unaligned = RuntimeSizing { rows: 100, sequences: 1, decode_rows: 64, logit_rows: 64, ..sizing() };
        assert!(RuntimeSizing { lanes: 2, ..unaligned }.validate(&config()).is_err());
        assert!(RuntimeSizing { lanes: 1, ..unaligned }.validate(&config()).is_ok());
        assert!(RuntimeSizing { rows: 4160, lanes: 1, ..sizing() }.validate(&config()).is_err());
    }

    /// #762 picks the waves for the topology and the unit for the waves; explicit waves or rows pin
    /// them. Six Sparks run three 2752-row waves, the 8192-row reserve rounded up to whole pages.
    #[test]
    fn prefill_shape_follows_the_adaptive_wave_defaults_unless_pinned() {
        let shape = |rows, waves, ranks| prefill_shape(rows, waves, ranks).map(|s| (s.unit_rows, s.waves, s.slot_rows()));
        for ranks in [2, 3, 4] { assert_eq!(shape(0, 0, ranks).unwrap(), (4096, 2, 8192)); }
        assert_eq!(shape(0, 0, 6).unwrap(), (2752, 3, 8256));
        assert_eq!(shape(0, 0, 0).unwrap(), (4096, 1, 4096));
        // Pinned waves take their own unit; pinned rows keep theirs.
        assert_eq!(shape(0, 2, 6).unwrap(), (4096, 2, 8192));
        assert_eq!(shape(0, 4, 4).unwrap(), (2048, 4, 8192));
        assert_eq!(shape(2048, 0, 4).unwrap(), (2048, 2, 4096));
        assert_eq!(shape(1024, 3, 2).unwrap(), (1024, 3, 3072));
        for (rows, waves) in [(100, 2), (4160, 1), (4096, 5), (32, 1)] { assert!(shape(rows, waves, 4).is_err(), "{rows} x {waves}"); }
    }

    /// A 2752-row wave needs no 2752-row export: the expert package runs it on the smallest capacity
    /// that holds it (4096), and the coordinator's programs take any rows up to their capacity.
    #[test]
    fn six_spark_units_run_on_the_4096_row_capacities() {
        assert_eq!(expert_workspace(&expert_manifest(), 6, 2752).unwrap(), 987_654_321);
        assert_eq!(expert_workspace(&expert_manifest(), 1, 2752).unwrap(), 987_654_321);
        let sizing = RuntimeSizing { rows: 2752, lanes: 3, spark_ranks: 6, ..sizing() };
        assert!(ProgramSizes::from_manifest(&programs_manifest(), sizing).is_ok());
        let ledger = RuntimeLedger::new(&config(), sizing, &ProgramSizes::from_manifest(&programs_manifest(), sizing).unwrap()).unwrap();
        assert_eq!(ledger.bytes("residual"), Some(3 * 2752 * 6144 * 4));
        assert_eq!(ledger.bytes("spark_intake"), Some(3 * 6 * 2752 * 6144 * 2));
        assert_eq!(sizing.execution_slots(), 3);
        // A program exported for fewer rows than a unit is refused.
        let mut manifest = programs_manifest();
        manifest["programs"].as_array_mut().unwrap().iter_mut().filter(|p| p["name"] == "minimax_norm")
            .for_each(|p| p["capacity_rows"] = 2048.into());
        assert!(ProgramSizes::from_manifest(&manifest, sizing).is_err());
    }

    #[test]
    fn decode_graph_inventory_covers_every_verify_row_count_exactly() {
        use crate::placement::inventory::Lifetime;
        // Target-only: one row per sequence; drafting: up to 8 rows per ring, within the tile.
        for (sequences, rings, rows) in [(16, 0, 16), (1, 0, 1), (16, 8, 64), (1, 1, 8), (4, 2, 18), (64, 0, 64)] {
            let graphs = decode_graphs(RuntimeSizing { sequences, ..sizing() }, rings).unwrap();
            assert_eq!(graphs.keys, (1..=rows).collect::<Vec<_>>());
            assert_eq!(graphs.inventory.lifetime, Lifetime::Startup);
            let captured = rows as u64 * DECODE_GRAPH_BYTES;
            assert_eq!(graphs.inventory.at_ready(0), captured);
            assert_eq!(graphs.inventory.bytes(0), captured + (captured * 10).div_ceil(100).max(16 << 20));
        }
        assert!(decode_graphs(RuntimeSizing { sequences: 4, ..sizing() }, 5).is_err());
        // The reservation is part of the fixed bytes the pool is sized after.
        let cfg = config();
        let programs = ProgramSizes::from_manifest(&programs_manifest(), sizing()).unwrap();
        let ledger = RuntimeLedger::new(&cfg, sizing(), &programs).unwrap();
        let none = Reservations::default().fixed_bytes(&ledger).unwrap();
        assert_eq!(Reservations { graphs: 7, ..Default::default() }.fixed_bytes(&ledger).unwrap(), none + 7);
    }

    #[test]
    fn minimax_local_golden_admits_no_transport_storage() {
        let local = RuntimeSizing { spark_ranks: 0, ..sizing() };
        let programs = ProgramSizes::from_manifest(&programs_manifest(), local).unwrap();
        let ledger = RuntimeLedger::new(&config(), local, &programs).unwrap();
        assert_eq!(ledger.bytes("spark_intake"), Some(256));
        for spark_ranks in [1, 5, 7] {
            assert!(RuntimeSizing { spark_ranks, ..local }.validate(&config()).is_err());
        }
        for spark_ranks in [2, 3, 4, 6] {
            assert!(RuntimeSizing { spark_ranks, ..local }.validate(&config()).is_ok());
        }
    }

    /// Golden's recipe: exactly its layers, no Spark ceiling and no startup graphs, reserving
    /// only the weights, its local experts, their package scratch and the headroom.
    #[test]
    fn exact_layers_without_startup_graphs_reserve_only_their_own_bytes() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        let catalog = crate::read_expert_catalog(dir.path())?;
        let ring = crate::plan::layout::spark_ring_geometry(1);
        let sizing = RuntimeSizing { sequences: 1, context: 4096, spark_ranks: 2, logit_rows: 256, ..sizing() };
        let (local, spark) = (vec![3, 5], vec![4, 6]);
        let recipe = Recipe::new(&cfg, &checkpoint, &catalog, &programs_manifest(), Some(&expert_manifest()), Request {
            sizing, kv_format: Qwen4KvCache::Fp8, local_layers: LocalLayers::Exactly { local: local.clone(), spark: spark.clone() },
            spark_budget: u64::MAX, spark_vision_bytes: 0, headroom: 2 << 30, draft_bytes: 0, draft_rings: 0, draft_staging: 0,
            host_embedding: false, startup_graphs: false, rings: cuteafd_core::SparkRings { max_frame_bytes: ring.max_frame_bytes,
                depth: ring.depth, slot_bytes: ring.slot_bytes, alignment: ring.alignment } })?;
        assert_eq!((&recipe.onboarding.local_layers, &recipe.onboarding.spark_layers), (&local, &spark));
        assert!(recipe.graphs.keys.is_empty() && recipe.graphs.inventory.bytes(0) == 0);
        let weights = super::super::coordinator::resident_weights(&checkpoint, false)?;
        let r = recipe.reservations;
        assert_eq!((r.weights, r.local_experts, r.expert_workspace, r.headroom, r.graphs), (weights.resident_bytes,
            expert_resident_bytes(&catalog, &local, 1, 0)?, expert_workspace(&expert_manifest(), 1, 256)?, 2 << 30, 0));
        assert_eq!(r.media + r.pending_code + r.runtime_baseline + r.startup, 0);
        Ok(())
    }

    /// One Spark endpoint carries decode and every wave; each rank runs one execution slot per wave
    /// in flight, every slot the worker's whole workspace (its BF16 I/O at a unit's rows and the
    /// package scratch), so a second wave reserves exactly one more.
    #[test]
    fn every_wave_in_flight_reserves_one_execution_slot_per_rank() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        let catalog = crate::read_expert_catalog(dir.path())?;
        let ring = crate::plan::layout::spark_ring_geometry(1);
        let recipe = |lanes, package: Option<&Value>| Recipe::new(&cfg, &checkpoint, &catalog, &programs_manifest(), package, Request {
            sizing: RuntimeSizing { rows: 4096, lanes, context: 4096, ..sizing() }, kv_format: Qwen4KvCache::Fp8,
            local_layers: LocalLayers::Bottom(0), spark_budget: u64::MAX, spark_vision_bytes: 0, headroom: 0, draft_bytes: 0,
            draft_rings: 0, draft_staging: 0, host_embedding: false, startup_graphs: false,
            rings: cuteafd_core::SparkRings { max_frame_bytes: ring.max_frame_bytes, depth: ring.depth,
                slot_bytes: ring.slot_bytes, alignment: ring.alignment } });
        let intake = 4096 * (6144 * 2 + 4 * 8 + 6144 * 2);
        for package in [None, Some(expert_manifest())] {
            let scratch = if package.is_some() { 987_654_321 } else { REMOTE_SCRATCH_ESTIMATE_BYTES };
            let one = recipe(1, package.as_ref())?;
            assert_eq!(one.endpoints, [RuntimeSizing { rows: 4096, ..sizing() }.spark_endpoint()?]);
            assert_eq!((one.execution_slots, one.slot_workspace), (1, intake + scratch));
            for lanes in [2, 3, 4] {
                let waves = recipe(lanes, package.as_ref())?;
                assert_eq!(waves.endpoints, one.endpoints, "one link at every wave count");
                assert_eq!(waves.execution_slots, lanes as u8);
                assert_eq!(waves.remote_reserved - one.remote_reserved, (lanes as u64 - 1) * (intake + scratch));
                assert_eq!(waves.working_set().prefill_lanes, 1);
            }
        }
        Ok(())
    }

    #[test]
    fn every_manifest_gap_or_capacity_drift_is_refused() {
        let base = programs_manifest();
        for index in 0..PROGRAM_NAMES.len() {
            let mut manifest = base.clone();
            manifest["programs"].as_array_mut().unwrap().remove(index);
            assert!(matches!(ProgramSizes::from_manifest(&manifest, sizing()), Err(AdmissionError::Manifest(_))));
        }
        for (name, path, value) in [("minimax_index_decode", "/capacity_rows", 32),
            ("minimax_attention_decode", "/capacity_rows", 128),
            ("minimax_index_prefill", "/geometry/max_pages", 100),
            ("minimax_attention_decode", "/geometry/max_batch", 32)] {
            let mut manifest = base.clone();
            let entry = manifest["programs"].as_array_mut().unwrap().iter_mut().find(|p| p["name"] == name).unwrap();
            *entry.pointer_mut(path).unwrap() = value.into();
            assert!(ProgramSizes::from_manifest(&manifest, sizing()).is_err(), "{name} {path}");
        }
        let mut manifest = base.clone();
        manifest["programs"][0]["scratch_bytes_at_capacity"]["scratch"] = Value::Null;
        assert!(ProgramSizes::from_manifest(&manifest, sizing()).is_err());
        let mut manifest = base.clone();
        let duplicate = manifest["programs"][0].clone();
        manifest["programs"].as_array_mut().unwrap().push(duplicate);
        assert!(ProgramSizes::from_manifest(&manifest, sizing()).is_err());
    }

    #[test]
    fn explicit_pool_cannot_clamp_or_drop_reservations() {
        let cfg = config();
        let sizing = sizing();
        let programs = ProgramSizes::from_manifest(&programs_manifest(), sizing).unwrap();
        let ledger = RuntimeLedger::new(&cfg, sizing, &programs).unwrap();
        let reservations = Reservations { weights: 1000, local_experts: 3000,
            expert_workspace: 4000, media: 5000, pending_code: 6000, runtime_baseline: 7000, headroom: 8000,
            graphs: 9000, startup: 0 };
        let fixed = ledger.fixed_bytes + 43_000;
        let page = 60 * (2 * 32768 + 8448);
        let available = fixed + 3 * page;
        let admitted = admit(&cfg, sizing, &ledger, reservations, available, Some(192), Qwen4KvCache::Fp8).unwrap();
        assert_eq!((admitted.fixed_bytes, admitted.pool_pages, admitted.total_bytes), (fixed, 3, available));
        assert_eq!(admitted.cache_bytes, 3 * page);
        let automatic = admit(&cfg, sizing, &ledger, reservations, available + 1, None, Qwen4KvCache::Fp8).unwrap();
        assert_eq!(automatic.pool_tokens, 192);
        assert!(matches!(admit(&cfg, sizing, &ledger, reservations, available - 1, Some(192), Qwen4KvCache::Fp8),
            Err(AdmissionError::Budget { .. })));
        assert!(admit(&cfg, sizing, &ledger, reservations, available, Some(193), Qwen4KvCache::Fp8).is_err());
        assert!(admit(&cfg, sizing, &ledger, reservations, available, Some(192), Qwen4KvCache::Bf16).is_err());
        assert!(matches!(admit(&cfg, sizing, &ledger, Reservations { startup: available + 1, ..reservations },
            available, None, Qwen4KvCache::Fp8), Err(AdmissionError::Budget { .. })));
    }

    #[test]
    fn prepared_scales_replace_source_storage_with_a_bounded_load_peak() {
        let dir = tempfile::tempdir().unwrap();
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let weights = super::super::coordinator::resident_weights(&Checkpoint::open(dir.path()).unwrap(), false).unwrap();
        assert_eq!(scale_prep_staging_bytes(&weights), 24576 * 6144 / 32);
    }

    #[test]
    fn native_expert_workspace_uses_the_selected_capacity() {
        let mut manifest = expert_manifest();
        assert_eq!(expert_workspace(&manifest, 1, 256).unwrap(), 102_837_248);
        assert_eq!(expert_workspace(&manifest, 3, 257).unwrap(), 987_654_321);
        assert!(expert_workspace(&manifest, 3, 4097).is_err());
        // Six Sparks run 512-wide slices; five have no package layout.
        assert_eq!(expert_workspace(&manifest, 6, 256).unwrap(), 102_837_248);
        assert!(expert_workspace(&manifest, 5, 256).is_err());
        manifest["layouts"]["tp3"]["activation"] = "silu_bf16_product".into();
        assert!(expert_workspace(&manifest, 3, 256).is_err());
    }
}
