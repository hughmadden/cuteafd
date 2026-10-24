//! Allocation recipe shared by native serving and manifest-backed planning.
use std::collections::BTreeMap;
use super::coordinator::RankWeightLayout;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeSizing {
    pub rows: usize,
    pub decode_rows: usize,
    pub sequences: usize,
    pub context: u64,
    pub spark_ranks: usize,
    /// Rows one head launch projects: decode and draft rows, or every prefill
    /// row when full prefill logits are served. Prefill otherwise projects
    /// only the rows whose logits a request needs.
    pub logit_rows: usize,
}
impl RuntimeSizing {
    /// Head rows for a server: every prefill row with full prefill logits,
    /// otherwise decode rows or the drafter's block rows, whichever is wider.
    pub fn head_rows(rows: usize, full_prefill_logits: bool, draft_rows: usize) -> usize {
        if full_prefill_logits { rows } else { rows.min(64).max(draft_rows).min(rows) }
    }
    pub fn validate(self, cfg: &MiniMaxM3Config) -> Result<()> {
        if !(1..=4096).contains(&self.rows) || !(1..=64).contains(&self.decode_rows)
            || self.sequences == 0 || self.sequences > self.decode_rows || self.sequences > self.rows
            || self.logit_rows < self.decode_rows || self.logit_rows > self.rows {
            return Err(AdmissionError::Unsupported("row capacity or sequence count"));
        }
        if self.context == 0 || self.context > cfg.max_position as u64 {
            return Err(AdmissionError::Unsupported("context outside the checkpoint extent"));
        }
        if !matches!(self.spark_ranks, 0 | 2 | 3 | 4) {
            return Err(AdmissionError::Unsupported("expert world outside local, TP2, TP3 or TP4"));
        }
        cfg.release_geometry().map_err(|e| manifest(e.to_string()))
    }
}

pub const PROGRAM_NAMES: &[&str] = &[
    "norm", "q_norm_rope", "k_norm_rope", "index_q_norm_rope", "index_k_norm_rope",
    "rope", "scale_prep", "kv_split", "swiglu_i3072", "swiglu_i12288", "router_scores",
    "route", "residual", "index_q_quant", "index_k_quant", "index_k_pack", "kv_pack",
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
/// element). A prefill lane's view of one starts at its first row.
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
        let (r, d, h) = (sizing.rows as u64, programs.decode_capacity(), cfg.hidden as u64);
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
        add("spark_intake", &[sizing.spark_ranks as u64, product(&[r, h, 2])?.max(256)])?;
        let fixed_bytes = sum(allocations.iter().map(|a| a.bytes))?;
        Ok(Self { allocations, fixed_bytes })
    }
    pub fn bytes(&self, name: &str) -> Option<u64> { self.allocations.iter().find(|a| a.name == name).map(|a| a.bytes) }
}

/// Page-table entries per extend launch: every page of the admitted context,
/// padded so each launch's row starts 16-byte aligned.
pub fn extend_table_stride(context: u64) -> u64 { context.div_ceil(PAGE_ROWS).next_multiple_of(4) }

/// Most extend launches one prefill step makes: each request's rows split
/// into tiles of at most `tile` rows.
pub fn extend_launches(sizing: RuntimeSizing, tile: u64) -> u64 {
    sizing.sequences as u64 + (sizing.rows as u64).div_ceil(tile)
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
    /// Additional absolute peak above the same pre-weight sample.
    pub startup: u64,
}
impl Reservations {
    pub fn fixed_bytes(self, ledger: &RuntimeLedger) -> Result<u64> {
        sum([ledger.fixed_bytes, self.weights, self.local_experts,
            self.expert_workspace, self.media, self.pending_code, self.runtime_baseline, self.headroom])
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
    let cache = MiniMaxCacheLayout::for_runtime(cfg, 1, sizing.context, kv_format)?;
    let fixed_bytes = reservations.fixed_bytes(ledger)?;
    let needed = fixed_bytes.max(reservations.startup);
    if needed > available { return Err(AdmissionError::Budget { needed, available }); }
    let page_bytes = cache.ranks[0].page_bytes;
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
    if !matches!(world, 1..=4) { return Err(AdmissionError::Unsupported("expert package TP degree")); }
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

#[derive(Debug, Clone)]
pub struct Onboarding {
    pub local_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    pub local_bytes: u64,
    pub spark_bytes: Vec<u64>,
}
/// Minimum bottom routed prefix meeting every remote owner's allocation ceiling.
/// `remote_reserved` includes scratch, transport, runtime and any remote media.
pub fn onboard_layers(catalog: &crate::OfficialV41Catalog, cfg: &MiniMaxM3Config, sizing: RuntimeSizing,
    budget: u64, remote_reserved: u64, requested: Option<usize>) -> Result<Onboarding> {
    sizing.validate(cfg)?;
    let layers = cfg.moe.iter().enumerate().filter_map(|(layer, &moe)| moe.then_some(layer)).collect::<Vec<_>>();
    if requested.is_some_and(|n| n >= layers.len()) { return Err(AdmissionError::Unsupported("all-local or excessive routed layer count")); }
    let candidates = requested.map_or_else(|| (0..layers.len()).collect(), |count| vec![count]);
    let mut shortfall = remote_reserved;
    for count in candidates {
        let remote = (0..sizing.spark_ranks).map(|rank| expert_resident_bytes(catalog, &layers[count..], sizing.spark_ranks, rank))
            .collect::<Result<Vec<_>>>()?;
        let needed = remote.iter().map(|&bytes| sum([bytes, remote_reserved])).collect::<Result<Vec<_>>>()?
            .into_iter().max().unwrap_or(remote_reserved);
        shortfall = needed;
        if needed <= budget {
            return Ok(Onboarding { local_bytes: expert_resident_bytes(catalog, &layers[..count], 1, 0)?,
                local_layers: layers[..count].to_vec(), spark_layers: layers[count..].to_vec(), spark_bytes: remote });
        }
    }
    Err(AdmissionError::Budget { needed: shortfall, available: budget })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::families::minimax_m3::config::tests::release_config;
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
        let layouts = (1..=4).map(|world| (format!("tp{world}"), serde_json::json!({
            "hidden":6144,"intermediate":3072,"experts":128,"top_k":4,"tp":world,"slice":3072/world,
            "input":"bf16","weights":"nvfp4","activation":"swigluoai_uninterleave","projection":"bf16",
            "swiglu_limit":7.0,"abi_version":5,"capacities":[
                {"capacity":256,"scratch_bytes":102837248},{"capacity":4096,"scratch_bytes":987654321}]})))
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({"layouts":layouts})
    }
    fn sizing() -> RuntimeSizing { RuntimeSizing { rows: 256, decode_rows: 64, sequences: 16, context: 1 << 20, spark_ranks: 3, logit_rows: 64 } }
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
    fn minimax_row_buffers_hold_one_row_per_batch_row() {
        let sizing = sizing();
        let programs = ProgramSizes::from_manifest(&programs_manifest(), sizing).unwrap();
        let ledger = RuntimeLedger::new(&config(), sizing, &programs).unwrap();
        // Lane views stride these by row: each holds exactly rows x its row bytes.
        for &(name, width, bytes) in ROW_BUFFERS {
            assert_eq!(ledger.bytes(name), Some((sizing.rows as u64 * width * bytes).max(256)), "{name}");
        }
        assert_eq!(ledger.bytes("norm"), Some(sizing.rows as u64 * config().hidden as u64 * 2));
    }

    #[test]
    fn minimax_local_golden_admits_no_transport_storage() {
        let local = RuntimeSizing { spark_ranks: 0, ..sizing() };
        let programs = ProgramSizes::from_manifest(&programs_manifest(), local).unwrap();
        let ledger = RuntimeLedger::new(&config(), local, &programs).unwrap();
        assert_eq!(ledger.bytes("spark_intake"), Some(256));
        for spark_ranks in [1, 5, 6] {
            assert!(RuntimeSizing { spark_ranks, ..local }.validate(&config()).is_err());
        }
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
            expert_workspace: 4000, media: 5000, pending_code: 6000, runtime_baseline: 7000, headroom: 8000, startup: 0 };
        let fixed = ledger.fixed_bytes + 34_000;
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
        let weights = RankWeightLayout::from_checkpoint(&Checkpoint::open(dir.path()).unwrap(), 0, 1).unwrap();
        assert_eq!(scale_prep_staging_bytes(&weights), 24576 * 6144 / 32);
    }

    #[test]
    fn native_expert_workspace_uses_the_selected_capacity() {
        let mut manifest = expert_manifest();
        assert_eq!(expert_workspace(&manifest, 1, 256).unwrap(), 102_837_248);
        assert_eq!(expert_workspace(&manifest, 3, 257).unwrap(), 987_654_321);
        assert!(expert_workspace(&manifest, 3, 4097).is_err());
        manifest["layouts"]["tp3"]["activation"] = "silu_bf16_product".into();
        assert!(expert_workspace(&manifest, 3, 256).is_err());
    }
}
