//! Dense hybrid Qwen admission. Every byte is future to the supplied baseline;
//! the executor supplies native packed sizes and exported workspace costs.
use crate::families::{qwen::HybridAttention, qwen3_5::Qwen35Config};
use crate::placement::*;
use anyhow::{ensure, Result};
use cuteafd_core::DType;
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
/// Only the initial single-device dense executor; not a claim of TP2 support.
pub const SINGLE: ExecutorModes = ExecutorModes { family: "qwen3_5", modes: &[WHOLE],
    hops: false, attention: AttentionPlacement::Heads };

#[derive(Debug, Clone, Copy)]
pub struct LayerBytes {
    pub weights: u64,
    /// Active recurrence and convolution slots, excluding retained prefix marks.
    pub state: u64,
    pub replay: u64,
    pub prefix: u64,
}

/// Drafter-owned allocations, excluding the target embedding and vocabulary head.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrafterBytes {
    pub owned: u64,
    pub scratch: u64,
}

impl DrafterBytes {
    fn movable(self) -> Movable {
        Movable { id: MovableId::Drafter, allowed: vec![0], expert_arena: false,
            parts: vec![Bytes2 { resident: self.owned, staging: 0 },
                Bytes2 { resident: self.scratch, staging: 0 }], conditional: vec![] }
    }
}

pub struct DenseInputs<'a> {
    pub cfg: &'a Qwen35Config,
    pub gpu: GpuBudget,
    pub layers: &'a [LayerBytes],
    /// Embedding, head, native norm, workspaces, graphs and pending loaded code.
    /// Callers retain the categories and names of their actual allocations.
    pub fixed: Vec<Demand>,
    pub drafter: Option<DrafterBytes>,
    pub requested_pool: Option<u64>,
    pub max_context: u64,
    /// Checkpoint/export-native K|V record width (BF16 initially, native FP8 later).
    pub record_bytes: u64,
    pub pool_overhead: u64,
    pub pool_bounded: Vec<BoundedDemand>,
}

pub fn request(inputs: DenseInputs<'_>) -> Result<PlacementRequest> {
    ensure!(!inputs.cfg.attention.is_empty() && inputs.layers.len() == inputs.cfg.attention.len(),
        "dense Qwen layer byte inventory differs from checkpoint");
    ensure!(inputs.record_bytes > 0 && inputs.max_context > 0, "invalid dense Qwen cache geometry");
    ensure!(inputs.fixed.iter().all(|item| item.gpu == 0)
        && inputs.pool_bounded.iter().all(|item| item.gpu == 0),
        "single-device dense Qwen demand names a peer");
    let layers = inputs.cfg.attention.iter().zip(inputs.layers).map(|(&kind, bytes)| {
        ensure!(bytes.weights > 0, "dense Qwen layer has no native weight admission");
        let mut extra = Vec::new();
        let kv = match kind {
            HybridAttention::Gdn => {
                ensure!(bytes.state > 0 && bytes.replay > 0, "dense GDN lacks exact state/replay admission");
                for (category, group, value) in [(Category::Kv, "GDN state", bytes.state),
                    (Category::Workspace, "GDN replay", bytes.replay),
                    (Category::Prefix, "GDN exact marks", bytes.prefix)] {
                    extra.push(LayerExtraDemand::new(category, group, value, LayerResidency::Owner, Basis::Exact));
                }
                0
            }
            HybridAttention::Full => {
                ensure!(bytes.state == 0 && bytes.replay == 0 && bytes.prefix == 0,
                    "full GQA must not carry fabricated recurrent state");
                inputs.record_bytes.checked_mul(PAGE_ROWS).ok_or_else(|| anyhow::anyhow!("dense KV unit size overflow"))?
            }
        };
        Ok(LayerDemand { kind: if kind == HybridAttention::Gdn { AttentionClass::Gdn } else { AttentionClass::Gqa },
            weights: ModeBytes { whole: bytes.weights, ..Default::default() },
            kv_unit: KvDemand { unit_bytes_whole: kv, ..Default::default() },
            fixed_bytes: ModeBytes::default(), extra, context_indexer: false, colocate: None,
            experts: None, modes: vec![WHOLE] })
    }).collect::<Result<_>>()?;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: vec![inputs.gpu], spark_ranks: 0, peer_access: false },
        attention_placement: None, context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool: PoolPolicy::resolve(&[inputs.gpu.capacity_bytes], inputs.max_context,
            inputs.requested_pool, PAGE_ROWS, true),
        layers, pool_overhead: vec![inputs.pool_overhead], pool_bounded: inputs.pool_bounded,
        fixed: inputs.fixed, movables: inputs.drafter.map(DrafterBytes::movable).into_iter().collect(),
        expert_workspace: 0, tp2_workspace: [0; 2],
        onboard: Onboard::Auto, expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![WHOLE], ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: inputs.cfg.hidden as u64 * 2,
            rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: SINGLE,
    })
}

/// Native admission uses the same solver in offline planning and device startup.
pub fn placement(inputs: DenseInputs<'_>) -> Result<Placement> {
    let result = solve(&request(inputs)?)?;
    SINGLE.check(&result)?;
    Ok(result)
}

pub const PREFILL_ROWS: usize = 4096;
pub const VERIFY_ROWS: usize = 64;
pub const HEAD_WORKSPACE: usize = 4 << 20;

/// Exact allocation recipe, excluding checkpoint weights and the KV pool.
/// Dense execution currently launches directly; it captures no CUDA graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuffers {
    pub planes: [usize; 13],
    pub logits: usize,
    pub ids: usize,
    pub head: usize,
    pub commit: usize,
    pub state_per_layer: usize,
    pub replay_per_layer: usize,
    pub verify: [usize; 3],
    pub prefix: usize,
    pub mark_bytes: usize,
    pub marks: usize,
}

impl NativeBuffers {
    pub fn new(cfg: &Qwen35Config, scratch: usize, slots: usize, marks: usize) -> Result<Self> {
        ensure!((1..=VERIFY_ROWS).contains(&slots), "dense state slots must be 1..64");
        // Bound all products below before computing the native allocation recipe.
        ensure!(cfg.hidden <= 16_777_216 && cfg.vocab <= 16_777_216
            && cfg.heads <= 1024 && cfg.head_dim <= 256 && cfg.attention.len() <= 4096,
            "dense allocation geometry exceeds supported bounds");
        let g = cfg.gdn;
        ensure!(g.key_heads > 0 && g.key_heads <= 1024 && g.value_heads > 0 && g.value_heads <= 1024
            && g.key_dim == 128 && g.value_dim == 128 && g.conv_kernel == 4
            && cfg.max_context <= 16_777_216, "unsupported dense native state geometry");
        let state_per_layer = g.state_bytes();
        let replay_row = g.key_heads * g.key_dim * 4 + g.value_heads * g.value_dim * 4
            + g.value_heads * 8 + g.conv_width() * 2;
        let replay_per_layer = (VERIFY_ROWS * replay_row).div_ceil(1024) * 1024;
        let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count();
        let mark_bytes = gdn.checked_mul(state_per_layer).ok_or_else(|| anyhow::anyhow!("dense mark overflow"))?;
        let hidden = PREFILL_ROWS * cfg.hidden * 2;
        let query = PREFILL_ROWS * cfg.heads * cfg.head_dim * 2;
        let planes = [hidden, hidden, hidden, query, query, query, scratch,
            PREFILL_ROWS * 4, PREFILL_ROWS * 4, PREFILL_ROWS * 12, PREFILL_ROWS * 8,
            PREFILL_ROWS * 8, VERIFY_ROWS * cfg.max_context.div_ceil(64) * 4].map(|n| n.max(256));
        Ok(Self { planes, logits: VERIFY_ROWS * cfg.vocab * 4, ids: PREFILL_ROWS * 4,
            head: HEAD_WORKSPACE, commit: 3 * VERIFY_ROWS * 4, state_per_layer, replay_per_layer,
            verify: [mark_bytes.checked_mul(slots).ok_or_else(|| anyhow::anyhow!("dense verify mark overflow"))?.max(256),
                VERIFY_ROWS * cfg.vocab * 4, cfg.attention.len() * VERIFY_ROWS * cfg.hidden * 2],
            prefix: marks.checked_mul(mark_bytes).ok_or_else(|| anyhow::anyhow!("dense prefix overflow"))?, mark_bytes, marks })
    }

    pub fn fixed(&self) -> Vec<Demand> {
        [(Category::Workspace, "native step planes and scratch", self.planes.iter().sum()),
            (Category::Workspace, "vocabulary head workspace", self.head),
            (Category::Workspace, "logits", self.logits), (Category::Workspace, "token ids", self.ids),
            (Category::Workspace, "GDN commit table", self.commit),
            (Category::Prefix, "exact retained prefix marks", self.prefix),
            (Category::Workspace, "block verify marks, logits and taps", self.verify.iter().sum()),
            (Category::Runtime, "head and runtime growth", 512 << 20)]
            .into_iter().map(|(category, group, bytes)| Demand::new(0, category, group, bytes as u64, Basis::Exact)).collect()
    }
}

/// Header-only packed native BF16 allocations, including widened small GDN operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWeights {
    pub layers: Vec<u64>,
    pub norm: u64,
    pub head: u64,
    pub embedding: u64,
}

impl NativeWeights {
    pub fn bf16(checkpoint: &crate::plan::checkpoint::Checkpoint, cfg: &Qwen35Config,
        norm_dtype: cuteafd_core::DType) -> Result<Self> {
        use cuteafd_core::DType;
        ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "invalid native GDN norm dtype");
        let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint lacks {name}"));
        let bytes = |name: &str, widen: bool, norm: bool| -> Result<u64> {
            let t = tensor(name)?;
            let dtype = &t.meta.dtype;
            ensure!((widen && matches!(dtype, DType::Bf16 | DType::F32))
                || (!widen && *dtype == if norm { norm_dtype.clone() } else { DType::Bf16 }),
                "{name}: unsupported native dense BF16 operand {dtype:?}");
            Ok(if widen && *dtype == DType::Bf16 { t.meta.byte_length * 2 } else { t.meta.byte_length })
        };
        let mut layers = Vec::new();
        for (index, &kind) in cfg.attention.iter().enumerate() {
            let p = format!("model.language_model.layers.{index}.");
            let one = |name: &str| bytes(&format!("{p}{name}.weight"), false, false).map(|n| n.max(256));
            let pack = |names: &[&str]| -> Result<u64> {
                names.iter().try_fold(0u64, |n, name| Ok(n + bytes(&format!("{p}{name}.weight"), false, false)?))
                    .map(|n| n.max(256))
            };
            let mut total = one("input_layernorm")? + one("post_attention_layernorm")?
                + one("mlp.down_proj")? + pack(&["mlp.gate_proj", "mlp.up_proj"])?;
            total += if kind == HybridAttention::Gdn {
                pack(&["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"])?
                    + one("linear_attn.out_proj")?
                    + bytes(&format!("{p}linear_attn.conv1d.weight"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.A_log"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.dt_bias"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.norm.weight"), false, true)?.max(256)
            } else {
                pack(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])?
                    + one("self_attn.o_proj")? + one("self_attn.q_norm")? + one("self_attn.k_norm")?
            };
            layers.push(total);
        }
        let embedding = bytes("model.language_model.embed_tokens.weight", false, false)?;
        Ok(Self { layers, norm: bytes("model.language_model.norm.weight", false, false)?.max(256),
            head: bytes(if cfg.tied_embeddings { "model.language_model.embed_tokens.weight" } else { "lm_head.weight" }, false, false)?.max(256), embedding })
    }

    pub fn demands(&self, cfg: &Qwen35Config, buffers: &NativeBuffers, slots: usize) -> (Vec<LayerBytes>, Vec<Demand>) {
        let layers = cfg.attention.iter().zip(&self.layers).map(|(&kind, &weights)| LayerBytes { weights,
            state: if kind == HybridAttention::Gdn { (buffers.state_per_layer * slots) as u64 } else { 0 },
            replay: if kind == HybridAttention::Gdn { buffers.replay_per_layer as u64 } else { 0 }, prefix: 0 }).collect();
        let mut fixed = buffers.fixed();
        fixed.extend([(Category::Embedding, "token embedding", self.embedding),
            (Category::Weights, "final norm", self.norm), (Category::Weights, "vocabulary head", self.head)]
            .into_iter().map(|(category, group, bytes)| Demand::new(0, category, group, bytes, Basis::Exact)));
        (layers, fixed)
    }
}

/// CPU-readable contract shared with runtime native ABI validation.
#[derive(Debug, Clone)]
pub struct NativeProgram {
    pub name: String,
    pub geometry: serde_json::Value,
    pub capacity_rows: u64,
    pub pointers: Vec<String>,
    pub scalar_kinds: Vec<u8>,
    pub scratch: std::collections::HashMap<String, u64>,
}

impl NativeProgram {
    pub fn from_manifest(entry: &serde_json::Value) -> Result<Self> {
        use anyhow::Context;
        let name = entry["name"].as_str().context("native program name missing")?.to_owned();
        let pointers = entry["pointers"].as_array().context("native pointers missing")?.iter()
            .map(|p| p["name"].as_str().map(str::to_owned).context("native pointer name missing"))
            .collect::<Result<_>>()?;
        let scalar_kinds = entry["scalars"].as_array().context("native scalars missing")?.iter()
            .map(|p| match p["type"].as_str() {
                Some("int32") => Ok(b'i'), Some("int64") => Ok(b'l'), Some("float32") => Ok(b'f'),
                _ => anyhow::bail!("{name}: unsupported native scalar"),
            }).collect::<Result<_>>()?;
        let scratch = entry["scratch_bytes_at_capacity"].as_object().context("native scratch inventory missing")?
            .iter().map(|(key, n)| Ok((key.clone(), n.as_u64().context("invalid native scratch size")?)))
            .collect::<Result<_>>()?;
        Ok(Self { name, geometry: entry["geometry"].clone(),
            capacity_rows: entry["capacity_rows"].as_u64().context("native capacity missing")?,
            pointers, scalar_kinds, scratch })
    }
}

fn integer(spec: &NativeProgram, field: &str, expected: usize) -> Result<()> {
    ensure!(spec.geometry[field].as_u64() == Some(expected as u64),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

fn number(spec: &NativeProgram, field: &str, expected: f64) -> Result<()> {
    ensure!(spec.geometry[field].as_f64() == Some(expected),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

pub fn norm_dtype(spec: &NativeProgram) -> Result<DType> {
    match spec.geometry["norm_dtype"].as_str() {
        Some("torch.float32") => Ok(DType::F32),
        Some("torch.bfloat16") => Ok(DType::Bf16),
        _ => anyhow::bail!("{}: missing or unsupported GDN norm dtype", spec.name),
    }
}

pub fn validate(cfg: &Qwen35Config, op: &str, rows: usize, spec: &NativeProgram) -> Result<()> {
    if rows > 0 {
        ensure!(spec.capacity_rows as usize >= rows, "{} capacity is smaller than {rows}", spec.name);
        integer(spec, "max_rows", rows)?;
    }
    if matches!(op, "norm" | "add" | "gdn" | "attn_producer" | "attn_o" | "ffn") {
        integer(spec, "hidden", cfg.hidden)?;
    }
    match op {
        "norm" => {
            number(spec, "eps", cfg.rms_norm_eps)?;
            ensure!(spec.geometry["zero_centered"] == true, "{}: dense norm must be zero-centered", spec.name);
        }
        "gdn" => {
            integer(spec, "key_heads", cfg.gdn.key_heads)?;
            integer(spec, "value_heads", cfg.gdn.value_heads)?;
            integer(spec, "head_dim", cfg.gdn.key_dim)?;
            integer(spec, "in_width", cfg.gdn.conv_width() + cfg.gdn.value_width() + 2 * cfg.gdn.value_heads)?;
            number(spec, "eps", cfg.rms_norm_eps)?;
            ensure!(spec.geometry["output_gate"] == "silu" && spec.geometry["fp8_weights"] == false,
                "{}: dense BF16 GDN needs SiLU and native BF16 projections", spec.name);
            norm_dtype(spec)?;
        }
        "gdn_commit" => {
            integer(spec, "key_heads", cfg.gdn.key_heads)?;
            integer(spec, "value_heads", cfg.gdn.value_heads)?;
            integer(spec, "channels", cfg.gdn.conv_width())?;
            integer(spec, "replay_rows", VERIFY_ROWS)?;
            integer(spec, "replay_bytes", NativeBuffers::new(cfg, 0, 1, 0)?.replay_per_layer)?;
        }
        "attn_producer" | "full_gqa" => {
            integer(spec, "heads", cfg.heads)?;
            integer(spec, "kv_heads", cfg.kv_heads)?;
            integer(spec, "head_dim", cfg.head_dim)?;
            integer(spec, "record_bytes", 2 * cfg.kv_heads * cfg.head_dim * 2)?;
            ensure!(spec.geometry["indexed"] == false, "{}: dense attention must not use QSA", spec.name);
            if op == "attn_producer" {
                integer(spec, "in_width", 2 * (cfg.heads + cfg.kv_heads) * cfg.head_dim)?;
                integer(spec, "rope_dim", cfg.rope_dim)?;
                number(spec, "rope_theta", cfg.rope_theta)?;
            } else {
                integer(spec, "page_rows", 64)?;
                integer(spec, "max_splits", 32)?;
                ensure!(spec.geometry["causal"] == true && spec.geometry["empty_rows_zero"] == true,
                    "{}: dense attention requires causal traversal and zero inactive rows", spec.name);
            }
        }
        "attn_o" => {
            integer(spec, "width", cfg.heads * cfg.head_dim)?;
            ensure!(spec.geometry["fp8_weights"] == false, "{}: expected native BF16 output projection", spec.name);
        }
        "ffn" => integer(spec, "intermediate", cfg.intermediate)?,
        "add" => (),
        _ => anyhow::bail!("unknown dense Qwen op {op}"),
    }
    Ok(())
}

pub fn validate_abi(op: &str, rows: usize, spec: &NativeProgram) -> Result<()> {
    let mut pointers: Vec<&str> = match op {
        "norm" => vec!["x", "weight", "out"],
        "add" => vec!["a", "b", "out"],
        "gdn_commit" => vec!["state", "conv_state", "replay", "tables"],
        "gdn" => vec!["x", "w_in", "conv_w", "a_log", "dt_bias", "norm_w", "w_out",
            "conv_state", "state", "slots", "seq_first", "out"],
        "attn_producer" => vec!["x", "w_in", "q_norm", "k_norm", "rope_positions", "kv_slots",
            "kv_cache", "query", "gate", "scratch"],
        "full_gqa" => vec!["query", "kv_cache", "positions", "page_table", "out", "scratch"],
        "attn_o" => vec!["attn", "gate", "w_o", "out", "scratch"],
        "ffn" => vec!["x", "w_gate_up", "w_down", "out", "scratch"],
        _ => anyhow::bail!("unknown dense Qwen ABI {op}"),
    };
    if op == "gdn" {
        if rows == VERIFY_ROWS { pointers.push("replay"); }
        pointers.push("scratch");
    }
    ensure!(spec.pointers == pointers, "{}: native pointer order differs from dense {op} ABI", spec.name);
    let scalars = match op {
        "gdn_commit" | "full_gqa" => 3,
        "gdn" if rows == VERIFY_ROWS => 2,
        _ => 1,
    };
    ensure!(spec.scalar_kinds == vec![b'i'; scalars], "{}: dense scalar ABI differs", spec.name);
    if pointers.contains(&"scratch") {
        ensure!(spec.scratch.get("scratch").is_some_and(|&n| n > 0 && n % 1024 == 0),
            "{}: missing or unaligned scratch admission", spec.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen::{GdnGeometry, GdnOutputGate};

    fn cfg() -> Qwen35Config {
        Qwen35Config { hidden: 4096, intermediate: 12288, vocab: 248320,
            attention: vec![HybridAttention::Gdn, HybridAttention::Full], heads: 16, kv_heads: 4, head_dim: 256,
            gdn: GdnGeometry { key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: GdnOutputGate::Silu }, rope_dim: 64, rope_theta: 1e7,
            rms_norm_eps: 1e-6, max_context: 262144, mtp_layers: 0, tied_embeddings: false }
    }

    fn inputs<'a>(cfg: &'a Qwen35Config, layers: &'a [LayerBytes], scratch: u64) -> DenseInputs<'a> {
        DenseInputs { cfg, layers, gpu: GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 1 << 30,
            baseline: Baseline::Measured { free_bytes: 14 << 30 } },
            fixed: vec![Demand::new(0, Category::Workspace, "native AOT scratch", scratch, Basis::Exact)],
            drafter: None, requested_pool: None, max_context: 262144, record_bytes: 4096,
            pool_overhead: 4, pool_bounded: Vec::new() }
    }

    #[test]
    fn native_scratch_and_exact_marks_are_charged_before_dense_pool() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 2 << 20 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let req = request(inputs(&cfg, &layers, 2 << 30))?;
        assert!(req.layers.iter().all(|layer| layer.experts.is_none()));
        assert_eq!(req.layers[0].kv_unit.unit_bytes_whole, 0);
        assert_eq!(req.layers[1].kv_unit.unit_bytes_whole, 64 * 4096);
        let base = solve(&req)?;
        assert!(base.layers.iter().all(|layer| layer.experts == ExpertHome::Dense));
        assert_eq!(base.pool_tokens, 2 << 20);
        let constrained = solve(&request(inputs(&cfg, &layers, 5 << 30))?)?;
        assert!(constrained.pool_tokens < base.pool_tokens && constrained.pool_tokens >= 262144);
        assert!(constrained.items[0].iter().any(|item| item.group == "native AOT scratch" && item.bytes == 5 << 30));
        let mut explicit = inputs(&cfg, &layers, 5 << 30);
        explicit.requested_pool = Some(2 << 20);
        assert!(solve(&request(explicit)?).is_err());
        Ok(())
    }

    #[test]
    fn drafter_owned_and_scratch_are_charged_once_before_pool_admission() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 0 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let base = solve(&request(inputs(&cfg, &layers, 2 << 30))?)?;
        let mut with_draft = inputs(&cfg, &layers, 2 << 30);
        with_draft.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30 });
        let placement = solve(&request(with_draft)?)?;
        assert_eq!(placement.movables, [(MovableId::Drafter, 0)]);
        assert!(placement.pool_tokens < base.pool_tokens && placement.pool_tokens >= 262144);
        let charged: u64 = placement.items[0].iter().filter(|item| item.category == Category::Drafter)
            .map(|item| item.bytes).sum();
        assert_eq!(charged, 4 << 30);
        let mut explicit = inputs(&cfg, &layers, 2 << 30);
        explicit.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30 });
        explicit.requested_pool = Some(base.pool_tokens);
        assert!(solve(&request(explicit)?).is_err());
        let mut overflow = inputs(&cfg, &layers, 0);
        overflow.drafter = Some(DrafterBytes { owned: u64::MAX, scratch: 1 });
        assert!(solve(&request(overflow)?).is_err());
        Ok(())
    }

    #[test]
    fn admission_rejects_missing_or_fabricated_layer_costs() {
        let cfg = cfg();
        assert!(request(inputs(&cfg, &[], 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 0, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 1, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
    }
}
