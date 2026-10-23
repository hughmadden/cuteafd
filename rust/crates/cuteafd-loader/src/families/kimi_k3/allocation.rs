//! CPU allocation layouts shared by Kimi checkpoint planning and native owners.
use super::config::{KimiAttention, KimiK3Config};
use crate::placement::{Demand, GraphSet, Lifetime};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeScratch {
    pub kda: usize,
    pub mla: [usize; 3],
}

impl NativeScratch {
    pub fn combine(kda: [usize; 2], mla: impl IntoIterator<Item = [usize; 3]>) -> Result<Self> {
        let mut scratch = Self { kda: *kda.iter().max().unwrap(), mla: [0; 3] };
        for requirements in mla {
            for (current, required) in scratch.mla.iter_mut().zip(requirements) {
                *current = (*current).max(required);
            }
        }
        scratch.bytes()?;
        Ok(scratch)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.mla.iter().try_fold(self.kda, |sum, bytes| sum.checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("Kimi native scratch extent overflow")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLaneLayout {
    pub rows: usize,
    pub max_context: usize,
    pub operands: BTreeMap<&'static str, usize>,
    pub arena_offsets: BTreeMap<&'static str, usize>,
    pub arena_bytes: usize,
}

pub const SCRATCH_PHASES: [&[&str]; 3] = [
    &["KDA normalized", "KDA scratch"],
    &["MLA query", "MLA q residual", "MLA attention", "MLA LSE",
        "MLA producer scratch", "MLA attention scratch", "MLA output scratch"],
    &["gate-up", "activated"],
];
/// Live operands at consecutive rank-local kernel boundaries. Projection scratch
/// is a separate owner; all these phases execute on the captured lane stream.
pub const CORE_PHASES: &[&[&str]] = &[
    &["K6 q", "K6 k", "K6 v", "K6 gate"],
    &["K6 q", "K6 k", "K6 v", "K6 gate", "K6 normalized", "K6 KDA scratch"],
    &["K6 normalized"],
    &["K6 q_a", "K6 kv_a", "MLA q residual", "K6 MLA scratch"],
    &["K6 q_b", "MLA q residual", "MLA query"],
    &["MLA query", "MLA attention", "MLA LSE", "MLA attention scratch"],
    &["MLA attention", "K6 gate", "K6 normalized"],
    &["K6 FFN gate", "K6 FFN up"],
    &["K6 FFN gate", "K6 FFN up", "gate-up"],
    &["gate-up", "activated"],
    &["activated"],
];
pub const CORE_MEET_PLANES: [&str; 3] = ["K6 attention partial", "K6 FFN partial", "K6 reduced"];
pub const ARENA_ALIGNMENT: usize = 1024;

impl NativeLaneLayout {
    pub fn new(cfg: &KimiK3Config, rows: usize, max_context: usize,
        scratch: NativeScratch) -> Result<Self> {
        cfg.release_geometry()?;
        ensure!((1..=4096).contains(&rows) && (1..=1_048_576).contains(&max_context),
            "Kimi lane row/context capacity");
        let mut operands = BTreeMap::new();
        let mut plane = |name, shape: &[usize]| -> Result<()> {
            let bytes = shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size)
                .ok_or_else(|| anyhow::anyhow!("Kimi lane {name} extent overflow")))?;
            ensure!(operands.insert(name, bytes.max(16)).is_none(), "duplicate Kimi lane operand");
            Ok(())
        };
        let hidden = [rows, cfg.hidden, 2];
        plane("residual arena", &[9, rows, cfg.hidden, 2])?;
        for name in ["delta", "normalized", "attention output", "ffn output", "shared output", "routed output", "combined output"] {
            plane(name, &hidden)?;
        }
        let intermediate = (cfg.dense_intermediate / 2).max(cfg.moe_intermediate * cfg.shared_experts / 2);
        plane("gate-up", &[rows, 2, intermediate, 2])?;
        plane("activated", &[rows, intermediate, 2])?;
        for name in ["latent input", "expert combined", "latent normalized"] {
            plane(name, &[rows, cfg.latent, 2])?;
        }
        plane("router logits", &[rows, cfg.experts, 4])?;
        for name in ["route ids", "route weights"] { plane(name, &[rows, cfg.topk, 4])?; }
        plane("expert wire", &[rows, (3584 + 3584 / 32)])?;
        plane("KDA normalized", &[rows, 48, 128, 2])?;
        for name in ["KDA slots", "KDA first"] { plane(name, &[rows, 4])?; }
        plane("MLA slots", &[rows, 8])?;
        plane("MLA query", &[rows, cfg.heads, 576, 2])?;
        plane("MLA q residual", &[rows, cfg.q_lora_rank, 2])?;
        plane("MLA attention", &[rows, cfg.heads, 512, 2])?;
        plane("MLA LSE", &[rows, cfg.heads, 4])?;
        // Decode/verify permits one request per row, at most the 64-row
        // specialization. Wide prefill is one sequence. Cache pages live outside.
        let requests = rows.min(64);
        plane("MLA page table", &[requests, max_context.div_ceil(64), 4])?;
        plane("MLA lengths", &[requests, 4])?;
        plane("MLA query offsets", &[requests + 1, 4])?;
        plane("KDA scratch", &[scratch.kda])?;
        for (name, bytes) in ["MLA producer scratch", "MLA attention scratch", "MLA output scratch"]
            .into_iter().zip(scratch.mla) { plane(name, &[bytes])?; }
        let mut arena_offsets = BTreeMap::new();
        let mut arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT)
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase alignment overflow"))?;
                ensure!(arena_offsets.insert(name, offset).is_none(), "duplicate Kimi arena operand");
                end = offset.checked_add(operands[name])
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase extent overflow"))?;
            }
            arena_bytes = arena_bytes.max(end);
        }
        let layout = Self { rows, max_context, operands, arena_offsets, arena_bytes };
        layout.bytes()?;
        Ok(layout)
    }

    /// Preserve every BF16 offset. Place transient K6 operands into holes whose
    /// previous reader has completed on the same lane stream. Peer meet outputs
    /// remain separate allocations because the exchange stream reads them.
    pub fn with_core_stages(mut self, stages: BTreeMap<&'static str, usize>) -> Result<Self> {
        let mut transient = stages.iter().filter(|(name, _)| !CORE_MEET_PLANES.contains(name))
            .map(|(&name, &bytes)| (name, bytes)).collect::<Vec<_>>();
        transient.sort_by_key(|&(name, bytes)| (std::cmp::Reverse(bytes), name));
        ensure!(stages.keys().all(|name| !self.operands.contains_key(name)), "duplicate Kimi core arena operand");
        self.operands.extend(stages);
        let phases = self.core_phases();
        for (name, bytes) in transient {
            let conflicts = phases.iter().map(Vec::as_slice).chain(SCRATCH_PHASES)
                .filter(|phase| phase.contains(&name)).flat_map(|phase| phase.iter().copied())
                .filter(|other| *other != name && self.arena_offsets.contains_key(other))
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(CORE_PHASES.iter().any(|phase| phase.contains(&name)), "Kimi stage lacks a live phase: {name}");
            let mut candidates = vec![0];
            for other in &conflicts {
                candidates.push(self.arena_offsets[other].checked_add(self.operands[other])
                    .context("Kimi core arena candidate overflow")?);
            }
            candidates.sort_unstable();
            let mut selected = None;
            for start in candidates {
                let start = start.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT).context("Kimi core arena alignment overflow")?;
                let end = start.checked_add(bytes).context("Kimi core arena extent overflow")?;
                if conflicts.iter().all(|other| end <= self.arena_offsets[other]
                    || start >= self.arena_offsets[other] + self.operands[other]) {
                    selected = Some((start, end)); break;
                }
            }
            let (start, end) = selected.context("Kimi core arena placement absent")?;
            self.arena_offsets.insert(name, start);
            self.arena_bytes = self.arena_bytes.max(end);
        }
        self.bytes()?;
        Ok(self)
    }

    /// Resolve optional larger split scratch to the fused scratch fallback used
    /// by the runtime. Every listed operand is live through that kernel boundary.
    pub fn core_phases(&self) -> Vec<Vec<&'static str>> {
        CORE_PHASES.iter().map(|phase| phase.iter().map(|&name| match name {
            "K6 KDA scratch" if !self.operands.contains_key(name) => "KDA scratch",
            "K6 MLA scratch" if !self.operands.contains_key(name) => "MLA producer scratch",
            _ => name,
        }).collect()).collect()
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .try_fold(self.arena_bytes, |sum, (_, bytes)| sum.checked_add(*bytes).context("Kimi lane extent overflow"))
    }
}

impl NativeScratch {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let scratch = |name: &str| -> Result<usize> {
            let found = manifest["programs"].as_array().context("Kimi program inventory missing")?
                .iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(found.len() == 1, "Kimi requires one {name} scratch inventory");
            usize::try_from(found[0]["scratch_bytes_at_capacity"]["scratch"].as_u64()
                .with_context(|| format!("Kimi missing {name} scratch extent"))?).context("Kimi scratch overflow")
        };
        let mut mla = Vec::new();
        for suffix in ["", "_h96"] {
            for (mode, rows) in [("decode", 64), ("verify", 64), ("extend", 4096)] {
                mla.push([scratch(&format!("kimik3_mla_producer{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_dense_mla_{mode}{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_mla_output{suffix}_m{rows}"))?]);
            }
        }
        Self::combine([scratch("kimik3_kda_m64")?, scratch("kimik3_kda_m4096")?], mla)
    }
}

pub const GRAPH_RESERVE: &str = "Kimi startup graph reserve";
pub const GRAPH_ROWS: [usize; 5] = [1, 4, 16, 64, 4096];

/// Speculative verify rows replay at their exact width (no padding: every
/// row writes KDA replay records and MLA cache slots). `verify` is the
/// drafter's block (anchor + drafts), 0 without speculation.
pub fn verify_graph_rows(verify: usize, capacity: usize) -> std::ops::RangeInclusive<usize> {
    2..=verify.min(64).min(capacity)
}

/// Every captured row count: the plain buckets plus each exact verify width.
/// Plain attention (and multi-sequence KDA) keys stay on the plain buckets; the
/// added widths capture speculative single-sequence attention, residual and FFN.
pub fn graph_rows(verify: usize, capacity: usize) -> Vec<usize> {
    let mut rows = GRAPH_ROWS.into_iter().chain(verify_graph_rows(verify, capacity))
        .filter(|&r| r <= capacity).collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    rows
}

pub fn graphs(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<GraphSet> {
    ensure!((1..=2).contains(&lanes) && (1..=4096).contains(&capacity)
        && (1..=64).contains(&slots) && verify <= 64, "Kimi graph geometry");
    let mut counts = [0; 2];
    for (rank, count) in counts.iter_mut().enumerate() {
        for rows in graph_rows(verify, capacity) {
            *count += (3 * cfg.layers + usize::from(rank == 1)) as u64;
            for layer in 0..cfg.layers {
                if resident(rank, layer) {
                    if GRAPH_ROWS.contains(&rows) {
                        *count += 1 + u64::from(cfg.attention[layer] == KimiAttention::Kda
                            && (2..=slots).contains(&rows));
                    }
                    *count += u64::from(verify_graph_rows(verify, capacity).contains(&rows));
                }
            }
        }
        *count *= lanes as u64;
    }
    Ok(GraphSet::new(&counts, 256 << 10, 10, 64 << 20,
        counts.iter().sum(), Lifetime::Startup))
}

pub fn graph_demand(set: &GraphSet, rank: usize) -> Demand {
    let mut demand = set.demand(rank);
    demand.group = GRAPH_RESERVE.into();
    demand.basis = Basis::Estimated;
    demand
}

pub fn kda_demands(cfg: &KimiK3Config, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    kda_demands_for_layers(cfg.kda_layers(), slots, marks, lanes, rank)
}

pub fn kda_demands_for_layers(layers: usize, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    ensure!((1..=64).contains(&slots) && marks > 0 && (1..=2).contains(&lanes)
        && rank < 2, "Kimi KDA state geometry");
    let state = 48 * 128 * 128 * 4 + 3 * 3 * 6144 * 2;
    let replay = 64 * 48 * (3 * 128 * 4 + 4 + 3 * 128 * 2);
    Ok([(Category::Kv, "Kimi KDA active state", layers * slots * state),
        (Category::Workspace, "Kimi KDA replay", layers * lanes * replay),
        (Category::Prefix, "Kimi exact KDA marks", layers * marks * state),
        (Category::Workspace, "Kimi KDA commit tables", lanes * slots * 3 * 4)]
        .map(|(category, group, bytes)| Demand::new(rank, category, group, bytes as u64, Basis::Exact)))
}

pub fn io_extents(vocab: usize, rank: usize, rows: usize, core_head: bool) -> [usize; 3] {
    if rank == 0 { [rows * 4, 0, 0] }
    else { [0, rows.min(64) * vocab * 4, if core_head { 0 } else { 4 << 20 }] }
}

/// One return slot per prefill wave in flight, each a unit of latent rows
/// from all six ranks (`prefill_return_bytes`).
pub fn spark_demand(shape: crate::serving_capacity::PrefillShape, lane: usize) -> Demand {
    Demand::new(1, Category::Workspace, format!("Kimi Spark lane {lane} latent intake"),
        crate::serving_capacity::prefill_return_bytes(shape.waves, shape.unit_rows, 6, 3584), Basis::Exact)
}

/// Kimi's lane operands are row planes of one prefill call: each unit owns its
/// rows of them. With `waves` in flight a unit is at most `rows / waves` rows
/// (page aligned), and each in-flight wave holds one return slot of that size.
pub fn prefill_shape(rows: usize, waves: usize) -> Result<crate::serving_capacity::PrefillShape> {
    prefill_shape_with(rows, waves, None)
}

/// `unit` overrides the derived unit rows; it must be page aligned and the
/// lane must hold `waves` of them (one wave of up to the lane is always valid).
pub fn prefill_shape_with(rows: usize, waves: usize, unit: Option<usize>)
    -> Result<crate::serving_capacity::PrefillShape> {
    ensure!(waves > 0 && rows >= 64 * waves, "Kimi lane rows {rows} hold fewer than {waves} units");
    let derived = if waves == 1 { rows.next_multiple_of(64).min(4096) } else { rows / waves / 64 * 64 };
    let unit = unit.unwrap_or(derived);
    ensure!(unit * waves <= rows.next_multiple_of(64), "Kimi lane rows {rows} hold fewer than {waves} units of {unit}");
    crate::serving_capacity::PrefillShape { unit_rows: unit as u64, waves: waves as u64 }.validate()
}

pub fn exchange_demand(rows: usize, rank: u8, f32_partials: bool, progress: usize) -> Result<Demand> {
    let recv = super::core::exchange_recv_bytes(rows, f32_partials)?;
    let bytes = recv + 256 + 256 + (((crate::placement::HOP_SLOTS + 1) * 16) as usize).max(256) + progress;
    Ok(Demand::new(rank, Category::Workspace, "Kimi peer/abort/hop controls", bytes as u64, Basis::Exact))
}

pub fn code_demand(rank: usize, pending: u64, cublas_bytes: u64, lanes: usize) -> Result<Demand> {
    ensure!(rank < 2 && (1..=2).contains(&lanes), "Kimi code allowance rank/lane geometry");
    let bytes = pending.checked_add(if rank == 1 { lanes as u64 * cublas_bytes } else { 0 })
        .context("Kimi future code/handle allowance overflow")?;
    Ok(Demand::new(rank as u8, Category::Workspace, "Kimi future loaded code and private head handles", bytes, Basis::Formula))
}

/// Exact FirstCapacity TP2 owner, using the shared EXL3 capacity arenas.
pub fn expert_workspace(package: &std::path::Path, rows: usize) -> Result<u64> {
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
        .capacities(rows)?;
    let manifests = capacities.into_iter().map(|capacity| -> Result<serde_json::Value> {
        let path = package.join(format!("m{capacity}/v41_exl3.json"));
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)
            .with_context(|| format!("read {}", path.display()))?)?;
        ensure!(manifest["hidden"] == 3584 && manifest["capacity"] == capacity,
            "Kimi TP2 expert workspace geometry/capacity mismatch");
        Ok(manifest)
    }).collect::<Result<Vec<_>>>()?;
    Ok(crate::serving_capacity::exl3_workspace_bytes(&manifests, true)? + (rows * 3584 * 4) as u64)
}

pub fn expert_package(manifest: &std::path::Path) -> Result<std::path::PathBuf> {
    let parent = manifest.parent().context("Kimi manifest has no parent")?;
    [parent.join("exl3/exl3-kimik3-k23/rtx-tp2"),
        parent.join("../lib/exl3/exl3-kimik3-k23/rtx-tp2")].into_iter()
        .find(|path| path.is_dir()).context("Kimi installed TP2 expert workspace manifests missing")
}

pub fn baseline_demands(cfg: &KimiK3Config, manifest: &serde_json::Value,
    rows: usize, context: usize, slots: usize, verify: usize, core: &super::core::CoreInventory,
    heads: bool) -> Result<Vec<Demand>> {
    let layout = NativeLaneLayout::new(cfg, rows, context, NativeScratch::from_manifest(manifest)?)?;
    let owners = crate::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    let graphs = graphs(cfg, 1, rows, slots, verify, |rank, layer| cfg.attention[layer] == KimiAttention::Kda
        || heads || rank == usize::from(owners[layer]))?;
    let mut demands = vec![spark_demand(prefill_shape(rows, 2)?, 0)];
    for rank in 0..2 {
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 native operands",
            layout.bytes()? as u64, Basis::Exact));
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 token IO",
            io_extents(cfg.vocab_size, rank, rows, core.get_weight("language_model.lm_head").is_some())
                .into_iter().sum::<usize>() as u64, Basis::Exact));
        demands.push(graph_demand(&graphs, rank));
        demands.push(exchange_demand(rows, rank as u8, super::core::needs_f32_partials(cfg, core), 0)?);
        // Active state and retained marks are charged by the caller's mark policy.
        demands.extend(kda_demands(cfg, slots, 1, 1, rank as u8)?.into_iter()
            .filter(|d| d.category == Category::Workspace));
    }
    Ok(demands)
}
