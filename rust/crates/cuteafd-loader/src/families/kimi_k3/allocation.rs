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

pub fn graphs(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<GraphSet> {
    ensure!((1..=2).contains(&lanes) && (1..=4096).contains(&capacity)
        && (1..=64).contains(&slots), "Kimi graph geometry");
    let mut counts = [0; 2];
    for (rank, count) in counts.iter_mut().enumerate() {
        for rows in GRAPH_ROWS.into_iter().filter(|&r| r <= capacity) {
            *count += (3 * cfg.layers + usize::from(rank == 1)) as u64;
            for layer in 0..cfg.layers {
                if resident(rank, layer) {
                    *count += 1 + u64::from(cfg.attention[layer] == KimiAttention::Kda
                        && (2..=slots).contains(&rows));
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

pub fn spark_demand(rows: usize, lane: usize) -> Demand {
    Demand::new(1, Category::Workspace, format!("Kimi Spark lane {lane} latent intake"),
        crate::serving_capacity::spark_intake_bytes(1, 6, rows as u64, 3584), Basis::Exact)
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
    rows: usize, context: usize, slots: usize, core: &super::core::CoreInventory,
    heads: bool) -> Result<Vec<Demand>> {
    let layout = NativeLaneLayout::new(cfg, rows, context, NativeScratch::from_manifest(manifest)?)?;
    let owners = crate::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    let graphs = graphs(cfg, 1, rows, slots, |rank, layer| cfg.attention[layer] == KimiAttention::Kda
        || heads || rank == usize::from(owners[layer]))?;
    let mut demands = vec![spark_demand(rows, 0)];
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
