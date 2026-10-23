//! Planning contracts for cache paths that have no native executor yet.
use super::admission::AdmissionError;
use crate::families::minimax_m3::{config::MiniMaxM3Config, reference};
use crate::plan::{layout::LayoutOptions, Hint, PlanReport};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, Item, MemoryLayout};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum KvBacking { #[default] Gpu, Host, Spark }
impl std::str::FromStr for KvBacking {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value { "gpu" => Ok(Self::Gpu), "host" => Ok(Self::Host), "spark" => Ok(Self::Spark),
            _ => Err("KV placement must be gpu, host or spark") }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CachePlan {
    pub nvfp4: bool,
    pub backing: KvBacking,
    /// Total hot-cache budget, including its slot metadata.
    pub hot_bytes: u64,
    pub host_bytes: u64,
}
impl Default for CachePlan {
    fn default() -> Self { Self { nvfp4: false, backing: KvBacking::Gpu, hot_bytes: 1 << 30, host_bytes: 192 << 30 } }
}
impl CachePlan {
    pub fn proposed(self) -> bool { self.nvfp4 || self.backing != KvBacking::Gpu }
}

#[derive(Debug, Clone, Copy)]
pub struct Records {
    pub kv: u64,
    pub index: u64,
    pub globals: u64,
    /// Proposed device mapping: 16 bytes per layer, head and 128-token block.
    pub directory: u64,
}
impl Records {
    pub fn new(cfg: &MiniMaxM3Config, nvfp4: bool) -> Self {
        let heads = (cfg.layers * cfg.kv_heads * 2) as u64;
        Self { kv: if nvfp4 { heads * (cfg.head_dim / 2 + cfg.head_dim / 16) as u64 }
            else { reference::kv_bytes_per_token(cfg, 1) as u64 },
            index: reference::index_bytes_per_token(cfg) as u64,
            globals: if nvfp4 { heads * 4 } else { 0 },
            directory: (cfg.layers * cfg.kv_heads * 16 / cfg.msa.block) as u64 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pool {
    pub tokens: u64,
    pub gpu_bytes: u64,
    pub backing_bytes: u64,
    pub hot_bytes: u64,
}

/// Payload geometry is exact; paging metadata and capacity are design estimates.
/// `remote_bytes` is the total backing budget (host or all Spark ranks).
pub fn pool(records: Records, options: CachePlan, available: u64, remote_bytes: u64,
    requested: Option<u64>) -> Result<Pool, AdmissionError> {
    let hot = if options.backing == KvBacking::Gpu { 0 } else { options.hot_bytes };
    let fixed = hot.checked_add(records.globals).ok_or(AdmissionError::Overflow("cache fixed bytes"))?;
    if fixed >= available { return Err(AdmissionError::Budget { needed: fixed + 1, available }); }
    let token_bytes = records.index + if options.backing == KvBacking::Gpu { records.kv } else { records.directory };
    let gpu_tokens = (available - fixed) / token_bytes / 128 * 128;
    let fitting = if options.backing == KvBacking::Gpu { gpu_tokens }
        else { gpu_tokens.min(remote_bytes / records.kv / 128 * 128) };
    let tokens = requested.filter(|&n| n > 0).unwrap_or(fitting);
    if tokens == 0 || tokens % 128 != 0 || tokens > fitting {
        return Err(AdmissionError::Unsupported("proposed KV pool must fit GPU/backing budgets and align to 128 tokens"));
    }
    Ok(Pool { tokens, gpu_bytes: fixed + tokens * token_bytes,
        backing_bytes: if options.backing == KvBacking::Gpu { 0 } else { tokens * records.kv }, hot_bytes: hot })
}

pub(super) fn apply(cfg: &MiniMaxM3Config, memory: &mut MemoryLayout, report: &mut PlanReport,
    opts: &LayoutOptions) -> anyhow::Result<()> {
    use anyhow::ensure;
    let options = opts.minimax_cache;
    ensure!(opts.native_mtp_layers > 0, "MiniMax cache proposals require the bundled DSpark");
    ensure!(opts.minimax_kv == crate::families::qwen4::Qwen4KvCache::Fp8,
        "MiniMax cache proposals support FP8 or NVFP4, not BF16");
    let records = Records::new(cfg, options.nvfp4);
    let sparks = memory.devices.iter().filter(|d| d.kind == DeviceKind::Spark).collect::<Vec<_>>();
    let ranks = sparks.len() as u64;
    // Equal sharding is bounded by the most occupied rank, including vision.
    let remote = if options.backing == KvBacking::Spark {
        sparks.iter().map(|d| d.capacity_bytes.saturating_sub(d.used_bytes())).min().unwrap_or(0)
            / (records.kv / ranks * 128) * (records.kv / ranks * 128) * ranks
    } else { options.host_bytes };
    let gpu = memory.devices.iter_mut().find(|d| d.kind == DeviceKind::Rtx).unwrap();
    gpu.items.retain(|i| i.category != Category::Kv);
    let available = gpu.capacity_bytes.saturating_sub(gpu.used_bytes());
    if options.backing != KvBacking::Gpu {
        // At least every layer's selected set plus one insertion block; 32
        // metadata bytes per head block. This is a capacity bound, not a hit rate.
        let blocks = cfg.layers as u64 * cfg.kv_heads as u64 * (cfg.msa.topk_blocks + 1) as u64;
        let block_bytes = records.kv / cfg.layers as u64 / cfg.kv_heads as u64 * 128;
        ensure!(options.hot_bytes >= blocks * (block_bytes + 32), "hot cache cannot hold one selected set per layer plus insertion blocks");
    }
    let result = pool(records, options, available, remote, opts.pool_tokens)?;
    gpu.kv_tokens = result.tokens;
    gpu.items.push(Item::new(Category::Kv, if options.backing == KvBacking::Gpu {
        "proposed NVFP4 K/V + FP8 index + per-head globals" } else {
        "proposed resident FP8 index + page directory + hot K/V cache + globals" },
        "NOT IMPLEMENTED", result.gpu_bytes, Basis::Estimated));
    memory.pool_tokens = result.tokens;
    if options.backing == KvBacking::Spark {
        for device in memory.devices.iter_mut().filter(|d| d.kind == DeviceKind::Spark) {
            device.items.push(Item::new(Category::Kv, "proposed full K/V backing shard", "NOT IMPLEMENTED",
                result.backing_bytes / ranks, Basis::Estimated));
            device.kv_tokens = result.tokens;
        }
    } else if options.backing == KvBacking::Host {
        report.bytes_by_owner.insert("proposed host KV backing".into(), result.backing_bytes);
    }
    report.placement_supported = false;
    report.hints.push(Hint { what: "MiniMax proposed cache execution is not implemented".into(),
        how: "Add and qualify NVFP4 pack/attention and/or device-driven selected-block paging; this capacity estimate cannot admit serving.".into() });
    memory.notes.push(format!("NOT IMPLEMENTED: {:?} backing, {} K/V B/token + {} index B/token; {} global bytes; {} hot-cache bytes; {} backing bytes; {} pool tokens (aggregate, per-sequence context cap {})",
        options.backing, records.kv, records.index, records.globals, result.hot_bytes, result.backing_bytes, result.tokens, cfg.max_position));
    if options.backing != KvBacking::Gpu {
        memory.notes.push(format!("Paging estimate includes {} B/token device directory; hot-cache budget includes slot metadata; no hit-rate or performance claim. Spark backing requires an unqualified device-initiated transport; GeForce GPUDirect RDMA is not assumed.", records.directory));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimax_nvfp4_records_and_paged_budgets() {
        let cfg = MiniMaxM3Config::from_hf(&crate::families::minimax_m3::config::tests::release_config()).unwrap();
        let fp8 = Records::new(&cfg, false);
        let fp4 = Records::new(&cfg, true);
        assert_eq!((fp8.kv, fp8.index, fp8.directory), (61_440, 7_920, 30));
        assert_eq!((fp4.kv + fp4.index, fp4.globals), (42_480, 1_920));
        let p = pool(fp4, CachePlan { nvfp4: true, ..Default::default() }, 5 << 30, 0, None).unwrap();
        assert_eq!(p.tokens, ((5u64 << 30) - 1920) / 42480 / 128 * 128);
        assert!(p.gpu_bytes <= 5 << 30);
        let options = CachePlan { backing: KvBacking::Host, ..Default::default() };
        let p = pool(fp8, options, 5 << 30, 2 << 30, None).unwrap();
        assert!(p.backing_bytes <= 2 << 30 && p.gpu_bytes <= 5 << 30);
        assert_eq!(p.backing_bytes, p.tokens * 61_440);
        assert!(pool(fp8, options, 1 << 30, 2 << 30, None).is_err());
        assert!(pool(fp8, options, 5 << 30, 2 << 30, Some(p.tokens + 128)).is_err());
        assert!(pool(fp8, options, 5 << 30, 2 << 30, Some(129)).is_err());
    }
}
