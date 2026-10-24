//! NVFP4 K/V (`--kv-cache nvfp4`): a planning contract without a native executor.
use super::admission::AdmissionError;
use crate::families::minimax_m3::{config::MiniMaxM3Config, reference};
use crate::plan::{layout::LayoutOptions, Hint, PlanReport};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, Item, MemoryLayout};

#[derive(Debug, Clone, Copy)]
pub struct Records {
    pub kv: u64,
    pub index: u64,
    pub globals: u64,
}
impl Records {
    pub fn new(cfg: &MiniMaxM3Config, nvfp4: bool) -> Self {
        let heads = (cfg.layers * cfg.kv_heads * 2) as u64;
        Self { kv: if nvfp4 { heads * (cfg.head_dim / 2 + cfg.head_dim / 16) as u64 }
            else { reference::kv_bytes_per_token(cfg, 1) as u64 },
            index: reference::index_bytes_per_token(cfg) as u64,
            globals: if nvfp4 { heads * 4 } else { 0 } }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pool {
    pub tokens: u64,
    pub gpu_bytes: u64,
}

/// Payload geometry is exact; allocations, padding and capacity are design estimates.
pub fn pool(records: Records, available: u64, requested: Option<u64>) -> Result<Pool, AdmissionError> {
    let fixed = records.globals;
    if fixed >= available { return Err(AdmissionError::Budget { needed: fixed + 1, available }); }
    let token_bytes = records.index + records.kv;
    let fitting = (available - fixed) / token_bytes / 128 * 128;
    let tokens = requested.filter(|&n| n > 0).unwrap_or(fitting);
    if tokens == 0 || tokens % 128 != 0 || tokens > fitting {
        return Err(AdmissionError::Unsupported("proposed KV pool must fit the GPU budget and align to 128 tokens"));
    }
    Ok(Pool { tokens, gpu_bytes: fixed + tokens * token_bytes })
}

pub(super) fn apply(cfg: &MiniMaxM3Config, memory: &mut MemoryLayout, report: &mut PlanReport,
    opts: &LayoutOptions) -> anyhow::Result<()> {
    use anyhow::ensure;
    ensure!(opts.native_mtp_layers > 0, "MiniMax cache proposals require the bundled DSpark");
    ensure!(opts.minimax_kv == crate::families::qwen4::Qwen4KvCache::Fp8,
        "MiniMax cache proposals support FP8 or NVFP4, not BF16");
    let records = Records::new(cfg, true);
    let gpu = memory.devices.iter_mut().find(|d| d.kind == DeviceKind::Rtx).unwrap();
    gpu.items.retain(|i| i.category != Category::Kv);
    let available = gpu.capacity_bytes.saturating_sub(gpu.used_bytes());
    let result = pool(records, available, opts.pool_tokens)?;
    gpu.kv_tokens = result.tokens;
    gpu.items.push(Item::new(Category::Kv, "proposed NVFP4 K/V + FP8 index + per-head globals",
        "NOT IMPLEMENTED", result.gpu_bytes, Basis::Estimated));
    memory.pool_tokens = result.tokens;
    report.placement_supported = false;
    report.hints.push(Hint { what: "MiniMax proposed cache execution is not implemented".into(),
        how: "Add and qualify NVFP4 pack/attention; this capacity estimate cannot admit serving.".into() });
    memory.notes.push(format!("NOT IMPLEMENTED: NVFP4 K/V, {} K/V B/token + {} index B/token; {} global bytes; {} pool tokens (aggregate, per-sequence context cap {})",
        records.kv, records.index, records.globals, result.tokens, if opts.context_tokens == 0 { cfg.max_position as u64 } else { opts.context_tokens }));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimax_nvfp4_records_and_budget() {
        let cfg = MiniMaxM3Config::from_hf(&crate::families::minimax_m3::config::tests::release_config()).unwrap();
        let fp8 = Records::new(&cfg, false);
        let fp4 = Records::new(&cfg, true);
        assert_eq!((fp8.kv, fp8.index), (61_440, 7_920));
        assert_eq!((fp4.kv + fp4.index, fp4.globals), (42_480, 1_920));
        let p = pool(fp4, 5 << 30, None).unwrap();
        assert_eq!(p.tokens, ((5u64 << 30) - 1920) / 42480 / 128 * 128);
        assert!(p.gpu_bytes <= 5 << 30);
        assert!(pool(fp4, 1920, None).is_err());
        assert!(pool(fp4, 5 << 30, Some(p.tokens + 128)).is_err());
        assert!(pool(fp4, 5 << 30, Some(129)).is_err());
        assert_eq!(pool(fp4, 5 << 30, Some(128)).unwrap().gpu_bytes, 1920 + 128 * 42_480);
    }
}
