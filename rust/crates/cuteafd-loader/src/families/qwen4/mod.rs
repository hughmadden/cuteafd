//! Qwen 3.8 Flash Next (qwen4_exp) family: configuration and the PLE n-gram
//! hashing for the generic engine.
pub mod config;
pub mod ngram;
pub mod resident;
pub mod rope;
pub use rope::{ImageSpan, RopeError, RopePositions};
pub use config::{Qwen4Attention, Qwen4Config};
pub use ngram::{NgramHasher, NgramHistory};

/// Whole-step local owners are serial; Spark table slots follow the shared depth.
pub fn prefill_limits(local: bool) -> crate::serving_capacity::prefill_waves::PrefillLimits {
    crate::serving_capacity::prefill_waves::PrefillLimits {
        max_waves: if local { 1 } else { 4 },
        ..crate::plan::Family::prefill_limits(&crate::plan::families::qwen::QWEN4)
    }
}

/// Native verify capacity shared by scheduler geometry and retained-source admission.
pub const VERIFY_ROWS: usize = 64;

/// Worst-case source ownership with no assumed GPU progress. Unit hooks still
/// restage metadata (Q4); every such group needs its own source until read.
#[allow(clippy::too_many_arguments)]
pub fn upload_shape(rows: usize, decode: bool, pages: usize, pools: usize, ple_rows: usize,
    layers: usize, units: usize, spark: bool, mtp: bool, ple_ids: bool)
    -> anyhow::Result<crate::shared::token_table::UploadShape> {
    use anyhow::Context;
    let tables = StepTableBytes::new(rows, decode, pages, pools, ple_rows)?;
    let mut shape = crate::shared::token_table::UploadShape::default();
    let planes = [tables.positions, tables.rope_positions, tables.block_rope_positions,
        tables.kv_slots, tables.slots, tables.seq_first, tables.pool_slots,
        tables.cache_lengths, tables.page_table, tables.pool_table];
    let hooks = if units > 0 { layers.checked_mul(if spark { 3 } else { 2 })
        .and_then(|n| n.checked_add(2)).context("Qwen upload hook count overflow")? } else { 1 };
    let groups = hooks.checked_mul(units.max(1)).context("Qwen upload group count overflow")?;
    shape.append(&planes, groups)?;
    if ple_ids { shape.append(&[tables.ple_ids], groups)?; }
    shape.append(&[tables.ids], units.max(1))?;
    if mtp && units == 0 {
        if decode {
            // shared/qwen/scheduler.rs (resource::trim_budget) trims sum(depth + 1) to geometry.decode_rows;
            // qwen4/serve.rs supplies VERIFY_ROWS through engine::DECODE_ROWS.
            // Pending catch-up and the subsequent draft chain each have at most
            // VERIFY_ROWS rows. Per-call alignment is bounded by one-row calls.
            let row = StepTableBytes::new(1, true, pages, pools, ple_rows)?;
            let mtp_planes = [row.positions, row.rope_positions, row.block_rope_positions,
                row.kv_slots, row.slots, row.seq_first, row.pool_slots, row.cache_lengths,
                row.page_table, row.pool_table, 4, 4];
            shape.append(&mtp_planes, 2 * VERIFY_ROWS)?;
        } else {
            shape.append(&planes, units.max(1))?;
            shape.append(&[rows.checked_mul(4).context("Qwen MTP upload rows overflow")?, tables.ids], units.max(1))?;
            // Packed MTP members divide the admitted rows/page tables, but each
            // independently queued call retains twelve immutable source ranges.
            shape.fragment(12, rows.saturating_sub(1))?;
        }
    }
    Ok(shape)
}

/// Separate prefill MTP workspace: no target hooks or PLE ids use this owner.
pub fn mtp_upload_shape(rows: usize, pages: usize, pools: usize, calls: usize)
    -> anyhow::Result<crate::shared::token_table::UploadShape> {
    let tables = StepTableBytes::new(rows, false, pages, pools, 0)?;
    let mut shape = crate::shared::token_table::UploadShape::default();
    shape.append(&[tables.positions, tables.rope_positions, tables.block_rope_positions,
        tables.kv_slots, tables.slots, tables.seq_first, tables.pool_slots,
        tables.cache_lengths, tables.page_table, tables.pool_table, tables.ids, tables.ids], calls)?;
    // Every live unit may split into one MTP call per row. Page lists are
    // disjoint within a unit, so only fence/alignment overhead multiplies.
    let extra = rows.saturating_sub(1).checked_mul(calls)
        .ok_or_else(|| anyhow::anyhow!("Qwen packed MTP call count overflow"))?;
    shape.fragment(12, extra)?;
    Ok(shape)
}

/// Destination extents shared by workspace allocation and ordered H2D admission.
#[derive(Debug)]
pub struct StepTableBytes {
    pub positions: usize, pub rope_positions: usize, pub block_rope_positions: usize,
    pub kv_slots: usize, pub slots: usize, pub seq_first: usize, pub pool_slots: usize,
    pub cache_lengths: usize, pub page_table: usize, pub pool_table: usize,
    pub ids: usize, pub ple_ids: usize,
}

impl StepTableBytes {
    pub fn new(rows: usize, decode: bool, pages: usize, pool_pages: usize, ple_rows: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        let mul = |a: usize, b: usize| a.checked_mul(b).context("Qwen host table size overflow");
        let table_rows = if decode { rows } else { 1 };
        Ok(Self { positions: mul(rows, 8)?, rope_positions: mul(rows, 12)?,
            block_rope_positions: mul(rows, 12)?, kv_slots: mul(rows, 8)?,
            slots: mul(rows, 4)?, seq_first: mul(rows, 4)?, pool_slots: mul(rows, 8)?,
            cache_lengths: mul(rows, 4)?, page_table: mul(mul(table_rows, pages)?, 4)?,
            pool_table: mul(mul(table_rows, pool_pages)?, 4)?, ids: mul(rows, 4)?,
            ple_ids: mul(mul(rows, ple_rows)?, 8)? })
    }


}

/// Immutable full-attention records. Index keys remain BF16 in either format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen4KvCache {
    Bf16,
    /// E4M3 K then V, followed by per-token/per-head FP32 K and V descales.
    #[default]
    Fp8,
}

impl Qwen4KvCache {
    /// Separate K or V plane with inline head descales; align vector loads.
    pub const fn plane_record_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::Bf16 => kv_heads * head_dim * 2,
            Self::Fp8 => (kv_heads * (head_dim + 4) + 15) / 16 * 16,
        }
    }

    pub const fn record_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::Bf16 => 2 * kv_heads * head_dim * 2,
            Self::Fp8 => 2 * kv_heads * (head_dim + 4),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unsupported Qwen KV format {0:?}: expected bf16 or fp8")]
pub struct QwenKvFormatError(String);

impl std::str::FromStr for Qwen4KvCache {
    type Err = QwenKvFormatError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => Err(QwenKvFormatError(value.into())),
        }
    }
}

#[cfg(test)]
mod kv_record_tests {
    use super::Qwen4KvCache;

    #[test]
    fn retained_source_schedule_covers_units_and_live_verify_rows() -> anyhow::Result<()> {
        use super::{upload_shape, mtp_upload_shape, VERIFY_ROWS};
        use crate::shared::token_table::{UploadCursor, UploadShape};
        let plain = upload_shape(4096, false, 16384, 4096, 16, 48, 3, true, false, false)?;
        assert_eq!(plain.bytes, 143_572_992);
        assert_eq!(plain.capacity_bytes()?, 143_638_512);
        assert_eq!(plain, upload_shape(4096, false, 16384, 4096, 16, 48, 3, true, true, false)?,
            "separate MTP workspace must not duplicate source charges on target");
        let one = upload_shape(4096, false, 16384, 4096, 16, 48, 1, true, false, false)?;
        assert_eq!(plain.bytes, 3 * one.bytes, "diagnostic wave1 still uses per-hook tables");
        let decode = upload_shape(VERIFY_ROWS, true, 16384, 4096, 16, 48, 0, true, true, false)?;
        let mut cursor = UploadCursor::new(decode)?;
        let planes = |rows: usize| [rows * 8, rows * 12, rows * 12, rows * 8, rows * 4,
            rows * 4, rows * 8, rows * 4, rows * 16384 * 4, rows * 4096 * 4];
        for bytes in planes(VERIFY_ROWS).into_iter().chain([VERIFY_ROWS * 4]) {
            assert!(cursor.reserve(bytes)?.is_some());
        }
        // Most fragmented legal pending/chain schedule: one live row per call.
        for _ in 0..2 * VERIFY_ROWS {
            for bytes in planes(1).into_iter().chain([4, 4]) {
                assert!(cursor.reserve(bytes)?.is_some());
            }
        }
        assert!(cursor.reserve(16)?.is_none(), "charge has exactly the scheduler source-fence count");
        let mut expected = UploadShape::default();
        let mut prefill_planes = planes(4096);
        prefill_planes[8] = 16384 * 4;
        prefill_planes[9] = 4096 * 4;
        expected.append(&prefill_planes, 3)?;
        expected.append(&[4096 * 4, 4096 * 4], 3)?;
        expected.fragment(12, (4096 - 1) * 3)?;
        assert_eq!(mtp_upload_shape(4096, 16384, 4096, 3)?, expected);
        Ok(())
    }

    #[test]
    fn packed_mtp_sources_cover_all_members_without_gpu_progress() -> anyhow::Result<()> {
        use super::{mtp_upload_shape, upload_shape};
        use crate::shared::token_table::UploadCursor;
        let rows = 64;
        let waves = 3;
        let planes = |n: usize, pages: usize, pools: usize| [n * 8, n * 12, n * 12,
            n * 8, n * 4, n * 4, n * 8, n * 4, pages * 4, pools * 4, n * 4, n * 4];
        let mut cursor = UploadCursor::new(mtp_upload_shape(rows, rows, rows, waves)?)?;
        for _ in 0..waves {
            // One-row members have independent, disjoint page lists. None of
            // their twelve source fences may be recycled before completion.
            for _ in 0..rows {
                for bytes in planes(1, 1, 1) { assert!(cursor.reserve(bytes)?.is_some()); }
            }
        }
        assert!(cursor.reserve(16)?.is_none());
        let mut local = UploadCursor::new(upload_shape(rows, false, rows, rows, 0, 48, 0, false, true, false)?)?;
        for bytes in planes(rows, rows, rows).into_iter().take(10).chain([rows * 4]) {
            assert!(local.reserve(bytes)?.is_some());
        }
        for _ in 0..rows {
            for bytes in planes(1, 1, 1) { assert!(local.reserve(bytes)?.is_some()); }
        }
        assert!(local.reserve(16)?.is_none());
        assert!(mtp_upload_shape(2, 1, 1, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn planar_records_keep_inline_scales_and_vector_alignment() {
        for (heads, dim, fp8_bytes) in [(2, 192, 400), (4, 192, 784), (2, 256, 528)] {
            assert_eq!(Qwen4KvCache::Bf16.plane_record_bytes(heads, dim), heads * dim * 2);
            assert_eq!(Qwen4KvCache::Fp8.plane_record_bytes(heads, dim), fp8_bytes);
            assert_eq!(fp8_bytes % 16, 0);
            assert!(fp8_bytes >= heads * (dim + 4));
            assert!(fp8_bytes < heads * (dim + 4) + 16);
        }
    }
}

impl std::fmt::Display for Qwen4KvCache {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" })
    }
}
