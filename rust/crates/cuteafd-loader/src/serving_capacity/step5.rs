//! Exact storage layout for the current dense-summary Step-5 kernel ABI.
//! This is cache storage, not serving admission: executable workspace, graph,
//! transport and weight owners must contribute their actual reservations too.
use super::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry};
use crate::plan::families::step5::Config;

pub const PAGE_ROWS: u64 = 64;
pub const REGION_ROWS: u64 = 8;
pub const PROXY_DIM: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSpan {
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlidingState {
    pub layer: usize,
    pub mtp: bool,
    pub key: ByteSpan,
    pub value: ByteSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionState {
    pub layer: usize,
    /// BF16 [8,256], including deterministic unused slots, on every head rank.
    pub pending_key: ByteSpan,
    pub pending_z: ByteSpan,
    /// Published complete-region summaries; only the lead rank owns the table.
    pub summaries: Option<ByteSpan>,
}

/// Offsets within one sequence's active allocation and within one mark slot.
/// Keeping the same offsets makes snapshots byte copies, never reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankState {
    pub sliding: Vec<SlidingState>,
    pub regions: Vec<RegionState>,
    pub length: ByteSpan,
    pub mtp_lengths: Vec<ByteSpan>,
    pub mtp_hidden: Option<ByteSpan>,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step5CacheLayout {
    pub ranks: Vec<RankState>,
    pub geometry: FamilyCacheGeometry,
    pub context: u64,
    pub mtp_hidden_rows: u64,
    window: u64,
}

fn invalid(what: &'static str) -> CacheGeometryError {
    CacheGeometryError::Unsupported { family: "step5", what }
}

fn mul(values: &[u64]) -> Result<u64, CacheGeometryError> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b)
        .ok_or(CacheGeometryError::Overflow("Step-5 cache storage")))
}

impl RankState {
    fn reserve(&mut self, bytes: u64) -> Result<ByteSpan, CacheGeometryError> {
        let span = ByteSpan { offset: self.bytes, bytes };
        self.bytes = self.bytes.checked_add(bytes).and_then(|end| end.checked_add(255))
            .map(|end| end & !255)
            .ok_or(CacheGeometryError::Overflow("Step-5 sequence state"))?;
        Ok(span)
    }

    fn ring(&mut self, layer: usize, mtp: bool, bytes: u64) -> Result<(), CacheGeometryError> {
        let key = self.reserve(bytes)?;
        let value = self.reserve(bytes)?;
        self.sliding.push(SlidingState { layer, mtp, key, value });
        Ok(())
    }
}

impl Step5CacheLayout {
    /// BF16 K/V, BF16 pending K/Z, E4M3-byte dense summaries. MTP is owned by
    /// rank 0 with all heads; target K/V is head-split. No 256-row MiMo limits.
    pub fn new(cfg: &Config, ranks: usize, context: u64, mtp: usize) -> Result<Self, CacheGeometryError> {
        if ![1, 2].contains(&ranks) || cfg.groups % ranks != 0 || cfg.heads % ranks != 0
            || (cfg.window, cfg.groups, cfg.dim, cfg.hidden) != (512, 4, 192, 4096)
            || cfg.full.len() != cfg.layers + cfg.mtp || cfg.layers != 92
            || mtp > cfg.mtp || cfg.full[cfg.layers..].iter().any(|full| *full)
        {
            return Err(invalid("native coordinator or MTP geometry"));
        }
        if context == 0 || context > cfg.max_context {
            return Err(invalid("context outside checkpoint extent"));
        }
        let capacity = context.div_ceil(REGION_ROWS);
        let full = cfg.full[..cfg.layers].iter().filter(|full| **full).count() as u64;
        let half_record = mul(&[cfg.groups as u64 / ranks as u64, cfg.dim as u64, 2])?;
        let window = cfg.window as u64;
        let mtp_hidden_rows = if mtp == 0 { 0 } else { window + mtp as u64 + 1 };
        let mut states = Vec::with_capacity(ranks);
        let mut costs = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let mut state = RankState {
                sliding: Vec::new(), regions: Vec::new(),
                length: ByteSpan { offset: 0, bytes: 0 },
                mtp_lengths: Vec::new(), mtp_hidden: None, bytes: 0,
            };
            state.length = state.reserve(8)?;
            for layer in 0..cfg.layers {
                if cfg.full[layer] {
                    let pending_key = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let pending_z = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let summaries = if rank == 0 {
                        Some(state.reserve(mul(&[capacity, PROXY_DIM])?)?)
                    } else { None };
                    state.regions.push(RegionState { layer, pending_key, pending_z, summaries });
                } else {
                    state.ring(layer, false, mul(&[window, half_record])?)?;
                }
            }
            if rank == 0 && mtp > 0 {
                for stage in 0..mtp {
                    state.ring(cfg.layers + stage, true,
                        mul(&[window, cfg.groups as u64, cfg.dim as u64, 2])?)?;
                    let length = state.reserve(8)?;
                    state.mtp_lengths.push(length);
                }
                state.mtp_hidden = Some(state.reserve(mul(&[mtp_hidden_rows, cfg.hidden as u64, 2])?)?);
            }
            costs.push(RankCacheGeometry {
                persistent_unit_bytes: mul(&[PAGE_ROWS, full, half_record, 2])?,
                active_state_per_sequence_bytes: state.bytes,
                retained_mark_bytes: state.bytes,
                // Staged proposals belong to the actual lane workspace, not committed state.
                ..Default::default()
            });
            states.push(state);
        }
        Ok(Self {
            ranks: states,
            geometry: FamilyCacheGeometry {
                logical_unit_rows: PAGE_ROWS,
                placement: if ranks == 1 { KvPlacement::SingleDevice } else { KvPlacement::PartitionedHeads },
                ranks: costs,
            },
            context, mtp_hidden_rows, window,
        })
    }

    /// Copy all physical SWA/MTP ring bytes and pending K/Z, but only completed
    /// summary rows. Unfinished-region K/V remains in the paged CSA tail.
    /// Stage lengths and hidden history are copied, not reset/replayed.
    pub fn snapshot_spans(&self, rank: usize, frontier: u64) -> Result<Vec<ByteSpan>, CacheGeometryError> {
        if frontier > self.context {
            return Err(invalid("snapshot frontier outside admitted context"));
        }
        let state = self.ranks.get(rank).ok_or_else(|| invalid("snapshot rank"))?;
        let mut spans = vec![state.length];
        for ring in &state.sliding {
            spans.extend([ring.key, ring.value]);
        }
        for region in &state.regions {
            spans.extend([region.pending_key, region.pending_z]);
            if let Some(summary) = region.summaries {
                let bytes = mul(&[frontier / REGION_ROWS, PROXY_DIM])?;
                if bytes > 0 {
                    spans.push(ByteSpan { offset: summary.offset, bytes });
                }
            }
        }
        spans.extend(&state.mtp_lengths);
        if let Some(hidden) = state.mtp_hidden { spans.push(hidden); }
        Ok(spans)
    }

    /// The window has no spare committed history: snapshot at the exact current
    /// frontier before another commit can overwrite it.
    pub fn capture_reach(&self) -> u64 { 0 }

    pub fn window(&self) -> u64 { self.window }

    /// Partial CSA pages copy every retained K/V row, including the causal
    /// current region; summary state is copied separately by snapshot_spans.
    pub fn csa_tail_bytes(&self, rank: usize, rows: u64) -> Result<u64, CacheGeometryError> {
        if rows > PAGE_ROWS { return Err(invalid("CSA tail longer than a page")); }
        let geometry = self.geometry.ranks.get(rank).ok_or_else(|| invalid("CSA tail rank"))?;
        mul(&[geometry.persistent_unit_bytes / PAGE_ROWS, rows])
    }

    /// Retention follows the shared slot policy. Its active-lane floor can
    /// exceed the nominal mark budget, so charge every resulting slot.
    pub fn prefix_arena_bytes(&self, lanes: u64, entries: u64, budget: u64)
        -> Result<(u64, Vec<u64>), CacheGeometryError> {
        let mark = self.ranks.iter().try_fold(0u64, |sum, rank| sum.checked_add(rank.bytes)
            .ok_or(CacheGeometryError::Overflow("Step-5 combined mark")))?;
        let slots = cuteafd_core::prefix::mark_slots_for(lanes, entries, mark, budget);
        let bytes = self.ranks.iter().map(|rank| mul(&[slots, rank.bytes])).collect::<Result<_, _>>()?;
        Ok((slots, bytes))
    }
}
