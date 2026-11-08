//! Exact aligned regions used by the coordinator's physical weight arenas.
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeMap, ops::Range};

/// One physical CUDA request and its immutable tensor views. IDs survive
/// planning/loading so granularity owners charge backings, not views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightArenaLayout {
    id: String,
    bytes: usize,
    views: BTreeMap<String, WeightArenaView>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightArenaView { offset: usize, bytes: usize }
impl WeightArenaView {
    pub fn offset(&self) -> usize { self.offset }
    pub fn bytes(&self) -> usize { self.bytes }
}

impl WeightArenaLayout {
    /// Preserve cudaMalloc's 256-byte alignment for every consumer (including
    /// the native K6 dispatcher and TMA's 16-byte tensor-map bases).
    pub fn new(id: impl Into<String>, tensors: impl IntoIterator<Item = (String, usize)>) -> Result<Self> {
        let id = id.into();
        ensure!(!id.is_empty(), "empty weight arena ID");
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        ensure!(!tensors.is_empty(), "empty weight arena scope");
        ensure!(tensors.iter().all(|(name, bytes)| !name.is_empty() && *bytes > 0), "empty weight arena tensor");
        let packed = PackedWeightRegions::new(tensors.iter().map(|(_, bytes)| *bytes))?;
        let mut views = BTreeMap::new();
        for ((name, bytes), region) in tensors.into_iter().zip(packed.regions) {
            ensure!(views.insert(name, WeightArenaView { offset: region.start, bytes }).is_none(),
                "duplicate weight arena tensor");
        }
        Ok(Self { id, bytes: packed.bytes, views })
    }
    pub fn id(&self) -> &str { &self.id }
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn views(&self) -> &BTreeMap<String, WeightArenaView> { &self.views }
    pub fn view(&self, name: &str) -> Option<WeightArenaView> { self.views.get(name).copied() }
    pub fn payload_bytes(&self) -> usize { self.views.values().map(WeightArenaView::bytes).sum() }
    pub fn padding_bytes(&self) -> usize { self.bytes - self.payload_bytes() }
}

#[derive(Debug, Clone)]
pub struct PackedWeightRegions {
    pub regions: Vec<Range<usize>>,
    pub bytes: usize,
}

impl PackedWeightRegions {
    pub fn new(sizes: impl IntoIterator<Item = usize>) -> Result<Self> {
        let mut regions = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            bytes = bytes.checked_add(255).context("weight region alignment overflow")? & !255;
            let end = bytes.checked_add(size).context("weight region extent overflow")?;
            regions.push(bytes..end);
            bytes = end;
        }
        Ok(Self { regions, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_plane_order_alignment_and_final_extent() {
        let layout = PackedWeightRegions::new([3, 0, 257, 16]).unwrap();
        assert_eq!(layout.regions, [0..3, 256..256, 256..513, 768..784]);
        assert_eq!(layout.bytes, 784);
        assert_eq!(PackedWeightRegions::new([]).unwrap().bytes, 0);
        assert!(PackedWeightRegions::new([usize::MAX, 1]).is_err());
        assert!(PackedWeightRegions::new([256, usize::MAX]).is_err());
    }

    #[test]
    fn named_arena_views_match_the_single_physical_request() -> Result<()> {
        let layout = WeightArenaLayout::new("core/gpu0", [("payload".into(), 257), ("scale".into(), 2)])?;
        assert_eq!(layout.id(), "core/gpu0");
        assert_eq!(layout.bytes(), 514);
        assert_eq!(layout.payload_bytes(), 259);
        assert_eq!(layout.padding_bytes(), 255);
        assert_eq!(layout.view("scale").unwrap().offset(), 512);
        assert_eq!(layout.view("scale").unwrap().bytes(), 2);
        assert!(layout.view("absent").is_none());
        let mut regions = layout.views().values().copied().collect::<Vec<_>>();
        regions.sort_by_key(WeightArenaView::offset);
        let mut end = 0;
        for region in regions {
            assert_eq!(region.offset() % 256, 0);
            assert!(region.offset() >= end);
            end = region.offset() + region.bytes();
        }
        assert_eq!(end, layout.bytes());
        assert!(WeightArenaLayout::new("x", []).is_err());
        assert!(WeightArenaLayout::new("", [("a".into(), 1)]).is_err());
        assert!(WeightArenaLayout::new("x", [("".into(), 1)]).is_err());
        assert!(WeightArenaLayout::new("x", [("same".into(), 1), ("same".into(), 2)]).is_err());
        assert!(WeightArenaLayout::new("x", [("empty".into(), 0)]).is_err());
        assert!(WeightArenaLayout::new("x", [("a".into(), usize::MAX), ("b".into(), 1)]).is_err());
        Ok(())
    }
}
