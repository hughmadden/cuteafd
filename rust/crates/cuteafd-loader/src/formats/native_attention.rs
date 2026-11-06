//! Native attention geometry shared by checkpoint inventories and runtime owners.
use super::trellis_core::CoreSplit;
use anyhow::{Context, Result};

/// Unrotated selected-view attention with the same logical TP axes as K6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProjection { pub shape: [usize; 2], pub fp8: bool, pub split: CoreSplit }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLayout {
    pub stem: String, pub shape: [usize; 2], pub fp8: bool,
    pub rank: usize, pub ranks: usize, pub split: CoreSplit,
}
impl NativeLayout {
    pub fn bytes(&self) -> Result<u64> {
        let [n, k] = self.shape;
        let values = n.checked_mul(k).and_then(|n| n.checked_mul(if self.fp8 { 1 } else { 2 }))
            .context("native attention projection extent overflow")?.max(256);
        let scales = if self.fp8 { n.div_ceil(128).checked_mul(k.div_ceil(128)).and_then(|n| n.checked_mul(4))
            .context("native attention scale extent overflow")?.max(256) } else { 0 };
        Ok(values.checked_add(scales).context("native attention projection byte overflow")? as u64)
    }
}

/// FP8-compact is a conversion of native BF16, never a requantization of native FP8.
#[derive(Debug, thiserror::Error)]
#[error("{tensor}: fp8-compact attention requires native BF16, found {found}")]
pub struct UnsupportedFp8Conversion { pub tensor: String, pub found: String }
