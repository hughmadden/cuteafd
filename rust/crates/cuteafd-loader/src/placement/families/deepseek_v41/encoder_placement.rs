//! Process-lifetime ownership of V4.1's native encoder and source consumers.
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderPlacement {
    #[default]
    Ced,
    Tail14,
}
impl std::str::FromStr for EncoderPlacement {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ced" => Ok(Self::Ced), "tail14" => Ok(Self::Tail14),
            _ => Err(format!("invalid V4.1 encoder placement {value:?}: expected ced or tail14")),
        }
    }
}
impl std::fmt::Display for EncoderPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { Self::Ced => "ced", Self::Tail14 => "tail14" })
    }
}
impl EncoderPlacement {
    pub fn resolve(self, ranks: usize) -> Result<CachePlacement> {
        ensure!(matches!(ranks, 1 | 2), "V4.1 requires one or two coordinator GPUs");
        ensure!(self != Self::Tail14 || ranks == 2, "V4.1 tail14 requires two coordinator GPUs");
        let boundary = match self { Self::Ced => 20, Self::Tail14 => 14 };
        CachePlacement::new(std::array::from_fn(|layer| usize::from(ranks == 2 && layer >= boundary)))
    }
}

/// Canonical source ownership is also attention ownership. A map is immutable
/// after admission and shared by weights, producer storage, caches and hops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePlacement { attention: [usize; 40] }
impl CachePlacement {
    pub fn new(attention: [usize; 40]) -> Result<Self> {
        ensure!(attention.iter().all(|&gpu| gpu < 2), "cache GPU must be zero or one");
        for (source, end) in [(2, 8), (8, 14), (14, 20), (20, 40)] {
            ensure!(attention[source..end].iter().all(|&gpu| gpu == attention[source]),
                "compressed source {source} must share its GPU with attention consumers");
        }
        Ok(Self { attention })
    }
    pub fn encoder_decoder() -> Self { EncoderPlacement::Ced.resolve(2).expect("CED map") }
    pub fn attention(&self, layer: usize) -> Result<usize> {
        self.attention.get(layer).copied().ok_or_else(|| anyhow::anyhow!("invalid attention layer"))
    }
    pub fn layers(self) -> [usize; 40] { self.attention }
    pub fn sources(&self) -> [usize; 4] {
        [self.attention[2], self.attention[8], self.attention[14], self.attention[20]]
    }
    /// Window, C2-source and C1-source counts on a physical device.
    pub fn counts(self, gpu: usize) -> (u64, u64, u64) {
        (self.attention.iter().filter(|&&owner| owner == gpu).count() as u64,
         self.sources()[..3].iter().filter(|&&owner| owner == gpu).count() as u64,
         u64::from(self.sources()[3] == gpu))
    }
    pub fn modes(self) -> Vec<crate::placement::LayerMode> {
        self.attention.into_iter().map(|gpu| crate::placement::LayerMode::Whole {
            gpu: gpu as u8, ffn: crate::placement::FfnMode::Owner }).collect()
    }
    pub fn from_execution_plan(plan: &crate::placement::LayerExecutionPlan) -> Result<Self> {
        use crate::placement::{FfnMode, HopStorage, LayerMode, ResidualHome};
        ensure!(plan.modes().len() == 40, "V4.1 execution requires 40 layers");
        ensure!(plan.spec().storage == HopStorage::DirectInput && plan.spec().row_bytes == 40976,
            "V4.1 residual hops must land in the existing block inputs");
        let mut attention = [0; 40];
        for (layer, owner) in attention.iter_mut().enumerate() {
            let LayerMode::Whole { gpu, ffn: FfnMode::Owner } = plan.mode(layer)? else {
                anyhow::bail!("V4.1 layer {layer} requires whole attention and owner-reduced FFN")
            };
            ensure!(plan.homes()[layer + 1] == ResidualHome::Owned(gpu),
                "V4.1 residual must stay on the attention owner");
            *owner = usize::from(gpu);
        }
        Self::new(attention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legal_tail_moves_one_complete_source_group_and_one_direct_input_hop() -> Result<()> {
        use crate::placement::{plan_hops, hop_buffer_bytes, HopPoint};
        for (option, boundary, sources, counts) in [
            (EncoderPlacement::Ced, 20, [0, 0, 0, 1], [(20, 3, 0), (20, 0, 1)]),
            (EncoderPlacement::Tail14, 14, [0, 0, 1, 1], [(14, 2, 0), (26, 1, 1)]),
        ] {
            let map = option.resolve(2)?;
            assert_eq!(map.sources(), sources);
            assert_eq!([map.counts(0), map.counts(1)], counts);
            let spec = super::super::execution_hops(2);
            let hops = plan_hops(&map.modes(), &spec)?;
            assert_eq!(hops.len(), 1);
            assert_eq!(hops[0].at, HopPoint::BeforeLayer(boundary));
            assert_eq!(spec.row_bytes, 40976);
            assert_eq!(hop_buffer_bytes(&hops, &spec, 2), Some(vec![0, 0]));
            assert_eq!((map.attention(0)?, map.attention(39)?), (0, 1));
        }
        assert!(EncoderPlacement::Tail14.resolve(1).is_err());
        assert_eq!(EncoderPlacement::Ced.resolve(1)?.counts(0), (40, 3, 1));
        for boundary in [10, 19] {
            assert!(CachePlacement::new(std::array::from_fn(|l| usize::from(l >= boundary))).is_err());
        }
        assert!(CachePlacement::new(std::array::from_fn(|l| l % 2)).is_err());
        Ok(())
    }
}
