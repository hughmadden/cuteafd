//! Coordinator (RTX) ranks of a push group: the two coordinator GPUs join the
//! Spark group as ranks `sparks` and `sparks + 1`, each with its own ring and
//! admitted device arena. Plans reuse the shared builder; delivery reuses the
//! shared consumer and post-Done initialization. Nothing publishes before the
//! whole group commits; any failure drops every unpublished owner and the
//! whole group (Sparks included) reloads through the mount.
use super::{weight_arena::ArenaLayout, weight_validation::ValidationLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::{Builder, CheckedPlan, Destination}, OfficialV41Catalog, V41Exl3Layer,
    V41Exl3Residency};

/// One coordinator GPU's admitted expert layers, laid out as today's mount
/// arenas, one region per layer: EXL3 residencies of TP2 halves or of whole
/// (world 1) layers and draft stages, or native MXFP4 TP2 planes (the Spark
/// native images at world 2, packed after Done like a Spark rank's).
pub(crate) struct CoordinatorExpertPlan {
    pub residencies: Vec<V41Exl3Residency>,
    /// Each EXL3 residency's expert layer selection (the mount loader's own).
    pub selections: Vec<crate::shared::experts::layer::ExpertLayer>,
    /// Native planes per layer (empty for EXL3).
    pub native: Vec<cuteafd_loader::push_load::native::NativePlanes>,
    pub layers: Vec<usize>,
    pub gpu: usize,
    pub layout: ArenaLayout,
    pub validation: ValidationLayout,
    pub scales: super::push_native::ScalePlan,
}

impl CoordinatorExpertPlan {
    pub fn new(catalog: &OfficialV41Catalog, layers: &[usize], gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(gpu < 2, "coordinator TP2 has two GPUs");
        if catalog.exl3().is_none() { return Self::native(catalog, layers, gpu, admitted); }
        let manifest = catalog.exl3().context("coordinator TP2 push requires EXL3 checkpoint headers")?;
        let residencies = layers.iter().map(|&layer| manifest.residency(V41Exl3Layer::Backbone(layer), 2, gpu))
            .collect::<Result<Vec<_>>>()?;
        let selections = layers.iter().map(|&layer|
            crate::shared::experts::layer::ExpertLayer::BackboneTp2 { layer, rank: gpu }).collect();
        Self::exl3(residencies, selections, layers.to_vec(), gpu, admitted)
    }

    /// Whole expert layers on one coordinator GPU (`BackboneFull` layers and
    /// `Dspark` stages, world 1), exactly the local mount loader's residencies.
    pub fn whole(catalog: &OfficialV41Catalog, selections: &[crate::shared::experts::layer::ExpertLayer],
        admitted: usize) -> Result<Self> {
        use crate::shared::experts::layer::ExpertLayer;
        ensure!(selections.iter().all(|s| matches!(s, ExpertLayer::BackboneFull { .. } | ExpertLayer::Dspark { .. })),
            "a whole-layer coordinator plan takes full backbone layers and draft stages");
        let residencies = selections.iter().map(|&s| crate::shared::experts::exl3::planned_residency(catalog, s))
            .collect::<Result<Vec<_>>>()?;
        let layers = selections.iter().filter_map(|s| match s { ExpertLayer::BackboneFull { layer } => Some(*layer),
            _ => None }).collect();
        Self::exl3(residencies, selections.to_vec(), layers, 0, admitted)
    }

    fn exl3(residencies: Vec<V41Exl3Residency>, selections: Vec<crate::shared::experts::layer::ExpertLayer>,
        layers: Vec<usize>, gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(residencies.iter().all(|r| r.unit_map.is_none()), "coordinator EXL3 keeps the legacy residency");
        let sizes = residencies.iter().map(|r| r.device_arena_layout().map(|(_, bytes)| bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ArenaLayout::new(sizes, admitted)?;
        let validation = ValidationLayout::exl3(&residencies, layout.bytes() as u64, usize::MAX)?;
        let layout = layout.with_validation(validation.bytes(), validation.bytes())?;
        Ok(Self { residencies, selections, native: Vec::new(), layers, gpu, layout, validation,
            scales: Default::default() })
    }

    /// Native MXFP4 halves: each layer's four planes for intermediate half
    /// `gpu` (W1/W3 rows, W2 columns), exactly the mount's TP2 residency.
    fn native(catalog: &OfficialV41Catalog, layers: &[usize], gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(catalog.fp8().is_none() && catalog.nvfp4().is_none(),
            "coordinator native TP2 push takes MXFP4 checkpoints");
        let geometry = catalog.routed_experts();
        ensure!(geometry.intermediate % 2 == 0, "native TP2 halves need an even intermediate");
        let width = geometry.intermediate / 2;
        let planes = cuteafd_loader::push_load::native::NativePlanes::with_scale_codes(geometry.hidden as usize, width,
            width.div_ceil(128) * 128, geometry.experts as usize, false, catalog.mxfp4_scale_codes())?;
        let native = layers.iter().map(|_| planes.clone()).collect::<Vec<_>>();
        let layout = ArenaLayout::new(native.iter().map(|p| p.bytes), admitted)?;
        let validation = ValidationLayout::exl3(&[], layout.bytes() as u64, usize::MAX)?;
        let mut generated = Vec::new();
        let mut scale_layers = Vec::new();
        for (region, (&layer, planes)) in layers.iter().zip(&native).enumerate() {
            let base = layout.region(region)?.start;
            generated.extend(planes.generated_ranges().into_iter().map(|r| base + r.start..base + r.end));
            scale_layers.push((layer, planes.clone(), base));
        }
        let layout = layout.with_generated(generated)?;
        let scales = super::push_native::ScalePlan::new(catalog, scale_layers, 2, gpu)?;
        Ok(Self { residencies: Vec::new(), selections: Vec::new(), native, layers: layers.to_vec(), gpu, layout,
            validation, scales })
    }

    /// Device scratch the post-Done native plane transform needs (0 for EXL3).
    pub fn scratch_bytes(&self) -> usize { self.native.iter().map(|p| p.scratch_bytes()).max().unwrap_or(0) }
    pub fn logical_bytes(&self) -> u64 { self.scales.logical_bytes(&self.layout) }

    /// Entries for group rank `rank` (the coordinator GPU's rank in the group).
    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32) -> Result<()> {
        self.append_at(builder, catalog, rank, 0)
    }

    /// As `append`, with this plan's logical space starting at `base` of the
    /// rank's (the coordinator's dense operands come first).
    pub fn append_at(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32, base: u64) -> Result<()> {
        if !self.native.is_empty() {
            for (index, (planes, &layer)) in self.native.iter().zip(&self.layers).enumerate() {
                let range = self.layout.region(index)?;
                let destination = Destination { rank, offset: base + range.start as u64, bytes: range.len() as u64 };
                for expert in 0..planes.experts {
                    let names = catalog.expert_staging(cuteafd_loader::V41ExpertSelection::BackboneTp2 {
                        layer, expert, rank: self.gpu })?.tensor_names().to_vec();
                    builder.v41_resident(catalog, &names, planes, expert, 2, self.gpu as u32, destination, None)?;
                }
            }
            self.scales.append(builder, rank, base + self.scales.base(&self.layout))?;
            return Ok(());
        }
        for (index, residency) in self.residencies.iter().enumerate() {
            let range = self.layout.region(index)?;
            let (offsets, _) = residency.device_arena_layout()?;
            builder.exl3_residency(catalog, residency, &offsets, Destination {
                rank, offset: base + range.start as u64, bytes: range.len() as u64 })?;
        }
        let validation = ValidationLayout::exl3(&self.residencies, base + self.layout.bytes() as u64, usize::MAX)?;
        ensure!(validation.bytes() == self.validation.bytes(), "shifted validation differs");
        validation.append(builder, catalog, rank)
    }

    /// This rank's checked entries alone, against a group of `world` ranks.
    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &std::path::Path, world: usize, rank: usize)
        -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.logical_bytes();
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

