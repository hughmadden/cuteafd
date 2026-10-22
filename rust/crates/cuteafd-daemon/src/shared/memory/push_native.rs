//! Native MXFP4/NVFP4 rank images, sealed against worker-reported admission.
#[cfg(test)]
#[path = "push_native/tests.rs"]
mod tests;
use super::weight_arena::ArenaLayout;
use anyhow::{ensure, Context, Result};
use cuteafd_core::SparkSlice;
use cuteafd_loader::{OfficialV41Catalog, V41ExpertSelection,
    push_load::{builder::{Builder, CheckedPlan, Destination}, native::NativePlanes, PushClient, PushGroup}};
use cuteafd_transport::worker_selection::push::{Offer, Placement};
use sha2::{Digest, Sha256};
use std::{ops::Range, path::Path};

pub(crate) struct LayerPlan {
    pub slice: SparkSlice,
    pub planes: NativePlanes,
    pub scalars: Range<usize>,
}
pub(crate) struct RankPlan {
    pub layers: Vec<LayerPlan>,
    pub layout: ArenaLayout,
    pub placement: Vec<Placement>,
    pub scratch: usize,
    pub host_scratch: usize,
}

fn names(catalog: &OfficialV41Catalog, layer: usize, expert: usize, world: usize, rank: usize)
    -> Result<(Vec<String>, usize)> {
    let selection = if world == 1 { V41ExpertSelection::BackboneFull { layer, expert } }
        else { V41ExpertSelection::BackboneTp { layer, expert, world, rank } };
    if catalog.nvfp4().is_some() {
        let staging = catalog.nvfp4_expert_staging(selection)?;
        Ok((staging.tensor_names().to_vec(), staging.intermediate_size()))
    } else {
        let staging = catalog.expert_staging(selection)?;
        Ok((staging.tensor_names().to_vec(), staging.intermediate_size()))
    }
}

impl RankPlan {
    pub fn new(catalog: &OfficialV41Catalog, slices: &[SparkSlice], world: usize, rank: usize,
        admitted: usize) -> Result<Self> {
        ensure!(world > 0 && rank < world && !slices.is_empty()
            && slices.windows(2).all(|s| s[0].layer < s[1].layer), "invalid native push rank layers");
        ensure!(catalog.exl3().is_none() && catalog.fp8().is_none(), "native push cannot convert checkpoint format");
        let geometry = catalog.routed_experts();
        let mut layers = Vec::new();
        let mut placement = Vec::new();
        let mut scalar_bytes = 0usize;
        for slice in slices {
            ensure!(slice.unit_map.is_none() && slice.tiers.is_empty() && slice.projection_counts.is_empty(),
                "native push slice has compressed metadata");
            let planes = NativePlanes::new(geometry.hidden as usize, slice.width as usize,
                slice.stored_width as usize, geometry.experts as usize, catalog.nvfp4().is_some())?;
            ensure!(slice.first as usize == rank.checked_mul(planes.intermediate).context("native slice overflow")?,
                "native push slice start differs from rank");
            let mut hash = Sha256::new();
            hash.update(serde_json::to_vec(slice)?);
            hash.update(format!("{planes:?}"));
            for expert in 0..planes.experts {
                let (names, width) = names(catalog, slice.layer as usize, expert, world, rank)?;
                ensure!(width == planes.intermediate, "native push header width differs from admission");
                for name in names {
                    let source = Builder::v41_source(catalog, &name)?;
                    hash.update(serde_json::to_vec(&(name, source.path.strip_prefix(catalog.snapshot())?,
                        source.offset, source.bytes))?);
                }
            }
            let end = scalar_bytes.checked_add(planes.scalar_bytes()).context("native scalar tail overflow")?;
            layers.push(LayerPlan { slice: slice.clone(), planes, scalars: scalar_bytes..end });
            scalar_bytes = end;
            placement.push(Placement { slice: slice.clone(), fingerprint: format!("{:x}", hash.finalize()) });
        }
        let layout = ArenaLayout::new(layers.iter().map(|l| l.planes.bytes), admitted)?
            .with_validation(scalar_bytes, scalar_bytes)?;
        ensure!(layout.bytes() == admitted, "native push planes differ from mount admission");
        let scratch = layers.iter().map(|l| l.planes.scratch_bytes()).max().unwrap_or(0);
        // Scalar validation and launch-vector construction coexist with the delivered tail.
        let vectors = layers.iter().filter(|l| l.planes.nvfp4)
            .map(|l| l.planes.experts.checked_mul(128).context("native scalar scratch overflow"))
            .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        let host_scratch = scalar_bytes.checked_add(vectors).context("native host scratch overflow")?;
        Ok(Self { layers, layout, placement, scratch, host_scratch })
    }

    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, world: usize, rank: usize) -> Result<()> {
        for (region, layer) in self.layers.iter().enumerate() {
            let range = self.layout.region(region)?;
            let destination = Destination { rank: u32::try_from(rank)?, offset: range.start as u64, bytes: range.len() as u64 };
            let scalars = layer.planes.nvfp4.then_some(Destination { rank: destination.rank,
                offset: (self.layout.bytes() + layer.scalars.start) as u64, bytes: layer.scalars.len() as u64 });
            for expert in 0..layer.planes.experts {
                let (names, _) = names(catalog, layer.slice.layer as usize, expert, world, rank)?;
                builder.v41_resident(catalog, &names, &layer.planes, expert, world as u32, rank as u32, destination, scalars)?;
            }
        }
        Ok(())
    }

    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        arenas[rank] = self.layout.logical_bytes() as u64;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, world, rank)?;
        Ok(builder.finish()?)
    }
}

fn checked_plan(root: &Path, catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>) -> Result<CheckedPlan> {
    let sparks = assignments.len();
    ensure!(sparks > 0 && offers.len() == sparks + coordinator.map_or(0, |c| c.ranks),
        "native push offer world differs from admission");
    let mut plans = Vec::new();
    for (rank, (offer, assignment)) in offers[..sparks].iter().zip(assignments).enumerate() {
        let slices = offer.placement.iter().map(|p| p.slice.clone()).collect::<Vec<_>>();
        let plan = RankPlan::new(catalog, &slices, sparks, rank, usize::try_from(assignment.resident_bytes)?)?;
        super::push_exl3::validate_offer(offer, rank, assignment, &plan.placement, plan.layout.logical_bytes() as u64)?;
        plans.push(plan);
    }
    let mut builder = Builder::new(root, offers.iter().map(|o| o.arena_bytes).collect());
    for (rank, plan) in plans.iter().enumerate() { plan.append(&mut builder, catalog, sparks, rank)?; }
    if let Some(coordinator) = coordinator { (coordinator.append)(&mut builder, catalog, &offers[sparks..])?; }
    Ok(builder.finish()?)
}

pub(crate) fn prepare(snapshot: &Path, catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>) -> Result<PushGroup> {
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let checked = checked_plan(&root, catalog, assignments, offers, coordinator)?;
    ensure!(checked.plan.files.iter().all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
        "native push source is not a SparkNest shard");
    let ranks = offers.iter().map(|o| Ok(cuteafd_loader::push_load::proto::RankSpec {
        rank: u32::try_from(o.rank)?, host: o.host.clone(), arena_bytes: o.arena_bytes,
    })).collect::<Result<Vec<_>>>()?;
    let group = PushClient::default().create(ranks, checked.plan.files.clone())?;
    checked.upload(&group)?;
    Ok(group)
}
