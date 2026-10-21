//! Checkpoint-native EXLR8 rank plans and worker-reported placement seals.
#[cfg(test)]
#[path = "push_exlr8/tests.rs"]
mod tests;
use super::weight_arena::ArenaLayout;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{plan::exlr8::residency::{ExpertSet, Residency},
    push_load::{builder::{Builder, CheckedPlan, Destination}, PushClient, PushGroup}};
use cuteafd_transport::worker_selection::push::{Offer, Placement};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) struct RankPlan {
    pub residencies: Vec<Residency>,
    pub layout: ArenaLayout,
    pub placement: Vec<Placement>,
}

impl RankPlan {
    pub fn new(snapshot: &Path, manifest: &str, layers: &[usize], world: usize, rank: usize,
        admitted: usize) -> Result<Self> {
        let residencies = layers.iter().map(|&layer|
            Residency::plan(snapshot, manifest, layer, world, rank, ExpertSet::Routed)
                .map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
        Self::from_residencies(snapshot, residencies, admitted)
    }

    pub fn from_residencies(snapshot: &Path, residencies: Vec<Residency>, admitted: usize) -> Result<Self> {
        let layout = ArenaLayout::new(residencies.iter().map(|r| r.resident_bytes), admitted)?;
        let placement = residencies.iter().map(|r| {
            let slice = r.spark_slice()?;
            let sources = r.loads.iter().map(|unit| {
                let components = unit.source.iter().map(|source| Ok((
                    source.path.strip_prefix(snapshot)?, source.offset, source.bytes)))
                    .collect::<Result<Vec<_>>>()?;
                Ok((unit.expert, unit.unit, unit.projection, unit.bits, components,
                    unit.destination, unit.checksum))
            }).collect::<Result<Vec<_>>>()?;
            let fingerprint = format!("{:x}", Sha256::digest(serde_json::to_vec(&(
                &slice, &r.set_id, &r.manifest, r.resident_bytes, r.signs_offsets,
                r.pointers_offset, r.bits_offset, r.compact_offsets, sources))?));
            Ok(Placement { slice, fingerprint })
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { residencies, layout, placement })
    }

    pub fn append(&self, builder: &mut Builder, snapshot: &Path, rank: u32) -> Result<()> {
        for (index, residency) in self.residencies.iter().enumerate() {
            let region = self.layout.region(index)?;
            builder.exlr8(snapshot, residency, Destination { rank,
                offset: region.start as u64, bytes: region.len() as u64 })?;
        }
        Ok(())
    }

    pub fn checked(&self, snapshot: &Path, root: &Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        arenas[rank] = self.layout.logical_bytes() as u64;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, snapshot, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

pub(crate) fn prepare(snapshot: &Path, manifest: &str, layers: &[usize],
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer]) -> Result<PushGroup> {
    let started = std::time::Instant::now();
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    ensure!(offers.len() == assignments.len(), "EXLR8 push offer world differs from admission");
    let mut plans = Vec::with_capacity(offers.len());
    for start in (0..offers.len()).step_by(8) {
        std::thread::scope(|scope| -> Result<()> {
            let jobs = (start..offers.len().min(start + 8)).map(|rank| scope.spawn(move || {
                let plan = RankPlan::new(snapshot, manifest, layers, offers.len(), rank,
                    usize::try_from(assignments[rank].resident_bytes)?)?;
                super::push_exl3::validate_offer(&offers[rank], rank, &assignments[rank],
                    &plan.placement, plan.layout.logical_bytes() as u64)?;
                Ok::<_, anyhow::Error>(plan)
            })).collect::<Vec<_>>();
            for job in jobs {
                plans.push(job.join().map_err(|_| anyhow::anyhow!("EXLR8 push planner panicked"))??);
            }
            Ok(())
        })?;
    }
    let planned = started.elapsed();
    let mut builder = Builder::new(&root, offers.iter().map(|o| o.arena_bytes).collect());
    for (rank, plan) in plans.iter().enumerate() {
        plan.append(&mut builder, snapshot, u32::try_from(rank)?)?;
    }
    let appended = started.elapsed();
    let checked = builder.finish()?;
    let checked_at = started.elapsed();
    ensure!(checked.plan.files.iter().all(|file|
        cuteafd_loader::push_load::is_push_shard(&root.join(file))), "EXLR8 push source is not a SparkNest shard");
    let ranks = offers.iter().map(|offer| Ok(cuteafd_loader::push_load::proto::RankSpec {
        rank: u32::try_from(offer.rank)?, host: offer.host.clone(), arena_bytes: offer.arena_bytes,
    })).collect::<Result<Vec<_>>>()?;
    let source_checked = started.elapsed();
    let group = PushClient::default().create(ranks, checked.plan.files.clone())?;
    let created = started.elapsed();
    checked.upload(&group)?;
    tracing::info!(ranks = offers.len(), rank_plan_ms = planned.as_millis(),
        append_ms = (appended - planned).as_millis(), check_ms = (checked_at - appended).as_millis(),
        source_check_ms = (source_checked - checked_at).as_millis(),
        create_ms = (created - source_checked).as_millis(), upload_ms = (started.elapsed() - created).as_millis(),
        "Spark EXLR8 push group prepared");
    Ok(group)
}
