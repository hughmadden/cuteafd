//! Optional EXL3 startup transaction; all GPU owners remain on the worker thread.
#[cfg(test)]
mod tests;
use super::NativeExpertServiceConfig;
use crate::shared::{experts::{exl3::Exl3Weights, layer::ExpertLoadBudget}, memory::{
    device::Device, push_consumer, push_exl3::RankPlan, push_initialize,
    weight_arena::{ExpectedCoverage, LoadingArena},
}};
use anyhow::{ensure, Result};
use cuteafd_ffi::{push_ring::{RegisteredRing, DEFAULT_SLOT_BYTES, DEFAULT_SLOTS}, NativeLibrary};
use cuteafd_loader::{push_load::{JoinArgs, PushClient}, OfficialV41Catalog};
use cuteafd_transport::worker_selection::push::{self, Offer};
use std::{net::TcpStream, os::fd::AsRawFd, rc::Rc};

pub(super) fn load<'a>(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig, layers: &[usize], budgets: &[ExpertLoadBudget],
    resident: usize, stream: &mut TcpStream) -> Result<Option<Vec<Exl3Weights<'a>>>> {
    // Metadata only: failure here sends a mount offer, with no push allocation.
    let planned = (|| -> Result<_> {
        let (root, host) = cuteafd_loader::push_load::namespace(&config.snapshot)?;
        let plan = RankPlan::new(catalog, layers, config.world, config.rank, resident)?;
        let checked = plan.checked(catalog, &root, config.world, config.rank)?;
        ensure!(checked.plan.files.iter().all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
            "push source is not a SparkNest shard");
        // Do not exceed the already admitted mount loader's transient host peaks.
        let pinned = budgets.iter().map(|b| b.pinned_host_bytes).max().unwrap_or(0);
        let host_budget = budgets.iter().map(|b| b.read_scratch_bytes).max().unwrap_or(0);
        ensure!(DEFAULT_SLOTS * DEFAULT_SLOT_BYTES <= pinned && plan.validation.bytes() <= host_budget,
            "push ring or validation exceeds existing mount admission");
        let offer = Offer { rank: config.rank, host, arena_bytes: plan.layout.logical_bytes() as u64,
            placement: plan.placement.clone() };
        Ok((plan, checked, offer))
    })();
    if let Err(error) = &planned { tracing::info!(%error, "Spark push unavailable; offering mount"); }
    let credentials = push::offer(stream, planned.as_ref().ok().map(|(_, _, offer)| offer.clone()))?;
    let Some((group, token)) = credentials else { return Ok(None); };
    let (plan, checked, _) = planned?;
    receive(library, plan, &checked, budgets, config.rank, &PushClient::default(), stream, group, token)
}

fn receive<'a>(library: &'a NativeLibrary, plan: RankPlan,
    checked: &cuteafd_loader::push_load::builder::CheckedPlan, budgets: &[ExpertLoadBudget],
    rank_id: usize, client: &PushClient, stream: &mut TcpStream, group: u64, token: u64,
) -> Result<Option<Vec<Exl3Weights<'a>>>> {
    tracing::info!(group, rank = rank_id, "Spark push rank joining load group");
    let device = Device { library, id: library.cuda_get_device()? };
    let joined = (|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank_id)?)?;
        let arena = LoadingArena::new(device, plan.layout.clone(), expected)?;
        let ring = RegisteredRing::new(library, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)?;
        let rank = client.join(JoinArgs { group, token,
            rank: u32::try_from(rank_id)?, ring_fd: ring.fd().as_raw_fd(),
            slots: u32::try_from(ring.slots())?, slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, rank))
    })();
    push::joined(stream, joined.is_ok())?;
    let candidate = joined.and_then(|(arena, ring, rank)| {
        let delivered = push_consumer::consume(device, arena, ring, rank, DEFAULT_SLOTS)?;
        let initialize = plan.residencies.iter().enumerate().map(|(region, residency)|
            push_initialize::Layer::Exl3 { region, residency }).collect::<Vec<_>>();
        let ready = Rc::new(push_initialize::initialize(delivered, &initialize, Some(&plan.validation), 0)?);
        plan.residencies.into_iter().zip(budgets).enumerate().map(|(region, (layout, &budget))|
            Exl3Weights::from_push(ready.clone(), region, layout, budget)).collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate { tracing::info!(group, rank = rank_id, %error, "Spark push candidate rejected"); }
    if push::completed(stream, candidate.is_ok())? {
        let weights = candidate?;
        tracing::info!(group, rank = rank_id, layers = weights.len(), "Spark EXL3 push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    // Drop every rank-arena reference before the drained acknowledgment. The
    // coordinator releases mount retry only after all ranks cross this barrier.
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe mount retry");
    push::drained(stream)?;
    tracing::info!(group, rank = rank_id, "whole Spark push group discarded; loading through mount");
    Ok(None)
}
