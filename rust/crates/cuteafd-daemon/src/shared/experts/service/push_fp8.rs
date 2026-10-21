//! Optional native Spark transaction; prepared code/scratch survive mount fallback.
#[cfg(test)]
#[path = "push_fp8/tests.rs"]
mod tests;
use super::{modules::Fp8Group, NativeExpertServiceConfig};
use crate::shared::{
    experts::fp8::Fp8Layer,
    memory::{
        device::Device,
        push_consumer,
        push_fp8::RankPlan,
        push_initialize,
        weight_arena::{ExpectedCoverage, LoadingArena},
    },
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    push_ring::{RegisteredRing, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES},
    NativeLibrary,
};
use cuteafd_loader::{
    push_load::{JoinArgs, PushClient},
    OfficialV41Catalog,
};
use cuteafd_transport::worker_selection::push::{self, Offer};
use std::{net::TcpStream, os::fd::AsRawFd, rc::Rc};

pub(super) fn load<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
    groups: &[Fp8Group],
    resident: usize,
    stream: &mut TcpStream,
) -> Result<Option<Vec<Fp8Layer<'a>>>> {
    let planned = (|| -> Result<_> {
        let (root, host) = cuteafd_loader::push_load::namespace(&config.snapshot)?;
        let mut layers = groups
            .iter()
            .flat_map(|group| group.layers.iter().map(|&layer| (layer, group.slicing)))
            .collect::<Vec<_>>();
        layers.sort_unstable_by_key(|&(layer, _)| layer);
        let plan = RankPlan::new(catalog, &layers, config.world, config.rank, resident)?;
        let checked = plan.checked(catalog, &root, config.world, config.rank)?;
        ensure!(
            checked
                .plan
                .files
                .iter()
                .all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
            "native push source is not a SparkNest shard"
        );
        // Group scratch is already allocated; the free-memory query includes it.
        let available = library.cuda_allocatable_memory_info()?.0;
        let headroom = super::spark_runtime_headroom()?;
        let peak = super::push::admit_peak(
            plan.layout.bytes(),
            plan.host_scratch,
            plan.host_scratch,
            headroom,
            available,
        )?;
        tracing::info!(
            rank = config.rank,
            resident_bytes = plan.layout.bytes(),
            ring_bytes = DEFAULT_SLOTS * DEFAULT_SLOT_BYTES,
            headroom_bytes = headroom,
            load_peak_bytes = peak,
            available_bytes = available,
            "Spark native push admitted"
        );
        let offer = Offer {
            rank: config.rank,
            host,
            arena_bytes: plan.layout.logical_bytes() as u64,
            placement: plan.placement.clone(),
        };
        Ok((plan, checked, offer))
    })();
    if let Err(error) = &planned {
        tracing::info!(%error, "Spark native push unavailable; offering mount");
    }
    let credentials = push::offer(
        stream,
        planned.as_ref().ok().map(|(_, _, offer)| offer.clone()),
    )?;
    let Some((group, token)) = credentials else {
        return Ok(None);
    };
    let (plan, checked, _) = planned?;
    receive(
        library,
        plan,
        &checked,
        config.rank,
        &PushClient::default(),
        stream,
        group,
        token,
    )
}

fn receive<'a>(
    library: &'a NativeLibrary,
    plan: RankPlan,
    checked: &cuteafd_loader::push_load::builder::CheckedPlan,
    rank_id: usize,
    client: &PushClient,
    stream: &mut TcpStream,
    group: u64,
    token: u64,
) -> Result<Option<Vec<Fp8Layer<'a>>>> {
    let device = Device {
        library,
        id: library.cuda_get_device()?,
    };
    tracing::info!(
        group,
        rank = rank_id,
        "Spark native push rank joining load group"
    );
    let joined = (|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank_id)?)?;
        let arena = LoadingArena::new(device, plan.layout.clone(), expected)?;
        let ring = RegisteredRing::new(library, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)?;
        let rank = client.join(JoinArgs {
            group,
            token,
            rank: u32::try_from(rank_id)?,
            ring_fd: ring.fd().as_raw_fd(),
            slots: u32::try_from(ring.slots())?,
            slot_bytes: u32::try_from(ring.slot_bytes())?,
        })?;
        Ok((arena, ring, rank))
    })();
    push::joined(stream, joined.is_ok())?;
    let candidate = joined.and_then(|(arena, ring, rank)| {
        let delivered = push_consumer::consume(device, arena, ring, rank, DEFAULT_SLOTS)?;
        let initializers = plan
            .layers
            .iter()
            .map(|layer| &layer.initialize)
            .collect::<Vec<_>>();
        // Move descriptors only after consuming all fragments; no view exists before Done.
        let initializers = initializers
            .into_iter()
            .map(|layer| match layer {
                push_initialize::Layer::Fp8 {
                    widen,
                    scalars,
                    inputs,
                } => push_initialize::Layer::Fp8 {
                    widen: widen.clone(),
                    scalars: scalars.clone(),
                    inputs: inputs.clone(),
                },
                _ => unreachable!("native rank contains non-native initializer"),
            })
            .collect::<Vec<_>>();
        let ready = Rc::new(push_initialize::initialize(
            delivered,
            &initializers,
            None,
            0,
        )?);
        plan.layers
            .iter()
            .enumerate()
            .map(|(region, layer)| {
                Fp8Layer::from_push(
                    ready.clone(),
                    region,
                    layer,
                    plan.layout.region(region)?.start,
                )
            })
            .collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate {
        tracing::info!(group, rank = rank_id, %error, "Spark native push candidate rejected");
    }
    if push::completed(stream, candidate.is_ok())? {
        let layers = candidate?;
        tracing::info!(
            group,
            rank = rank_id,
            layers = layers.len(),
            "Spark native push committed; publishing ready rank arena"
        );
        return Ok(Some(layers));
    }
    drop(candidate);
    ensure!(
        !library.is_quarantined_after_failed_drain(),
        "failed push drain prevents safe mount retry"
    );
    push::drained(stream).context("native push fallback drain barrier")?;
    Ok(None)
}
