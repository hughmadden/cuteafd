//! V4.1 coordinator GPUs as push-group ranks (two-GPU serving): each GPU's
//! dense operands (landed checkpoint bytes, scale packs after Done, the FP8
//! head and dSpark experts packed on the GPU from bounded staging) and its
//! native TP2 routed halves arrive in the Spark push group. Dense records are
//! planned from headers before membership: a planning error offers no
//! coordinator ranks (the Sparks still push, the coordinator fills by mount).
use crate::shared::experts::rtx::{dense::{self, Records}, push::{CoordinatorPush, PushedExperts}};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_loader::OfficialV41Catalog;

/// Whether this launch may push its coordinator ranks (records are taken
/// while loading; unused records then fill through the mount).
pub(super) fn wanted(args: &crate::cli::NativeServeArgs, catalog: &OfficialV41Catalog) -> bool {
    !args.peers.is_empty() && catalog.exl3().is_none() && catalog.nvfp4().is_none() && catalog.fp8().is_none()
        && !args.tp2_query_projection && !args.tp2_output_projection && !args.tp2_dspark_experts
        && cuteafd_loader::push_load::namespace(&args.snapshot).is_ok()
}

/// The coordinator ranks the Spark admission starts with, or none (the
/// coordinator then fills its records through the mount): planned from the
/// recorded dense operands and the admitted TP2 halves, before membership.
pub(super) struct Planned<'a> {
    push: CoordinatorPush,
    dense: Option<(Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<dense::DenseArena<'a>>)>,
    sparks: usize,
}

#[allow(clippy::type_complexity)]
pub(super) fn plan<'a>(snapshot: &std::path::Path, catalog: &OfficialV41Catalog, devices: [Device<'a>; 2],
    records: &Records, tp2_layers: usize, tp2_budgets: [usize; 2], sparks: usize)
    -> Result<(Planned<'a>, Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)> {
    let dense = dense::plan_records(snapshot, records.clone(), &devices, sparks)?;
    let layers = (0..tp2_layers).collect::<Vec<_>>();
    let (push, streams, ranks) = CoordinatorPush::prepare(snapshot, Some(catalog), 2,
        (tp2_layers > 0).then(|| crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 { layers: layers.to_vec(), budgets: tp2_budgets.to_vec() }))?;
    Ok((Planned { push, dense: Some(dense), sparks }, streams, ranks))
}

impl<'a> Planned<'a> {
    /// Receives both GPUs' ranks (dense and TP2 halves) while the Sparks load.
    /// `Ok(None)`: the whole group fell back; the caller fills `records` and
    /// loads its TP2 halves through the mount.
    pub(super) fn receive(self, devices: [Device<'a>; 2], catalog: &OfficialV41Catalog)
        -> Result<Option<[PushedExperts<'a>; 2]>> {
        let Self { push, dense, sparks } = self;
        let started = std::time::Instant::now();
        let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), sparks, dense.map(Ok), || ());
        let pushed = pushed?;
        tracing::info!(pushed = pushed.is_some(), elapsed_ms = started.elapsed().as_millis(),
            "V4.1 coordinator push ranks received");
        pushed.map(|ranks| <[_; 2]>::try_from(ranks).map_err(|_| anyhow::anyhow!("two coordinator GPUs"))).transpose()
    }
}

/// Fills `records` through the mount on their GPUs (default streams).
pub(super) fn fill(library: &cuteafd_ffi::NativeLibrary, records: &Records) -> Result<()> {
    let started = std::time::Instant::now();
    dense::fill_records_by_mount(library, records, |_| std::ptr::null_mut())
        .context("V4.1 coordinator dense mount fill")?;
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    tracing::info!(bytes, operands, staged, steps, largest, finishes, elapsed_ms = started.elapsed().as_millis(),
        "V4.1 coordinator dense records filled through the mount");
    Ok(())
}
