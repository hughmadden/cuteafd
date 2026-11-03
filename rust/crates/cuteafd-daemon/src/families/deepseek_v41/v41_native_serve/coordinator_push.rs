//! V4.1 coordinator ranks receive device-resident dense operands and onboarded
//! experts in one committed group. Namespace launches require push; planning or
//! delivery failure never permits a mounted reread. Intentional host-mapped
//! embedding and Engram tables are not coordinator device operands.
use crate::shared::experts::rtx::{dense::{self, Records}, push::{CoordinatorPush, PushedExperts, PushPolicy}};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_loader::OfficialV41Catalog;

/// Whether this launch may push its coordinator ranks (records are taken
/// while loading; unused records then fill through the mount).
pub(super) fn wanted(args: &crate::cli::NativeServeArgs, catalog: &OfficialV41Catalog) -> Result<bool> {
    let policy = PushPolicy::for_snapshot(&args.snapshot)?;
    policy.plan(|| {
        anyhow::ensure!(!args.peers.is_empty() && catalog.exl3().is_none() && catalog.nvfp4().is_none()
            && catalog.fp8().is_none(),
            "V4.1 coordinator push does not support this weight placement yet");
        Ok(())
    }).map(|planned| planned.is_some()).map_err(Into::into)
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
        let pushed = PushPolicy::Required.committed(pushed)?;
        tracing::info!(pushed = pushed.is_some(), elapsed_ms = started.elapsed().as_millis(),
            "V4.1 coordinator push ranks received");
        pushed.map(|ranks| <[_; 2]>::try_from(ranks).map_err(|_| anyhow::anyhow!("two coordinator GPUs"))).transpose()
    }
}

/// One GPU's dense and whole-layer payload; with no Sparks this owns the driver.
pub(super) struct Whole<'a> {
    push: CoordinatorPush,
    dense: (Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<dense::DenseArena<'a>>),
    sparks: usize,
}

#[allow(clippy::type_complexity)]
pub(super) fn plan_whole<'a>(snapshot: &std::path::Path, catalog: &OfficialV41Catalog,
    device: Device<'a>, records: &Records,
    selected: &[(crate::shared::experts::layer::ExpertLayer, usize)], sparks: usize)
    -> Result<(Whole<'a>, Option<(Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)>)> {
    let dense = dense::plan_records(snapshot, records.clone(), &[device], sparks)?;
    let budget = selected.iter().try_fold(0usize, |sum, (_, bytes)| sum.checked_add(*bytes))
        .context("V4.1 whole expert resident bytes overflow")?;
    let experts = (!selected.is_empty()).then(|| crate::shared::experts::rtx::push::CoordinatorExperts::Whole {
        owners: vec![crate::shared::experts::rtx::push::WholeOwner { rank: 0,
            selections: selected.iter().map(|(selection, _)| *selection).collect(), budget }],
    });
    let (push, ranks) = if sparks == 0 {
        (CoordinatorPush::prepare_alone(snapshot, Some(catalog), 1, experts)?, None)
    } else {
        let (push, streams, ranks) = CoordinatorPush::prepare(snapshot, Some(catalog), 1, experts)?;
        (push, Some((streams, ranks)))
    };
    Ok((Whole { push, dense, sparks }, ranks))
}

impl<'a> Whole<'a> {
    pub(super) fn receive(self, device: Device<'a>, catalog: &OfficialV41Catalog,
        selected: &[(crate::shared::experts::layer::ExpertLayer, usize)], admitted: usize)
        -> Result<Vec<crate::shared::experts::layer::ExpertWeights<'a>>> {
        let Self { push, dense, sparks } = self;
        let (received, ()) = push.receive_with(vec![device], Some(catalog), sparks, Some(Ok(dense)), || ());
        let mut ranks = PushPolicy::Required.committed(received)?
            .context("V4.1 required whole payload")?;
        anyhow::ensure!(ranks.len() == 1, "V4.1 whole payload requires one coordinator rank");
        ranks.remove(0).whole_native(selected, admitted)
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

#[cfg(test)]
mod tests {
    use crate::families::deepseek_v41::v41_projection_tp2::{shard_tests, Kind, Wave, Weights};
    use crate::shared::experts::rtx::{dense::Recorder, push::tests::{assert_bytes_equal, mark_namespace,
        push_records_alone}};
    use crate::shared::memory::device::{Allocation, Device, Stream};
    use anyhow::{ensure, Context, Result};
    use cuteafd_ffi::CuteafdDeviceBuffer;

    /// TP2 q_b and o_b shards (`--tp2-query-projection`,
    /// `--tp2-output-projection`) under coordinator push: both GPUs record
    /// their halves while loading and receive them through the push protocol;
    /// every weight and packed scale equals the mount load's, and the TP2
    /// projection lanes on the pushed shards give the mounted shards' outputs
    /// bit for bit.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB (V4.1 FP8 matrix exports), two CUDA GPUs and CUTEAFD_PUSH_FIXTURE_DIR on NVMe"]
    fn projection_shards_push_byte_exact_on_two_gpus() -> Result<()> {
        let _ = tracing_subscriber::fmt().with_env_filter("info").with_test_writer().try_init();
        // SAFETY: the explicitly supplied architecture-matched library lives through the test.
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let fixtures = std::path::PathBuf::from(std::env::var("CUTEAFD_PUSH_FIXTURE_DIR")?);
        let root = tempfile::tempdir_in(&fixtures)?;
        let snapshot = root.path().canonicalize()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(&snapshot);
        mark_namespace(&snapshot)?;
        let catalog = cuteafd_loader::read_official_v41_role_catalog(&snapshot, "projections",
            |name| name.contains(".attn.wq_b.") || name.contains(".attn.wo_b."))?;
        let layers = [2usize, 20];
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for kind in [Kind::QueryB, Kind::OutputB] {
            for layer in layers { shard_tests::pattern(&catalog, layer, kind)?; }
            let load = || -> Result<[Vec<Weights<'_>>; 2]> {
                let [a, b] = [0usize, 1].map(|rank| layers.iter().map(|&layer| Weights::load(devices[rank], &catalog,
                    layer, kind, rank, Weights::load_peak_device_bytes(&lib, kind)?)).collect::<Result<Vec<_>>>());
                Ok([a?, b?])
            };
            let mounted = load()?;
            let recorder = Recorder::start(true)?.context("dense recorder")?;
            let pushed = load();
            let records = recorder.finish();
            let pushed = pushed?;
            let sockets = tempfile::tempdir_in(&fixtures)?;
            push_records_alone(&snapshot, sockets.path(), devices, records)?;
            for weights in pushed.iter().flatten() { weights.release_raw_scales(); }
            // Every shard lands byte for byte, and each GPU holds exactly the
            // bytes its loader charges (weight half and packed scales).
            let planned = Weights::device_bytes(&lib, kind)?;
            for rank in 0..2 {
                for (p, m) in pushed[rank].iter().zip(&mounted[rank]) {
                    assert_bytes_equal(&lib, p.buffers()[0], m.buffers()[0], &format!("{kind:?} layer {} rank {rank} weight", p.layer_index()))?;
                    assert_bytes_equal(&lib, p.buffers()[1], m.buffers()[1],
                        &format!("{kind:?} layer {} rank {rank} packed scales", p.layer_index()))?;
                    let delivered: usize = p.buffers().iter().map(|b| b.bytes).sum();
                    ensure!(delivered == planned, "{kind:?} layer {} rank {rank}: {delivered} B delivered, {planned} B charged",
                        p.layer_index());
                }
            }
            // Owner GPU0's TP2 lane over each shard set, on the same inputs.
            let (k, n) = kind.geometry();
            let capacity = 16u32;
            let budgets = Wave::device_bytes(&lib, kind, capacity, 0)?;
            let mut on_pushed = Wave::new([&pushed[0], &pushed[1]], capacity, 0, budgets)?;
            let mut on_mounted = Wave::new([&mounted[0], &mounted[1]], capacity, 0, budgets)?;
            let input = Allocation::new(devices[0], capacity as usize * k * 2)?;
            // The unsharded projection on GPU0, independent of the shard loader.
            let plan = devices[0].run(|| lib.v41_fp8_matrix_plan(capacity, k as u32, n as u32))?;
            let stream = Stream::new(devices[0])?;
            let scratch = Allocation::new(devices[0], plan.info().scratch_bytes as usize)?;
            let alpha = Allocation::new(devices[0], 16)?;
            let full_output = Allocation::new(devices[0], capacity as usize * n * 2)?;
            devices[0].run(|| unsafe { plan.initialize_scratch(scratch.buffer, alpha.buffer, stream.raw) })?;
            stream.drain()?;
            let mut max_error = 0f32;
            for (layer, rows, seed) in [(2usize, 1u32, 0usize), (20, 6, 7), (2, 16, 3)] {
                let host: Vec<u8> = (0..capacity as usize * k).flat_map(|i| {
                    let value = ((i + seed) % 31) as f32 / 32.0 - 0.5;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect();
                devices[0].run(|| lib.copy_h2d(input.buffer, &host))?;
                let names = shard_tests::names(layer, kind);
                let full = devices[0].own(|| crate::families::deepseek_v41::v41_tensors::NativeRtxTensors::load(&lib,
                    &catalog, &names, crate::families::deepseek_v41::v41_tensors::NativeRtxTensors::plan(&catalog, &names)?, 1 << 20))?;
                let full_scales = Allocation::new(devices[0], k * n / 32)?;
                devices[0].run(|| unsafe { lib.v41_fp8_matrix_kernel(1, k as u32, n as u32)?
                    .pack_scales(full.get().get(&names[1])?, full_scales.buffer, stream.raw) })?;
                devices[0].run(|| unsafe { plan.launch(input.buffer, full.get().get(&names[0])?, full_scales.buffer,
                    scratch.buffer, alpha.buffer, full_output.buffer, rows, stream.raw) })?;
                stream.drain()?;
                let mut expected = vec![0u8; rows as usize * n * 2];
                let full_rows = CuteafdDeviceBuffer { bytes: expected.len(), ..full_output.buffer };
                devices[0].run(|| lib.copy_d2h(&mut expected, full_rows))?;
                let mut outputs = Vec::new();
                for wave in [&mut on_pushed, &mut on_mounted] {
                    let output = runtime.block_on(unsafe { wave.execute(layer, rows, input.buffer) })?;
                    let mut bytes = vec![0u8; rows as usize * n * 2];
                    devices[0].run(|| lib.copy_d2h(&mut bytes, CuteafdDeviceBuffer { bytes: rows as usize * n * 2, ..output }))?;
                    outputs.push(bytes);
                }
                ensure!(outputs[0].iter().any(|&b| b != 0), "{kind:?} layer {layer}: projection output is all zero");
                ensure!(outputs[0] == outputs[1], "{kind:?} layer {layer} rows {rows}: pushed shards' lane differs from the mount's");
                let decode = |v: &[u8]| f32::from_bits(u32::from(u16::from_ne_bytes([v[0], v[1]])) << 16);
                for (a, b) in outputs[0].chunks_exact(2).zip(expected.chunks_exact(2)) {
                    let (a, b) = (decode(a), decode(b));
                    ensure!(a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4,
                        "{kind:?} layer {layer} rows {rows}: pushed TP2 lane {a} against the unsharded projection {b}");
                    max_error = max_error.max((a - b).abs());
                }
            }
            eprintln!("{kind:?}: layers {layers:?} pushed shards equal the mount's; their TP2 lane matches the mount's bit for bit \
                and the unsharded projection to {max_error}");
        }
        Ok(())
    }
}
