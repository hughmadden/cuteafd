use super::*;
use cuteafd_loader::push_load::builder::Builder;
use crate::shared::memory::device::Device;
use anyhow::Context;
use cuteafd_ffi::{native_library_lifetime_fixture::Fixture, push_ring::DEFAULT_SLOT_BYTES};
use cuteafd_loader::{plan::testing::{qwen4_config, qwen4_exl3, write_snapshot},
    push_load::{proto::{Event, Request}, slot, Listener, PushClient}};
use std::{fs::File, net::TcpStream, os::unix::fs::FileExt, time::Duration};

fn control_pair() -> Result<(TcpStream, TcpStream)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let client = TcpStream::connect(listener.local_addr()?)?;
    let (worker, _) = listener.accept()?;
    for stream in [&client, &worker] {
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    }
    Ok((client, worker))
}

fn read_control(stream: &TcpStream) -> Result<serde_json::Value> {
    use std::io::BufRead;
    let mut line = Vec::new();
    std::io::BufReader::new(stream).read_until(b'\n', &mut line)?;
    Ok(serde_json::from_slice(&line)?)
}

fn write_control(stream: &mut TcpStream, value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    stream.write_all(&bytes)?;
    Ok(())
}

/// Every run of `plan` for `rank`, packed into slots in plan order (a single
/// source streaming in file order).
fn pack(plan: &CheckedPlan, root: &std::path::Path, rank: u32, slot_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let sources = plan.plan.files.iter().map(|name| std::fs::read(root.join(name)))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut packed = Vec::new();
    let mut packer = slot::Packer::new(vec![0; slot_bytes]);
    for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
        let pattern = &plan.plan.patterns[entry.pattern as usize];
        for (src, dst) in pattern.runs() {
            let mut bytes = &sources[entry.file as usize][(entry.src_base + src) as usize
                ..(entry.src_base + src + u64::from(pattern.run)) as usize];
            let mut destination = entry.dst_base + dst;
            while !bytes.is_empty() {
                let n = packer.push(destination, bytes);
                if n == 0 { packed.push(packer.finish().0); packer = slot::Packer::new(vec![0; slot_bytes]); continue; }
                destination += n as u64;
                bytes = &bytes[n..];
            }
        }
    }
    if !packer.is_empty() { packed.push(packer.finish().0); }
    Ok(packed)
}

/// Two coordinator GPUs (group ranks 6 and 7 after six Spark ranks) receive
/// concurrently from one fake SparkNest, each into its own fake device. Both
/// publish mount-exact EXL3 TP2 halves on success; on a failed commit both drop
/// their arenas before the drained acknowledgment.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn coordinator_tp2_pair_is_mount_exact_on_two_devices_and_drops_before_retry() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 64 << 30)?; }
    std::fs::write(fixture.directory().join("events"), b"")?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.contains(".experts.")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        let image = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".trellis") {
                (0..source.bytes).map(|i| (i.wrapping_mul(29) >> 2) as u8 ^ 0x5a).collect()
            } else { [0u8, 0x3c].into_iter().cycle().take(source.bytes as usize).collect() };
        File::options().write(true).open(source.path)?.write_all_at(&image, source.offset)?;
    }
    let (sparks, world) = (6usize, 8usize);
    for fail in [false, true] {
        let plans = (0..2).map(|gpu| Tp2ExpertPlan::new(&catalog, &[0, 1], gpu, usize::MAX)).collect::<Result<Vec<_>>>()?;
        let checked = plans.iter().enumerate().map(|(gpu, plan)| plan.checked(&catalog, root.path(), world, sparks + gpu))
            .collect::<Result<Vec<_>>>()?;
        // Independent mount reference: the coordinator's legacy residency reads.
        let expected = plans.iter().map(|plan| plan.residencies.iter().map(|residency| -> Result<Vec<Vec<u8>>> {
            let mut image: Vec<Vec<u8>> = residency.buffers.iter().map(|b| vec![0; b.bytes.max(16)]).collect();
            let mut scratch = vec![0; residency.scratch_bytes(&catalog)?];
            for (index, load) in residency.loads.iter().enumerate() {
                let mut bytes = vec![0; load.bytes];
                residency.read_into(&catalog, index, &mut bytes, &mut scratch)?;
                for &(buffer, offset) in &load.destinations { image[buffer][offset..offset + load.bytes].copy_from_slice(&bytes); }
            }
            for (buffer, spec) in image.iter_mut().zip(&residency.buffers) {
                for (dst, word) in buffer.chunks_exact_mut(4).zip(&spec.initial_words) { dst.copy_from_slice(&word.to_le_bytes()); }
            }
            Ok(image)
        }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        let mut servers = Vec::new();
        let mut workers = Vec::new();
        for gpu in 0..2 {
            let rank = sparks + gpu;
            let socket = root.path().join(format!("{}-{gpu}.sock", if fail { "fail" } else { "good" }))
                .to_string_lossy().into_owned();
            let listener = Listener::bind(&socket)?;
            let packed = pack(&checked[gpu], root.path(), rank as u32, DEFAULT_SLOT_BYTES)?;
            let (mut coordinator, worker) = control_pair()?;
            workers.push((worker, socket));
            servers.push(std::thread::spawn(move || -> Result<()> {
                let conn = listener.accept()?;
                conn.set_timeout(Some(Duration::from_secs(10)))?;
                let (request, fd) = conn.next_request()?.context("rank did not join")?;
                let Request::Join { slots, slot_bytes, rank: joined, .. } = request else { anyhow::bail!("expected rank Join") };
                assert_eq!(joined as usize, rank);
                let file = File::from(fd.context("rank ring fd absent")?);
                conn.send_event(&Event::Joined).context("send Joined")?;
                assert_eq!(read_control(&coordinator).context("read joined")?["message"], "joined");
                for (index, bytes) in packed.into_iter().enumerate() {
                    let index = index % slots as usize;
                    file.write_all_at(&bytes, index as u64 * u64::from(slot_bytes))?;
                    conn.send_event(&Event::Filled { slot: index as u32, len: bytes.len() as u32 }).context("send Filled")?;
                    let (release, _) = conn.next_request().context("read Release")?.context("missing slot release")?;
                    assert!(matches!(release, Request::Release { slot } if slot == index as u32));
                }
                conn.send_event(&Event::Done).context("send Done")?;
                assert_eq!(read_control(&coordinator).context("read completed")?["success"], true);
                write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success":!fail})).context("write commit")?;
                if fail {
                    assert_eq!(read_control(&coordinator)?["message"], "drained");
                    write_control(&mut coordinator, &serde_json::json!({"message":"mount"}))?;
                }
                Ok(())
            }));
        }
        let before = fixture.events()?;
        // The fake DSO models CUDA queues process-wide, so the two ranks receive
        // one after the other here (each against its own concurrent fake source).
        let results = workers.into_iter().zip(plans).zip(&checked).enumerate().map(|(gpu, (((mut worker, socket), plan), checked))| {
            let client = PushClient::at(socket, Duration::from_secs(10))?;
            let device = Device { library: &library, id: gpu as i32 };
            receive_tp2(device, &catalog, plan, checked, sparks + gpu, &client, &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES)
        }).collect::<Vec<Result<_>>>();
        let served = servers.into_iter().map(|s| s.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked")).and_then(|r| r))
            .collect::<Vec<_>>();
        for (gpu, (served, received)) in served.iter().zip(&results).enumerate() {
            if let Err(e) = served { eprintln!("server {gpu}: {e:#}"); }
            if let Err(e) = received { eprintln!("rank {gpu}: {e:#}"); }
        }
        for served in served { served?; }
        let results = results.into_iter().collect::<Result<Vec<_>>>()?;
        if fail {
            assert!(results.iter().all(Option::is_none));
            assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 2, "both failed arenas freed before retry");
            continue;
        }
        for (gpu, (weights, image)) in results.into_iter().zip(expected).enumerate() {
            let weights = weights.context("successful push fell back")?.exl3()?;
            assert_eq!(weights.len(), 2);
            for (weight, image) in weights.iter().zip(image) {
                for (spec, expected) in weight.layout.buffers.iter().zip(image) {
                    let mut actual = vec![0; expected.len()];
                    Device { library: &library, id: gpu as i32 }.run(|| library.copy_d2h(&mut actual, weight.buffer(&spec.name)?))?;
                    assert_eq!(actual, expected, "GPU {gpu} {}", spec.name);
                }
            }
            drop(weights);
        }
        assert_eq!(fixture.events()?[before.len()..].matches('d').count(), 2, "each GPU's rank arena freed once");
    }
    Ok(())
}

/// Dense operands recorded in host memory (a stand-in for a family's operand
/// allocations): every planned byte must arrive, at its offset, before Done.
struct HostDense {
    image: Vec<u8>,
    planned: Vec<(u64, u64)>,
    delivered: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    device: i32,
}

impl crate::shared::memory::push_consumer::PushDestination for HostDense {
    type Delivered = ();
    fn belongs_to(&self, device: Device<'_>) -> bool { device.id == self.device }
    fn logical_bytes(&self) -> usize { self.image.len() }
    fn device_bytes(&self) -> usize { self.image.len() }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        anyhow::ensure!(self.planned.iter().any(|&(s, n)| offset >= s && offset + bytes <= s + n), "unplanned dense fragment");
        Ok(())
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        // The fake DSO copies host-to-"device" with memcpy, so a host pointer works here.
        Ok(cuteafd_ffi::CuteafdDeviceBuffer { ptr: self.image[offset..offset + bytes].as_ptr() as *mut _, bytes,
            device_id: self.device, flags: 0 })
    }
    fn copy_validation(&mut self, _: u64, _: &[u8]) -> Result<()> { anyhow::bail!("dense has no host tail") }
    fn copied(&mut self, _: u64, _: u64) -> Result<()> { Ok(()) }
    fn done(self) -> Result<()> {
        *self.delivered.lock().unwrap() = self.image.clone();
        Ok(())
    }
}

/// One GPU rank carrying dense operands in `[0, base)` and the TP2 halves from
/// `base`: both land exactly, through one ring, in one group.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn combined_rank_lands_dense_operands_and_tp2_halves_exactly() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 64 << 30)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors() {
        let image: Vec<u8> = if tensor.metadata.name.ends_with(".mcg") { 0xcbac1fedu32.to_le_bytes().to_vec() }
            else if tensor.metadata.name.ends_with(".suh") || tensor.metadata.name.ends_with(".svh") {
                [0u8, 0x3c].into_iter().cycle().take(tensor.metadata.byte_length as usize).collect()
            } else { (0..tensor.metadata.byte_length).map(|i| (i as u8).wrapping_mul(13) ^ 0x5a).collect() };
        File::options().write(true).open(root.path().join(&tensor.shard))?.write_all_at(&image, tensor.metadata.byte_offset)?;
    }
    // Dense part: two non-expert tensors landed whole, a gap between them.
    let dense_names: Vec<String> = catalog.tensors().iter().filter(|t| !t.metadata.name.contains(".experts."))
        .take(2).map(|t| t.metadata.name.clone()).collect();
    let mut planned = Vec::new();
    let mut cursor = 0u64;
    for name in &dense_names {
        let start = cursor.next_multiple_of(256);
        let bytes = catalog.tensor(name)?.metadata.byte_length;
        planned.push((start, bytes));
        cursor = start + bytes + 1;
    }
    let base = cursor.next_multiple_of(256);
    let (rank, world) = (6usize, 8usize);
    let plan = Tp2ExpertPlan::new(&catalog, &[0, 1], 0, usize::MAX)?;
    let checked_tp2 = plan.checked(&catalog, root.path(), world, rank)?;
    let mut arenas = vec![0; world];
    arenas[rank] = base + plan.layout.logical_bytes() as u64;
    let mut builder = Builder::new(root.path(), arenas);
    for (name, &(start, bytes)) in dense_names.iter().zip(&planned) {
        builder.contiguous(&Builder::v41_source(&catalog, name)?, 0, bytes,
            cuteafd_loader::push_load::builder::Destination { rank: rank as u32, offset: start, bytes })?;
    }
    plan.append_at(&mut builder, &catalog, rank as u32, base)?;
    let combined = builder.finish()?;
    let socket = root.path().join("combined.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let packed = pack(&combined, root.path(), rank as u32, DEFAULT_SLOT_BYTES)?;
    let (mut coordinator, mut worker) = control_pair()?;
    let server = std::thread::spawn(move || -> Result<()> {
        let conn = listener.accept()?;
        conn.set_timeout(Some(Duration::from_secs(10)))?;
        let (_, fd) = conn.next_request()?.context("rank did not join")?;
        let file = File::from(fd.context("rank ring fd absent")?);
        conn.send_event(&Event::Joined)?;
        assert_eq!(read_control(&coordinator)?["message"], "joined");
        for (index, bytes) in packed.into_iter().enumerate() {
            let slot = index % 8;
            file.write_all_at(&bytes, slot as u64 * DEFAULT_SLOT_BYTES as u64)?;
            conn.send_event(&Event::Filled { slot: slot as u32, len: bytes.len() as u32 })?;
            let (release, _) = conn.next_request()?.context("missing slot release")?;
            assert!(matches!(release, Request::Release { slot: s } if s == slot as u32));
        }
        conn.send_event(&Event::Done)?;
        assert_eq!(read_control(&coordinator)?["success"], true);
        write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success": true}))?;
        Ok(())
    });
    let delivered = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let dense = HostDense { image: vec![0; cursor as usize], planned: planned.clone(), delivered: delivered.clone(), device: 0 };
    let client = PushClient::at(socket, Duration::from_secs(10))?;
    let device = Device { library: &library, id: 0 };
    let weights = receive_rank(device, Some((&catalog, plan, &checked_tp2)), Some(DenseRank { destination: dense, base }),
        rank, &client, &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES)?;
    server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))??;
    let weights = weights.context("combined push fell back")?.exl3()?;
    assert_eq!(weights.len(), 2);
    let image = delivered.lock().unwrap().clone();
    for (name, &(start, bytes)) in dense_names.iter().zip(&planned) {
        let tensor = catalog.tensor(name)?;
        let mut expected = vec![0; bytes as usize];
        File::open(root.path().join(&tensor.shard))?.read_exact_at(&mut expected, tensor.metadata.byte_offset)?;
        assert_eq!(&image[start as usize..(start + bytes) as usize], &expected[..], "dense {name}");
    }
    for weight in &weights {
        let mut scratch = vec![0; weight.layout.scratch_bytes(&catalog)?];
        for (index, load) in weight.layout.loads.iter().enumerate() {
            let mut expected = vec![0; load.bytes];
            weight.layout.read_into(&catalog, index, &mut expected, &mut scratch)?;
            for &(buffer, offset) in &load.destinations {
                let mut actual = vec![0; load.bytes];
                let spec = &weight.layout.buffers[buffer];
                let buffer = weight.buffer(&spec.name)?;
                // SAFETY: the load's destination range lies inside this published buffer.
                let view = cuteafd_ffi::CuteafdDeviceBuffer { bytes: load.bytes,
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, ..buffer };
                library.copy_d2h(&mut actual, view)?;
                assert_eq!(actual, expected, "TP2 {}", load.tensor);
            }
        }
    }
    Ok(())
}

/// A dense planning failure on the coordinator must not abort the load: both
/// GPU ranks offer mount, so the group driver opens no group and sends every
/// rank (a Spark included) the mount decision, and the coordinator returns
/// `Ok(None)` (TP2 halves and dense operands then load through the mount).
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn dense_planning_error_sends_the_whole_group_to_mount() -> Result<()> {
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    // A SparkNest namespace (ADR-064 root identity, shards with hosts), so the
    // TP2 plans alone would be offered: only the dense error can decline them.
    let mark = |path: &std::path::Path, name: &std::ffi::CStr, value: &[u8]| -> Result<()> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        // SAFETY: both strings are NUL-terminated and live across the call; the
        // value pointer and length describe one live buffer.
        let status = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        anyhow::ensure!(status == 0, "setxattr: {}", std::io::Error::last_os_error());
        Ok(())
    };
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    for entry in std::fs::read_dir(root.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") { mark(&path, c"user.sparknest.hosts", b"[]")?; }
    }
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    dense_error_mount_group(&library, root.path(), &catalog, true)?;
    // Non-EXL3 groups carry dense-only coordinator ranks, not EXL3 TP2 halves.
    let root = tempfile::tempdir_in(fixture.directory())?;
    let catalog = crate::shared::memory::push_fp8::tests::fixture(root.path(),
        cuteafd_loader::formats::fp8_experts::ExpertFormat::Fp8Block128, false)?;
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    dense_error_mount_group(&library, root.path(), &catalog, false)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
    let catalog = crate::shared::memory::push_exlr8::tests::fixture_catalog(root.path())?;
    mark(root.path(), c"user.sparknest.cluster", br#"{"me":"coordinator"}"#)?;
    dense_error_mount_group(&library, root.path(), &catalog, false)?;
    Ok(())
}

fn dense_error_mount_group(library: &cuteafd_ffi::NativeLibrary, root: &std::path::Path,
    catalog: &OfficialV41Catalog, tp2: bool) -> Result<()> {
    let sparks = 1;
    if tp2 { for gpu in 0..2 {
        let plan = Tp2ExpertPlan::new(catalog, &[0, 1], gpu, usize::MAX)?;
        let checked = plan.checked(catalog, root, sparks + 2, sparks + gpu)?;
        assert!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
            "without the dense error the TP2 halves would be offered");
    } }
    let (push, coordinator_streams, _) = CoordinatorPush::prepare(root, Some(catalog), 2,
        tp2.then_some((&[0, 1][..], vec![usize::MAX; 2])))?;
    // One Spark rank offering push, then the two coordinator ranks, in group order.
    let (mut spark_driver, mut spark) = control_pair()?;
    let spark_thread = std::thread::spawn(move || cuteafd_transport::worker_selection::push::offer(&mut spark,
        Some(cuteafd_transport::worker_selection::push::Offer { rank: 0, host: "spark".into(), arena_bytes: 1,
            placement: Vec::new() })));
    let driver = std::thread::spawn(move || -> Result<bool> {
        let mut streams = vec![spark_driver.try_clone()?];
        streams.extend(coordinator_streams);
        let mut prepared = false;
        cuteafd_transport::worker_selection::push::coordinate(&mut streams, |_| {
            prepared = true;
            anyhow::bail!("no group may open after a declined coordinator rank")
        })?;
        drop(spark_driver.shutdown(std::net::Shutdown::Both));
        Ok(prepared)
    });
    let devices = [Device { library, id: 0 }, Device { library, id: 1 }];
    let dense: Option<Result<(Vec<DensePlan>, Vec<NoDense>)>> = Some(Err(anyhow::anyhow!("injected dense planning error")));
    let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), sparks, dense, || ());
    assert!(pushed?.is_none(), "a declined dense plan must fall back to mount");
    assert!(!driver.join().map_err(|_| anyhow::anyhow!("driver panicked"))??, "the group driver must not prepare a group");
    assert_eq!(spark_thread.join().map_err(|_| anyhow::anyhow!("spark panicked"))??, None,
        "the Spark rank receives the whole-group mount decision");
    Ok(())
}

/// The families plan their dense records before membership: a planning error
/// there offers no coordinator ranks, so the group driver sees the Spark
/// ranks alone and opens their push group (the coordinator then loads by
/// mount). The whole-group fallback stays only for errors after membership.
#[test]
fn pre_membership_dense_planning_error_offers_no_coordinator_ranks_and_sparks_push() -> Result<()> {
    let root = tempfile::tempdir()?;
    // The families' pre-membership step: planning fails, so nothing is offered.
    let planned = (|| -> Result<(CoordinatorPush, Vec<std::net::TcpStream>,
        crate::shared::memory::push_exl3::CoordinatorRanks)> {
        anyhow::bail!("injected dense planning error")
    })().map_err(|error| tracing::info!(%error, "coordinator push unavailable; mount")).ok();
    assert!(planned.is_none());
    let coordinator_streams = planned.map(|(_, streams, _)| streams).unwrap_or_default();
    // One Spark rank offering push, and no coordinator rank: the group opens.
    let (mut spark_driver, mut spark) = control_pair()?;
    let spark_thread = std::thread::spawn(move || cuteafd_transport::worker_selection::push::offer(&mut spark,
        Some(cuteafd_transport::worker_selection::push::Offer { rank: 0, host: "spark".into(), arena_bytes: 1,
            placement: Vec::new() })));
    let socket = root.path().join("group.sock").to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let server = std::thread::spawn(move || -> Result<usize> {
        let conn = listener.accept()?;
        conn.set_timeout(Some(Duration::from_secs(10)))?;
        let (request, _) = conn.next_request()?.context("no Create")?;
        let Request::Create { ranks, .. } = request else { anyhow::bail!("expected Create") };
        conn.send_event(&Event::Created { group: 7, token: 11 })?;
        Ok(ranks.len())
    });
    let mut streams = vec![spark_driver.try_clone()?];
    streams.extend(coordinator_streams);
    let mut offered = 0;
    let client = PushClient::at(socket, Duration::from_secs(10))?;
    let _ = cuteafd_transport::worker_selection::push::coordinate(&mut streams, |offers| {
        offered = offers.len();
        let ranks = offers.iter().map(|o| cuteafd_loader::push_load::proto::RankSpec {
            rank: o.rank as u32, host: o.host.clone(), arena_bytes: o.arena_bytes }).collect();
        Ok(client.create(ranks, Vec::new())?)
    });
    drop(spark_driver.shutdown(std::net::Shutdown::Both));
    assert_eq!(offered, 1, "the Spark rank alone is offered");
    assert_eq!(server.join().map_err(|_| anyhow::anyhow!("server panicked"))??, 1, "the group opens with the Spark rank");
    let credentials = spark_thread.join().map_err(|_| anyhow::anyhow!("spark panicked"))??;
    assert_eq!(credentials, Some((7, 11)), "the Spark rank receives the group's push credentials");
    Ok(())
}

/// A model of SparkNest's push credit flow (crates/nest-data/src/push.rs
/// `grant`/`flush`: today's policy as of 0.6.2, and the per-(source, rank)
/// floor of 1d3be3b), NOT a cuteafd guarantee: it documents why coordinator
/// ranks stalled the K4 and Flash push groups and checks the floor's fix.
///
/// - Each rank host has one incast budget (in slots) shared by its ranks; a
///   fill frees one unit; `grant(rank)` runs at join and on that rank's own
///   releases only, handing free ring slots round robin to EVERY source owing
///   the rank (needed yet or not) while the budget allows (at least one grant
///   if the host has none out).
/// - With the floor, `grant(rank)` first tops each owing source up to one held
///   grant of that rank, outside the budget, then runs the budget's round robin.
/// - Each source streams its slots in plan order and blocks in `flush` until it
///   holds a credit for that slot's rank. The engine releases each fill at once.
mod sparknest_credit_model {
    use std::collections::VecDeque;

    #[derive(Clone, Copy, PartialEq)]
    pub enum Policy {
        /// SparkNest 0.6.2: one budget per host, no floor.
        Budget,
        /// SparkNest 1d3be3b: each owing source held at one grant outside the budget.
        SourceFloor,
    }

    /// Returns true when every slot was delivered, false on a stall (no source
    /// can make progress and no grant can be issued).
    pub fn run(policy: Policy, host_of: &[usize], ring: usize, budget: usize, streams: &[Vec<usize>]) -> bool {
        let ranks = host_of.len();
        let hosts = host_of.iter().max().map_or(0, |h| h + 1);
        let sources = streams.len();
        let mut free = vec![ring; ranks];
        let mut out = vec![0usize; hosts];
        let mut next = vec![0usize; ranks];
        // credits[source][rank] = (budgeted, floor) grants held.
        let mut credits = vec![vec![(0usize, 0usize); ranks]; sources];
        let mut cursor = vec![0usize; sources];
        let owing: Vec<Vec<usize>> = (0..ranks).map(|r| (0..sources)
            .filter(|&s| streams[s].contains(&r)).collect()).collect();
        let grant = |r: usize, free: &mut Vec<usize>, out: &mut Vec<usize>, next: &mut Vec<usize>,
            credits: &mut Vec<Vec<(usize, usize)>>| {
            if owing[r].is_empty() { return; }
            if policy == Policy::SourceFloor {
                for &s in &owing[r] {
                    let held = credits[s][r].0 + credits[s][r].1;
                    if held == 0 && free[r] > 0 { free[r] -= 1; credits[s][r].1 += 1; }
                }
            }
            let h = host_of[r];
            while free[r] > 0 && (out[h] < budget || (out[h] == 0 && policy == Policy::Budget)) {
                let s = owing[r][next[r] % owing[r].len()];
                next[r] += 1;
                free[r] -= 1;
                out[h] += 1;
                credits[s][r].0 += 1;
            }
        };
        for r in 0..ranks { grant(r, &mut free, &mut out, &mut next, &mut credits); }
        let mut released: VecDeque<usize> = VecDeque::new();
        loop {
            let mut progressed = false;
            for s in 0..sources {
                while let Some(&r) = streams[s].get(cursor[s]) {
                    let held = &mut credits[s][r];
                    if held.1 > 0 { held.1 -= 1; }
                    else if held.0 > 0 { held.0 -= 1; out[host_of[r]] -= 1; } // a budgeted fill frees budget
                    else { break; }
                    cursor[s] += 1;
                    released.push_back(r);
                    progressed = true;
                }
            }
            while let Some(r) = released.pop_front() {
                free[r] += 1;
                grant(r, &mut free, &mut out, &mut next, &mut credits);
            }
            if cursor.iter().zip(streams).all(|(c, s)| *c == s.len()) { return true; }
            if !progressed { return false; }
        }
    }

    /// Plan-order streams: each source's files interleave both ranks' slots
    /// (a dense tensor's row shares for GPU0 then GPU1, file after file).
    pub fn interleaved(sources: usize, slots: usize, ranks: &[usize], ahead: usize) -> Vec<Vec<usize>> {
        (0..sources).map(|s| {
            let mut v = Vec::new();
            for block in 0..slots / ahead {
                for &r in ranks { v.extend(std::iter::repeat_n(r, ahead + (block + s) % 3)); }
            }
            v
        }).collect()
    }
}

#[test]
fn sparknest_model_two_ranks_on_one_host_stall_under_the_shared_budget() {
    use sparknest_credit_model::{interleaved, run, Policy};
    // Group ...667's shape: six sources streaming slots for two coordinator
    // ranks on one host (ring 128 slots, incast budget 64 slots).
    let streams = interleaved(6, 400, &[0, 1], 20);
    assert!(!run(Policy::Budget, &[0, 0], 128, 64, &streams),
        "one shared budget is expected to stall two rings on one host");
    assert!(run(Policy::SourceFloor, &[0, 0], 128, 64, &streams),
        "the per-(source, rank) floor outside the budget must complete");
}

/// The 1-RTX GLM Flash pair's shape (group ...669): one coordinator rank on its
/// own host beside two Spark ranks. One source (kiwi there) is in a long
/// stretch of coordinator tensors while the other five stream experts. Grants
/// go round robin to every source owing a rank, needed yet or not, so each
/// Spark rank's budget parks at the coordinator-busy source and the
/// coordinator's at the expert-busy ones: a stall under today's policy (as
/// SparkNest's grants_parked_at_busy_sources_do_not_deadlock_one_rank_per_host
/// reproduces at 1d3be3b); the per-(source, rank) floor (1d3be3b) completes.
#[test]
fn sparknest_model_one_coordinator_rank_beside_spark_ranks() {
    use sparknest_credit_model::{run, Policy};
    // Ranks 0 and 1 are Sparks, rank 2 the coordinator; source 3 holds a long
    // run of coordinator tensors before its expert files, the rest the reverse.
    let streams: Vec<Vec<usize>> = (0..6).map(|s| {
        let experts: Vec<usize> = (0..400).map(|i| i % 2).collect();
        let dense = vec![2; 300];
        if s == 3 { [dense, experts].concat() } else { [experts, dense].concat() }
    }).collect();
    let hosts = [0, 1, 2];
    assert!(!run(Policy::Budget, &hosts, 128, 64, &streams),
        "grants parked at busy sources are expected to stall under today's policy");
    assert!(run(Policy::SourceFloor, &hosts, 128, 64, &streams),
        "a per-(source, rank) credit floor outside the shared budget must complete");
}


/// One coordinator GPU's rank (dense records and/or TP2 halves, laid out as
/// the group driver lays them out) received through a ring from a fake
/// SparkNest that streams the rank's checked plan in plan order, then
/// committed. For installed-checkpoint goldens on real or fake devices.
pub(crate) fn receive_pushed<'a>(device: Device<'a>, catalog: &cuteafd_loader::OfficialV41Catalog,
    root: &std::path::Path, socket_dir: &std::path::Path, records: &crate::shared::experts::rtx::dense::Records,
    tp2: Option<Tp2ExpertPlan>, sparks: usize, gpu: usize) -> Result<PushedTp2<'a>> {
    use crate::shared::experts::rtx::dense::RankLayout;
    let rank = sparks + gpu;
    let layout = RankLayout::new(records.operands.clone(), records.steps.clone(), device.id)?
        .with_finishes(&records.finishes);
    let dense_plan = layout.plan();
    let (base, host) = (dense_plan.base, dense_plan.host_bytes);
    let tp2_bytes = tp2.as_ref().map_or(0, |p| p.layout.logical_bytes() as u64);
    let (host_base, total) = rank_bytes(base, tp2_bytes, host);
    // The rank's checked plan (built twice: one copy streams from the fake source).
    let build = || -> Result<CheckedPlan> {
        let mut arenas = vec![0; sparks + 2];
        arenas[rank] = total;
        let mut builder = Builder::new(root, arenas);
        (dense_plan.append)(&mut builder, rank as u32, host_base)?;
        if let Some(plan) = &tp2 { plan.append_at(&mut builder, catalog, rank as u32, base)?; }
        Ok(builder.finish()?)
    };
    let checked = build()?;
    // As plan_records: the dense destination's coverage from its own plan alone.
    let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(
        &layout.checked(root, sparks + 2, rank)?, rank as u32)?;
    let dense = crate::shared::experts::rtx::dense::DenseArena::new(device, layout, expected)?;
    let socket = socket_dir.join(format!("g{gpu}.sock")).to_string_lossy().into_owned();
    let listener = Listener::bind(&socket)?;
    let (mut coordinator, mut worker) = control_pair()?;
    for stream in [&coordinator, &worker] { stream.set_read_timeout(Some(Duration::from_secs(600)))?; }
    let streamed = build()?;
    let source_root = root.to_path_buf();
    let server = std::thread::spawn(move || -> Result<()> {
        let conn = listener.accept()?;
        conn.set_timeout(Some(Duration::from_secs(600)))?;
        let (request, fd) = conn.next_request()?.context("rank did not join")?;
        let Request::Join { slots, slot_bytes, .. } = request else { anyhow::bail!("expected rank Join") };
        let file = File::from(fd.context("rank ring fd absent")?);
        conn.send_event(&Event::Joined)?;
        assert_eq!(read_control(&coordinator)?["message"], "joined");
        // Streams the rank's runs in plan order, one slot at a time (bounded memory).
        let mut index = 0usize;
        let mut send = |bytes: Vec<u8>| -> Result<()> {
            let at = index % slots as usize;
            file.write_all_at(&bytes, at as u64 * u64::from(slot_bytes))?;
            conn.send_event(&Event::Filled { slot: at as u32, len: bytes.len() as u32 })?;
            let (release, _) = conn.next_request()?.context("missing slot release")?;
            anyhow::ensure!(matches!(release, Request::Release { slot } if slot == at as u32), "release order");
            index += 1;
            Ok(())
        };
        let mut sources = std::collections::HashMap::new();
        let mut packer = slot::Packer::new(vec![0; slot_bytes as usize]);
        for entry in streamed.plan.entries.iter().filter(|e| e.rank == rank as u32) {
            let source = sources.entry(entry.file).or_insert_with(||
                File::open(source_root.join(&streamed.plan.files[entry.file as usize])));
            let source = source.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
            let pattern = &streamed.plan.patterns[entry.pattern as usize];
            let mut run = vec![0u8; pattern.run as usize];
            for (src, dst) in pattern.runs() {
                source.read_exact_at(&mut run, entry.src_base + src)?;
                let (mut bytes, mut destination) = (&run[..], entry.dst_base + dst);
                while !bytes.is_empty() {
                    let n = packer.push(destination, bytes);
                    if n == 0 {
                        send(std::mem::replace(&mut packer, slot::Packer::new(vec![0; slot_bytes as usize])).finish().0)?;
                        continue;
                    }
                    destination += n as u64;
                    bytes = &bytes[n..];
                }
            }
        }
        if !packer.is_empty() { send(packer.finish().0)?; }
        conn.send_event(&Event::Done)?;
        anyhow::ensure!(read_control(&coordinator)?["success"] == true, "rank candidate rejected");
        write_control(&mut coordinator, &serde_json::json!({"message":"commit", "success": true}))?;
        Ok(())
    });
    let client = PushClient::at(socket.clone(), Duration::from_secs(600))?;
    // As the group driver: the TP2 arena's own checked entries (from offset 0).
    let tp2_checked = tp2.as_ref().map(|plan| plan.checked(catalog, root, sparks + 2, rank)).transpose()?;
    let tp2 = tp2.zip(tp2_checked.as_ref()).map(|(plan, checked)| (catalog, plan, checked));
    let received = receive_rank(device, tp2, Some(DenseRank { destination: dense, base }), rank, &client,
        &mut worker, (7, 11), 8, DEFAULT_SLOT_BYTES);
    if !server.is_finished() {
        // A rank that failed before joining never connects: wake the fake source.
        let _ = std::os::unix::net::UnixStream::connect(&socket);
    }
    let served = server.join().map_err(|_| anyhow::anyhow!("fake SparkNest panicked"))?;
    let received = received?;
    served?;
    received.context("pushed rank fell back")
}

/// `receive_pushed` (the installed-golden replay driver) on a synthetic
/// snapshot: dense operands, a staged step and TP2 halves in one rank.
#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn receive_pushed_replays_dense_records_and_tp2_halves() -> Result<()> {
    use crate::shared::experts::rtx::dense::{Job, Operand, Rect, Records, Step};
    let _ = tracing_subscriber::fmt().with_env_filter("info").with_test_writer().try_init();
    let fixture = Fixture::build()?;
    let library = fixture.load()?;
    fixture.configure_memory(&library, 0, 64 << 30)?;
    let root = tempfile::tempdir_in(fixture.directory())?;
    let (tensors, _) = qwen4_exl3(2, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(2);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for tensor in catalog.tensors().iter().filter(|t| t.metadata.name.ends_with(".mcg")) {
        let source = Builder::v41_source(&catalog, &tensor.metadata.name)?;
        File::options().write(true).open(source.path)?.write_all_at(&0xcbac1fedu32.to_le_bytes(), source.offset)?;
    }
    let device = Device { library: &library, id: 0 };
    let dense = catalog.tensors().iter().filter(|t| !t.metadata.name.contains(".experts.")).take(3)
        .map(|t| Rect { name: t.metadata.name.clone(), path: root.path().join(&t.shard), base: t.metadata.byte_offset,
            row_bytes: t.metadata.byte_length as usize, rows: 0..1, cols: 0..t.metadata.byte_length as usize })
        .collect::<Vec<_>>();
    let allocations = dense.iter().map(|r| crate::shared::memory::DeviceAllocation::new(&library, r.bytes().max(256)))
        .collect::<Result<Vec<_>>>()?;
    let mut records = Records::default();
    anyhow::ensure!(dense.len() == 2, "fixture has two dense tensors");
    records.operands.push(Operand { device: 0, buffer: allocations[0].buffer, lands: vec![(dense[0].clone(), 0)],
        bytes: dense[0].bytes() });
    records.steps.push(Step { device: 0, rects: vec![dense[1].clone()],
        jobs: vec![Job::Upload { at: 0, bytes: dense[1].bytes(), dst: allocations[1].buffer }] });
    let plan = Tp2ExpertPlan::new(&catalog, &[0, 1], 0, usize::MAX)?;
    let sockets = tempfile::tempdir_in(fixture.directory())?;
    let pushed = receive_pushed(device, &catalog, root.path(), sockets.path(), &records, Some(plan), 1, 0)?;
    assert_eq!(pushed.exl3()?.len(), 2);
    for (rect, allocation) in dense.iter().zip(&allocations) {
        let mut expected = vec![0; rect.bytes()];
        rect.read(&mut expected)?;
        let mut actual = vec![0; rect.bytes()];
        library.copy_d2h(&mut actual, cuteafd_ffi::CuteafdDeviceBuffer { bytes: rect.bytes(), ..allocation.buffer })?;
        assert_eq!(actual, expected, "{}", rect.name);
    }
    Ok(())
}
