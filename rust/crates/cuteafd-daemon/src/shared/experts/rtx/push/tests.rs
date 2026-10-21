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
            let weights = weights.context("successful push fell back")?;
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
    let weights = weights.context("combined push fell back")?;
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
    let sparks = 1;
    for gpu in 0..2 {
        let plan = Tp2ExpertPlan::new(&catalog, &[0, 1], gpu, usize::MAX)?;
        let checked = plan.checked(&catalog, root.path(), sparks + 2, sparks + gpu)?;
        assert!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.path().join(f))),
            "without the dense error the TP2 halves would be offered");
    }
    let (push, coordinator_streams, _) = CoordinatorPush::prepare(root.path(), Some(&catalog), 2,
        Some((&[0, 1], vec![usize::MAX; 2])))?;
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
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let dense: Option<Result<(Vec<DensePlan>, Vec<NoDense>)>> = Some(Err(anyhow::anyhow!("injected dense planning error")));
    let (pushed, ()) = push.receive_with(devices.to_vec(), Some(&catalog), sparks, dense, || ());
    assert!(pushed?.is_none(), "a declined dense plan must fall back to mount");
    assert!(!driver.join().map_err(|_| anyhow::anyhow!("driver panicked"))??, "the group driver must not prepare a group");
    assert_eq!(spark_thread.join().map_err(|_| anyhow::anyhow!("spark panicked"))??, None,
        "the Spark rank receives the whole-group mount decision");
    Ok(())
}

/// A model of SparkNest's push credit flow (crates/nest-data/src/push.rs,
/// `grant`/`flush` as of 615cc7a), NOT a cuteafd guarantee: it documents why
/// two coordinator rings on one rank host can deadlock under that policy, and
/// is the regression check for a fixed policy once SparkNest changes.
///
/// - Each rank host has one incast budget (in slots) shared by all its ranks;
///   a fill frees one unit; `grant(rank)` runs at join and on that rank's own
///   releases only, handing free ring slots to its sources round robin while
///   the host budget allows (at least one grant if the host has none out).
/// - Each source streams its slots in plan order (rank of each slot fixed by
///   the plan) and blocks in `flush` until it holds a credit for that rank.
/// - The engine releases each filled slot at once.
mod sparknest_credit_model {
    use std::collections::VecDeque;

    #[derive(Clone, Copy, PartialEq)]
    pub enum Policy {
        /// SparkNest today: freed budget is offered to the releasing rank only.
        ReleasingRank,
        /// Not enough: every rank on the host is offered freed budget, round
        /// robin (budget still parks at sources blocked on the other rank).
        AllRanks,
        /// A fix: every (source, rank) the plan pairs keeps at least one credit
        /// outside the shared budget, so no source blocks for want of a grant.
        SourceFloor,
    }

    /// Returns true when every slot was delivered, false on a stall (no source
    /// can make progress and no grant can be issued).
    pub fn run(policy: Policy, host_of: &[usize], ring: usize, budget: usize,
        streams: &[Vec<usize>]) -> bool {
        let ranks = host_of.len();
        let hosts = host_of.iter().max().map_or(0, |h| h + 1);
        let sources = streams.len();
        let mut free = vec![ring; ranks];
        let mut out = vec![0usize; hosts];
        let mut next = vec![0usize; ranks];
        // credits[source][rank]
        let mut credits = vec![vec![0usize; ranks]; sources];
        let mut cursor = vec![0usize; sources];
        let owing: Vec<Vec<usize>> = (0..ranks).map(|r| (0..sources)
            .filter(|&s| streams[s].contains(&r)).collect()).collect();
        let grant = |r: usize, free: &mut Vec<usize>, out: &mut Vec<usize>, next: &mut Vec<usize>,
            credits: &mut Vec<Vec<usize>>| {
            let h = host_of[r];
            if owing[r].is_empty() { return; }
            while free[r] > 0 && (out[h] < budget || out[h] == 0) {
                let s = owing[r][next[r] % owing[r].len()];
                next[r] += 1;
                free[r] -= 1;
                out[h] += 1;
                credits[s][r] += 1;
            }
        };
        // The floor: one credit per (source, rank) pair, outside the budget.
        let floor = |free: &mut Vec<usize>, credits: &mut Vec<Vec<usize>>| {
            for r in 0..ranks {
                for &s in &owing[r] {
                    if credits[s][r] == 0 && free[r] > 0 { free[r] -= 1; credits[s][r] += 1; }
                }
            }
        };
        let mut floored = vec![vec![false; ranks]; sources];
        if policy == Policy::SourceFloor {
            floor(&mut free, &mut credits);
            for r in 0..ranks { for &s in &owing[r] { floored[s][r] = credits[s][r] > 0; } }
        }
        for r in 0..ranks { grant(r, &mut free, &mut out, &mut next, &mut credits); }
        let mut released: VecDeque<usize> = VecDeque::new();
        loop {
            let mut progressed = false;
            for s in 0..sources {
                while let Some(&r) = streams[s].get(cursor[s]) {
                    if credits[s][r] == 0 { break; }
                    credits[s][r] -= 1;
                    cursor[s] += 1;
                    if floored[s][r] { floored[s][r] = false; } else { out[host_of[r]] -= 1; } // filled: its budget is free
                    released.push_back(r); // the engine releases at once
                    progressed = true;
                }
            }
            while let Some(r) = released.pop_front() {
                free[r] += 1;
                match policy {
                    Policy::ReleasingRank => grant(r, &mut free, &mut out, &mut next, &mut credits),
                    Policy::SourceFloor => {
                        // Refill the floor first (its credit is outside the budget), then the budget.
                        for &s in &owing[r] {
                            if credits[s][r] == 0 && free[r] > 0 && cursor[s] < streams[s].len() {
                                free[r] -= 1; credits[s][r] += 1; floored[s][r] = true;
                            }
                        }
                        let h = host_of[r];
                        for q in (0..ranks).filter(|&q| host_of[q] == h) {
                            grant(q, &mut free, &mut out, &mut next, &mut credits);
                        }
                    }
                    Policy::AllRanks => {
                        let h = host_of[r];
                        let mut order: Vec<usize> = (0..ranks).filter(|&q| host_of[q] == h).collect();
                        let at = order.iter().position(|&q| q == r).unwrap_or(0);
                        order.rotate_left(at);
                        for q in order { grant(q, &mut free, &mut out, &mut next, &mut credits); }
                    }
                }
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
    // Six sources streaming slots for two coordinator ranks on one host
    // (ring 128 slots, incast budget 64 slots).
    let streams = interleaved(6, 400, &[0, 1], 20);
    assert!(!run(Policy::ReleasingRank, &[0, 0], 128, 64, &streams),
        "SparkNest's releasing-rank grants are expected to stall two rings on one host");
    assert!(!run(Policy::AllRanks, &[0, 0], 128, 64, &streams),
        "sharing freed budget alone still parks credits at blocked sources");
    assert!(run(Policy::SourceFloor, &[0, 0], 128, 64, &streams),
        "a per-(source, rank) credit floor outside the shared budget must complete");
    // One rank per host (the Spark-only groups) completes under today's policy.
    assert!(run(Policy::ReleasingRank, &[0, 1], 128, 64, &streams));
}
