//! Two-GPU tests of the explicit-stream pushes (`PeerExchange::*_on`), the
//! pushing `PeerTransfer` and the chain's flag edge (`finish_after_flag`).
use super::{FlagRoute, PeerExchange};
use crate::shared::memory::chain::{self, StageChain};
use crate::shared::memory::device::{Allocation, Device, PeerTransfer, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::test_support::CudaStreamGate;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn library() -> Result<NativeLibrary> {
    // SAFETY: the explicitly selected native library outlives every owner below.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    for gpu in 0..2 {
        Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?;
    }
    Ok(library)
}

fn pattern(bytes: usize, seed: usize) -> Vec<u8> {
    (0..bytes).map(|i| (i.wrapping_mul(131) ^ seed.wrapping_mul(29) ^ (i >> 7)) as u8).collect()
}

fn read(device: Device<'_>, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes];
    device.run(|| device.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..buffer }))?;
    Ok(out)
}

fn write(device: Device<'_>, buffer: CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
    device.run(|| device.library.copy_h2d(CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer }, bytes))
}

/// Pushes on explicit streams land exactly, by SM and by copy engine, for
/// flat and strided planes; 1,000 alternating publications on two flags in
/// both directions keep their positional pairing.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn explicit_stream_pushes_land_exact_bytes() -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    const LANE: usize = 64 << 10;
    let exchange = PeerExchange::new_flags(&library, [0, 1], 2, LANE)?;
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    let source = Allocation::new(devices[0], 2 << 20)?;
    let destination = Allocation::new(devices[1], 2 << 20)?;
    let check = Allocation::new(devices[1], 2 << 20)?;
    for (round, bytes) in [16usize, 4096, LANE, LANE + 16, 2 << 20].into_iter().enumerate() {
        let data = pattern(bytes, round);
        write(devices[0], source.buffer, &data)?;
        devices[1].run(|| library.cuda_zero_bytes(destination.buffer, destination.buffer.bytes))?;
        exchange.push_to_on(0, streams[0].raw, 0, source.buffer.ptr, destination.buffer.ptr, bytes)?;
        exchange.wait_on(1, streams[1].raw, 0)?;
        // The read is ordered after the wait on the receiving stream.
        devices[1].run(|| unsafe { library.copy_d2d_async(check.buffer, destination.buffer, bytes, streams[1].raw) })?;
        streams[1].drain()?;
        streams[0].drain()?;
        ensure!(read(devices[1], check.buffer, bytes)? == data, "push of {bytes} B differs");
    }
    // Planes by SM (under the lane): a strided and a flat plane; by copy
    // engine (a 16 B lane): flat planes only, strided ones refused.
    let narrow = PeerExchange::new_flags(&library, [0, 1], 1, 16)?;
    let offset = |buffer: CuteafdDeviceBuffer, at: usize| unsafe { buffer.ptr.cast::<u8>().add(at).cast::<std::ffi::c_void>() };
    let strided = PeerPlane { destination: destination.buffer.ptr, source: source.buffer.ptr, rows: 5, row_bytes: 32,
        destination_pitch: 64, source_pitch: 48 };
    let error = narrow.push_planes_on(0, streams[0].raw, 0, &[strided]).unwrap_err().to_string();
    ensure!(error.contains("strided"), "a strided copy-engine plane was not refused: {error}");
    for (round, exchange) in [&exchange, &narrow].into_iter().enumerate() {
        let data = pattern(2 << 20, 17 + round);
        write(devices[0], source.buffer, &data)?;
        devices[1].run(|| library.cuda_zero_bytes(destination.buffer, destination.buffer.bytes))?;
        let first = if round == 0 { strided } else {
            PeerPlane { rows: 1, row_bytes: 32, destination_pitch: 32, source_pitch: 32, ..strided }
        };
        let planes = [
            first,
            PeerPlane { destination: offset(destination.buffer, 4096), source: offset(source.buffer, 8192).cast_const(),
                rows: 3, row_bytes: 1040, destination_pitch: 1040, source_pitch: 1040 },
        ];
        exchange.push_planes_on(0, streams[0].raw, 0, &planes)?;
        exchange.wait_on(1, streams[1].raw, 0)?;
        devices[1].run(|| unsafe { library.copy_d2d_async(check.buffer, destination.buffer, 8192, streams[1].raw) })?;
        streams[1].drain()?;
        streams[0].drain()?;
        let landed = read(devices[1], check.buffer, 8192)?;
        let mut expected = vec![0u8; 8192];
        for row in 0..first.rows as usize { expected[row * 64..row * 64 + 32].copy_from_slice(&data[row * 48..row * 48 + 32]); }
        expected[4096..4096 + 3 * 1040].copy_from_slice(&data[8192..8192 + 3 * 1040]);
        ensure!(landed == expected, "planes by {} differ", if round == 0 { "SM" } else { "copy engine" });
    }
    // Positional pairing across many publications, both directions.
    let back = Allocation::new(devices[0], 16 * 1000)?;
    let forth = Allocation::new(devices[1], 16 * 1000)?;
    let slots = [Allocation::new(devices[0], 16)?, Allocation::new(devices[1], 16)?];
    let landing = [Allocation::new(devices[0], 16)?, Allocation::new(devices[1], 16)?];
    for i in 0..1000usize {
        let from = i % 2;
        let to = 1 - from;
        let value: Vec<u8> = (0..16).map(|b| (i * 7 + b) as u8).collect();
        // Stage the value on the sender's stream, push it, land and keep a copy.
        devices[from].run(|| library.copy_h2d(slots[from].buffer, &value))?;
        exchange.push_to_on(from, streams[from].raw, from, slots[from].buffer.ptr, landing[to].buffer.ptr, 16)?;
        exchange.wait_on(to, streams[to].raw, from)?;
        let keep = if to == 1 { forth.buffer } else { back.buffer };
        let at = CuteafdDeviceBuffer { ptr: unsafe { keep.ptr.cast::<u8>().add(i * 16).cast() }, bytes: 16, ..keep };
        devices[to].run(|| unsafe { library.copy_d2d_async(at, landing[to].buffer, 16, streams[to].raw) })?;
        // Sequential host pacing: the next value is staged after this lands.
        streams[to].drain()?;
        streams[from].drain()?;
    }
    let [kept0, kept1] = [read(devices[0], back.buffer, 16 * 1000)?, read(devices[1], forth.buffer, 16 * 1000)?];
    for i in 0..1000usize {
        let kept = if i % 2 == 0 { &kept1 } else { &kept0 };
        ensure!((0..16).all(|b| kept[i * 16 + b] == (i * 7 + b) as u8), "publication {i} paired with another push");
    }
    eprintln!("explicit-stream pushes: SM and copy-engine extents, planes and 1,000 alternating publications exact");
    Ok(())
}

/// A pushing transfer queues its push on the gated producer and its `then`
/// behind the flag: the bytes `then` copies are the producer's, never the
/// destination's stale ones. Push and pull deliver the same bytes. Negative
/// control: without the flag wait, `then` reads the stale destination.
fn pushed_transfer(wait: bool) -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    const BYTES: usize = 655_360;
    let producer = Stream::new(devices[0])?;
    let fresh = Allocation::new(devices[0], BYTES)?;
    let source = Allocation::new(devices[0], BYTES)?;
    let mut staging = Allocation::new(devices[1], BYTES)?;
    let check = Allocation::new(devices[1], BYTES)?;
    let mut pushing = PeerTransfer::new(devices[0], devices[1])?;
    pushing.enable_push(Box::new(FlagRoute::new(&library, [0, 1], BYTES)?));
    let mut pulling = PeerTransfer::new(devices[0], devices[1])?;
    let chain = StageChain::on_devices(&library, &[0, 1])?;
    chain.set_device_order(true);
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut outputs = Vec::new();
    for (round, transfer) in [&mut pulling, &mut pushing].into_iter().enumerate() {
        let data = pattern(BYTES, 40 + round);
        write(devices[0], fresh.buffer, &data)?;
        write(devices[0], source.buffer, &vec![0x5a; BYTES])?;
        write(devices[1], staging.buffer, &vec![0xa5; BYTES])?;
        library.cuda_device_synchronize()?;
        // SAFETY: the gate is released by the thread below before any drain.
        let gate = devices[0].run(|| unsafe { CudaStreamGate::new(&library, producer.raw) })?;
        let release = gate.release_handle();
        let checked = check.buffer;
        let copied = std::thread::scope(|threads| -> Result<()> {
            threads.spawn(move || { std::thread::sleep(Duration::from_millis(50)); release.store(true, Ordering::Release); });
            // The producer's write of the fresh bytes sits behind the gate.
            devices[0].run(|| unsafe { library.copy_d2d_async(source.buffer, fresh.buffer, BYTES, producer.raw) })?;
            crate::shared::memory::device::SKIP_PUSH_WAIT.with(|skip| skip.set(!wait));
            let queued = runtime.block_on(chain.handle().scope(async {
                unsafe { transfer.copy_then(&source, &mut staging, &producer, BYTES, |landed, stream| {
                    library.copy_d2d_async(checked, landed, BYTES, stream)
                }).await }
            }));
            crate::shared::memory::device::SKIP_PUSH_WAIT.with(|skip| skip.set(false));
            queued?;
            chain.drain_bounded(Duration::from_secs(10))?;
            producer.drain()
        });
        drop(gate);
        copied?;
        library.cuda_device_synchronize()?;
        outputs.push(read(devices[1], check.buffer, BYTES)?);
        let expected = if round == 1 && !wait { vec![0xa5; BYTES] } else { data };
        ensure!(outputs[round] == expected, "{} transfer delivered {} bytes",
            if round == 0 { "pulled" } else { "pushed" }, if outputs[round][0] == 0xa5 { "stale" } else { "unexpected" });
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn pushed_transfer_waits_for_its_gated_producer() -> Result<()> { pushed_transfer(true) }

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn pushed_transfer_without_its_wait_reads_stale_bytes() -> Result<()> { pushed_transfer(false) }

/// `finish_after_flag`: a later stage joining the adopted head follows the
/// gated stage the push's stream joined, with no cross-device event. Negative
/// control: a receiver that adopts without the flag wait lets it overtake.
fn adopted_head(wait: bool) -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let [held, sender] = [Stream::new(devices[0])?, Stream::new(devices[0])?];
    let [receiver, later] = [Stream::new(devices[1])?, Stream::new(devices[1])?];
    let exchange = PeerExchange::new_flags(&library, [0, 1], 1, 4096)?;
    let token = [Allocation::new(devices[0], 16)?, Allocation::new(devices[1], 16)?];
    let done = devices[1].run(|| library.cuda_event_create_ordering())?;
    let chain = StageChain::on_devices(&library, &[0, 1])?;
    chain.set_device_order(true);
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    // Warm every launch ungated (lazy module loading).
    runtime.block_on(chain.handle().scope(async { unsafe {
        devices[0].run(|| { chain::join(&library, held.raw)?; chain::finish(&library, held.raw) })?;
        let mark = devices[0].run(|| chain::join_for_push(&library, sender.raw))?;
        exchange.push_to_on(0, sender.raw, 0, token[0].buffer.ptr, token[1].buffer.ptr, 16)?;
        devices[1].run(|| { exchange.wait_on(1, receiver.raw, 0)?; chain::finish_after_flag(&library, receiver.raw, mark) })
    } }))?;
    chain.drain()?;
    library.cuda_device_synchronize()?;
    // SAFETY: released below before any drain.
    let gate = devices[0].run(|| unsafe { CudaStreamGate::new(&library, held.raw) })?;
    let queued = runtime.block_on(chain.handle().scope(async { unsafe {
        devices[0].run(|| { chain::join(&library, held.raw)?; chain::finish(&library, held.raw) })?;
        let mark = devices[0].run(|| chain::join_for_push(&library, sender.raw))?;
        exchange.push_to_on(0, sender.raw, 0, token[0].buffer.ptr, token[1].buffer.ptr, 16)?;
        devices[1].run(|| {
            if wait { exchange.wait_on(1, receiver.raw, 0)?; }
            chain::finish_after_flag(&library, receiver.raw, mark)?;
            ensure!(chain::head_device() == Some(1), "the receiver did not adopt the head");
            chain::join(&library, later.raw)?;
            library.cuda_event_record(done, later.raw)
        })
    } }));
    let deadline = Instant::now() + Duration::from_millis(if wait { 300 } else { 3000 });
    let mut overtook = false;
    while Instant::now() < deadline {
        if devices[1].run(|| unsafe { library.cuda_event_query(done) })? { overtook = true; break; }
        std::thread::sleep(Duration::from_millis(5));
    }
    gate.release_handle().store(true, Ordering::Release);
    if !wait {
        // The unmatched publication is consumed so the exchange stays paired.
        exchange.wait_on(1, receiver.raw, 0)?;
    }
    drop(gate);
    queued?;
    chain.drain_bounded(Duration::from_secs(10))?;
    library.cuda_device_synchronize()?;
    devices[1].run(|| unsafe { library.cuda_event_destroy(done) })?;
    ensure!(overtook != wait, "the stage after the adopted head {} while the joined stage was held",
        if overtook { "completed" } else { "stayed pending" });
    // A finish between the push's join and its receiver refuses the adoption.
    let refused = runtime.block_on(chain.handle().scope(async { unsafe {
        let mark = devices[0].run(|| chain::join_for_push(&library, sender.raw))?;
        exchange.push_to_on(0, sender.raw, 0, token[0].buffer.ptr, token[1].buffer.ptr, 16)?;
        devices[0].run(|| chain::finish(&library, held.raw))?;
        devices[1].run(|| { exchange.wait_on(1, receiver.raw, 0)?; chain::finish_after_flag(&library, receiver.raw, mark) })
    } }));
    chain.drain_bounded(Duration::from_secs(10))?;
    library.cuda_device_synchronize()?;
    ensure!(refused.is_err_and(|error| error.to_string().contains("moved")), "a moved head was adopted");
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn adopted_head_follows_the_pushers_joined_stage() -> Result<()> { adopted_head(true) }

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn adopted_head_without_the_flag_wait_overtakes() -> Result<()> { adopted_head(false) }

/// Q0: does work on independent streams stall behind streams held by a
/// device wait? Holds K of 32 streams on GPU0 with a spin wait (a peer flag
/// released by GPU1), a cross-device event wait (released by a gate on
/// GPU1) or `cuStreamWaitValue32` on a host word, each followed by dependent
/// work; then queues the same work on every other stream and counts what
/// finished 20 ms later, before any release. The work is a short kernel, or
/// a 64 KiB pinned host-to-device copy (copy engine). Run once per
/// `CUDA_DEVICE_MAX_CONNECTIONS` (it is read when CUDA starts).
#[test]
#[ignore = "probe: requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn q0_held_streams_stall_independent_work() -> Result<()> {
    use crate::shared::memory::HostAllocation;
    const STREAMS: usize = 32;
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let connections = std::env::var("CUDA_DEVICE_MAX_CONNECTIONS").unwrap_or_else(|_| "8 (default)".into());
    let streams = (0..STREAMS).map(|_| Stream::new(devices[0])).collect::<Result<Vec<_>>>()?;
    let release_stream = Stream::new(devices[1])?;
    let operands = [Allocation::new(devices[0], 512)?, Allocation::new(devices[0], 512)?];
    let outputs = (0..STREAMS).map(|_| Allocation::new(devices[0], 512)).collect::<Result<Vec<_>>>()?;
    let events = (0..STREAMS).map(|_| devices[0].run(|| library.cuda_event_create_ordering())).collect::<Result<Vec<_>>>()?;
    let host = HostAllocation::new(&library, 64 << 10)?;
    let landings = (0..STREAMS).map(|_| Allocation::new(devices[0], 64 << 10)).collect::<Result<Vec<_>>>()?;
    let queue = |work: &str, stream: usize| devices[0].run(|| unsafe {
        if work == "kernel" {
            library.peer_add_bf16(operands[0].buffer.ptr, operands[1].buffer.ptr, outputs[stream].buffer.ptr, 256,
                streams[stream].raw)
        } else {
            library.copy_host_buffer_h2d_async(landings[stream].buffer, host.buffer, 64 << 10, streams[stream].raw)
        }
    });
    // Warm every kernel, copy and stream before any wait exists.
    for stream in 0..STREAMS { queue("kernel", stream)?; queue("copy", stream)?; }
    library.cuda_device_synchronize()?;
    let mut word = HostAllocation::new(&library, 64)?;
    let exchange = PeerExchange::new_flags(&library, [0, 1], 4, 4096)?;
    let mut report = Vec::new();
    for (work, hold) in ["kernel", "copy"].into_iter().flat_map(|work|
        ["spin wait", "event wait", "wait-value"].map(|hold| (work, hold))) {
        for held in [1usize, 2, 4] {
            word.bytes_mut().fill(0);
            let gate = if hold == "event wait" {
                // SAFETY: released below before the stream drains.
                Some(devices[1].run(|| unsafe { CudaStreamGate::new(&library, release_stream.raw) })?)
            } else { None };
            let gated = devices[1].run(|| library.cuda_event_create_ordering())?;
            if gate.is_some() { devices[1].run(|| unsafe { library.cuda_event_record(gated, release_stream.raw) })?; }
            for k in 0..held {
                match hold {
                    "spin wait" => exchange.wait_on(0, streams[k].raw, k)?,
                    "event wait" => devices[0].run(|| unsafe { library.cuda_stream_wait_event(streams[k].raw, gated) })?,
                    _ => devices[0].run(|| unsafe {
                        library.cuda_stream_wait_value32(streams[k].raw, word.buffer.ptr.cast::<u32>(), 1)
                    })?,
                }
                queue(work, k)?;
            }
            for stream in held..STREAMS {
                queue(work, stream)?;
                devices[0].run(|| unsafe { library.cuda_event_record(events[stream], streams[stream].raw) })?;
            }
            std::thread::sleep(Duration::from_millis(20));
            let mut stalled = Vec::new();
            for stream in held..STREAMS {
                if !devices[0].run(|| unsafe { library.cuda_event_query(events[stream]) })? { stalled.push(stream); }
            }
            // Release every hold, then drain.
            match hold {
                "spin wait" => for k in 0..held { exchange.signal_on(1, release_stream.raw, k)?; },
                "event wait" => gate.as_ref().unwrap().release_handle().store(true, Ordering::Release),
                _ => unsafe { (*word.buffer.ptr.cast::<std::sync::atomic::AtomicU32>()).store(1, Ordering::Release) },
            }
            drop(gate);
            for stream in &streams { stream.drain()?; }
            release_stream.drain()?;
            devices[1].run(|| unsafe { library.cuda_event_destroy(gated) })?;
            report.push(format!("{work} behind {hold} x{held}: {}/{} independent done while held, stalled streams {stalled:?}",
                STREAMS - held - stalled.len(), STREAMS - held));
        }
    }
    for event in events { devices[0].run(|| unsafe { library.cuda_event_destroy(event) })?; }
    eprintln!("Q0 (CUDA_DEVICE_MAX_CONNECTIONS={connections}, {STREAMS} streams on GPU0):\n  {}", report.join("\n  "));
    Ok(())
}
