//! The exchange worker against a fake GPU (counters a test thread advances)
//! and a fake transport: wave order, slot reuse by counters, a decode step
//! queued far ahead of its exchange, and failure release, without hardware.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// #1183: idle CE throughput at the actual Muse 2752-row reply size. No
/// network/model is involved; this is a prerequisite, not an overlap claim.
#[test]
#[ignore = "requires one idle GPU and CUTEAFD_PLANE_UPLOAD_LIBRARY; run alone (--exact)"]
fn gpu_reply_copy_engine_bandwidth() -> Result<()> {
    use crate::shared::memory::{DeviceAllocation, device::{Device, Event, Stream}};
    const BYTES: usize = 304_349_184;
    const RANKS: usize = 6;
    let path = library_path()?;
    // SAFETY: the opt-in path names the matching built native library.
    let library = unsafe { NativeLibrary::load(&path)? };
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let mut host = HostAllocation::new(&library, BYTES)?;
    for (i, byte) in host.bytes_mut().iter_mut().enumerate() { *byte = (i % 251) as u8; }
    let output = DeviceAllocation::new(&library, BYTES)?;
    let streams = (0..RANKS).map(|_| Stream::new(device)).collect::<Result<Vec<_>>>()?;
    let mut start = Event::new(device)?;
    let mut end = Event::new(device)?;
    let mut done = (0..RANKS).map(|_| Event::new(device)).collect::<Result<Vec<_>>>()?;
    for (name, copies, stream_count) in [("one", 1, 1), ("six_serial", RANKS, 1), ("six_streams", RANKS, RANKS)] {
        let chunk = BYTES / copies;
        let mut samples = Vec::new();
        for repeat in 0..6 {
            start.record(&streams[0])?;
            for stream in streams.iter().take(stream_count).skip(1) {
                // SAFETY: start is already recorded for this repeat; all owners survive drainage.
                unsafe { library.cuda_stream_wait_event(stream.raw, start.raw)?; }
            }
            for copy in 0..copies {
                let offset = copy * chunk;
                let mut dst = output.buffer;
                let mut src = host.buffer;
                // SAFETY: the disjoint slices are within the BYTES-sized live allocations.
                unsafe {
                    dst.ptr = dst.ptr.cast::<u8>().add(offset).cast();
                    src.ptr = src.ptr.cast::<u8>().add(offset).cast();
                }
                dst.bytes = chunk;
                src.bytes = chunk;
                // SAFETY: immutable pinned source and disjoint device destination outlive all copies.
                unsafe { library.copy_host_buffers_h2d_batch_async(&[dst], &[src], &[chunk], streams[copy % stream_count].raw)?; }
            }
            for index in 1..stream_count {
                done[index].record(&streams[index])?;
                // SAFETY: each current completion event is recorded before this fan-in wait.
                unsafe { library.cuda_stream_wait_event(streams[0].raw, done[index].raw)?; }
            }
            end.record(&streams[0])?;
            streams[0].drain()?;
            // SAFETY: both timing events completed; fan-in includes every copy stream.
            let ms = unsafe { library.cuda_event_elapsed_ms(start.raw, end.raw)? };
            ensure!(ms.is_finite() && ms > 0.0, "invalid CE timing {ms}");
            if repeat > 0 {
                let gbps = BYTES as f64 / (ms as f64 * 1e6);
                samples.push(gbps);
                eprintln!("reply_ce case={name} bytes={BYTES} copies={copies} streams={stream_count} repeat={repeat} ms={ms:.6} gbps={gbps:.3}");
            }
        }
        samples.sort_by(f64::total_cmp);
        eprintln!("reply_ce case={name} median_gbps={:.3} min_gbps={:.3} max_gbps={:.3}", samples[2], samples[0], samples[4]);
        let mut actual = vec![0u8; BYTES];
        library.copy_d2h(&mut actual, output.buffer)?;
        ensure!(actual == host.bytes(), "CE output differs for {name}");
    }
    Ok(())
}

/// #815: queue compute before arrival and hold representative kernel/H2D work
/// on other streams. Report whether a fresh CE stream can release compute;
/// always release the host words before draining, including error paths.
#[test]
#[ignore = "requires one idle GPU, CUDA_MODULE_LOADING=EAGER and CUTEAFD_PLANE_UPLOAD_LIBRARY"]
fn gpu_reply_ce_land_before_arrival_pressure() -> Result<()> {
    use crate::shared::memory::{DeviceAllocation, device::{Device, Event, Stream}};
    ensure!(std::env::var("CUDA_MODULE_LOADING").as_deref() == Ok("EAGER"), "run with CUDA_MODULE_LOADING=EAGER");
    let path = library_path()?;
    // SAFETY: the opt-in path names the matching built native library.
    let library = unsafe { NativeLibrary::load(&path)? };
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let host = HostAllocation::new(&library, 304_349_184)?;
    let output = DeviceAllocation::new(&library, host.buffer.bytes)?;
    struct ReleaseWords<'a>(&'a HostAllocation<'a>);
    impl Drop for ReleaseWords<'_> {
        fn drop(&mut self) {
            for offset in [0, 64] {
                // SAFETY: aligned atomics of the live mapped control allocation.
                unsafe { (*self.0.buffer.ptr.cast::<u8>().add(offset).cast::<AtomicU32>()).store(1, Ordering::Release); }
            }
        }
    }
    for pressure in [0, 4, 16] {
        let control = HostAllocation::new(&library, 128)?;
        let pressure_outputs = (0..=pressure).map(|_| DeviceAllocation::new(&library, 65536)).collect::<Result<Vec<_>>>()?;
        let streams = (0..=pressure).map(|_| Stream::new(device)).collect::<Result<Vec<_>>>()?;
        let copy_stream = Stream::new(device)?;
        let mut done = Event::new(device)?;
        // Warm the exact kernel and copy API before any memory wait exists.
        for (stream, target) in streams.iter().zip(&pressure_outputs) {
            // SAFETY: live disjoint destinations and immutable pinned source; drained below.
            unsafe {
                library.sm_copy_regions(&[(target.buffer.ptr, host.buffer.ptr.cast_const(), 65536)], stream.raw)?;
                library.copy_host_buffers_h2d_batch_async(&[target.buffer], &[host.buffer], &[65536], stream.raw)?;
            }
            stream.drain()?;
        }
        // Declared after every stream: unwinding releases words before stream Drop drains.
        let release = ReleaseWords(&control);
        let copied = control.buffer.ptr.cast::<u32>();
        let ready = control.buffer.ptr.cast::<u8>().wrapping_add(64).cast::<u32>();
        for (index, (stream, target)) in streams.iter().zip(&pressure_outputs).enumerate() {
            // SAFETY: current compute waits for copied; pressure waits for a host release.
            // All sources/words survive stream drainage, and release covers errors too.
            unsafe {
                library.cuda_stream_wait_value32(stream.raw, if index == 0 { copied } else { ready }, 1)?;
                library.sm_copy_regions(&[(target.buffer.ptr, host.buffer.ptr.cast_const(), 65536)], stream.raw)?;
                library.copy_host_buffers_h2d_batch_async(&[target.buffer], &[host.buffer], &[65536], stream.raw)?;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
        // Simulated host-observed arrival: NO unsatisfied wait on the CE stream.
        // SAFETY: the immutable pinned source and full destination are retained to drain.
        unsafe {
            library.copy_host_buffers_h2d_batch_async(&[output.buffer], &[host.buffer], &[host.buffer.bytes], copy_stream.raw)?;
            library.cuda_stream_write_value32(copy_stream.raw, copied, 1)?;
        }
        done.record(&copy_stream)?;
        let started = Instant::now();
        let completed = loop {
            // SAFETY: the current completion event was recorded above and remains live.
            if unsafe { library.cuda_event_query(done.raw)? } { break true; }
            if started.elapsed() >= Duration::from_millis(50) { break false; }
            std::thread::sleep(Duration::from_millis(1));
        };
        eprintln!("reply_ce_pressure pressure_streams={pressure} completed_before_host_release={completed} observed_ms={:.3}", started.elapsed().as_secs_f64() * 1e3);
        drop(release);
        copy_stream.drain()?;
        for stream in &streams { stream.drain()?; }
        let mut actual = vec![0u8; host.buffer.bytes];
        library.copy_d2h(&mut actual, output.buffer)?;
        ensure!(actual == host.bytes(), "CE output differs under {pressure}-stream pressure");
    }
    Ok(())
}

/// #1183: the Muse dependency shape, not synthetic blocked CE pressure.
/// One compute stream, depth-three slots, 19 ms replies and 750 us rank skew;
/// compare total ring time, including the CE current-record host handshake.
#[test]
#[ignore = "requires one idle GPU, CUDA_MODULE_LOADING=EAGER and CUTEAFD_PLANE_UPLOAD_LIBRARY"]
fn gpu_reply_ce_current_event_depth_three() -> Result<()> {
    use crate::shared::memory::{DeviceAllocation, device::{Device, Event, Stream}};
    const RANKS: usize = 6;
    const SLOTS: usize = 3;
    const WAVES: u32 = 18;
    const CHUNK: usize = 50_724_864;
    const BYTES: usize = RANKS * CHUNK;
    const STAGING: usize = 2752 * (2 * 8 * 4 + 4224);
    const TABLES: usize = 22_144;
    ensure!(std::env::var("CUDA_MODULE_LOADING").as_deref() == Ok("EAGER"), "run with CUDA_MODULE_LOADING=EAGER");
    let path = library_path()?;
    // SAFETY: the opt-in path names the matching built native library.
    let library = Arc::new(unsafe { NativeLibrary::load(&path)? });
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let mut host = HostAllocation::new(&library, BYTES)?;
    for (i, byte) in host.bytes_mut().iter_mut().enumerate() { *byte = (i % 251) as u8; }
    let output = DeviceAllocation::new(&library, SLOTS * BYTES)?;
    let input = DeviceAllocation::new(&library, STAGING)?;
    let staging = HostAllocation::new(&library, SLOTS * STAGING)?;
    let table = DeviceAllocation::new(&library, SLOTS * TABLES)?;
    let descriptors = HostAllocation::new(&library, Pulls::bytes(SLOTS))?;
    let pulls = Pulls { base: descriptors.buffer.ptr as usize };
    let stream = Stream::new(device)?;
    let copy = Stream::new(device)?;
    let copied = (0..SLOTS).map(|_| Event::new(device)).collect::<Result<Vec<_>>>()?;
    let mut staged = (0..SLOTS).map(|_| Event::new(device)).collect::<Result<Vec<_>>>()?;
    // All allocation, lazy loading and ordered-upload reservations precede waits.
    library.tp2_native_initialize()?;
    library.peer_exchange_initialize()?;
    library.copy_h2d(input.buffer, &vec![0x35; STAGING])?;
    for slot in 0..SLOTS {
        let target = cuteafd_ffi::CuteafdDeviceBuffer { ptr: (table.buffer.ptr as usize + slot * TABLES) as *mut c_void,
            bytes: TABLES, ..table.buffer };
        // SAFETY: fixed admitted table destinations on the compute stream.
        unsafe { library.reserve_h2d_ordered(target, stream.raw)?; }
        for rank in 0..RANKS {
            pulls.write(slot, rank, &[PullEntry { source: (host.buffer.ptr as usize + rank * CHUNK) as u64,
                offset: 0, bytes: CHUNK as u64 }]);
        }
    }
    // Warm exact launch shapes, including the SM path, before either arm.
    // SAFETY: live, disjoint destinations and immutable sources, drained below.
    unsafe {
        library.sm_copy_regions(&[(staging.buffer.ptr, input.buffer.ptr.cast_const(), STAGING as u64)], stream.raw)?;
        library.pull_frames(&[pulls.at(0, 0).cast_const()], &[(output.buffer.ptr, CHUNK as u64)], stream.raw)?;
        library.copy_host_buffers_h2d_batch_async(&[CuteafdDeviceBuffer { bytes: BYTES, ..output.buffer }],
            &[host.buffer], &[BYTES], copy.raw)?;
    }
    stream.drain()?;
    copy.drain()?;
    let mut results = Vec::new();
    for ce in [false, true] {
        let control = HostAllocation::new(&library, Counter::words(SLOTS) * LINE)?;
        let words = Words { base: control.buffer.ptr as usize, slots: SLOTS };
        let stop = Arc::new(AtomicBool::new(false));
        let (send, receive) = std::sync::mpsc::channel::<(u32, usize)>();
        let (recorded, acknowledgement) = std::sync::mpsc::channel::<u32>();
        let worker_library = library.clone();
        let worker_stop = stop.clone();
        let (source, destination, copy_raw) = (host.buffer.ptr as usize, output.buffer.ptr as usize, copy.raw as usize);
        let events: Vec<usize> = copied.iter().map(|event| event.raw as usize).collect();
        let started = Instant::now();
        let mut host_wait = Duration::ZERO;
        let mut max_ahead = 0usize;
        let mut first_land_gpu_staged = 0usize;
        let result = std::thread::scope(|scope| -> Result<()> {
            let worker = scope.spawn(move || -> Result<()> {
                struct ReleaseOnExit(Words);
                impl Drop for ReleaseOnExit {
                    fn drop(&mut self) {
                        for rank in 0..RANKS { self.0.store(Counter::Arrived(rank), WAVES); }
                        self.0.store(Counter::Landed, WAVES);
                    }
                }
                // A worker error must not leave the engine's SM/table upload stuck.
                let _release = ReleaseOnExit(words);
                worker_library.cuda_set_device(0)?;
                let mut jobs = VecDeque::<(u32, usize, Option<Instant>, usize)>::new();
                let mut closed = false;
                let mut progress = Instant::now();
                loop {
                    if worker_stop.load(Ordering::Acquire) { return Ok(()); }
                    loop {
                        match receive.try_recv() {
                            Ok((sequence, slot)) => { jobs.push_back((sequence, slot, None, 0)); progress = Instant::now(); }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => { closed = true; break; }
                        }
                    }
                    if closed && jobs.is_empty() { break; }
                    ensure!(progress.elapsed() < Duration::from_secs(2), "fixture worker progress timeout");
                    // Post all staged/free waves, so reply delays overlap at depth three.
                    for (sequence, slot, arrival, _) in &mut jobs {
                        if arrival.is_none() && reached(words.load(Counter::Staged(*slot)), *sequence)
                            && reached(words.load(Counter::Consumed(*slot)), (*sequence - 1) / SLOTS as u32) {
                            *arrival = Some(Instant::now() + Duration::from_millis(19));
                            progress = Instant::now();
                        }
                    }
                    let Some((sequence, slot, Some(arrival), index)) = jobs.front_mut() else {
                        std::thread::sleep(Duration::from_micros(50)); continue;
                    };
                    if Instant::now() < *arrival + Duration::from_micros(*index as u64 * 150) {
                        std::thread::sleep(Duration::from_micros(50)); continue;
                    }
                    let (sequence, slot) = (*sequence, *slot);
                    let rank = [1usize, 4, 0, 5, 3, 2][*index];
                    if ce {
                        let dst = CuteafdDeviceBuffer { ptr: (destination + slot * BYTES + rank * CHUNK) as *mut c_void,
                            bytes: CHUNK, device_id: 0, flags: 0 };
                        let src = cuteafd_ffi::CuteafdHostBuffer { ptr: (source + rank * CHUNK) as *mut c_void,
                            bytes: CHUNK, flags: cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED };
                        // SAFETY: receive source retained through final drain; slot's prior consumer completed.
                        unsafe { worker_library.copy_host_buffers_h2d_batch_async(&[dst], &[src], &[CHUNK], copy_raw as *mut c_void)?; }
                    }
                    words.store(Counter::Arrived(rank), sequence);
                    *index += 1;
                    progress = Instant::now();
                    if *index != RANKS { continue; }
                    jobs.pop_front();
                    if ce {
                        // SAFETY: current record is submitted BEFORE acknowledgement; no future-record wait.
                        unsafe { worker_library.cuda_event_record(events[slot] as *mut c_void, copy_raw as *mut c_void)?; }
                    }
                    words.store(Counter::Landed, sequence);
                    if ce && recorded.send(sequence).is_err() { return Ok(()); }
                }
                Ok(())
            });
            let engine = (|| -> Result<()> {
                let mut pending = VecDeque::new();
                for sequence in 1..=WAVES {
                    let slot = (sequence as usize - 1) % SLOTS;
                    let previous = sequence.saturating_sub(SLOTS as u32);
                    // Under the fallback, a reused slot's prior wave was host-acknowledged at land.
                    if ce { ensure!(reached(words.load(Counter::Landed), previous), "unsatisfied staging reuse wait before CE"); }
                    // SAFETY: live slot staging/control and input; reuse waits precede SM staging.
                    unsafe {
                        library.cuda_stream_wait_value32(stream.raw, words.ptr(Counter::Landed), previous)?;
                        library.sm_copy_regions(&[((staging.buffer.ptr as usize + slot * STAGING) as *mut c_void,
                            input.buffer.ptr.cast_const(), STAGING as u64)], stream.raw)?;
                        library.cuda_stream_write_value32(stream.raw, words.ptr(Counter::Staged(slot)), sequence)?;
                    }
                    staged[slot].record(&stream)?;
                    send.send((sequence, slot))?;
                    pending.push_back((sequence, slot));
                    max_ahead = max_ahead.max(pending.len());
                    if pending.len() < SLOTS && sequence < WAVES { continue; }
                    loop {
                        let (sequence, slot) = pending.pop_front().context("fixture oldest wave")?;
                        if ce {
                            let waiting = Instant::now();
                            ensure!(acknowledgement.recv_timeout(Duration::from_secs(2))? == sequence, "fixture event generation mismatch");
                            host_wait += waiting.elapsed();
                            if sequence == 1 {
                                for event in &staged {
                                    // SAFETY: all three current staging records precede first land.
                                    first_land_gpu_staged += usize::from(unsafe { library.cuda_event_query(event.raw)? });
                                }
                                ensure!(first_land_gpu_staged == SLOTS, "host land wait prevented depth-three GPU run-ahead");
                            }
                            // SAFETY: worker recorded THIS generation before acknowledging; slot not reused until consumed.
                            unsafe { library.cuda_stream_wait_event(stream.raw, copied[slot].raw)?; }
                        } else {
                            for rank in 0..RANKS {
                                // SAFETY: same host-release/per-rank SM pulls as today's Waves::land.
                                unsafe {
                                    library.cuda_stream_wait_value32(stream.raw, words.ptr(Counter::Arrived(rank)), sequence)?;
                                    library.pull_frames(&[pulls.at(slot, rank).cast_const()],
                                        &[((output.buffer.ptr as usize + slot * BYTES + rank * CHUNK) as *mut c_void, CHUNK as u64)], stream.raw)?;
                                }
                            }
                            // SAFETY: the worker publishes all completed replies from the host.
                            unsafe { library.cuda_stream_wait_value32(stream.raw, words.ptr(Counter::Landed), sequence)?; }
                        }
                        let target = CuteafdDeviceBuffer { ptr: (table.buffer.ptr as usize + slot * TABLES) as *mut c_void,
                            bytes: TABLES, ..table.buffer };
                        // SAFETY: actual Muse table-upload API after the landing dependency; reserved before waits.
                        unsafe {
                            library.copy_h2d_ordered(target, &vec![0x51; TABLES], stream.raw)?;
                            library.cuda_stream_write_value32(stream.raw, words.ptr(Counter::Consumed(slot)),
                                (sequence - 1) / SLOTS as u32 + 1)?;
                        }
                        if sequence < WAVES - SLOTS as u32 + 1 || pending.is_empty() { break; }
                    }
                }
                Ok(())
            })();
            drop(send);
            if engine.is_err() {
                stop.store(true, Ordering::Release);
                for rank in 0..RANKS { words.store(Counter::Arrived(rank), WAVES); }
                words.store(Counter::Landed, WAVES);
            }
            let worker_result = worker.join().map_err(|_| anyhow::anyhow!("fixture worker panicked"))?;
            engine?;
            worker_result?;
            Ok(())
        });
        // A failed worker must release every current/future SM wait before RAII stream drainage.
        if result.is_err() {
            for rank in 0..RANKS { words.store(Counter::Arrived(rank), WAVES); }
            words.store(Counter::Landed, WAVES);
        }
        stream.drain()?;
        copy.drain()?;
        result?;
        let elapsed = started.elapsed();
        for slot in 0..SLOTS {
            let mut actual = vec![0u8; BYTES];
            let view = CuteafdDeviceBuffer { ptr: (output.buffer.ptr as usize + slot * BYTES) as *mut c_void,
                bytes: BYTES, ..output.buffer };
            library.copy_d2h(&mut actual, view)?;
            ensure!(actual == host.bytes(), "fixture slot {slot} differs, CE={ce}");
        }
        eprintln!("reply_ce_ring ce={ce} waves={WAVES} depth={SLOTS} bytes={BYTES} arrival_delay_us=19000 rank_skew_us=750 elapsed_ms={:.3} wave_ms={:.3} host_land_wait_ms={:.3} max_ahead={max_ahead}",
            elapsed.as_secs_f64() * 1e3, elapsed.as_secs_f64() * 1e3 / WAVES as f64, host_wait.as_secs_f64() * 1e3);
        results.push(elapsed.as_secs_f64());
    }
    eprintln!("reply_ce_ring speedup={:.6}", results[0] / results[1]);
    Ok(())
}

/// The "GPU": the control words, written by the engine side of the test as
/// a stream would (staged after copies, consumed after a reduce) and read by
/// the worker; the worker publishes `landed` here.
struct World {
    words: Vec<AtomicU32>,
    slots: usize,
    /// Posted sequences, in order, and the slot each went to.
    posted: Mutex<Vec<(u32, usize)>>,
    /// Per slot: the sequence currently posted into it and not yet landed.
    occupied: Mutex<Vec<Option<u32>>>,
    fail_post: Option<u32>,
    fail_poll: Option<u32>,
    released: AtomicBool,
    polls: AtomicUsize,
}

impl World {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self { words: (0..Counter::words(slots)).map(|_| AtomicU32::new(0)).collect(), slots,
            posted: Mutex::new(Vec::new()), occupied: Mutex::new(vec![None; slots]), fail_post: None, fail_poll: None,
            released: AtomicBool::new(false), polls: AtomicUsize::new(0) })
    }
    fn load(&self, counter: Counter) -> u32 { self.words[counter.index(self.slots)].load(Ordering::Acquire) }
    fn store(&self, counter: Counter, value: u32) { self.words[counter.index(self.slots)].store(value, Ordering::Release) }
}

struct Fake { world: Arc<World> }
struct FakeWave { sequence: u32, slot: usize, polls: usize }

impl Backend for Fake {
    type Wave = FakeWave;
    fn counter(&self, counter: Counter) -> u32 { self.world.load(counter) }
    fn post(&mut self, job: Job) -> Result<FakeWave> {
        if self.world.fail_post == Some(job.sequence) { anyhow::bail!("injected post failure"); }
        let mut occupied = self.world.occupied.lock().unwrap();
        assert!(occupied[job.slot].is_none(), "slot {} reposted before its wave landed", job.slot);
        // The planes are free: every fenced slot's previous reduce was consumed.
        assert!(job.consumed.iter().all(|&(slot, uses)| reached(self.world.load(Counter::Consumed(slot)), uses)));
        if let Source::Staged { .. } = job.source {
            assert!(reached(self.world.load(Counter::Staged(job.slot)), job.sequence), "posted before staged");
        }
        let mut posted = self.world.posted.lock().unwrap();
        if let Some(&(previous, _)) = posted.last() { assert_eq!(job.sequence, previous.wrapping_add(1), "posts out of order"); }
        posted.push((job.sequence, job.slot));
        occupied[job.slot] = Some(job.sequence);
        Ok(FakeWave { sequence: job.sequence, slot: job.slot, polls: 0 })
    }
    fn poll(&mut self, wave: &mut FakeWave, _info: &WaveInfo) -> Result<bool> {
        let slot = wave.slot;
        self.world.polls.fetch_add(1, Ordering::Relaxed);
        wave.polls += 1;
        if self.world.fail_poll == Some(wave.sequence) && wave.polls == 2 { anyhow::bail!("injected receive failure"); }
        if wave.polls < 3 { return Ok(false); }
        self.world.occupied.lock().unwrap()[slot] = None;
        Ok(true)
    }
    fn publish(&mut self, sequence: u32) {
        if !reached(self.world.load(Counter::Landed), sequence) { self.world.store(Counter::Landed, sequence); }
        self.world.released.store(true, Ordering::Release);
    }
}

fn job(sequence: u32, slot: usize, consumed: u32) -> Job {
    Job { sequence, slot, rows: 1, payload_bytes: 64, consumed: vec![(slot, consumed)], slice: (slot * 64, 64),
        release: consumed + 1, sent: Instant::now(),
        source: Source::Staged { layout: RouteStaging::new(1, 8, 16).unwrap(), combined_reply: false,
            build: Box::new(|_, _| anyhow::bail!("the fake posts without building")) } }
}

/// Spins (bounded) until `counter` reaches `target`, as a stream wait would.
fn wait(world: &World, counter: Counter, target: u32) -> Result<()> {
    let started = Instant::now();
    while !reached(world.load(counter), target) {
        ensure!(started.elapsed() < Duration::from_secs(5), "{counter:?} never reached {target}");
        std::thread::yield_now();
    }
    Ok(())
}

/// The engine as a single stream: a decode step of `layers` layers queued on
/// it as send (wait for the slot's previous landing, stage, write staged),
/// land (wait for landed), reduce (write consumed). The host side (sending
/// jobs) runs ahead of the "stream" thread, which executes the queued ops in
/// order: exactly the run-ahead serving does.
fn drive(world: &Arc<World>, slots: usize, layers: u32, timeout: Duration)
    -> (Result<()>, Result<Stop>, Option<u32>) {
    #[derive(Debug)]
    enum Op { WaitLanded(u32), Staged(usize, u32), Consumed(usize, u32) }
    let (jobs, inbox) = mpsc::channel();
    let (ops, queue) = mpsc::channel::<Op>();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, timeout, false));
        let stream = scope.spawn(|| -> Result<()> {
            for op in queue {
                match op {
                    Op::WaitLanded(sequence) => wait(world, Counter::Landed, sequence)?,
                    Op::Staged(slot, sequence) => world.store(Counter::Staged(slot), sequence),
                    Op::Consumed(slot, uses) => world.store(Counter::Consumed(slot), uses),
                }
            }
            Ok(())
        });
        // The host queues the whole step without waiting for anything.
        let mut uses = vec![0u32; slots];
        for sequence in 1..=layers {
            let slot = (sequence as usize - 1) % slots;
            ops.send(Op::WaitLanded(sequence.wrapping_sub(slots as u32))).unwrap();
            ops.send(Op::Staged(slot, sequence)).unwrap();
            jobs.send(Command::Wave(job(sequence, slot, uses[slot]))).unwrap();
            ops.send(Op::WaitLanded(sequence)).unwrap();
            uses[slot] += 1;
            ops.send(Op::Consumed(slot, uses[slot])).unwrap();
        }
        drop(ops);
        let stream = stream.join().unwrap();
        drop(jobs);
        let (worker, highest) = worker.join().unwrap();
        (stream, worker, highest)
    })
}

#[test]
fn a_decode_step_queued_ahead_lands_every_layer_in_order() {
    for slots in 1..=4 {
        let world = World::new(slots);
        let (stream, worker, highest) = drive(&world, slots, 80, Duration::from_secs(5));
        stream.unwrap();
        worker.unwrap();
        assert_eq!(highest, Some(80));
        let posted = world.posted.lock().unwrap();
        assert_eq!(posted.iter().map(|p| p.0).collect::<Vec<_>>(), (1..=80).collect::<Vec<_>>());
        assert!(posted.iter().all(|&(sequence, slot)| slot == (sequence as usize - 1) % slots));
        assert_eq!(world.load(Counter::Landed), 80);
    }
}

#[test]
fn a_wave_waits_for_its_staged_copies_and_its_slots_previous_reduce() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        // Wave 2 reuses slot 0, whose wave 1 is landed but not yet reduced.
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        wait(&world, Counter::Landed, 1).unwrap();
        jobs.send(Command::Wave(job(2, 0, 1))).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(world.posted.lock().unwrap().len(), 1, "posted before staged");
        world.store(Counter::Staged(0), 2);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(world.posted.lock().unwrap().len(), 1, "posted into planes a pending reduce still reads");
        world.store(Counter::Consumed(0), 1);
        wait(&world, Counter::Landed, 2).unwrap();
        drop(jobs);
        let (result, _) = worker.join().unwrap();
        result.unwrap();
    });
}

#[test]
fn a_failure_releases_every_queued_and_later_wait() {
    for (fail_post, fail_poll) in [(Some(3), None), (None, Some(2)), (Some(1), None), (None, Some(40))] {
        let mut world = World::new(2);
        Arc::get_mut(&mut world).unwrap().fail_post = fail_post;
        Arc::get_mut(&mut world).unwrap().fail_poll = fail_poll;
        let (jobs, inbox) = mpsc::channel();
        std::thread::scope(|scope| {
            let world2 = world.clone();
            let worker = scope.spawn(move || {
                let mut backend = Fake { world: world2 };
                let (result, highest) = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
                assert!(result.is_err());
                release_until_closed(&mut backend, &inbox, highest);
            });
            // The engine keeps queueing (it does not know yet) and waits on
            // the GPU: every wait must release.
            let mut uses = [0u32; 2];
            for sequence in 1..=60u32 {
                let slot = (sequence as usize - 1) % 2;
                wait(&world, Counter::Landed, sequence.wrapping_sub(2)).unwrap();
                world.store(Counter::Staged(slot), sequence);
                if jobs.send(Command::Wave(job(sequence, slot, uses[slot]))).is_err() { break; }
                wait(&world, Counter::Landed, sequence).unwrap();
                uses[slot] += 1;
                world.store(Counter::Consumed(slot), uses[slot]);
            }
            drop(jobs);
            worker.join().unwrap();
        });
        assert!(world.released.load(Ordering::Acquire));
    }
}

#[test]
fn closing_with_waves_in_flight_fails_instead_of_hanging() {
    let world = World::new(2);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        // Wave 2's copies never complete: the engine errored before.
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        jobs.send(Command::Wave(job(2, 1, 0))).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        drop(jobs);
        let (result, highest) = worker.join().unwrap();
        assert!(result.is_err() && highest == Some(2));
    });
}

#[test]
fn a_stalled_gpu_times_out_instead_of_hanging() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    let (result, _) = std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_millis(50), false));
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        let joined = worker.join().unwrap();
        drop(jobs);
        joined
    });
    assert!(format!("{:#}", result.unwrap_err()).contains("no progress"));
}

#[test]
fn an_idle_worker_blocks_instead_of_spinning() {
    let world = World::new(1);
    let (jobs, inbox) = mpsc::channel();
    std::thread::scope(|scope| {
        let mut backend = Fake { world: world.clone() };
        let worker = scope.spawn(move || run_worker(&mut backend, &inbox, Duration::from_secs(5), false));
        world.store(Counter::Staged(0), 1);
        jobs.send(Command::Wave(job(1, 0, 0))).unwrap();
        wait(&world, Counter::Landed, 1).unwrap();
        let polls = world.polls.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(50));
        // Blocked on its channel: no further progress passes while idle.
        assert_eq!(world.polls.load(Ordering::Relaxed), polls);
        // The next request wakes it and lands.
        world.store(Counter::Staged(0), 2);
        world.store(Counter::Consumed(0), 1);
        jobs.send(Command::Wave(job(2, 0, 1))).unwrap();
        wait(&world, Counter::Landed, 2).unwrap();
        drop(jobs);
        worker.join().unwrap().0.unwrap();
    });
}

#[test]
fn counters_compare_cyclically_across_wraparound() {
    assert!(reached(5, 5) && reached(6, 5) && !reached(4, 5));
    assert!(reached(1, u32::MAX) && !reached(u32::MAX, 1));
    // A slot's first use waits for `sequence - waves`, already reached.
    for waves in 1..=4u32 { assert!(reached(0, 1u32.wrapping_sub(waves))); }
    assert!(reached(0, 0u32.wrapping_sub(3)));
}

/// No family exchanges inline any more: every Spark wave goes through the
/// link's persistent exchange, and nothing waits on the host for routes or
/// a receive around it.
#[test]
fn no_family_exchanges_on_the_engine_thread() {
    let banned = [
        [".dis", "patch(&"].concat(), ["block_", "on(self.moe_land"].concat(), ["block_", "on(spark_land"].concat(),
        [".rece", "ive(wave"].concat(), ["receive_", "wave("].concat(), ["with_", "waves("].concat(),
        ["with_link_", "waves("].concat(),
    ];
    for (family, source) in [
        ("glm5", include_str!("../families/glm5/engine.rs")),
        ("glm5 layers", include_str!("../families/glm5/engine/layers.rs")),
        ("glm5 mod", include_str!("../families/glm5/mod.rs")),
        ("glm5_flash", include_str!("../families/glm5_flash/engine.rs")),
        ("glm5_flash mod", include_str!("../families/glm5_flash/mod.rs")),
        ("mimo_v2", include_str!("../families/mimo_v2/engine.rs")),
        ("mimo_v2 mod", include_str!("../families/mimo_v2/mod.rs")),
        ("qwen4", include_str!("../families/qwen4/engine.rs")),
        ("step5", include_str!("../families/step5/execution.rs")),
        ("deepseek_v4", include_str!("../families/deepseek_v4/engine.rs")),
        ("deepseek_v4 prefill", include_str!("../families/deepseek_v4/engine/prefill.rs")),
        ("kimi_k3", include_str!("../families/kimi_k3/spark.rs")),
        ("deepseek_v41", include_str!("../families/deepseek_v41/v41_experts/coordinator.rs")),
    ] {
        for needle in &banned {
            assert!(!source.contains(needle.as_str()), "{family} still exchanges inline ({needle})");
        }
    }
}

/// A reply frame of the echo world: a mapped pinned buffer from a pool. When
/// the exchange releases it, it is scribbled over (as the NIC would rewrite a
/// reposted receive) and returned, so a pull that ran after the release reads
/// garbage and the byte-exact check fails.
struct EchoFrame { host: usize, bytes: usize, capacity: usize, rank: usize, buffers: Arc<EchoBuffers> }

/// The echo world's pinned frame buffers, freed when the last frame and the
/// world are gone.
struct EchoBuffers {
    library: Arc<NativeLibrary>,
    /// The buffers (host address, bytes) to free.
    owned: Mutex<Vec<(usize, usize)>>,
    /// Per rank: free buffers (host address, bytes) of its ring.
    pools: Vec<Mutex<Vec<(usize, usize)>>>,
    /// Frames released so far (every rank).
    released: AtomicUsize,
}
impl Drop for EchoBuffers {
    fn drop(&mut self) {
        for (ptr, bytes) in self.owned.get_mut().unwrap().drain(..) {
            let mut buffer = cuteafd_ffi::CuteafdHostBuffer { ptr: ptr as *mut c_void, bytes,
                flags: cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED | cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_MAPPED };
            if let Err(error) = self.library.free_host_buffer(&mut buffer) { eprintln!("freeing an echo frame: {error:#}"); }
        }
    }
}
impl ReplyFrame for EchoFrame {
    fn len(&self) -> usize { self.bytes }
    fn device_address(&self) -> Option<u64> { Some(self.host as u64) }
}
impl Drop for EchoFrame {
    fn drop(&mut self) {
        // SAFETY: a pinned buffer of `buffers`, which this frame keeps alive.
        unsafe { std::ptr::write_bytes(self.host as *mut u8, 0x7f, self.capacity) };
        self.buffers.released.fetch_add(1, Ordering::SeqCst);
        self.buffers.pools[self.rank].lock().unwrap().push((self.host, self.capacity));
    }
}

/// An in-process "Spark world" for the GPU exchange test: each rank answers a
/// request with its rows scaled by `rank + 1` (BF16 bit patterns), from its
/// own thread-free reply queue, a few polls after the post; rank r arrives
/// `r * stagger` after rank 0. Ranks in `landing` write their rows straight
/// into device planes, the way GPUDirect landing does (here by a blocking H2D
/// copy the "NIC" finishes before it reports the wave); the others arrive in
/// pooled mapped pinned frames ([`EchoFrame`]), at most `depth` per rank held
/// at once (a receive ring).
struct EchoWorld {
    library: std::sync::Arc<NativeLibrary>,
    ranks: usize,
    row_bytes: usize,
    landing: u8,
    /// Each rank's reserve range base (an integer: it crosses to the worker).
    ranges: Vec<usize>,
    /// Each layer's input rows: a request must carry exactly its layer's.
    inputs: Vec<Vec<u8>>,
    /// The exchange's counters, set once it started (diagnosis).
    words: Arc<std::sync::OnceLock<Words>>,
    /// A rank's rows arrive this long after the post (a Spark round trip).
    delay: Duration,
    /// The worker thread's CPU seconds and wall time over the run.
    cpu: Option<(f64, Instant)>,
    spent: Option<(f64, f64)>,
    /// Rank r's rows arrive `r * stagger` after rank 0's.
    stagger: Duration,
    buffers: Arc<EchoBuffers>,
}
struct EchoWave { offset: usize, rows: Vec<Vec<u8>>, polls: usize, posted: Instant, sent: u8 }

impl EchoWorld {
    /// A ring of `depth` replies per rank (two frame buffers of
    /// `frame_bytes` each per reply, offset by 4 bytes, so frames start 4-byte
    /// aligned like real payloads after their row indices).
    #[allow(clippy::too_many_arguments)]
    fn new(library: Arc<NativeLibrary>, ranks: usize, row_bytes: usize, landing: u8, ranges: Vec<usize>,
        inputs: Vec<Vec<u8>>, words: Arc<std::sync::OnceLock<Words>>, delay: Duration, stagger: Duration,
        depth: usize, frame_bytes: usize) -> Result<Self> {
        let buffers = Arc::new(EchoBuffers { library: library.clone(), owned: Mutex::new(Vec::new()),
            pools: (0..ranks).map(|_| Mutex::new(Vec::new())).collect(), released: AtomicUsize::new(0) });
        for rank in 0..ranks {
            for _ in 0..2 * depth {
                let host = library.alloc_host_buffer(frame_bytes + 16)?;
                buffers.owned.lock().unwrap().push((host.ptr as usize, host.bytes));
                buffers.pools[rank].lock().unwrap().push((host.ptr as usize + 4, frame_bytes));
            }
        }
        Ok(Self { library, ranks, row_bytes, landing, ranges, inputs, words, delay, cpu: None, spent: None,
            stagger, buffers })
    }
    fn frame(&self, rank: usize, rows: &[u8]) -> Result<EchoFrame> {
        let (host, capacity) = self.buffers.pools[rank].lock().unwrap().pop()
            .with_context(|| format!("rank {rank}'s receive ring ran dry"))?;
        ensure!(rows.len() <= capacity, "echo frame too small");
        // SAFETY: a free pooled pinned buffer of `capacity` bytes.
        unsafe { std::ptr::copy_nonoverlapping(rows.as_ptr(), host as *mut u8, rows.len()) };
        Ok(EchoFrame { host, bytes: rows.len(), capacity, rank, buffers: self.buffers.clone() })
    }
}

fn library_path() -> Result<std::ffi::OsString> {
    std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")
}

fn rank_rows(hidden: &[u8], rank: usize) -> Vec<u8> {
    hidden.chunks_exact(2).flat_map(|word| {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16) * (rank as f32 + 1.0);
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }).collect()
}

impl Transport for EchoWorld {
    type Wave = EchoWave;
    type Frame = EchoFrame;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[cuteafd_transport::LandingSlice]>)
        -> Result<EchoWave> {
        // A landing link places every wave; a host-receive link names none.
        ensure!(landing.is_some() == (self.landing != 0), "landing slices: {landing:?}, landing ranks {:#b}", self.landing);
        let offset = landing.map_or(0, |slices| slices[0].offset);
        ensure!(landing.is_none_or(|slices| slices.len() == self.ranks && slices.iter().all(|s| s.offset == offset
            && s.bytes >= request.header.row_count as usize * self.row_bytes)), "landing slices disagree: {landing:?}");
        let hidden = request.hidden_payload.as_ref();
        ensure!(hidden.len() == request.header.row_count as usize * self.row_bytes, "echo rows differ");
        let layer = request.header.layer_id as usize;
        if hidden != self.inputs[layer].as_slice() {
            // Snapshot now, then again later: still arriving (own layer
            // later) or overwritten (a later layer)?
            let words = self.words.get().copied();
            let counters = || words.map(|w| (w.load(Counter::Landed), w.load(Counter::Staged(0)), w.load(Counter::Consumed(0))));
            let mut report = Vec::new();
            for delay_us in [0u64, 50, 500, 5000, 20000] {
                std::thread::sleep(Duration::from_micros(delay_us));
                let snapshot: Vec<u8> = hidden.to_vec();
                let matches = self.inputs.iter().position(|input| input.as_slice() == snapshot.as_slice());
                let differ = snapshot.iter().zip(&self.inputs[layer]).filter(|(a, b)| a != b).count();
                let first = snapshot.iter().zip(&self.inputs[layer]).position(|(a, b)| a != b);
                report.push(format!("+{delay_us}us: differ {differ} first {first:?} matches {matches:?} \
                    (landed, staged, consumed) {:?}", counters()));
            }
            anyhow::bail!("request for layer {layer}: {}", report.join("; "));
        }
        if self.cpu.is_none() { self.cpu = Some((cuteafd_ffi::thread_cpu_seconds()?, Instant::now())); }
        Ok(EchoWave { offset, rows: (0..self.ranks).map(|rank| rank_rows(hidden, rank)).collect(), polls: 0,
            posted: Instant::now(), sent: 0 })
    }
    fn poll(&mut self, wave: &mut EchoWave, sink: &mut dyn FnMut(usize, u32, EchoFrame) -> Result<()>)
        -> Result<Option<u8>> {
        wave.polls += 1;
        if wave.polls < 3 { return Ok(None); }
        let age = wave.posted.elapsed();
        for (rank, rows) in wave.rows.iter().enumerate() {
            if wave.sent & (1 << rank) != 0 || age < self.delay + self.stagger * rank as u32 { continue; }
            wave.sent |= 1 << rank;
            if self.landing & (1 << rank) != 0 {
                let ptr = self.ranges[rank] + wave.offset;
                self.library.copy_h2d(CuteafdDeviceBuffer { ptr: ptr as *mut c_void, bytes: rows.len(),
                    ..Default::default() }, rows)?;
            } else {
                // Two frames per rank, row-indexed, as chunked replies arrive.
                let half = (rows.len() / self.row_bytes / 2) * self.row_bytes;
                let (first, second) = (self.frame(rank, &rows[..half])?, self.frame(rank, &rows[half..])?);
                sink(rank, 0, first)?;
                sink(rank, (half / self.row_bytes) as u32, second)?;
            }
        }
        let all = if self.ranks == 8 { u8::MAX } else { (1u8 << self.ranks) - 1 };
        if wave.sent != all { return Ok(None); }
        if let Some((cpu, wall)) = self.cpu {
            self.spent = Some((cuteafd_ffi::thread_cpu_seconds()? - cpu, wall.elapsed().as_secs_f64()));
        }
        Ok(Some(self.landing))
    }
    fn timeout(&self) -> Duration { Duration::from_secs(10) }
    fn abandon(&mut self) {}
    fn retained_reply_slots(&self) -> Vec<usize> {
        // Two frames per reply here: a ring of `2 * depth` frames holds `depth` waves.
        self.buffers.pools.iter().map(|pool| pool.lock().unwrap().len() / 2).collect()
    }
}

/// GPU gate: a whole decode step (80 layers, one slot) and prefill runs
/// (several slots) queued ahead on one stream through the real exchange:
/// counters in mapped memory, stream waits and writes, staging copies,
/// pinned-rank uploads behind the wait, and the reduce. Each case runs its
/// layers at its fixed cut (slices placed per wave); the reduced output of
/// every layer must equal the host reference byte for byte. Opt in with
/// `CUTEAFD_PLANE_UPLOAD_LIBRARY` (the built native library).
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_exchange_queues_a_step_ahead_and_lands_byte_exact() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&path)? });
    cuteafd_transport::set_verbs_host_native_library_path(std::path::PathBuf::from(&path))?;
    // The compact reducer sums rows of the process-wide expert hidden size
    // (5120 unless a family sets another; this test process never does), so
    // the planes are V4.1-shaped. Top-6 also takes the byte path of the
    // staging copy (24-byte route rows are not 16-byte multiples).
    let (hidden, topk) = (5120usize, 6usize);
    let row_bytes = hidden * 2;
    // (ranks, landing ranks, slots, rows, layers, delay, per-rank stagger).
    // The 800 us case models a Spark round trip and reports the worker
    // thread's CPU share. Slot rows scale the shared 2752-row unit down to
    // test size; 128 rows x 10 KiB x 4 pinned ranks crosses the per-rank pull
    // threshold (#815), so those waves pull rank by rank as each arrives.
    for (ranks, landing, slots, max_rows, layers, delay_us, stagger_us) in [
        (3usize, 0u8, 1usize, 16usize, 80u32, 0u64, 0u64),
        (4, 0b0101, 1, 64, 80, 0, 0), (3, 0b111, 1, 1, 80, 0, 0),
        (4, 0, 3, 128, 36, 0, 0), (4, 0, 3, 128, 36, 200, 300), (4, 0, 4, 64, 36, 0, 0),
        (6, 0b100001, 3, 128, 36, 0, 100),
        (4, 0b1111, 3, 128, 36, 0, 0), (3, 0b111, 1, 1, 80, 800, 0)] {
        let (delay, stagger) = (Duration::from_micros(delay_us), Duration::from_micros(stagger_us));
        let cuts = [(slots, max_rows)];
        let reserve = slots * max_rows;
        let total = layers as usize;
        let ranges_owned = (0..ranks).map(|_| DeviceAllocation::new(&library, reserve * row_bytes))
            .collect::<Result<Vec<_>>>()?;
        let ranges: Vec<CuteafdDeviceBuffer> = ranges_owned.iter().map(|range| range.buffer).collect();
        let max_layout = RouteStaging::new(max_rows, topk, row_bytes)?;
        let arenas = (0..slots).map(|_| StagingArena::new(max_layout.bytes)).collect::<Result<Vec<_>>>()?;
        let words = Arc::new(std::sync::OnceLock::new());
        // Each layer's rows at its cut's width (the widest rows the layer may carry).
        let widths: Vec<usize> = cuts.iter().flat_map(|&(_, rows)| std::iter::repeat_n(rows, layers as usize)).collect();
        let hosts: Vec<Vec<u8>> = (0..total).map(|layer| (0..widths[layer] * hidden).flat_map(|i| {
            // Multiples of 0.5 in [-2, 2]: every rank partial (x rank + 1) and
            // every sum over up to six ranks is exact in BF16, so the reference
            // does not depend on the reducer's summation order.
            // The first eight values carry the layer's bits (0 or 0.5), so
            // every layer's rows differ.
            let value = if i < 8 { ((layer >> i) & 1) as f32 / 2.0 } else { ((i * 7 + layer * 13) % 9) as f32 / 2.0 - 2.0 };
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        }).collect()).collect();
        // A ring as deep as the admitted slots: the exchange holds a wave's
        // frames until its reduce, so a shallower ring would run dry.
        let echo = EchoWorld::new(library.clone(), ranks, row_bytes, landing,
            ranges.iter().map(|range| range.ptr as usize).collect(), hosts.clone(), words.clone(), delay, stagger,
            slots, max_rows.div_ceil(2) * row_bytes)?;
        let geometry = crate::shared::wave_geometry::SlotGeometry::new(slots, max_rows)?;
        let intake = Intake { prefill: true, ranks, row_bytes, ranges, geometry, landing, placed: true };
        let storage = std::rc::Rc::new(ExchangeStorage::new(&library, &intake)?);
        let mut exchange = Waves::start(&library, intake, echo, Vec::new(), arenas, false, storage.clone())?;
        let _ = words.set(exchange.words());
        let stream = crate::shared::spark_intake::WarmStream::new(library.as_ref())?;
        let ids = DeviceAllocation::new(&library, max_rows * topk * 4)?;
        let weights = DeviceAllocation::new(&library, max_rows * topk * 4)?;
        // One input and one output per layer, all live until the end.
        let inputs = (0..total).map(|layer| DeviceAllocation::new(&library, widths[layer] * row_bytes)).collect::<Result<Vec<_>>>()?;
        let outputs = (0..total).map(|layer| DeviceAllocation::new(&library, widths[layer] * row_bytes)).collect::<Result<Vec<_>>>()?;
        let mut expected = Vec::new();
        for (layer, input) in inputs.iter().enumerate() {
            let host = &hosts[layer];
            library.copy_h2d(input.buffer, host)?;
            let sum: Vec<u8> = (0..widths[layer] * hidden).flat_map(|i| {
                let total: f32 = (0..ranks).map(|rank| {
                    let word = rank_rows(&host[2 * i..2 * i + 2], rank);
                    f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
                }).sum();
                ((total.to_bits() >> 16) as u16).to_le_bytes()
            }).collect();
            expected.push(sum);
        }
        let mut tickets = std::collections::VecDeque::new();
        let land = |exchange: &mut Waves<'_>, layer: usize, ticket: WaveTicket| -> Result<()> {
            let slot = ticket.slot();
            exchange.land(ticket, stream.raw())?;
            // SAFETY: planes, output and the null shared plane are live.
            unsafe { exchange.reduce(slot, std::ptr::null(),
                outputs[layer].buffer.ptr.cast(), widths[layer], stream.raw()) }
        };
        for (index, &(waves, rows)) in cuts.iter().enumerate() {
            assert_eq!(exchange.waves(), waves);
            let layout = RouteStaging::new(rows, topk, row_bytes)?;
            // The host queues every layer of this cut before the GPU runs the
            // first: send, land, reduce, consumed, back to back, with up to
            // `waves` waves in flight (prefill units).
            for layer in index * layers as usize..(index + 1) * layers as usize {
                if tickets.len() == waves {
                    let (layer, ticket) = tickets.pop_front().unwrap();
                    land(&mut exchange, layer, ticket)?;
                }
                let (h, l) = (hidden, layer);
                // SAFETY: ids, weights and the layer's input are live device buffers of the layout's extents.
                let ticket = unsafe { exchange.send(rows, rows * row_bytes, layout, [ids.buffer, weights.buffer, inputs[layer].buffer],
                    stream.raw(), false, Box::new(move |routes, wire| ExpertProtocolV2Request::new_bytes(l as u64 + 1, 1, l as u32,
                        h as u32, cuteafd_transport::ExpertV2Dtype::Bf16, (0..rows as u32).map(|row|
                            cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                            source_kind: cuteafd_transport::ExpertV2SourceKind::Decode, source_request_id: 1,
                            token_position: row.into(), route_offset: row * topk as u32, route_count: topk as u32 })
                            .collect(), routes, wire)))? };
                tickets.push_back((layer, ticket));
            }
        }
        while let Some((layer, ticket)) = tickets.pop_front() { land(&mut exchange, layer, ticket)?; }
        stream.finish()?;
        healthy()?;
        let all = &expected;
        for (layer, (output, expected)) in outputs.iter().zip(all).enumerate() {
            let rows = widths[layer];
            let mut actual = vec![0u8; rows * row_bytes];
            library.copy_d2h(&mut actual, output.buffer)?;
            if actual != *expected {
                let first = actual.iter().zip(expected.iter()).position(|(a, e)| a != e).unwrap();
                let bad = actual.chunks(row_bytes).zip(expected.chunks(row_bytes)).filter(|(a, e)| a != e).count();
                panic!("ranks={ranks} landing={landing:#b} cuts={cuts:?} layer {layer} differs: first byte {first}, \
                    {bad} of {rows} rows");
            }
        }
        ensure!(exchange.outstanding.is_empty() && exchange.unread.iter().all(|unread| !unread),
            "the exchange closes at rest: every wave landed and was consumed");
        let (echo, frames, result) = exchange.close::<EchoWorld>()?;
        result?;
        assert!(frames.is_empty(), "a closed exchange whose waves were all consumed holds no frames");
        if landing != (1 << ranks) - 1 {
            assert_eq!(echo.buffers.released.load(Ordering::SeqCst), 2 * total * (ranks - landing.count_ones() as usize),
                "every pinned frame went back to its ring");
        }
        let share = echo.spent.map(|(cpu, wall)| format!(", worker CPU {:.0}% of {:.1} ms", 100.0 * cpu / wall, wall * 1e3))
            .filter(|_| delay_us > 0).unwrap_or_default();
        eprintln!("exchange ranks={ranks} landing={landing:#b} cuts={cuts:?}: {total} layers byte-exact{share}");
    }
    Ok(())
}

/// #784: terminal teardown closes the exchange with waves still in flight
/// (an engine that stopped mid-step) and gets its transport back to quiesce
/// and release the QPs; the worker reports the abandoned waves and releases
/// every wait it was queued, so no stream blocks behind them.
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_terminal_close_returns_the_transport_with_waves_in_flight() -> Result<()> {
    terminal_close_with_codec(false)
}

#[test]
#[ignore = "requires a GPU and native codec exports; run alone (--exact)"]
fn gpu_compressed_close_retains_reducer_and_reset_reuses_storage() -> Result<()> {
    cuteafd_core::spark_reply::set_requested(Some(cuteafd_core::SparkReplyFormat::Nvfp4))
        .map_err(anyhow::Error::msg)?;
    terminal_close_with_codec(true)
}

fn terminal_close_with_codec(compressed: bool) -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&path)? });
    cuteafd_transport::set_verbs_host_native_library_path(std::path::PathBuf::from(&path))?;
    let (hidden, topk, ranks, rows, waves, layers) = (5120usize, 6usize, 3usize,
        if compressed { 65usize } else { 4usize }, 2usize, 4usize);
    let row_bytes = hidden * 2;
    let owner = crate::shared::spark_intake::SparkIntake::with_waves(&library,
        crate::shared::spark_intake::IntakeMode::Pinned, ranks, rows, row_bytes, waves)?;
    let ranges = owner.ranges();
    let layout = RouteStaging::new(rows, topk, row_bytes)?;
    let arenas = (0..waves).map(|_| StagingArena::new(layout.bytes)).collect::<Result<Vec<_>>>()?;
    let words = Arc::new(std::sync::OnceLock::new());
    let hosts: Vec<Vec<u8>> = (0..layers).map(|_| vec![0u8; rows * row_bytes]).collect();
    // Replies take a second: every wave is still in flight when teardown closes.
    let echo = EchoWorld::new(library.clone(), ranks, row_bytes, 0, ranges.iter().map(|range| range.ptr as usize).collect(),
        hosts, words.clone(), Duration::from_secs(1), Duration::ZERO, waves, rows * row_bytes)?;
    let geometry = crate::shared::wave_geometry::SlotGeometry::new(waves, rows)?;
    let intake = Intake { prefill: true, ranks, row_bytes, ranges, geometry, landing: 0, placed: true };
    let ledger_owner = "test/reply-codec-terminal-close";
    let device = library.cuda_get_device()?;
    let receipt = || {
        cuteafd_ffi::memory_ledger::snapshot().rows.iter().filter(|row|
            row.key.space == cuteafd_ffi::memory_ledger::Space::Device && row.key.device == device
                && row.key.owner == Some(ledger_owner) && row.key.scope == "spark-intake/reply_codec")
            .fold((0usize, 0usize), |(bytes, allocations), row| (bytes + row.bytes, allocations + row.allocations))
    };
    assert_eq!(receipt(), (0, 0));
    let storage = {
        let _owner = cuteafd_ffi::memory_ledger::owner(ledger_owner);
        owner.exchange_storage(&intake)?
    };
    let expected_receipt = (0, 0);
    assert_eq!(receipt(), expected_receipt, "fused decode/reduce allocates no codec scratch");
    let weak = std::rc::Rc::downgrade(&storage);
    let control = storage.control.buffer.ptr;
    let pull_block = storage.pull_block.as_ref().unwrap().buffer.ptr;
    let reducer = storage.reply_reduce.as_ref().map(std::rc::Rc::as_ptr);
    assert_eq!(reducer.is_some(), compressed);
    assert!(owner.exchange_storage(&intake).is_err(), "live exchange storage must not be reset");
    let mut exchange = Waves::start(&library, intake, echo, Vec::new(), arenas, false, storage)?;
    let _ = words.set(exchange.words());
    let stream = crate::shared::spark_intake::WarmStream::new(library.as_ref())?;
    let ids = DeviceAllocation::new(&library, rows * topk * 4)?;
    let weights = DeviceAllocation::new(&library, rows * topk * 4)?;
    let input = DeviceAllocation::new(&library, rows * row_bytes)?;
    // The worker checks each request against the layer's (zero) input rows;
    // a fresh allocation is not zeroed.
    library.cuda_zero_bytes(input.buffer, rows * row_bytes)?;
    let mut tickets = Vec::new();
    for layer in 0..waves {
        let h = hidden;
        // SAFETY: ids, weights and input are live device buffers of the layout's extents.
        tickets.push(unsafe { exchange.send(rows, rows * row_bytes, layout, [ids.buffer, weights.buffer, input.buffer], stream.raw(), compressed && rows > 64,
            Box::new(move |routes, wire| ExpertProtocolV2Request::new_bytes(layer as u64 + 1, 1, layer as u32,
                h as u32, cuteafd_transport::ExpertV2Dtype::Bf16, (0..rows as u32).map(|row|
                    cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                    source_kind: if compressed { cuteafd_transport::ExpertV2SourceKind::Prefill }
                        else { cuteafd_transport::ExpertV2SourceKind::Decode }, source_request_id: 1,
                    token_position: row.into(), route_offset: row * topk as u32, route_count: topk as u32 })
                    .collect(), routes, wire)))? });
    }
    std::thread::sleep(Duration::from_millis(20));
    assert!(!tickets.iter().any(|ticket| exchange.landed(ticket)), "replies must still be in flight");
    // The first wave's landing is queued as a layer's collect would queue it:
    // a device wait on the reply, a second away. The second stays unlanded.
    let first = tickets.remove(0);
    exchange.land(first, stream.raw())?;
    let started = std::time::Instant::now();
    let (echo, frames, result) = exchange.close::<EchoWorld>()?;
    assert!(frames.is_empty(), "no reply arrived, so none is held");
    assert!(started.elapsed() < Duration::from_millis(900), "close must not wait out the in-flight replies");
    let error = result.expect_err("closing with waves in flight reports them");
    assert!(format!("{error:#}").contains("outstanding"), "{error:#}");
    assert_eq!(echo.ranks, ranks, "the transport comes back to the link");
    // Both queued waits and queued pull descriptor loads still own storage
    // after the consuming close returned, before any stream synchronization.
    assert!(weak.upgrade().is_some());
    assert_eq!(weak.upgrade().unwrap().control.buffer.ptr, control);
    assert_eq!(weak.upgrade().unwrap().pull_block.as_ref().unwrap().buffer.ptr, pull_block);
    assert_eq!(receipt(), expected_receipt, "close retains queued codec storage before drainage");
    // The landing wait queued behind an abandoned wave was released (a
    // worker that did not release it would leave this stream blocked).
    stream.finish()?;
    let geometry = Intake { prefill: true, ranks, row_bytes, ranges: owner.ranges(), geometry: owner.geometry(),
        landing: 0, placed: true };
    let reset = owner.exchange_storage(&geometry)?;
    assert_eq!(reset.control.buffer.ptr, control, "reset reuses counters");
    assert_eq!(reset.pull_block.as_ref().unwrap().buffer.ptr, pull_block, "reset reuses descriptors");
    assert_eq!(reset.reply_reduce.as_ref().map(std::rc::Rc::as_ptr), reducer, "reset reuses the prebound reducer");
    assert_eq!(receipt(), expected_receipt, "reset allocates no additional codec storage");
    assert_eq!(reset.control.bytes().iter().copied().max(), Some(0));
    assert_eq!(reset.pull_block.as_ref().unwrap().bytes().iter().copied().max(), Some(0));
    drop(reset);
    owner.terminal_release_payloads();
    assert!(weak.upgrade().is_none(), "only proved terminal drainage releases storage");
    assert_eq!(receipt(), (0, 0), "terminal release removes the codec ledger owner");
    eprintln!("reply codec receipt: bytes={} allocations={} ranks={ranks} slots={waves} rows={rows}; retained through close, reused on reset, released after drain",
        expected_receipt.0, expected_receipt.1);
    assert!(started.elapsed() < Duration::from_millis(900), "the queued landing waited out the reply");
    drop(tickets);
    Ok(())
}

/// #784: no link path drops its started exchange without taking the
/// transport back (the exchange owns it once started).
#[test]
fn link_teardown_and_reset_take_the_transport_back_from_the_exchange() {
    let link = include_str!("spark_intake.rs");
    for name in ["pub(crate) fn terminal_quiesce(&mut self)", "pub(crate) fn reset_connections(&mut self)"] {
        let body = link.split(name).nth(1).unwrap().split("\n    }\n").next().unwrap();
        let stop = if name.contains("reset") { "exchange.cancel::<SparkExperts>()?" } else { "exchange.close::<SparkExperts>()?" };
        assert!(body.contains(stop) && body.contains("self.transport = Some(transport);")
            && !body.contains("drop(exchange)"), "{name}: {body}");
    }
    // A reset re-opens the sessions before the exchange restarts: its device
    // landing mask is read from the live sessions, as after bootstrap.
    let reset = link.split("pub(crate) fn reset_connections(&mut self)").nth(1).unwrap().split("\n    }\n").next().unwrap();
    let reopened = reset.find("transport.bootstrap(&request)?").expect("reset re-opens the sessions");
    assert!(reopened < reset.find("self.start()").unwrap(), "{reset}");
}

/// Block-mode pacing spins through the hybrid window: a decode wave (Kimi
/// K3's 0.6-0.7 ms per-layer round) never sleeps, however early it is in its
/// expected duration; only a wait past 10 ms sleeps, and only while early.
#[test]
fn pacing_spins_through_the_hybrid_window_before_sleeping() {
    let window = cuteafd_transport::wait_policy::BLOCK_SPIN.as_secs_f64();
    assert_eq!(window, 0.010);
    // A Kimi layer round: early in its mean, but inside the window.
    assert!(!Pace::early(Some(700e-6), 100e-6));
    assert!(!Pace::early(Some(700e-6), 400e-6));
    // A long wave: still spins for the window, then sleeps while early.
    assert!(!Pace::early(Some(0.050), 0.005));
    assert!(Pace::early(Some(0.050), 0.012));
    // Late in its expected duration: spins again.
    assert!(!Pace::early(Some(0.050), 0.040));
    // No mean yet: never sleeps.
    assert!(!Pace::early(None, 0.020));
}

/// A CPU receive ring for the [`Native`] backend tests (#815): each rank holds
/// `depth` receive slots; a reply takes a free one, and the transport refuses
/// to post a wave while any rank has none free. Frames are plain host
/// buffers; dropping one returns its slot.
struct Ring {
    ranks: usize,
    depth: usize,
    rows: usize,
    row_bytes: usize,
    /// Per rank: free slot buffers (address, bytes).
    free: Vec<Arc<Mutex<Vec<(usize, usize)>>>>,
    /// Per rank: arrivals per poll pass (rank r answers on pass r + 1).
    posted: Arc<AtomicUsize>,
    _buffers: Vec<Vec<u8>>,
    fail_poll_after: Option<usize>,
    /// Polls before a wave's replies start arriving (a slow wire).
    hold_polls: usize,
    payload_bytes: Option<usize>,
    request_bytes: Option<Vec<u8>>,
    landing_slices: Vec<(usize, usize)>,
    /// Waves given up after a failure / for a deliberate reset.
    abandoned: Arc<AtomicUsize>,
    cancelled: Arc<AtomicUsize>,
}
struct RingFrame { address: usize, bytes: usize, capacity: usize, free: Arc<Mutex<Vec<(usize, usize)>>> }
impl ReplyFrame for RingFrame {
    fn len(&self) -> usize { self.bytes }
    fn device_address(&self) -> Option<u64> { Some(self.address as u64) }
}
impl Drop for RingFrame {
    fn drop(&mut self) { self.free.lock().unwrap().push((self.address, self.capacity)); }
}
struct RingWave { passes: usize, sent: u8, bytes: usize }

impl Ring {
    fn new(ranks: usize, depth: usize, rows: usize, row_bytes: usize) -> Self {
        let mut buffers = Vec::new();
        let free = (0..ranks).map(|_| {
            let slots = (0..depth).map(|_| {
                let mut buffer = vec![0u8; rows * row_bytes];
                let entry = (buffer.as_mut_ptr() as usize, buffer.len());
                buffers.push(buffer);
                entry
            }).collect();
            Arc::new(Mutex::new(slots))
        }).collect();
        Self { ranks, depth, rows, row_bytes, free, posted: Arc::new(AtomicUsize::new(0)), _buffers: buffers,
            fail_poll_after: None, hold_polls: 0, payload_bytes: None, request_bytes: None, landing_slices: Vec::new(), abandoned: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicUsize::new(0)) }
    }
}

impl Transport for Ring {
    type Wave = RingWave;
    type Frame = RingFrame;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[cuteafd_transport::LandingSlice]>)
        -> Result<RingWave> {
        // A wave needs a free receive on every rank (an RC SEND into an empty
        // receive queue stalls the link).
        for (rank, free) in self.free.iter().enumerate() {
            ensure!(!free.lock().unwrap().is_empty(), "rank {rank}'s receive ring is dry at post");
        }
        self.posted.fetch_add(1, Ordering::SeqCst);
        self.request_bytes = request.encode().ok();
        self.landing_slices = landing.unwrap_or_default().iter().map(|s| (s.offset, s.bytes)).collect();
        Ok(RingWave { passes: 0, sent: 0, bytes: self.payload_bytes.unwrap_or(request.header.row_count as usize * self.row_bytes) })
    }
    fn poll(&mut self, wave: &mut RingWave, sink: &mut dyn FnMut(usize, u32, RingFrame) -> Result<()>)
        -> Result<Option<u8>> {
        wave.passes += 1;
        if let Some(limit) = self.fail_poll_after { ensure!(wave.passes <= limit, "injected receive failure"); }
        for rank in 0..self.ranks {
            if wave.sent & (1 << rank) != 0 || wave.passes <= rank + self.hold_polls { continue; }
            let (address, bytes) = self.free[rank].lock().unwrap().pop()
                .with_context(|| format!("rank {rank}'s receive ring ran dry"))?;
            wave.sent |= 1 << rank;
            ensure!(wave.bytes <= bytes, "reply larger than its receive");
            sink(rank, 0, RingFrame { address, bytes: wave.bytes, capacity: bytes, free: self.free[rank].clone() })?;
        }
        Ok((wave.sent.count_ones() as usize == self.ranks).then_some(0))
    }
    fn timeout(&self) -> Duration { Duration::from_secs(5) }
    fn abandon(&mut self) { self.abandoned.fetch_add(1, Ordering::SeqCst); }
    fn cancel(&mut self) { self.cancelled.fetch_add(1, Ordering::SeqCst); }
    fn retained_reply_slots(&self) -> Vec<usize> { vec![self.depth; self.ranks] }
}

/// The real [`Native`] backend over a [`Ring`], with host memory for the
/// mapped control and descriptor blocks.
fn ring_backend(ring: Ring, slots: usize) -> (Native<Ring>, Vec<u8>, Vec<u8>) {
    let mut control = vec![0u8; Counter::words(slots) * LINE + LINE];
    let mut pulls = vec![0u8; Pulls::bytes(slots) + PULL_STRIDE];
    let words = Words { base: (control.as_mut_ptr() as usize).next_multiple_of(LINE), slots };
    let pulls_at = Pulls { base: (pulls.as_mut_ptr() as usize).next_multiple_of(16) };
    let ranks = ring.ranks;
    let row_bytes = ring.row_bytes;
    (Native { transport: ring, reply_format: cuteafd_core::SparkReplyFormat::Bf16, reply_payloads: vec![0; slots],
        prefill_rows: 0, compressed_prefill_rows: 0, mixed_fallback_rows: 0, timing: false, arenas: Vec::new(), words, pulls: pulls_at, slots, ranks, row_bytes, landing: 0,
        placed: true, receiving: (0..ranks).map(|_| (0, Vec::new(), Vec::new())).collect(), retained: VecDeque::new() },
        control, pulls)
}

fn built_job(sequence: u32, slot: usize, rows: usize, uses: &[u32], fences: Vec<(usize, u32)>) -> Job {
    let mut request = ExpertProtocolV2Request::new_bytes(sequence.into(), 1, 0, 4, cuteafd_transport::ExpertV2Dtype::Bf16,
        Vec::new(), Vec::new(), bytes::Bytes::new()).unwrap();
    // The fake ring sizes each reply by the request's rows.
    request.header.row_count = rows as u32;
    Job { sequence, slot, rows, payload_bytes: rows * 16, source: Source::Built(request), consumed: fences, slice: (0, rows * 16),
        release: uses[slot] + 1, sent: Instant::now() }
}

#[test]
fn exchange_timing_posts_and_arrivals_are_opt_in() -> Result<()> {
    struct Writer(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    for timing in [false, true] {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let writer = bytes.clone();
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || Writer(writer.clone())).finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let (mut backend, _control, _pulls) = ring_backend(Ring::new(2, 1, 1, 16), 1);
        backend.timing = timing;
        let job = built_job(42, 0, 1, &[0], Vec::new());
        let info = WaveInfo { sequence: 42, slot: 0, rows: 1, payload_bytes: 16, slice: job.slice, release: 1 };
        let mut wave = backend.post(job)?;
        while !backend.poll(&mut wave, &info)? {}
        let output = String::from_utf8(bytes.lock().unwrap().clone())?;
        if timing {
            let posted = output.lines().find(|line| line.contains("spark exchange wave posted")).unwrap();
            for field in ["sequence=42", "slot=0", "request_id=42", "layer_id=0", "rows=1", "payload_bytes=16"] {
                assert!(posted.contains(field), "missing {field}: {posted}");
            }
            for rank in 0..2 {
                assert!(output.lines().any(|line| line.contains("spark exchange rank arrived")
                    && line.contains("sequence=42") && line.contains(&format!("rank={rank}"))), "{output}");
            }
        } else { assert!(output.is_empty(), "{output}"); }
    }
    Ok(())
}

/// #815: reply frames stay in their receive slots until their wave's reduce
/// is consumed, so the ring bounds the waves in flight. At the deepest
/// admission (4 slots, every one landed and unread while the engine queues
/// ahead), with decode-sized and prefill-sized waves interleaved on the one
/// link as a decode lane and prefill units share it, a ring of exactly 4
/// never blocks a post on a slot that only a later wave's consumed frees; the
/// arrivals and descriptors are published per rank, and every frame goes back
/// once consumed. The bound is tight: one receive fewer runs the ring dry.
#[test]
fn retained_frames_never_starve_the_ring_at_the_deepest_admission() {
    for slots in 1..=4usize {
        let (ranks, rows, row_bytes) = (4usize, 8usize, 16usize);
        let ring = Ring::new(ranks, slots, rows, row_bytes);
        check_reply_ring(&ring.retained_reply_slots(), 0b1111, slots).unwrap();
        let free = ring.free.clone();
        let (mut backend, _control, _pulls) = ring_backend(ring, slots);
        let world_words = backend.words;
        let (jobs, inbox) = mpsc::channel();
        let total = 40u32;
        std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                let result = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
                (result, backend)
            });
            // The engine: every slot's wave queued at once (as prefill units
            // queue ahead), each reduced only `slots` waves later.
            let mut uses = vec![0u32; slots];
            let mut unread = VecDeque::new();
            for sequence in 1..=total {
                let slot = (sequence as usize - 1) % slots;
                if unread.len() == slots {
                    let (old, old_slot): (u32, usize) = unread.pop_front().unwrap();
                    let started = Instant::now();
                    while !reached(world_words.load(Counter::Landed), old) {
                        if worker.is_finished() { panic!("worker ended early: {:?}", worker.join().unwrap().0 .0); }
                        assert!(started.elapsed() < Duration::from_secs(5), "wave {old} never landed ({slots} slots)");
                        std::thread::yield_now();
                    }
                    // Every rank arrived and published its frame before `landed`.
                    for rank in 0..ranks {
                        assert!(reached(world_words.load(Counter::Arrived(rank)), old), "rank {rank} of wave {old}");
                    }
                    uses[old_slot] += 1;
                    world_words.store(Counter::Consumed(old_slot), uses[old_slot]);
                }
                // Odd waves decode-sized (one row), even ones prefill-sized.
                let width = if sequence % 2 == 1 { 1 } else { rows };
                jobs.send(Command::Wave(built_job(sequence, slot, width, &uses, vec![(slot, uses[slot])]))).unwrap();
                unread.push_back((sequence, slot));
            }
            while let Some((old, old_slot)) = unread.pop_front() {
                let started = Instant::now();
                while !reached(world_words.load(Counter::Landed), old) {
                    assert!(started.elapsed() < Duration::from_secs(5), "wave {old} never landed ({slots} slots)");
                    std::thread::yield_now();
                }
                uses[old_slot] += 1;
                world_words.store(Counter::Consumed(old_slot), uses[old_slot]);
            }
            // Closing at rest returns the frames of every consumed reduce.
            drop(jobs);
            let (result, backend) = worker.join().unwrap();
            result.0.unwrap();
            assert_eq!(backend.transport.posted.load(Ordering::SeqCst), total as usize);
            assert!(backend.retained.is_empty() && free.iter().all(|rank| rank.lock().unwrap().len() == slots),
                "{slots} slots: every frame went back to its ring");
        });
    }
}

/// Negative control for the bound: with one receive fewer than the admitted
/// slots, the deepest admission (every slot landed, none reduced yet) leaves
/// the next post without a receive on every rank.
#[test]
fn one_receive_fewer_than_the_admitted_slots_runs_the_ring_dry() {
    let (ranks, rows, row_bytes, slots) = (4usize, 8usize, 16usize, 4usize);
    let ring = Ring::new(ranks, slots - 1, rows, row_bytes);
    let (mut backend, _control, _pulls) = ring_backend(ring, slots);
    let uses = vec![0u32; slots];
    let mut held = Vec::new();
    for sequence in 1..=slots as u32 {
        let slot = sequence as usize - 1;
        let job = built_job(sequence, slot, if sequence % 2 == 1 { 1 } else { rows }, &uses, vec![(slot, 0)]);
        let info = WaveInfo { sequence, slot, rows: job.rows, payload_bytes: job.payload_bytes, slice: job.slice, release: 1 };
        match backend.post(job) {
            Ok(mut wave) => {
                while backend.poll(&mut wave, &info).unwrap() == false {}
                held.push(sequence);
            }
            Err(error) => {
                assert_eq!(sequence as usize, slots, "the ring ran dry before the deepest wave: {error:#}");
                assert!(format!("{error:#}").contains("dry"), "{error:#}");
                return;
            }
        }
    }
    panic!("a ring of {} held {} unreduced waves", slots - 1, held.len());
}

/// A ring shallower than the admitted slots is refused at start: the
/// exchange would otherwise hold every receive with frames that only a later
/// wave's consumed frees.
#[test]
fn a_ring_shallower_than_the_admitted_waves_is_refused() {
    assert!(check_reply_ring(&[4, 4, 3, 4], 0b1111, 4).is_err());
    assert!(check_reply_ring(&[4, 4, 3, 4], 0b1011, 4).is_ok(), "a GPU-landing rank keeps no frames");
    assert!(check_reply_ring(&[0, 0], 0b11, 4).is_ok(), "copied frames keep no receive");
    check_reply_ring(&[8; 6], 0b111111, 4).unwrap();
    let error = check_reply_ring(&[2, 8], 0b11, 3).unwrap_err();
    assert!(format!("{error:#}").contains("rank 0 keeps 2 reply frames"), "{error:#}");
}

/// A rank's descriptor and arrival are published once its rows are all in,
/// before the wave lands; a pull reads the frames the descriptor names.
#[test]
fn each_pinned_rank_publishes_its_frames_before_the_wave_lands() {
    let (ranks, rows, row_bytes) = (3usize, 4usize, 16usize);
    let ring = Ring::new(ranks, 1, rows, row_bytes);
    let (mut backend, _control, _pulls) = ring_backend(ring, 1);
    let job = built_job(1, 0, rows, &[0], vec![(0, 0)]);
    let info = WaveInfo { sequence: 1, slot: 0, rows, payload_bytes: rows * row_bytes, slice: (0, rows * row_bytes), release: 1 };
    let mut wave = backend.post(job).unwrap();
    // Pass 1: rank 0 only.
    assert!(backend.poll(&mut wave, &info).unwrap() == false);
    assert!(reached(backend.words.load(Counter::Arrived(0)), 1) && !reached(backend.words.load(Counter::Arrived(1)), 1));
    // SAFETY: the test's descriptor block.
    let published = unsafe { *backend.pulls.at(0, 0) };
    assert_eq!(published.count, 1);
    assert_eq!((published.frame[0].offset, published.frame[0].bytes), (0, (rows * row_bytes) as u64));
    assert_eq!(unsafe { *backend.pulls.at(0, 1) }.count, 0, "rank 1 not published before it arrived");
    assert!(backend.poll(&mut wave, &info).unwrap() == false);
    assert!(backend.poll(&mut wave, &info).unwrap(), "the third pass completes the wave");
    assert_eq!(backend.retained.len(), 1);
    assert_eq!(backend.retained[0].frames.len(), ranks, "the frames stay until the reduce");
    assert!(!backend.release_consumed(), "not consumed yet");
    backend.words.store(Counter::Consumed(0), 1);
    assert!(backend.release_consumed() && backend.retained.is_empty());
}

/// #815 failure path: a receive failure mid-wave (one rank arrived and
/// published, the rest never will) clears every descriptor before it releases
/// the waits, so a pull that runs afterwards copies nothing, and the frames
/// already received are handed back with the transport (the link keeps them
/// until its streams drained) instead of being released under a running pull.
#[test]
fn a_failure_mid_wave_clears_the_pulls_and_hands_the_frames_back() {
    let (ranks, rows, row_bytes) = (3usize, 4usize, 16usize);
    let mut ring = Ring::new(ranks, 2, rows, row_bytes);
    ring.fail_poll_after = Some(2);
    let free = ring.free.clone();
    let (mut backend, _control, _pulls) = ring_backend(ring, 2);
    let words = backend.words;
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Command::Wave(built_job(1, 0, rows, &[0, 0], vec![(0, 0), (1, 0)]))).unwrap();
    jobs.send(Command::Wave(built_job(2, 1, rows, &[0, 0], vec![(1, 0)]))).unwrap();
    let (result, highest) = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
    assert!(format!("{:#}", result.unwrap_err()).contains("injected receive failure"));
    // Ranks 0 and 1 arrived (passes 1-2) and published; rank 2 never did.
    for rank in 0..2 { assert!(reached(words.load(Counter::Arrived(rank)), 1)); }
    // SAFETY: the test's descriptor block.
    assert!(unsafe { (*backend.pulls.at(0, 0)).count } > 0);
    drop(jobs);
    release_until_closed(&mut backend, &inbox, highest);
    for slot in 0..2 {
        for rank in 0..MAX_INTAKE_RANKS {
            // SAFETY: as above.
            assert_eq!(unsafe { (*backend.pulls.at(slot, rank)).count }, 0, "slot {slot} rank {rank} still pullable");
        }
    }
    for rank in 0..ranks { assert!(reached(words.load(Counter::Arrived(rank)), 2), "rank {rank}'s waits released"); }
    assert!(reached(words.load(Counter::Landed), 2));
    // The arrived frames are still held (not back in the ring) until handed over.
    assert_eq!(free[0].lock().unwrap().len(), 1, "rank 0's frame is held, not reposted under a pull");
    let frames = backend.frames();
    assert_eq!(frames.len(), 2, "ranks 0 and 1's frames of the failed wave");
    drop(frames);
    assert!(free.iter().all(|rank| rank.lock().unwrap().len() == 2));
}

/// GPU gate (#815): a pinned rank's pull is ordered by its own arrival wait,
/// and the reduce behind it by `landed`. Warm the exact calls first (eager
/// module loading: a lazily loaded kernel behind a pending wait stalls the
/// host), then queue a wide wave whose rank 1 arrives 300 ms after rank 0:
/// the stream's output stays pending until rank 1's arrival, and the reduce is
/// byte-exact. Negative control: the same pull queued without its arrival
/// wait (and before `landed`) runs at once and reads the receive slot before
/// the reply is in (the echo world parks stale bytes there), so its output is
/// wrong. Run with `CUDA_MODULE_LOADING=EAGER`.
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_pinned_pull_waits_for_its_rank_and_the_control_clobbers() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    ensure!(std::env::var("CUDA_MODULE_LOADING").as_deref() == Ok("EAGER"), "run with CUDA_MODULE_LOADING=EAGER");
    let path = library_path()?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&path)? });
    cuteafd_transport::set_verbs_host_native_library_path(std::path::PathBuf::from(&path))?;
    let (hidden, topk, ranks, rows) = (5120usize, 6usize, 2usize, 64usize);
    let row_bytes = hidden * 2;
    for control in [false, true] {
        let ranges_owned = (0..ranks).map(|_| DeviceAllocation::new(&library, rows * row_bytes))
            .collect::<Result<Vec<_>>>()?;
        let ranges: Vec<CuteafdDeviceBuffer> = ranges_owned.iter().map(|range| range.buffer).collect();
        let layout = RouteStaging::new(rows, topk, row_bytes)?;
        let arenas = vec![StagingArena::new(layout.bytes)?];
        let words = Arc::new(std::sync::OnceLock::new());
        let inputs_host: Vec<Vec<u8>> = (0..2).map(|layer| (0..rows * hidden).flat_map(|i| {
            let value = ((i * 7 + layer * 13) % 9) as f32 / 2.0 - 2.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        }).collect()).collect();
        // Layer 0 warms every call (no stagger needed); layer 1 is the probe:
        // rank 1 replies 300 ms after rank 0.
        let mut echo = EchoWorld::new(library.clone(), ranks, row_bytes, 0, ranges.iter().map(|r| r.ptr as usize).collect(),
            inputs_host.clone(), words.clone(), Duration::ZERO, Duration::ZERO, 1, (rows / 2) * row_bytes)?;
        // Stale bytes in every receive buffer: a pull before the reply reads them.
        for pool in &echo.buffers.pools {
            for &(host, bytes) in pool.lock().unwrap().iter() {
                // SAFETY: a free pooled pinned buffer.
                unsafe { std::ptr::write_bytes(host as *mut u8, 0x55, bytes) };
            }
        }
        let buffers = echo.buffers.clone();
        echo.stagger = Duration::ZERO;
        let geometry = crate::shared::wave_geometry::SlotGeometry::new(1, rows)?;
        let intake = Intake { prefill: true, ranks, row_bytes, ranges, geometry, landing: 0, placed: true };
        let stagger = Arc::new(AtomicUsize::new(0));
        let storage = std::rc::Rc::new(ExchangeStorage::new(&library, &intake)?);
        let mut exchange = Waves::start(&library, intake, StaggerSwitch { echo, stagger: stagger.clone() }, Vec::new(),
            arenas, false, storage.clone())?;
        let _ = words.set(exchange.words());
        let stream = crate::shared::spark_intake::WarmStream::new(library.as_ref())?;
        let ids = DeviceAllocation::new(&library, rows * topk * 4)?;
        let weights = DeviceAllocation::new(&library, rows * topk * 4)?;
        let inputs = (0..2).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
        let outputs = (0..2).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
        for (input, host) in inputs.iter().zip(&inputs_host) { library.copy_h2d(input.buffer, host)?; }
        let done = library.cuda_event_create()?;
        for layer in 0..2usize {
            if layer == 1 { stagger.store(300_000, Ordering::SeqCst); }
            let h = hidden;
            // SAFETY: ids, weights and the layer's input are live device buffers of the layout's extents.
            let ticket = unsafe { exchange.send(rows, rows * row_bytes, layout, [ids.buffer, weights.buffer, inputs[layer].buffer],
                stream.raw(), false, Box::new(move |routes, wire| ExpertProtocolV2Request::new_bytes(layer as u64 + 1, 1,
                    layer as u32, h as u32, cuteafd_transport::ExpertV2Dtype::Bf16, (0..rows as u32).map(|row|
                        cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                        source_kind: cuteafd_transport::ExpertV2SourceKind::Decode, source_request_id: 1,
                        token_position: row.into(), route_offset: row * topk as u32, route_count: topk as u32 })
                        .collect(), routes, wire)))? };
            let slot = ticket.slot();
            if control && layer == 1 {
                // The pull without its arrival wait and ahead of `landed`:
                // it runs as soon as the stream reaches it.
                let started = Instant::now();
                while !reached(exchange.words().load(Counter::Arrived(0)), ticket.sequence) {
                    ensure!(started.elapsed() < Duration::from_secs(5), "rank 0 never arrived");
                    std::thread::yield_now();
                }
                // SAFETY: the test's live planes and descriptors; the control
                // reads rank 1's receive slot before its reply is in.
                unsafe { exchange.pull(ticket.sequence, slot, rows, 0b11, false, stream.raw())? };
                exchange.outstanding.front_mut().unwrap().2 = 0b11;
            }
            exchange.land(ticket, stream.raw())?;
            // SAFETY: planes, output and the null shared plane are live.
            unsafe {
                exchange.reduce(slot, std::ptr::null(),
                    outputs[layer].buffer.ptr.cast(), rows, stream.raw())?;
                library.cuda_event_record(done, stream.raw())?;
            }
            if layer == 1 && !control {
                std::thread::sleep(Duration::from_millis(100));
                // SAFETY: a live event of this device.
                ensure!(!unsafe { library.cuda_event_query(done)? },
                    "the reduce ran before rank 1's reply arrived");
            }
            stream_sync(&library, stream.raw())?;
        }
        let expected: Vec<u8> = (0..rows * hidden).flat_map(|i| {
            let host = &inputs_host[1];
            let total: f32 = (0..ranks).map(|rank| {
                let word = rank_rows(&host[2 * i..2 * i + 2], rank);
                f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
            }).sum();
            ((total.to_bits() >> 16) as u16).to_le_bytes()
        }).collect();
        let mut actual = vec![0u8; rows * row_bytes];
        library.copy_d2h(&mut actual, outputs[1].buffer)?;
        if control {
            ensure!(actual != expected, "negative control: a pull without its arrival wait read the reply anyway");
        } else {
            ensure!(actual == expected, "the ordered pull's reduce differs from the reference");
        }
        ensure!(exchange.outstanding.is_empty() && exchange.unread.iter().all(|unread| !unread),
            "the exchange closes at rest: every wave landed and was consumed");
        let (_, frames, result) = exchange.close::<StaggerSwitch>()?;
        result?;
        drop(frames);
        stream.finish()?;
        // SAFETY: the event is idle (the stream drained).
        unsafe { library.cuda_event_destroy(done)? };
        drop(buffers);
        eprintln!("pinned pull ordering, control={control}: ok");
    }
    Ok(())
}

fn stream_sync(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    // SAFETY: the caller's live stream.
    unsafe { library.cuda_stream_synchronize(stream) }
}

/// The echo world with a stagger the test changes between waves (µs).
struct StaggerSwitch { echo: EchoWorld, stagger: Arc<AtomicUsize> }
impl Transport for StaggerSwitch {
    type Wave = EchoWave;
    type Frame = EchoFrame;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[cuteafd_transport::LandingSlice]>)
        -> Result<EchoWave> {
        self.echo.stagger = Duration::from_micros(self.stagger.load(Ordering::SeqCst) as u64);
        self.echo.dispatch(request, landing)
    }
    fn poll(&mut self, wave: &mut EchoWave, sink: &mut dyn FnMut(usize, u32, EchoFrame) -> Result<()>)
        -> Result<Option<u8>> { self.echo.poll(wave, sink) }
    fn timeout(&self) -> Duration { self.echo.timeout() }
    fn abandon(&mut self) {}
    fn retained_reply_slots(&self) -> Vec<usize> { self.echo.retained_reply_slots() }
}

/// A link's first `waves` sends wait for sequence 0 (landed at start), never
/// on a wrapped value; later sends wait for the slot's previous wave.
#[test]
fn a_slots_first_use_waits_for_nothing_beyond_the_initial_landing() {
    for waves in 1..=4usize {
        for sequence in 1..=waves as u32 { assert_eq!(super::staging_reuse(sequence, waves), 0, "{waves} {sequence}"); }
        assert_eq!(super::staging_reuse(waves as u32 + 1, waves), 1);
        assert_eq!(super::staging_reuse(100, waves), 100 - waves as u32);
    }
}

/// #784 reset after a cancelled request: the link stops its exchange with
/// waves still outstanding (their replies never consumed). The worker gives
/// them up as a cancel, not a failure: the process stays healthy (a failure
/// is terminal for every later request), the transport's waves are dropped
/// without the failure record, and every queued wait is released so no
/// stream waits on a wave that will never land. A real failure still records.
#[test]
fn a_reset_with_waves_outstanding_cancels_without_poisoning_health() {
    let (ranks, rows, row_bytes) = (2usize, 4usize, 16usize);
    let mut ring = Ring::new(ranks, 2, rows, row_bytes);
    // Replies stay out long enough for the cancel to find both waves in flight.
    ring.hold_polls = usize::MAX / 2;
    let (cancelled, abandoned) = (ring.cancelled.clone(), ring.abandoned.clone());
    let (mut backend, _control, _pulls) = ring_backend(ring, 2);
    let words = backend.words;
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Command::Wave(built_job(1, 0, rows, &[0, 0], vec![(0, 0)]))).unwrap();
    jobs.send(Command::Wave(built_job(2, 1, rows, &[0, 0], vec![(1, 0)]))).unwrap();
    jobs.send(Command::Cancel).unwrap();
    let healthy_before = cuteafd_transport::health::failure_reason().is_none();
    let (result, highest) = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
    assert_eq!(result.as_ref().ok(), Some(&Stop::Cancelled), "{result:?}");
    assert_eq!(highest, Some(2));
    drop(jobs);
    finish_worker(&mut backend, &inbox, result, highest).expect("a cancel is not a failure");
    assert_eq!((cancelled.load(Ordering::SeqCst), abandoned.load(Ordering::SeqCst)), (1, 0));
    if healthy_before {
        assert!(cuteafd_transport::health::failure_reason().is_none(), "a cancel marked the process unhealthy");
    }
    // Every wait over the given-up waves is released.
    assert!(reached(words.load(Counter::Landed), 2));
    for rank in 0..ranks { assert!(reached(words.load(Counter::Arrived(rank)), 2), "rank {rank}'s wait"); }

    // Control: closing (not cancelling) with waves outstanding is a failure,
    // abandons the waves and is reported.
    let mut ring = Ring::new(ranks, 2, rows, row_bytes);
    ring.hold_polls = usize::MAX / 2;
    let (cancelled, abandoned) = (ring.cancelled.clone(), ring.abandoned.clone());
    let (mut backend, _control, _pulls) = ring_backend(ring, 2);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Command::Wave(built_job(1, 0, rows, &[0, 0], vec![(0, 0)]))).unwrap();
    drop(jobs);
    let (result, highest) = run_worker(&mut backend, &inbox, Duration::from_secs(5), false);
    let error = finish_worker(&mut backend, &inbox, result, highest).unwrap_err();
    assert!(format!("{error:#}").contains("waves outstanding"), "{error:#}");
    assert_eq!((cancelled.load(Ordering::SeqCst), abandoned.load(Ordering::SeqCst)), (0, 1));
}

/// W48-12: a link whose ranks all land in device memory has no pull
/// descriptor block (base 0). A failed wave must still release its waiters
/// without touching descriptors (p10.4.2 wrote through a null pointer here:
/// SIGSEGV in `spark-exchange` when a parked prefill's client disconnected).
#[test]
fn fail_without_a_descriptor_block_writes_no_descriptor() {
    let slots = 128;
    let (mut native, _control, _pulls) = ring_backend(Ring::new(2, 2, 4, 16), slots);
    native.pulls = Pulls { base: 0 };
    native.landing = 0b11;
    Pulls { base: 0 }.clear(slots);
    Backend::fail(&mut native, 7);
    for rank in 0..2 {
        assert!(reached(native.words.load(Counter::Arrived(rank)), 7), "rank {rank} not released");
    }
}

/// A mixed-format reply uses its exact extent, not the padded worst-mix stride.
/// The same backend covers compact replies without changing their request bytes.
#[test]
fn wave_payload_is_independent_of_reserved_slot_stride() -> Result<()> {
    for (rows, row_bytes, payload_bytes) in [(16, 16, 16 * 16), (1700, 34864, 1700 * 18432)] {
        let ring = Ring::new(1, 1, 1, 16);
        let (mut backend, _control, _pulls) = ring_backend(ring, 1);
        backend.row_bytes = row_bytes;
        backend.transport.payload_bytes = Some(payload_bytes);
        // The fake frame points at a small host buffer; this CPU test never
        // reads its bytes, only the descriptor published by the real backend.
        backend.transport.free[0].lock().unwrap()[0].1 = payload_bytes;
        let mut job = built_job(1, 0, rows, &[0], vec![(0, 0)]);
        job.payload_bytes = payload_bytes;
        job.slice = (123456, 2752 * row_bytes);
        let info = WaveInfo { sequence: 1, slot: 0, rows, payload_bytes, slice: job.slice, release: 1 };
        let mut wave = backend.post(job)?;
        assert!(backend.poll(&mut wave, &info)?);
        // SAFETY: ring_backend owns this mapped descriptor for the test's life.
        let descriptor = unsafe { *backend.pulls.at(0, 0) };
        assert_eq!((descriptor.frame[0].offset, descriptor.frame[0].bytes), (0, payload_bytes as u64));
        assert!(reached(backend.words.load(Counter::Arrived(0)), 1));
        assert_eq!(backend.retained[0].frames[0].bytes, payload_bytes);
        assert!(validate_payload(payload_bytes, payload_bytes - 1).is_err());
    }
    assert!(validate_payload(0, 1234).is_err());
    Ok(())
}

#[test]
fn native_post_selects_reply_wire_and_preserves_bf16_slot_geometry() -> Result<()> {
    use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
    use cuteafd_transport::{ExpertProtocolV2RowDescriptor, ExpertV2Dtype,
        ExpertV2SourceKind::{Decode, MtpVerify, Prefill},
        EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32,
        EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3};
    let (ranks, slots, slot_rows) = (3usize, 2usize, 130usize);
    let cases = [
        (vec![(65, Prefill)], true),
        (vec![(64, Prefill)], false),
        (vec![(64, Prefill), (64, Prefill)], false),
        (vec![(65, Prefill), (65, Prefill)], true),
        (vec![(65, Prefill), (64, Prefill)], false),
        (vec![(65, Prefill), (1, Decode)], false),
        (vec![(65, Prefill), (1, MtpVerify)], false),
        (vec![(65, Decode)], false),
        (vec![(65, MtpVerify)], false),
    ];
    for (family, hidden) in [("v41", 5120usize), ("v4-flash", 4096), ("v4-pro", 7168),
        ("glm53", 6144), ("glm-flash", 4096), ("glm-flash-exlr8", 4096),
        ("mimo-flash", 4096), ("mimo-pro", 6144), ("kimi", 3584),
        ("qwen", 2560), ("step5", 4096), ("ml4", 6144)] {
    let row_bytes = hidden * 2;
    for selected in [Bf16, Fp8, Nvfp4] {
        for landing in [0, (1 << ranks) - 1] {
            let (mut backend, _control, _pulls) = ring_backend(Ring::new(ranks, slots, slot_rows, row_bytes), slots);
            backend.reply_format = selected;
            backend.landing = landing;
            for (index, (sources, eligible)) in cases.iter().enumerate() {
                let origins: Vec<_> = sources.iter().enumerate().flat_map(|(id, &(rows, kind))|
                    std::iter::repeat_n((id as u64 + 1, kind), rows)).collect();
                let rows = origins.len();
                let request = ExpertProtocolV2Request::new_bytes(index as u64 + 1, 1, 0, hidden as u32, ExpertV2Dtype::Bf16,
                    origins.iter().enumerate().map(|(row, &(source_request_id, source_kind))| ExpertProtocolV2RowDescriptor {
                        row_id: row as u64, source_kind, source_request_id, token_position: row as u64,
                        route_offset: row as u32, route_count: 1 }).collect(),
                    (0..rows as u32).map(|row| ExpertProtocolV2RouteEntry {
                        row_index: row, expert_id: 0, gate_weight: 1.0 }).collect(),
                    bytes::Bytes::from(vec![0; rows * row_bytes]))?;
                let before = request.encode()?;
                let actual = if *eligible { selected } else { Bf16 };
                let payload_bytes = rows * actual.row_bytes(hidden).unwrap();
                backend.transport.payload_bytes = Some(payload_bytes);
                let sequence = index as u32 + 1;
                let slot = index % slots;
                let slice = (slot * slot_rows * row_bytes, slot_rows * row_bytes);
                let mut job = built_job(sequence, slot, rows, &[0, 0], vec![(slot, 0)]);
                job.source = Source::Built(request);
                job.payload_bytes = rows * row_bytes;
                job.slice = slice;
                let mut wave = backend.post(job)?;
                let encoded = backend.transport.request_bytes.as_ref().unwrap();
                let posted = ExpertProtocolV2Request::decode(encoded)?;
                let (mode, flags) = match actual {
                    Bf16 => (0, 0),
                    Fp8 => (1, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32),
                    Nvfp4 => (2, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3),
                };
                assert_eq!(posted.header.flags, flags, "{family} {selected} {sources:?}");
                assert_eq!(backend.words.load(Counter::ReplyFormat(slot)), mode);
                assert_eq!(backend.payload_bytes(slot, rows * row_bytes), payload_bytes);
                if actual == Bf16 { assert_eq!(encoded, &before, "fallback/default preserves the request bytes"); }
                if landing != 0 {
                    assert_eq!(backend.transport.landing_slices, vec![(slice.0, payload_bytes); ranks]);
                } else {
                    assert!(backend.transport.landing_slices.is_empty());
                    let info = WaveInfo { sequence, slot, rows, payload_bytes, slice, release: sequence };
                    for _ in 0..ranks { backend.poll(&mut wave, &info)?; }
                    for rank in 0..ranks {
                        // SAFETY: ring_backend retains the mapped descriptor block until the backend drops.
                        let descriptor = unsafe { *backend.pulls.at(slot, rank) };
                        assert_eq!(descriptor.count, 1);
                        assert_eq!((descriptor.frame[0].offset, descriptor.frame[0].bytes), (0, payload_bytes as u64));
                    }
                    backend.words.store(Counter::Consumed(slot), sequence);
                    assert!(backend.release_consumed());
                }
            }
            assert_eq!(backend.compressed_prefill_rows,
                if selected == Bf16 { 0 } else { 195 }, "{family} {selected}");
        }
    }
    }
    Ok(())
}

#[test]
fn native_compressed_poll_rejects_wrong_frame_extent() -> Result<()> {
    use cuteafd_core::SparkReplyFormat::{Fp8, Nvfp4};
    for format in [Fp8, Nvfp4] {
        let (hidden, rows) = (32usize, 65usize);
        let expected = rows * format.row_bytes(hidden).unwrap();
        for actual in [expected - 1, expected + 1] {
            let mut ring = Ring::new(1, 1, rows, hidden * 2);
            ring.payload_bytes = Some(actual);
            let (mut backend, _control, _pulls) = ring_backend(ring, 1);
            let job = built_job(1, 0, rows, &[0], vec![(0, 0)]);
            let mut wave = backend.post(job)?;
            let info = WaveInfo { sequence: 1, slot: 0, rows, payload_bytes: expected,
                slice: (0, rows * hidden * 2), release: 1 };
            let error = backend.poll(&mut wave, &info).unwrap_err();
            assert!(format!("{error:#}").contains("variable reply plane must arrive whole"), "{error:#}");
            assert!(!reached(backend.words.load(Counter::Arrived(0)), 1));
        }
    }
    Ok(())
}

#[test]
fn compact_wave_keeps_wire_bytes_and_reserved_landing_offsets() -> Result<()> {
    for rows in [1, 16] {
        let request = ExpertProtocolV2Request::new_bytes(1, 1, 0, 8, cuteafd_transport::ExpertV2Dtype::Bf16,
            (0..rows as u32).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                row_id: row.into(), source_kind: cuteafd_transport::ExpertV2SourceKind::Decode,
                source_request_id: 1, token_position: row.into(), route_offset: row, route_count: 1 }).collect(),
            (0..rows as u32).map(|row| ExpertProtocolV2RouteEntry { row_index: row, expert_id: 0, gate_weight: 1.0 }).collect(),
            bytes::Bytes::from(vec![0; rows * 16]))?;
        let before = request.encode()?;
        let ring = Ring::new(1, 1, rows, 16);
        let (mut backend, _control, _pulls) = ring_backend(ring, 1);
        backend.landing = 1;
        let mut job = built_job(1, 0, rows, &[0], vec![(0, 0)]);
        job.source = Source::Built(request);
        job.slice = (8192, 4096);
        backend.post(job)?;
        assert_eq!(backend.transport.request_bytes.as_deref(), Some(before.as_slice()));
        assert_eq!(backend.transport.landing_slices, vec![(8192, rows * 16)]);
    }
    Ok(())
}

#[test]
fn staging_mode_parses_the_three_settings_and_refuses_others() {
    assert_eq!(StagingMode::parse(None).unwrap(), StagingMode::Auto);
    assert_eq!(StagingMode::parse(Some(" ")).unwrap(), StagingMode::Auto);
    assert_eq!(StagingMode::parse(Some("auto")).unwrap(), StagingMode::Auto);
    assert_eq!(StagingMode::parse(Some("sm")).unwrap(), StagingMode::Sm);
    assert_eq!(StagingMode::parse(Some("copy-engine")).unwrap(), StagingMode::CopyEngine);
    let error = StagingMode::parse(Some("ce")).unwrap_err().to_string();
    assert!(error.contains("CUTEAFD_SPARK_STAGING=ce"), "{error}");
}

#[test]
fn slow_sm_staging_needs_held_copy_engine_work_and_a_slow_largest_wave() {
    use cuteafd_ffi::peer_exchange::StagingProbe;
    // A 4090 D behind an EPYC root complex (measured): 64/128/256 KiB, SM vs copy engine.
    let mut slow_host = StagingProbe { copy_engine_held: 1, probes: 3, crossover_bytes: 32 << 10, ..Default::default() };
    for (i, (bytes, sm, ce)) in [(64u64 << 10, 33.1f32, 14.1f32), (128 << 10, 61.6, 14.1), (256 << 10, 118.5, 22.8)]
        .into_iter().enumerate() {
        slow_host.bytes[i] = bytes; slow_host.sm_us[i] = sm; slow_host.copy_engine_us[i] = ce;
    }
    assert!(slow_sm_staging(&slow_host));
    // SM keeps up at the largest wave (within 2x): prefill stays unpaced.
    let mut fast = slow_host;
    fast.sm_us[2] = 40.0;
    assert!(!slow_sm_staging(&fast));
    // Waits that don't hold copy-engine work, or no probe at all, never pace prefill.
    assert!(!slow_sm_staging(&StagingProbe { copy_engine_held: 0, ..slow_host }));
    assert!(!slow_sm_staging(&StagingProbe { probes: 0, ..slow_host }));
    let mut unmeasured = slow_host;
    unmeasured.copy_engine_us[2] = 0.0;
    assert!(!slow_sm_staging(&unmeasured));
}

#[test]
fn staging_route_uses_the_copy_engine_only_where_waits_hold_it() {
    use cuteafd_ffi::peer_exchange::StagingProbe;
    let held = StagingProbe { copy_engine_held: 1, probes: 9, crossover_bytes: 16 << 10, ..Default::default() };
    let never = StagingProbe { crossover_bytes: 0, ..held };
    let loose = StagingProbe { copy_engine_held: 0, probes: 0, crossover_bytes: 0, ..Default::default() };
    assert_eq!(staging_route(StagingMode::Auto, &held).unwrap(), StagingRoute::CopyEngineFrom(16 << 10));
    assert_eq!(staging_route(StagingMode::Auto, &never).unwrap(), StagingRoute::Sm);
    assert_eq!(staging_route(StagingMode::Auto, &loose).unwrap(), StagingRoute::Sm);
    assert_eq!(staging_route(StagingMode::Sm, &held).unwrap(), StagingRoute::Sm);
    assert_eq!(staging_route(StagingMode::CopyEngine, &never).unwrap(), StagingRoute::CopyEngineFrom(0));
    assert!(staging_route(StagingMode::CopyEngine, &loose).is_err());

    let route = StagingRoute::CopyEngineFrom(16 << 10);
    assert!(!route.copy_engine((16 << 10) - 1));
    assert!(route.copy_engine(16 << 10));
    assert!(StagingRoute::CopyEngineFrom(0).copy_engine(1));
    assert!(!StagingRoute::Sm.copy_engine(usize::MAX));
}

#[test]
fn copy_engine_staging_waits_until_every_earlier_wave_landed() {
    let route = StagingRoute::CopyEngineFrom(16 << 10);
    // Large enough and caught up: the copy engine stages.
    assert!(route.copy_engine_now(16 << 10, 9, 10));
    assert!(route.copy_engine_now(1 << 20, 12, 10));
    // An earlier wave still in flight: its landing may be queued ahead on the
    // stream, so SM copies stage instead of parking the copy engine.
    assert!(!route.copy_engine_now(1 << 20, 8, 10));
    // Below the crossover, or with the SM route, never the copy engine.
    assert!(!route.copy_engine_now((16 << 10) - 1, 9, 10));
    assert!(!StagingRoute::Sm.copy_engine_now(1 << 20, 9, 10));
    // The first wave has nothing earlier; sequences wrap.
    assert!(route.copy_engine_now(1 << 20, 0, 1));
    assert!(route.copy_engine_now(1 << 20, u32::MAX, 0));
    assert!(!route.copy_engine_now(1 << 20, u32::MAX - 1, 0));
}

/// Microseconds from the last reply frame in pinned host memory to the GPU
/// holding the whole wave in its planes (the host sees the completion), per
/// path, median of `iterations`: how a pinned reply wave of `ranks` frames
/// of `plane_bytes` reaches the reduce on this host.
#[derive(Debug, Clone)]
struct ReplyPathTimes {
    /// The exchange's host-ordered landing: the host queues the SM pull of
    /// the frames in place once they are in.
    pull_us: f64,
    /// The exchange's device-ordered landing: the pull queued behind a
    /// stream wait on a host word, which the host then stores.
    wait_pull_us: f64,
    /// A copy-engine upload straight from each frame, queued once it is in.
    upload_us: f64,
    /// `SparkIntake::receive`'s staged path: a CPU copy of the frames into
    /// pinned staging, then one upload per rank.
    staged_us: f64,
    /// The host-ordered SM pull on a chosen grid: (blocks, us).
    grids: Vec<(u32, f64)>,
    /// An SM copy of the frames with their addresses as kernel arguments
    /// (no device read of the descriptors), queued once they are in.
    regions_us: f64,
}

/// Pull grids the reply bench tries beside the capacity-sized default.
const PROBE_GRIDS: [u32; 6] = [4, 8, 16, 32, 64, 128];

fn reply_path_times(library: &NativeLibrary, ranks: usize, plane_bytes: usize, iterations: usize) -> Result<ReplyPathTimes> {
    use crate::shared::memory::DeviceAllocation;
    use cuteafd_ffi::peer_exchange::{PullEntry, PullFrames};
    ensure!((1..=MAX_INTAKE_RANKS).contains(&ranks) && plane_bytes % 16 == 0, "invalid reply probe shape");
    let mut frames = (0..ranks).map(|_| HostAllocation::new(library, plane_bytes)).collect::<Result<Vec<_>>>()?;
    for (rank, frame) in frames.iter_mut().enumerate() {
        for (index, byte) in frame.bytes_mut().iter_mut().enumerate() { *byte = (index * 7 + rank * 13) as u8; }
    }
    let planes = (0..ranks).map(|_| DeviceAllocation::new(library, plane_bytes)).collect::<Result<Vec<_>>>()?;
    let mut staging = HostAllocation::new(library, ranks * plane_bytes)?;
    let mut block = HostAllocation::new(library, ranks * std::mem::size_of::<PullFrames>() + LINE)?;
    ensure!(library.cuda_host_buffer_device_alias(block.buffer)?.ptr == block.buffer.ptr,
        "mapped pinned memory is not device-visible at its host address");
    let word = block.buffer.ptr.cast::<u8>().wrapping_add(ranks * std::mem::size_of::<PullFrames>()).cast::<u32>();
    let replies: Vec<*const PullFrames> = (0..ranks).map(|rank|
        block.buffer.ptr.cast::<PullFrames>().wrapping_add(rank).cast_const()).collect();
    {
        let descriptors = block.bytes_mut();
        for (rank, frame) in frames.iter().enumerate() {
            let mut reply = PullFrames { count: 1, ..Default::default() };
            reply.frame[0] = PullEntry { source: frame.buffer.ptr as u64, offset: 0, bytes: plane_bytes as u64 };
            let at = rank * std::mem::size_of::<PullFrames>();
            // SAFETY: inside the descriptor block; PullFrames is plain old data.
            unsafe { std::ptr::write_unaligned(descriptors[at..].as_mut_ptr().cast::<PullFrames>(), reply) };
        }
    }
    let targets: Vec<(*mut c_void, u64)> = planes.iter().map(|plane| (plane.buffer.ptr, plane_bytes as u64)).collect();
    let stream = crate::shared::spark_intake::WarmStream::new(library)?;
    let done = library.cuda_event_create()?;
    let spin = || -> Result<()> {
        // SAFETY: a live event recorded on the probe stream.
        while !unsafe { library.cuda_event_query(done)? } { std::hint::spin_loop(); }
        Ok(())
    };
    let check = |planes: &[DeviceAllocation<'_>]| -> Result<()> {
        for (plane, frame) in planes.iter().zip(&frames) {
            let mut landed = vec![0u8; plane_bytes];
            library.copy_d2h(&mut landed, plane.buffer)?;
            ensure!(landed == frame.bytes()[..plane_bytes], "a reply path landed other bytes");
            library.cuda_zero_bytes(plane.buffer, plane_bytes)?;
        }
        Ok(())
    };
    let median = |samples: &mut Vec<f64>| { samples.sort_by(f64::total_cmp); samples[samples.len() / 2] };
    let mut times: Vec<Vec<f64>> = (0..5 + PROBE_GRIDS.len()).map(|_| Vec::new()).collect();
    let regions: Vec<(*mut c_void, *const c_void, u64)> = planes.iter().zip(&frames)
        .map(|(plane, frame)| (plane.buffer.ptr, frame.buffer.ptr.cast_const(), plane_bytes as u64)).collect();
    for iteration in 0..iterations + 20 {
        for (path, samples) in times.iter_mut().enumerate() {
            let sequence = (iteration * 4 + path + 1) as u32;
            if path == 1 {
                // SAFETY: the word and descriptors live in the mapped block, the
                // planes on this device; the host stores the word below.
                unsafe {
                    library.cuda_stream_wait_value32(stream.raw(), word, sequence)?;
                    library.pull_frames(&replies, &targets, stream.raw())?;
                    library.cuda_event_record(done, stream.raw())?;
                }
            }
            let started = std::time::Instant::now();
            // SAFETY (every arm): frames, staging, descriptors and planes stay
            // allocated until the event completes, which each arm waits for.
            match path {
                0 => unsafe { library.pull_frames(&replies, &targets, stream.raw())?; },
                4 => unsafe { library.sm_copy_regions(&regions, stream.raw())?; },
                5.. => unsafe { library.pull_frames_blocks(&replies, &targets, PROBE_GRIDS[path - 5], stream.raw())?; },
                // SAFETY: an aligned word of the mapped block, read by the queued wait.
                1 => unsafe { std::ptr::write_volatile(word, sequence) },
                2 => for (plane, frame) in planes.iter().zip(&frames) {
                    unsafe { library.copy_host_buffer_h2d_async(plane.buffer, frame.buffer, plane_bytes, stream.raw())?; }
                },
                3 => {
                    let bytes = staging.bytes_mut();
                    for (rank, frame) in frames.iter().enumerate() {
                        crate::shared::spark_intake::copy_parallel(&mut bytes[rank * plane_bytes..][..plane_bytes],
                            &frame.bytes()[..plane_bytes]);
                    }
                    for (rank, plane) in planes.iter().enumerate() {
                        let source = cuteafd_ffi::CuteafdHostBuffer {
                            ptr: staging.buffer.ptr.cast::<u8>().wrapping_add(rank * plane_bytes).cast(),
                            bytes: plane_bytes, ..staging.buffer };
                        unsafe { library.copy_host_buffer_h2d_async(plane.buffer, source, plane_bytes, stream.raw())?; }
                    }
                }
            }
            if path != 1 { unsafe { library.cuda_event_record(done, stream.raw())?; } }
            spin()?;
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            if iteration >= 20 { samples.push(elapsed); }
            if iteration == 0 { check(&planes)?; }
        }
    }
    // SAFETY: the event is idle (every record above completed).
    unsafe { library.cuda_event_destroy(done)?; }
    stream.finish()?;
    let grids = PROBE_GRIDS.iter().zip(&mut times[5..]).map(|(&blocks, samples)| (blocks, median(samples))).collect();
    Ok(ReplyPathTimes { pull_us: median(&mut times[0]), wait_pull_us: median(&mut times[1]),
        upload_us: median(&mut times[2]), staged_us: median(&mut times[3]), regions_us: median(&mut times[4]), grids })
}

/// How a V4.1 reply wave (four ranks, 10 KiB a row) reaches the reduce on
/// this host at C1 (1 row), verify (6) and C16 (16) widths: the exchange's
/// SM pull of the frames in place (host- or device-ordered) against a
/// copy-engine upload from the frames and the CPU staging copy
/// (`SparkIntake::receive`, which no Spark family calls since the exchange).
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY); measures this host's reply paths"]
fn reply_wave_paths_on_this_host() -> Result<()> {
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = unsafe { NativeLibrary::load(&path)? };
    library.cuda_set_device(0)?;
    library.peer_exchange_initialize()?;
    eprintln!("| Wave (4 ranks) | SM pull, host-ordered | SM pull behind a stream wait | Copy-engine upload | \
        CPU staging + upload | SM copy, frames by value |");
    eprintln!("|---|---:|---:|---:|---:|---:|");
    let mut grids = Vec::new();
    for rows in [1usize, 6, 16, 64] {
        let times = reply_path_times(&library, 4, rows * 10240, 400)?;
        eprintln!("| {} KiB ({rows} rows) | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |", rows * 40, times.pull_us,
            times.wait_pull_us, times.upload_us, times.staged_us, times.regions_us);
        grids.push((rows, times.grids));
    }
    eprintln!("\nHost-ordered SM pull by grid (us):\n| Wave | {} |", PROBE_GRIDS.map(|b| format!("{b} blocks")).join(" | "));
    eprintln!("|---|{}", "---:|".repeat(PROBE_GRIDS.len()));
    for (rows, times) in grids {
        eprintln!("| {} KiB | {} |", rows * 40, times.iter().map(|(_, us)| format!("{us:.1}")).collect::<Vec<_>>().join(" | "));
    }
    Ok(())
}

#[test]
fn landing_route_decides_per_probed_frame_size() {
    let sample = |kib: usize, pull_us: f64, copy_engine_us: f64| LandingSample { frame_bytes: kib << 10, pull_us,
        copy_engine_us };
    // The copy engine's dip at 20 KiB frames keeps the pull there only.
    let probed = [sample(10, 16.2, 11.1), sample(20, 19.4, 27.0), sample(40, 23.8, 14.7), sample(160, 60.8, 42.6)];
    let route = landing_route(LandingMode::Auto, &probed);
    assert_eq!(route, LandingRoute::Probed(vec![(10 << 10, true), (20 << 10, false), (40 << 10, true), (160 << 10, true)]));
    for (frame, copy) in [(1, true), (10 << 10, true), ((10 << 10) + 4, false), (20 << 10, false), (30 << 10, true),
        (160 << 10, true), (40 << 20, true)] {
        assert_eq!(route.copy_engine(frame), copy, "frame of {frame} bytes");
    }
    // A frame past the largest probe takes the largest's decision.
    let slow = landing_route(LandingMode::Auto, &[sample(10, 9.0, 7.0), sample(80, 50.0, 60.0)]);
    assert!(slow.copy_engine(10 << 10) && !slow.copy_engine(1 << 30));
    assert_eq!(landing_route(LandingMode::Auto, &[]), LandingRoute::Pull);
    assert_eq!(landing_route(LandingMode::Sm, &probed), LandingRoute::Pull);
    assert!(!LandingRoute::Pull.copy_engine(1) && LandingRoute::CopyEngine.copy_engine(1 << 30));
    assert_eq!(landing_route(LandingMode::CopyEngine, &[]), LandingRoute::CopyEngine);
    // The pull is the default; the copy engine is opt-in.
    for (raw, mode) in [(None, LandingMode::Sm), (Some(""), LandingMode::Sm), (Some("auto"), LandingMode::Auto),
        (Some(" sm "), LandingMode::Sm), (Some("copy-engine"), LandingMode::CopyEngine)] {
        assert_eq!(LandingMode::parse(raw).unwrap(), mode);
    }
    assert!(LandingMode::parse(Some("pinned")).is_err());
    // The probed frames follow the link's rows: every width to six rows,
    // capped at the widest wave and at 1 MiB per rank.
    assert_eq!(landing_probe_frames(10240, 6), [1, 2, 3, 4, 6].map(|rows| rows * 10240));
    assert_eq!(landing_probe_frames(10240, 4096).last(), Some(&(96 * 10240)));
    assert_eq!(landing_probe_frames(1 << 21, 4), [1 << 20]);
}

/// The landing probe on this host: both routes' times at every probed size
/// and the route it picks (`CUTEAFD_SPARK_LANDING=auto`), for four and six
/// pinned ranks.
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY); measures this host's landing routes"]
fn landing_probe_on_this_host() -> Result<()> {
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = unsafe { NativeLibrary::load(&path)? };
    library.cuda_set_device(0)?;
    library.peer_exchange_initialize()?;
    for ranks in [4usize, 6] {
        let samples = probe_landing(&library, ranks, &landing_probe_frames(10240, 4096), 24)?;
        eprintln!("\n{ranks} pinned ranks of V4.1 rows (10 KiB):");
        eprintln!("| Rows | Wave | SM pull us | Copy engine us | Route |\n|---|---|---:|---:|---|");
        for sample in &samples {
            eprintln!("| {} | {} KiB | {:.1} | {:.1} | {} |", sample.frame_bytes / 10240, sample.frame_bytes * ranks >> 10,
                sample.pull_us, sample.copy_engine_us,
                if sample.copy_engine_us < sample.pull_us { "copy engine" } else { "SM pull" });
        }
    }
    Ok(())
}

/// The GPU expert tier's result joins a slot's BF16 reply reduction as the
/// plane after the ranks' (`ReplyReduction::launch_with`, which
/// `Waves::reduce_with` queues): ranks in order, then the tier's plane, then
/// the shared expert, summed in FP32 and rounded once, byte-identical to that
/// sum on the CPU; without the plane the binding reduces as `launch` does.
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_reply_reduction_adds_the_tier_plane_after_the_ranks() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = unsafe { NativeLibrary::load(&path)? };
    let hidden = 5120usize;
    let bf16 = |value: f32| { let bits = value.to_bits(); ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16 };
    let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut state = 0x5eed_7e1du64;
    let mut value = move || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        // Mixed magnitudes so FP32 order and the single rounding both matter.
        ((state >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * [1.0f32, 64.0, 1.0 / 64.0][(state % 3) as usize]
    };
    let stream = crate::shared::spark_intake::WarmStream::new(&library)?;
    let mode = HostAllocation::new(&library, LINE)?;
    for (ranks, rows) in [(4usize, 1usize), (4, 6), (6, 96), (6, 2048)] {
        let count = rows * hidden;
        // Rank planes, then the tier's plane, then the shared expert's.
        let host: Vec<Vec<u16>> = (0..ranks + 2).map(|_| (0..count).map(|_| bf16(value())).collect()).collect();
        let buffers = host.iter().map(|plane| -> Result<DeviceAllocation<'_>> {
            let allocation = DeviceAllocation::new(&library, count * 2)?;
            library.copy_h2d(allocation.buffer, &plane.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            Ok(allocation)
        }).collect::<Result<Vec<_>>>()?;
        let output = DeviceAllocation::new(&library, count * 2)?;
        let mut planes = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (pointer, buffer) in planes.iter_mut().zip(&buffers[..ranks]) { *pointer = buffer.buffer.ptr.cast_const().cast(); }
        let reduction = ReplyReduction { planes, ranks: ranks as u32, hidden, capacity: rows, fused: None,
            mode: mode.buffer.ptr.cast() };
        let (tier, shared) = (buffers[ranks].buffer.ptr.cast_const().cast::<u16>(), buffers[ranks + 1].buffer.ptr.cast_const().cast());
        for extra in [None, Some(tier)] {
            // SAFETY: every plane, the shared values and the output hold `rows`
            // rows on this device; the stream is synchronized before reading.
            unsafe {
                reduction.launch_with(&library, extra, shared, output.buffer.ptr.cast(), rows, stream.raw())?;
                library.cuda_stream_synchronize(stream.raw())?;
            }
            let mut raw = vec![0u8; count * 2];
            library.copy_d2h(&mut raw, output.buffer)?;
            let actual: Vec<u16> = raw.chunks_exact(2).map(|word| u16::from_le_bytes([word[0], word[1]])).collect();
            let expected: Vec<u16> = (0..count).map(|index| {
                let mut sum = widen(host[0][index]);
                for plane in &host[1..ranks] { sum += widen(plane[index]); }
                if extra.is_some() { sum += widen(host[ranks][index]); }
                bf16(sum + widen(host[ranks + 1][index]))
            }).collect();
            ensure!(actual == expected, "{ranks} ranks, {rows} rows, tier plane {}: {} of {count} values differ",
                extra.is_some(), actual.iter().zip(&expected).filter(|(a, b)| a != b).count());
        }
        eprintln!("reply reduction: {ranks} rank planes + the tier's, {rows} rows, byte-identical to the CPU sum");
    }
    // A full eight-rank wave has no plane left for the tier.
    let reduction = ReplyReduction { planes: [std::ptr::null(); MAX_INTAKE_RANKS], ranks: MAX_INTAKE_RANKS as u32,
        hidden, capacity: 1, fused: None, mode: mode.buffer.ptr.cast() };
    // SAFETY: refused before any launch.
    let refused = unsafe { reduction.launch_with(&library, Some(std::ptr::NonNull::<u16>::dangling().as_ptr()),
        std::ptr::null(), std::ptr::NonNull::<u16>::dangling().as_ptr(), 1, stream.raw()) };
    ensure!(refused.is_err(), "an eight-rank reduction accepted the tier's plane");
    stream.finish()
}

/// Compressed Spark replies and the GPU expert tier together: a slot's fused
/// reduction decodes each rank's FP8 or NVFP4 wire plane (or reads BF16, as
/// the slot's mode word says), rounds it to BF16, and sums ranks in order,
/// then the tier's BF16 plane (`expert_reply_reduce_tier`), then shared, in
/// FP32 rounded once: byte-identical to that decode and sum on the CPU, and
/// without the plane to `expert_reply_reduce`'s. Unlike the BF16 reducer's
/// eight planes, a full eight-rank wave still takes the tier's plane. The
/// wire bytes are random codes, so no quantizer is needed (sm_89 has none).
#[test]
#[ignore = "requires a GPU and the native library with codec exports (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_fused_reply_reduction_adds_the_tier_plane_after_the_ranks() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = unsafe { NativeLibrary::load(&path)? };
    let fused = std::rc::Rc::new(FusedReplyReduce::bind(&library)?);
    let hidden = 5120usize;
    let bf16 = |value: f32| { let bits = value.to_bits(); ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16 };
    let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let e4m3 = |code: u8| {
        let (exponent, mantissa) = (i32::from((code >> 3) & 15), f32::from(code & 7) / 8.0);
        let magnitude = if exponent == 0 { mantissa * 2f32.powi(-6) } else { (1.0 + mantissa) * 2f32.powi(exponent - 7) };
        if code & 0x80 != 0 { -magnitude } else { magnitude }
    };
    let fp4 = |code: u8| {
        let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
        if code & 8 != 0 { -magnitude } else { magnitude }
    };
    let mut state = 0x7e1d_f05eu64;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let stream = crate::shared::spark_intake::WarmStream::new(&library)?;
    let mode = HostAllocation::new(&library, LINE)?;
    for format in [Fp8, Nvfp4, Bf16] {
        for (ranks, rows) in [(4usize, 65usize), (6, 130), (8, 65)] {
            let count = rows * hidden;
            let row_bytes = format.row_bytes(hidden).context("reply row bytes")?;
            // Each rank's wire plane and its CPU decode, each value BF16-rounded.
            let (wires, decoded): (Vec<Vec<u8>>, Vec<Vec<f32>>) = (0..ranks).map(|_| {
                let mut wire = vec![0u8; rows * row_bytes];
                let mut values = vec![0f32; count];
                for row in 0..rows {
                    let line = &mut wire[row * row_bytes..(row + 1) * row_bytes];
                    match format {
                        Fp8 => {
                            // E4M3 codes without NaN; UE8M0 scales 2^-17..2^13.
                            for code in &mut line[..hidden] { *code = loop { let c = next() as u8; if c & 0x7f != 0x7f { break c; } }; }
                            for scale in &mut line[hidden..] { *scale = 110 + (next() % 31) as u8; }
                            for col in 0..hidden {
                                let scale = f32::from_bits(u32::from(line[hidden + col / 32]) << 23);
                                values[row * hidden + col] = widen(bf16(e4m3(line[col]) * scale));
                            }
                        }
                        Nvfp4 => {
                            // Packed E2M1 pairs (low nibble first); positive normal E4M3 scales.
                            for pair in &mut line[..hidden / 2] { *pair = next() as u8; }
                            for scale in &mut line[hidden / 2..] { *scale = ((1 + next() % 14) << 3 | next() % 8) as u8; }
                            for col in 0..hidden {
                                let code = (line[col / 2] >> (4 * (col % 2))) & 15;
                                values[row * hidden + col] = widen(bf16(fp4(code) * e4m3(line[hidden / 2 + col / 16])));
                            }
                        }
                        Bf16 => for col in 0..hidden {
                            let value = bf16(((next() >> 11) as f32 / (1u64 << 53) as f32 - 0.5)
                                * [1.0f32, 64.0, 1.0 / 64.0][(next() % 3) as usize]);
                            line[2 * col..2 * col + 2].copy_from_slice(&value.to_le_bytes());
                            values[row * hidden + col] = widen(value);
                        },
                    }
                }
                (wire, values)
            }).unzip();
            // The tier's plane and the shared expert's, BF16 at mixed magnitudes.
            let extras: Vec<Vec<u16>> = (0..2).map(|_| (0..count).map(|_| bf16(((next() >> 11) as f32
                / (1u64 << 53) as f32 - 0.5) * [1.0f32, 64.0, 1.0 / 64.0][(next() % 3) as usize])).collect()).collect();
            let upload = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
                let allocation = DeviceAllocation::new(&library, bytes.len())?;
                library.copy_h2d(allocation.buffer, bytes)?;
                Ok(allocation)
            };
            let planes_on_device = wires.iter().map(|wire| upload(wire)).collect::<Result<Vec<_>>>()?;
            let extras_on_device = extras.iter().map(|plane| upload(&plane.iter().flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?;
            let output = DeviceAllocation::new(&library, count * 2)?;
            let mut planes = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
            for (pointer, plane) in planes.iter_mut().zip(&planes_on_device) { *pointer = plane.buffer.ptr.cast_const().cast(); }
            // SAFETY: the mapped mode word is read by the launches below, after this write.
            unsafe { mode.buffer.ptr.cast::<u32>().write_volatile(match format { Bf16 => 0, Fp8 => 1, Nvfp4 => 2 }) };
            let reduction = ReplyReduction { planes, ranks: ranks as u32, hidden, capacity: rows,
                fused: Some(fused.clone()), mode: mode.buffer.ptr.cast() };
            let tier = extras_on_device[0].buffer.ptr.cast_const().cast::<u16>();
            for (extra, shared) in [(None, true), (Some(tier), true), (Some(tier), false)] {
                let shared_plane = if shared { extras_on_device[1].buffer.ptr.cast_const().cast() } else { std::ptr::null() };
                // SAFETY: every plane, the tier's and shared values and the
                // output hold `rows` rows on this device; synchronized before reading.
                unsafe {
                    reduction.launch_with(&library, extra, shared_plane, output.buffer.ptr.cast(), rows, stream.raw())?;
                    library.cuda_stream_synchronize(stream.raw())?;
                }
                let mut raw = vec![0u8; count * 2];
                library.copy_d2h(&mut raw, output.buffer)?;
                let actual: Vec<u16> = raw.chunks_exact(2).map(|word| u16::from_le_bytes([word[0], word[1]])).collect();
                let expected: Vec<u16> = (0..count).map(|index| {
                    let mut sum = decoded[0][index];
                    for plane in &decoded[1..] { sum += plane[index]; }
                    if extra.is_some() { sum += widen(extras[0][index]); }
                    if shared { sum += widen(extras[1][index]); }
                    bf16(sum)
                }).collect();
                ensure!(actual == expected, "{format} {ranks} ranks, {rows} rows, tier plane {}, shared {shared}: {} of \
                    {count} values differ", extra.is_some(), actual.iter().zip(&expected).filter(|(a, b)| a != b).count());
            }
            eprintln!("fused reply reduction: {format}, {ranks} rank planes + the tier's, {rows} rows, byte-identical to the CPU decode and sum");
        }
    }
    stream.finish()
}

/// Host-ordered collects through the real exchange (as `finish_values` runs
/// them): the host waits for each wave, queuing arrived ranks of wide waves
/// meanwhile (`pull_arrived`), then lands, reduces and consumes. Landed by
/// copy engine and by SM pull alike, every layer's reduce equals the host
/// reference byte for byte, at decode, verify and rank-by-rank widths, with
/// GPU-landing ranks mixed in.
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY)"]
fn gpu_host_ordered_collects_land_byte_exact_by_either_route() -> Result<()> {
    use crate::shared::memory::DeviceAllocation;
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&path)? });
    cuteafd_transport::set_verbs_host_native_library_path(std::path::PathBuf::from(&path))?;
    let (hidden, topk) = (5120usize, 6usize);
    let row_bytes = hidden * 2;
    // (ranks, landing ranks, slots, rows per wave, layers, per-rank stagger us).
    for route in [LandingRoute::CopyEngine, LandingRoute::Pull] {
        for (ranks, landing, slots, rows, layers, stagger_us) in [(4usize, 0u8, 1usize, 1usize, 24u32, 0u64),
            (4, 0, 2, 6, 24, 0), (4, 0, 1, 16, 24, 50), (4, 0, 2, 128, 12, 200), (6, 0b100001, 1, 6, 24, 0)] {
            let reserve = slots * rows;
            let ranges_owned = (0..ranks).map(|_| DeviceAllocation::new(&library, reserve * row_bytes))
                .collect::<Result<Vec<_>>>()?;
            let ranges: Vec<CuteafdDeviceBuffer> = ranges_owned.iter().map(|range| range.buffer).collect();
            let layout = RouteStaging::new(rows, topk, row_bytes)?;
            let arenas = (0..slots).map(|_| StagingArena::new(layout.bytes)).collect::<Result<Vec<_>>>()?;
            let words = Arc::new(std::sync::OnceLock::new());
            let total = layers as usize;
            let hosts: Vec<Vec<u8>> = (0..total).map(|layer| (0..rows * hidden).flat_map(|i| {
                // Multiples of 0.5 in [-2, 2]: the rank sums are exact in BF16.
                let value = ((i * 7 + layer * 13) % 9) as f32 / 2.0 - 2.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect()).collect();
            let echo = EchoWorld::new(library.clone(), ranks, row_bytes, landing,
                ranges.iter().map(|range| range.ptr as usize).collect(), hosts.clone(), words.clone(),
                Duration::ZERO, Duration::from_micros(stagger_us), slots, rows.div_ceil(2) * row_bytes)?;
            let geometry = crate::shared::wave_geometry::SlotGeometry::new(slots, rows)?;
            let intake = Intake { prefill: true, ranks, row_bytes, ranges, geometry, landing, placed: true };
            let storage = std::rc::Rc::new(ExchangeStorage::new(&library, &intake)?);
            let mut exchange = Waves::start(&library, intake, echo, Vec::new(), arenas, false, storage)?;
            let _ = words.set(exchange.words());
            exchange.set_landing_route(route.clone());
            let stream = crate::shared::spark_intake::WarmStream::new(library.as_ref())?;
            let ids = DeviceAllocation::new(&library, rows * topk * 4)?;
            let weights = DeviceAllocation::new(&library, rows * topk * 4)?;
            let inputs = (0..total).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
            let outputs = (0..total).map(|_| DeviceAllocation::new(&library, rows * row_bytes)).collect::<Result<Vec<_>>>()?;
            for (layer, input) in inputs.iter().enumerate() {
                library.copy_h2d(input.buffer, &hosts[layer])?;
                let (h, l) = (hidden, layer);
                // SAFETY: ids, weights and the layer's input are live device buffers of the layout's extents.
                let ticket = unsafe { exchange.send(rows, rows * row_bytes, layout, [ids.buffer, weights.buffer, input.buffer],
                    stream.raw(), false, Box::new(move |routes, wire| ExpertProtocolV2Request::new_bytes(l as u64 + 1, 1, l as u32,
                        h as u32, cuteafd_transport::ExpertV2Dtype::Bf16, (0..rows as u32).map(|row|
                            cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                            source_kind: cuteafd_transport::ExpertV2SourceKind::Decode, source_request_id: 1,
                            token_position: row.into(), route_offset: row * topk as u32, route_count: topk as u32 })
                            .collect(), routes, wire)))? };
                if layer == 0 {
                    assert_eq!(exchange.copies(&ticket, 1, stream.raw())?, route != LandingRoute::Pull);
                }
                // The host-ordered collect: wait for the wave, queuing arrived ranks.
                let started = Instant::now();
                while !exchange.landed(&ticket) {
                    healthy()?;
                    exchange.pull_arrived(&ticket, stream.raw())?;
                    ensure!(started.elapsed() < Duration::from_secs(20), "wave {layer} never landed");
                    std::hint::spin_loop();
                }
                let slot = ticket.slot();
                exchange.land(ticket, stream.raw())?;
                // SAFETY: planes and output are live; no shared plane.
                unsafe { exchange.reduction(slot).launch(&library, std::ptr::null(),
                    outputs[layer].buffer.ptr.cast(), rows, stream.raw())?; }
                exchange.consumed(slot, stream.raw())?;
            }
            stream.finish()?;
            healthy()?;
            for (layer, output) in outputs.iter().enumerate() {
                let expected: Vec<u8> = (0..rows * hidden).flat_map(|i| {
                    let total: f32 = (0..ranks).map(|rank| {
                        let word = rank_rows(&hosts[layer][2 * i..2 * i + 2], rank);
                        f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16)
                    }).sum();
                    ((total.to_bits() >> 16) as u16).to_le_bytes()
                }).collect();
                let mut actual = vec![0u8; rows * row_bytes];
                library.copy_d2h(&mut actual, output.buffer)?;
                ensure!(actual == expected, "{route:?} ranks={ranks} landing={landing:#b} rows={rows} layer {layer} differs");
            }
            ensure!(exchange.outstanding.is_empty() && exchange.unread.iter().all(|unread| !unread),
                "the exchange closes at rest: every wave landed and was consumed");
            let (_, frames, result) = exchange.close::<EchoWorld>()?;
            result?;
            ensure!(frames.is_empty(), "a closed exchange whose waves were all consumed holds no frames");
            eprintln!("host-ordered {route:?}: ranks={ranks} landing={landing:#b} slots={slots} rows={rows}: \
                {total} layers byte-exact");
        }
    }
    Ok(())
}

/// Microseconds from the host storing a wave's landed word to the GPU
/// holding `ranks` pinned frames of `frame_bytes` each, with the landing
/// queued ahead behind a stream wait on that word (a device-ordered pass):
/// `per_rank` launches one copy per rank (a wide wave's rank-by-rank
/// landing), else one for all ranks. Median of `iterations`.
fn device_ordered_landing_us(library: &NativeLibrary, ranks: usize, frame_bytes: usize, per_rank: bool,
    route: &dyn Fn(&[*const cuteafd_ffi::peer_exchange::PullFrames], &[(*mut c_void, u64)],
        &[(*mut c_void, *const c_void, u64)], *mut c_void) -> Result<()>, iterations: usize) -> Result<f64> {
    use crate::shared::memory::DeviceAllocation;
    use cuteafd_ffi::peer_exchange::{PullEntry, PullFrames};
    let mut frames = (0..ranks).map(|_| HostAllocation::new(library, frame_bytes)).collect::<Result<Vec<_>>>()?;
    for (rank, frame) in frames.iter_mut().enumerate() {
        for (index, byte) in frame.bytes_mut().iter_mut().enumerate().step_by(4096) { *byte = (index + rank) as u8; }
    }
    let planes = (0..ranks).map(|_| DeviceAllocation::new(library, frame_bytes)).collect::<Result<Vec<_>>>()?;
    let mut block = HostAllocation::new(library, ranks * std::mem::size_of::<PullFrames>() + LINE)?;
    let word = block.buffer.ptr.cast::<u8>().wrapping_add(ranks * std::mem::size_of::<PullFrames>()).cast::<u32>();
    let replies: Vec<*const PullFrames> = (0..ranks).map(|rank|
        block.buffer.ptr.cast::<PullFrames>().wrapping_add(rank).cast_const()).collect();
    {
        let descriptors = block.bytes_mut();
        for (rank, frame) in frames.iter().enumerate() {
            let mut reply = PullFrames { count: 1, ..Default::default() };
            reply.frame[0] = PullEntry { source: frame.buffer.ptr as u64, offset: 0, bytes: frame_bytes as u64 };
            // SAFETY: inside the descriptor block; PullFrames is plain old data.
            unsafe { std::ptr::write_unaligned(descriptors[rank * std::mem::size_of::<PullFrames>()..].as_mut_ptr()
                .cast::<PullFrames>(), reply) };
        }
    }
    let targets: Vec<(*mut c_void, u64)> = planes.iter().map(|plane| (plane.buffer.ptr, frame_bytes as u64)).collect();
    let regions: Vec<(*mut c_void, *const c_void, u64)> = planes.iter().zip(&frames)
        .map(|(plane, frame)| (plane.buffer.ptr, frame.buffer.ptr.cast_const(), frame_bytes as u64)).collect();
    let stream = crate::shared::spark_intake::WarmStream::new(library)?;
    let done = library.cuda_event_create()?;
    let mut samples = Vec::with_capacity(iterations);
    for iteration in 0..iterations + 3 {
        let sequence = iteration as u32 + 1;
        // SAFETY: the word and descriptors live in the mapped block, frames and
        // planes until the event completes, which each iteration waits for.
        unsafe {
            library.cuda_stream_wait_value32(stream.raw(), word, sequence)?;
            if per_rank {
                for rank in 0..ranks {
                    route(&replies[rank..rank + 1], &targets[rank..rank + 1], &regions[rank..rank + 1], stream.raw())?;
                }
            } else {
                route(&replies, &targets, &regions, stream.raw())?;
            }
            library.cuda_event_record(done, stream.raw())?;
        }
        std::thread::sleep(Duration::from_micros(200));
        let started = Instant::now();
        // SAFETY: an aligned word of the mapped block, read by the queued wait.
        unsafe { std::ptr::write_volatile(word, sequence) };
        // SAFETY: a live event recorded above.
        while !unsafe { library.cuda_event_query(done)? } { std::hint::spin_loop(); }
        if iteration >= 3 { samples.push(started.elapsed().as_secs_f64() * 1e6); }
        if iteration == 0 {
            for (plane, frame) in planes.iter().zip(&frames) {
                let mut landed = vec![0u8; frame_bytes];
                library.copy_d2h(&mut landed, plane.buffer)?;
                ensure!(landed == frame.bytes()[..frame_bytes], "a device-ordered landing route landed other bytes");
                library.cuda_zero_bytes(plane.buffer, frame_bytes)?;
            }
        }
    }
    // SAFETY: idle; every record completed.
    unsafe { library.cuda_event_destroy(done)?; }
    stream.finish()?;
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

/// Device-ordered landings on this host: the SM pull (each block reads its
/// rank's descriptor once) against an SM copy of the frames with their
/// addresses as kernel arguments (no descriptor reads: the ceiling), both
/// queued behind the wait a device-ordered pass queues. Decode and verify
/// waves land every rank in one launch; a prefill unit's wide wave lands rank
/// by rank (one launch each).
#[test]
#[ignore = "requires a GPU and the native library (CUTEAFD_PLANE_UPLOAD_LIBRARY); measures this host's device-ordered landings"]
fn device_ordered_landing_on_this_host() -> Result<()> {
    let path = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY").context("set CUTEAFD_PLANE_UPLOAD_LIBRARY")?;
    // SAFETY: the opt-in path names the built native library for this GPU.
    let library = unsafe { NativeLibrary::load(&path)? };
    library.cuda_set_device(0)?;
    library.peer_exchange_initialize()?;
    let pull = |replies: &[*const cuteafd_ffi::peer_exchange::PullFrames], targets: &[(*mut c_void, u64)],
        _: &[(*mut c_void, *const c_void, u64)], stream: *mut c_void| unsafe { library.pull_frames(replies, targets, stream) };
    let by_value = |_: &[*const cuteafd_ffi::peer_exchange::PullFrames], _: &[(*mut c_void, u64)],
        regions: &[(*mut c_void, *const c_void, u64)], stream: *mut c_void| -> Result<()> {
        for chunk in regions.chunks(4) { unsafe { library.sm_copy_regions(chunk, stream)?; } }
        Ok(())
    };
    eprintln!("| Wave | Landing | SM pull us | By-value SM copy us |\n|---|---|---:|---:|");
    for rows in [1usize, 6, 8, 24] {
        let bytes = rows * 10240;
        let (a, b) = (device_ordered_landing_us(&library, 4, bytes, false, &pull, 100)?,
            device_ordered_landing_us(&library, 4, bytes, false, &by_value, 100)?);
        eprintln!("| {} KiB ({rows} rows x 4 ranks) | one launch | {a:.1} | {b:.1} |", bytes * 4 >> 10);
    }
    for rows in [256usize, 2752] {
        let bytes = rows * 10240;
        let (a, b) = (device_ordered_landing_us(&library, 4, bytes, true, &pull, 12)?,
            device_ordered_landing_us(&library, 4, bytes, true, &by_value, 12)?);
        eprintln!("| {:.1} MiB ({rows} rows x 4 ranks) | rank by rank | {a:.1} ({:.1} GB/s) | {b:.1} ({:.1} GB/s) |",
            (bytes * 4) as f64 / (1 << 20) as f64, (bytes * 4) as f64 / a / 1e3, (bytes * 4) as f64 / b / 1e3);
    }
    Ok(())
}

#[test]
fn bf16_units_under_compressed_settings_dispatch_zero_reply_decode_launches() {
    use cuteafd_transport::ExpertV2SourceKind::{Decode, MtpVerify, Prefill};
    for available in [false, true] {
        for (rows, kind) in [(1, Decode), (96, Decode), (192, MtpVerify), (64, Prefill)] {
            assert!(!compressed_reduction(single_reply_eligible(rows, kind), available));
        }
        assert_eq!(compressed_reduction(single_reply_eligible(65, Prefill), available), available);
        for rows in [vec![(1, true); 64].into_iter().chain(vec![(2, true); 65]).collect::<Vec<_>>(),
            vec![(1, true); 65].into_iter().chain(vec![(2, false); 96]).collect()] {
            assert!(!compressed_reduction(combined_reply_eligible(rows), available));
        }
    }
}
