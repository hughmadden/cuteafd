//! One process-wide owner of miss reads; no caller memory crosses this thread.
use super::{invalid, uring, MappedTableError, ReadPriority, Result};
use io_uring::{opcode, types, IoUring};
use std::{collections::VecDeque, fs::File, os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::{Arc, Condvar, Mutex, OnceLock, atomic::{AtomicBool, Ordering}, mpsc},
    time::{Duration, Instant}};

const SLOTS: usize = 16;
const DECODE_SLOTS: usize = 2;
pub(super) const BUFFER_BYTES: usize = 1 << 20;
const WAKE: u64 = u64::MAX;

#[derive(Clone)]
pub(super) struct Read { pub file: Arc<File>, pub offset: u64 }
#[derive(Default)]
struct Buffer { bytes: Vec<u8>, reads: Vec<Read> }
struct Reply { buffer: Buffer, result: Result<uring::ReadStats> }
struct Request { buffer: Buffer, width: usize, priority: ReadPriority, reply: mpsc::SyncSender<Reply> }
#[derive(Default)]
struct Queue {
    decode: VecDeque<Request>, prefill: VecDeque<Request>, buffers: Vec<Buffer>,
    borrowed: usize, prefill_borrowed: usize, decode_waiters: usize,
}
struct Shared { queue: Mutex<Queue>, available: Condvar, wake: OwnedFd, ready: AtomicBool }
pub(super) struct Reactor { shared: Arc<Shared> }
static REACTOR: OnceLock<Mutex<Option<Arc<Reactor>>>> = OnceLock::new();

fn error(context: &str, source: std::io::Error) -> MappedTableError {
    MappedTableError::Io { context: context.into(), source }
}
fn signal(fd: &OwnedFd) -> Result<()> {
    let value = 1u64;
    loop {
        // SAFETY: eventfd consumes exactly one initialized u64; fd stays owned.
        let result = unsafe { libc::write(fd.as_raw_fd(), (&value as *const u64).cast(), 8) };
        if result == 8 { return Ok(()); }
        let source = std::io::Error::last_os_error();
        if source.kind() == std::io::ErrorKind::WouldBlock { return Ok(()); }
        if source.kind() != std::io::ErrorKind::Interrupted { return Err(error("waking table read reactor", source)); }
    }
}
fn clear_signal(fd: &OwnedFd) {
    let mut value = 0u64;
    loop {
        // SAFETY: eventfd writes one u64 into live, sized storage.
        let result = unsafe { libc::read(fd.as_raw_fd(), (&mut value as *mut u64).cast(), 8) };
        if result == 8 { return; }
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted { return; }
    }
}

pub(super) fn global() -> Result<Arc<Reactor>> {
    let mut global = REACTOR.get_or_init(|| Mutex::new(None)).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(reactor) = global.as_ref() { return Ok(Arc::clone(reactor)); }
    // SAFETY: eventfd returns a fresh descriptor, with no borrowed pointers.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 { return Err(error("creating table reactor eventfd", std::io::Error::last_os_error())); }
    // SAFETY: fd is freshly allocated and transferred exactly once.
    let wake = unsafe { OwnedFd::from_raw_fd(fd) };
    let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), available: Condvar::new(), wake, ready: AtomicBool::new(false) });
    let owner = Arc::clone(&shared);
    let (started, ready) = mpsc::sync_channel(1);
    std::thread::Builder::new().name("table-read-reactor".into()).spawn(move || run(owner, started))
        .map_err(|source| error("starting table read reactor", source))?;
    ready.recv().map_err(|_| invalid("table read reactor exited during startup"))?;
    let reactor = Arc::new(Reactor { shared });
    *global = Some(Arc::clone(&reactor));
    Ok(reactor)
}
impl Reactor {
    pub(super) fn ready(&self) -> bool { self.shared.ready.load(Ordering::Acquire) }

    pub(super) fn read(&self, reads: &[uring::Read<'_>], output: &mut [u8], width: usize,
        priority: ReadPriority) -> Result<uring::ReadStats> {
        if width == 0 || width > BUFFER_BYTES { return Err(invalid("table reactor row exceeds buffer budget")); }
        let decode = priority == ReadPriority::Decode;
        let mut stats = uring::ReadStats::default();
        let mut failure = None;
        // Stream arbitrarily large gathers with a two-buffer window. Returning
        // one buffer never drains the entire submission pipeline.
        let mut chunks = reads.chunks(uring::MAX_BATCH.min(BUFFER_BYTES / width)).peekable();
        let mut pending = VecDeque::new();
        while chunks.peek().is_some() || !pending.is_empty() {
          'fill: while chunks.peek().is_some() && pending.len() < 2 {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            if decode { queue.decode_waiters += 1; }
            while queue.borrowed == SLOTS || (!decode &&
                (queue.prefill_borrowed == SLOTS - DECODE_SLOTS || queue.decode_waiters > 0)) {
                if !pending.is_empty() {
                    if decode { queue.decode_waiters -= 1; }
                    break 'fill;
                }
                queue = self.shared.available.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
            if decode { queue.decode_waiters -= 1; } else { queue.prefill_borrowed += 1; }
            queue.borrowed += 1;
            let mut buffer = queue.buffers.pop().unwrap_or_default();
            drop(queue);
            let chunk = chunks.next().expect("pending chunk");
            if buffer.bytes.is_empty() { buffer.bytes.resize(BUFFER_BYTES, 0); }
            buffer.reads.clear();
            buffer.reads.extend(chunk.iter().map(|read| Read { file: Arc::clone(&read.part.file),
                offset: read.part.file_offset + read.row * width as u64 }));
            let (send, receive) = mpsc::sync_channel(1);
            let request = Request { buffer, width, priority, reply: send };
            {
                let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
                if decode { queue.decode.push_back(request); } else { queue.prefill.push_back(request); }
            }
            // A nonblocking eventfd is latched even when the owner is between
            // inspecting queues and arming PollAdd. No polling or lost wakeup.
            if let Err(error) = signal(&self.shared.wake) {
                tracing::error!(%error, "cannot wake table reactor; refusing an unbounded wait");
                std::process::abort();
            }
            pending.push_back((chunk, receive));
          }
            let (chunk, receive) = pending.pop_front().expect("submitted read batch");
            let reply = receive.recv().map_err(|_| invalid("table read reactor exited"))?;
            let result = reply.result;
            if result.is_ok() {
                for (read, bytes) in chunk.iter().zip(reply.buffer.bytes.chunks_exact(width)) {
                    output[read.slot * width..(read.slot + 1) * width].copy_from_slice(bytes);
                }
            }
            let mut queue = self.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            let mut buffer = reply.buffer;
            buffer.reads.clear();
            queue.buffers.push(buffer);
            queue.borrowed -= 1;
            if !decode { queue.prefill_borrowed -= 1; }
            self.shared.available.notify_all();
            drop(queue);
            let batch = match result {
                Ok(batch) => batch,
                Err(error) => { failure.get_or_insert(error); chunks.by_ref().for_each(drop); continue; }
            };
            stats.miss_ns += batch.miss_ns;
            stats.miss_max_ns = stats.miss_max_ns.max(batch.miss_max_ns);
            for (total, value) in stats.miss_histogram.iter_mut().zip(batch.miss_histogram) { *total += value; }
        }
        failure.map_or(Ok(stats), Err)
    }
}

struct Job { request: Request, next: usize, left: usize, started: Instant,
    error: Option<MappedTableError>, stats: uring::ReadStats }
#[derive(Clone, Copy)]
struct Active { job: usize, row: usize, done: usize }
struct Engine {
    ring: IoUring, jobs: Vec<Option<Job>>, active: Vec<Option<Active>>,
    retry: VecDeque<Active>, free: Vec<usize>, outstanding: usize, wake_armed: bool, decode_streak: usize,
}
impl Engine {
    fn new(ring: IoUring) -> Self {
        let capacity = (ring.params().sq_entries() as usize - 1).min(uring::worker_limit().expect("validated ring worker limit") as usize);
        Self { ring, jobs: (0..SLOTS).map(|_| None).collect(), active: vec![None; capacity],
            retry: VecDeque::new(), free: (0..capacity).rev().collect(), outstanding: 0, wake_armed: false, decode_streak: 0 }
    }
    fn arm(&mut self, fd: &OwnedFd) -> Result<()> {
        if !self.wake_armed {
            let entry = opcode::PollAdd::new(types::Fd(fd.as_raw_fd()), libc::POLLIN as _).build().user_data(WAKE);
            // SAFETY: PollAdd holds no memory pointer; Shared owns the fd beyond this engine.
            unsafe { self.ring.submission().push(&entry) }.map_err(|_| invalid("reactor wake queue full"))?;
            self.wake_armed = true;
        }
        Ok(())
    }
    fn accept(&mut self, shared: &Shared) {
        let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        while let Some(index) = self.jobs.iter().position(Option::is_none) {
            let Some(request) = queue.decode.pop_front().or_else(|| queue.prefill.pop_front()) else { break; };
            let left = request.buffer.reads.len();
            self.jobs[index] = Some(Job { request, next: 0, left, started: Instant::now(), error: None, stats: Default::default() });
        }
    }
    fn enqueue(&mut self, active: Active) -> Result<()> {
        let index = self.free.pop().ok_or_else(|| invalid("reactor capacity exceeded"))?;
        let job = self.jobs[active.job].as_mut().expect("live read job");
        let read = &job.request.buffer.reads[active.row];
        let width = job.request.width;
        // SAFETY: only this owner touches the buffer, it never resizes while in
        // flight, and each active row is disjoint. Engine::drop drains all SQEs
        // before any job buffer or Arc<File> can be released, including unwind.
        let pointer = unsafe { job.request.buffer.bytes.as_mut_ptr().add(active.row * width + active.done) };
        let entry = opcode::Read::new(types::Fd(read.file.as_raw_fd()), pointer, (width - active.done) as u32)
            .offset(read.offset + active.done as u64).build().flags(io_uring::squeue::Flags::ASYNC).user_data(index as u64);
        // SAFETY: the owned job and the draining engine retain fd and destination.
        unsafe { self.ring.submission().push(&entry) }.map_err(|_| invalid("reactor read queue full"))?;
        self.active[index] = Some(active);
        self.outstanding += 1;
        Ok(())
    }
    fn schedule(&mut self) -> Result<()> {
        let reserve = (self.active.len() / 8).clamp(1, 64);
        while self.outstanding < self.active.len() {
            let decode = self.jobs.iter().position(|job| job.as_ref().is_some_and(|job|
                job.request.priority == ReadPriority::Decode && job.next < job.request.buffer.reads.len()));
            let prefill = self.jobs.iter().position(|job| job.as_ref().is_some_and(|job|
                job.request.priority == ReadPriority::Prefill && job.next < job.request.buffer.reads.len()));
            // Bound starvation under a continuous decode stream, without using
            // decode's reserved admission capacity for prefill.
            let prefill_allowed = self.outstanding < self.active.len() - reserve;
            let selected = if self.decode_streak >= 256 && prefill_allowed { prefill.or(decode) }
                else { decode.or(if prefill_allowed { prefill } else { None }) };
            if let Some(id) = selected {
                let job = self.jobs[id].as_mut().expect("selected job");
                let row = job.next;
                job.next += 1;
                if job.request.priority == ReadPriority::Decode { self.decode_streak += 1; } else { self.decode_streak = 0; }
                self.enqueue(Active { job: id, row, done: 0 })?;
            } else if let Some(at) = self.retry.iter().position(|active| prefill_allowed ||
                self.jobs[active.job].as_ref().is_some_and(|job| job.request.priority == ReadPriority::Decode)) {
                let active = self.retry.remove(at).expect("selected retry");
                self.enqueue(active)?;
            } else { break; }
        }
        Ok(())
    }
    fn complete(&mut self, shared: &Shared) -> Result<()> {
        while let Some(cqe) = self.ring.completion().next() {
            if cqe.user_data() == WAKE {
                self.wake_armed = false;
                clear_signal(&shared.wake);
                continue;
            }
            let id = usize::try_from(cqe.user_data()).map_err(|_| invalid("invalid reactor completion"))?;
            let mut active = self.active.get_mut(id).and_then(Option::take).ok_or_else(|| invalid("unknown reactor completion"))?;
            self.outstanding -= 1;
            self.free.push(id);
            let job = self.jobs[active.job].as_mut().expect("completion job retained");
            let count = cqe.result();
            if count > 0 && count as usize <= job.request.width - active.done {
                active.done += count as usize;
                if active.done < job.request.width { self.retry.push_back(active); continue; }
                let elapsed = job.started.elapsed().as_nanos() as u64;
                job.stats.miss_ns += elapsed;
                job.stats.miss_max_ns = job.stats.miss_max_ns.max(elapsed);
                job.stats.miss_histogram[uring::tier(elapsed)] += 1;
            } else {
                job.error.get_or_insert_with(|| error("table reactor read", if count < 0 {
                    std::io::Error::from_raw_os_error(-count)
                } else { std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short immutable table") }));
            }
            job.left -= 1;
            if job.left == 0 {
                let job = self.jobs[active.job].take().expect("completed job retained");
                let result = job.error.map_or(Ok(job.stats), Err);
                // A cancelled caller may have dropped the receiver. All writes
                // are complete before its owned buffer is dropped here.
                let _ = job.request.reply.send(Reply { buffer: job.request.buffer, result });
            }
        }
        Ok(())
    }
    fn wait(&mut self) -> Result<()> {
        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => return Ok(()),
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {},
                Err(source) => return Err(error("waiting for table reactor", source)),
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        while self.outstanding > 0 {
            while let Some(cqe) = self.ring.completion().next() {
                if cqe.user_data() != WAKE { self.outstanding -= 1; }
            }
            if self.outstanding > 0 && self.wait().is_err() {
                tracing::error!("cannot drain owned table read buffers; refusing unsafe teardown");
                std::process::abort();
            }
        }
        for job in self.jobs.iter_mut().filter_map(Option::take) {
            let _ = job.request.reply.send(Reply { buffer: job.request.buffer, result: Err(invalid("table reactor batch failed; retry via mmap")) });
        }
    }
}
fn reject(shared: &Shared) {
    let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
    for request in queue.decode.drain(..).collect::<Vec<_>>().into_iter().chain(queue.prefill.drain(..)) {
        let _ = request.reply.send(Reply { buffer: request.buffer, result: Err(invalid("table reactor temporarily unavailable; retry via mmap")) });
    }
}
fn run(shared: Arc<Shared>, started: mpsc::SyncSender<()>) {
    let mut engine = match uring::new_ring() {
        Ok(ring) => { shared.ready.store(true, Ordering::Release); Some(Engine::new(ring)) },
        Err(error) => { tracing::warn!(%error, "table reactor setup failed; batches can retry"); None },
    };
    let _ = started.send(());
    let mut retry_after = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(owner) = engine.as_mut() {
            let step = owner.complete(&shared).and_then(|()| {
                owner.accept(&shared);
                owner.arm(&shared.wake)?;
                owner.schedule()?;
                owner.wait()
            });
            if let Err(error) = step {
                tracing::warn!(%error, "table reactor batch failed; draining before retry");
                shared.ready.store(false, Ordering::Release);
                drop(engine.take());
                reject(&shared);
                retry_after = Instant::now() + Duration::from_secs(1);
            }
        } else {
            // Setup can fail under transient memory pressure. Sleep in the
            // eventfd poll, then retry at most once per second of demand.
            let mut poll = libc::pollfd { fd: shared.wake.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: poll borrows one initialized descriptor for this call.
            let result = unsafe { libc::poll(&mut poll, 1, -1) };
            if result < 0 { continue; }
            clear_signal(&shared.wake);
            if Instant::now() >= retry_after {
                retry_after = Instant::now() + Duration::from_secs(1);
                match uring::new_ring() {
                    Ok(ring) => { engine = Some(Engine::new(ring)); shared.ready.store(true, Ordering::Release); },
                    Err(error) => tracing::warn!(%error, "table reactor setup retry failed"),
                }
            }
            if engine.is_none() { reject(&shared); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn concurrent_callers_share_one_owner_and_stream_beyond_pool_volume() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut file = tempfile::NamedTempFile::new()?;
        let payload: Vec<_> = (0..4096).map(|i| (i % 251) as u8).collect();
        file.write_all(&payload)?;
        // SAFETY: the test keeps its immutable file alive until all callers join.
        let part = Arc::new(unsafe { super::super::MappedRows::open(file.path(), 0, 16, 256)? });
        let owner = global()?;
        if !owner.ready() { return Ok(()); }
        let mut threads = Vec::new();
        for caller in 0..4 {
            let part = Arc::clone(&part);
            let owner = Arc::clone(&owner);
            let expected = payload.clone();
            threads.push(std::thread::spawn(move || -> Result<()> {
                assert!(Arc::ptr_eq(&owner, &global()?));
                // One gather exceeds the entire 16 MiB pool; it must stream.
                let count = if caller == 0 { 70_000 } else { 257 };
                let reads: Vec<_> = (0..count).map(|slot| uring::Read { part: &part, row: (slot % 16) as u64, slot }).collect();
                let mut output = vec![0; count * 256];
                let priority = if caller == 0 { ReadPriority::Prefill } else { ReadPriority::Decode };
                owner.read(&reads, &mut output, 256, priority)?;
                for (slot, row) in output.chunks_exact(256).enumerate() {
                    assert_eq!(row, &expected[(slot % 16) * 256..(slot % 16 + 1) * 256]);
                }
                Ok(())
            }));
        }
        for thread in threads { thread.join().expect("read caller panic")?; }
        let queue = owner.shared.queue.lock().unwrap();
        assert_eq!((queue.borrowed, queue.prefill_borrowed, queue.decode_waiters), (0, 0, 0));
        assert!(queue.buffers.iter().map(|buffer| buffer.bytes.len()).sum::<usize>() <= SLOTS * BUFFER_BYTES);
        Ok(())
    }

    #[test]
    fn scheduling_reserves_decode_capacity_and_completes_abandoned_replies() -> Result<()> {
        // This isolated engine uses no global owner and is dropped after draining.
        let ring = match uring::new_ring() { Ok(ring) => ring, Err(_) => return Ok(()) };
        let mut engine = Engine::new(ring);
        let file = Arc::new(File::open("/dev/zero").map_err(|source| error("opening test data", source))?);
        let request = |priority, count| {
            let (reply, receiver) = mpsc::sync_channel(1);
            drop(receiver); // caller cancellation must never free in-flight storage
            let buffer = Buffer { bytes: vec![0; count], reads: (0..count).map(|_| Read { file: Arc::clone(&file), offset: 0 }).collect() };
            Job { request: Request { buffer, width: 1, priority, reply }, next: 0, left: count,
                started: Instant::now(), error: None, stats: Default::default() }
        };
        let limit = engine.active.len();
        let reserve = (limit / 8).clamp(1, 64);
        engine.jobs[0] = Some(request(ReadPriority::Prefill, limit * 2));
        engine.schedule()?;
        assert_eq!(engine.outstanding, limit - reserve);
        engine.jobs[1] = Some(request(ReadPriority::Decode, reserve));
        engine.schedule()?;
        assert_eq!(engine.outstanding, limit);
        assert_eq!(engine.jobs[1].as_ref().unwrap().next, reserve);
        // Drop drains even though no receiver exists. This also tests shutdown
        // with buffered reads still queued rather than submitted.
        drop(engine);
        Ok(())
    }
}
