//! Bounded cold-path RGB8/BF16 TCP channel; no CUDA work runs on the scheduler.
use super::{EncodeJob as NativeJob, EncoderService};
use cuteafd_core::ImageKey;
use cuteafd_engine::media::{EncodeJob, EncodeOutput, EncoderClient, EncoderTicket, MediaError};
use cuteafd_loader::media::EncoderId;
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const MAGIC: &[u8; 8] = b"CAFDVI01";
const MAX_PATCHES: u32 = 16_384;
const MAX_WIDTH: u32 = 16_384;
const MAX_ERROR: usize = 1024;
type Result<T> = std::result::Result<T, MediaError>;
fn error(value: impl std::fmt::Display) -> MediaError { MediaError::Encoder(value.to_string()) }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderHandshake {
    pub encoder_id: EncoderId,
    pub max_patches: u32,
    pub output_width: u32,
    pub patch_size: u32,
    pub merge_size: u32,
    /// The placement contract, shared by coordinator and every replica.
    pub plan_hash: [u8; 32],
}
impl EncoderHandshake {
    fn validate(&self) -> Result<()> {
        if self.max_patches == 0 || self.max_patches > MAX_PATCHES
            || self.output_width == 0 || self.output_width > MAX_WIDTH
            || ![14, 16].contains(&self.patch_size) || self.merge_size != 2 {
            return Err(error("invalid encoder capacity/geometry"));
        }
        Ok(())
    }
    fn compatible(&self, expected: &Self) -> Result<()> {
        self.validate()?;
        expected.validate()?;
        if self.encoder_id != expected.encoder_id || self.plan_hash != expected.plan_hash
            || self.output_width != expected.output_width || self.patch_size != expected.patch_size
            || self.merge_size != expected.merge_size || self.max_patches < expected.max_patches {
            return Err(error("vision encoder identity/placement/capacity mismatch"));
        }
        Ok(())
    }
    fn write(&self, stream: &mut Wire) -> Result<()> {
        stream.write_all(MAGIC).map_err(error)?;
        stream.write_all(&self.encoder_id.0).map_err(error)?;
        stream.write_all(&self.plan_hash).map_err(error)?;
        for value in [self.max_patches, self.output_width, self.patch_size, self.merge_size] { put_u32(stream, value)?; }
        Ok(())
    }
    fn read(stream: &mut Wire) -> Result<Self> {
        let magic = read_array::<8>(stream)?;
        if &magic != MAGIC { return Err(error("unsupported vision wire version")); }
        let encoder_id = EncoderId(read_array(stream)?);
        let plan_hash = read_array(stream)?;
        let value = Self { encoder_id, plan_hash, max_patches: get_u32(stream)?, output_width: get_u32(stream)?, patch_size: get_u32(stream)?, merge_size: get_u32(stream)? };
        value.validate()?;
        Ok(value)
    }
    fn validate_job(&self, job: &EncodeJob) -> Result<usize> {
        let ([t, h, w], rgb8) = job.image_input()?;
        let patches = u64::from(h) * u64::from(w);
        let rgb_bytes = patches * u64::from(self.patch_size).pow(2) * 3;
        if t != 1 || h == 0 || w == 0 || h % self.merge_size != 0 || w % self.merge_size != 0
            || patches > u64::from(self.max_patches) || job.tokens as u64 != patches / u64::from(self.merge_size).pow(2)
            || job.hidden_width != self.output_width as usize || rgb8.len() as u64 != rgb_bytes {
            return Err(error("invalid image grid/RGB8/output geometry"));
        }
        job.feature_bytes()
    }
}
fn read_array<const N: usize>(s: &mut Wire) -> Result<[u8; N]> {
    let mut bytes = [0; N]; s.read_exact(&mut bytes).map_err(error)?; Ok(bytes)
}
fn get_u32(s: &mut Wire) -> Result<u32> { Ok(u32::from_le_bytes(read_array(s)?)) }
fn get_u64(s: &mut Wire) -> Result<u64> { Ok(u64::from_le_bytes(read_array(s)?)) }
fn put_u32(s: &mut Wire, n: u32) -> Result<()> { s.write_all(&n.to_le_bytes()).map_err(error) }
fn put_u64(s: &mut Wire, n: u64) -> Result<()> { s.write_all(&n.to_le_bytes()).map_err(error) }
// Each handshake/frame has one absolute deadline, even when bytes trickle in.
struct Wire {
    socket: TcpStream,
    timeout: Duration,
    deadline: Instant,
}
impl Wire {
    fn new(socket: TcpStream, timeout: Duration) -> Result<Self> {
        configure(&socket, timeout)?;
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| error("vision timeout too large"))?;
        Ok(Self { socket, timeout, deadline })
    }
    fn frame(&mut self) { self.deadline = Instant::now() + self.timeout; }
    fn remaining(&self) -> std::io::Result<Duration> {
        self.deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "vision frame deadline"))
    }
}
impl Read for Wire {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.socket.set_read_timeout(Some(self.remaining()?))?;
        self.socket.read(bytes)
    }
}
impl Write for Wire {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.socket.set_write_timeout(Some(self.remaining()?))?;
        self.socket.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.socket.flush() }
}
fn configure(s: &TcpStream, timeout: Duration) -> Result<()> {
    if timeout.is_zero() { return Err(error("vision timeout must be positive")); }
    s.set_nodelay(true).map_err(error)?;
    s.set_read_timeout(Some(timeout)).map_err(error)?;
    s.set_write_timeout(Some(timeout)).map_err(error)
}
struct Work {
    job: EncodeJob,
    reply: mpsc::SyncSender<Result<EncodeOutput>>,
    cancelled: Arc<AtomicBool>,
}
struct Replica {
    queue: Option<mpsc::SyncSender<Work>>,
    socket: TcpStream,
    failed: Arc<AtomicBool>,
    owner: Option<JoinHandle<()>>,
}
struct OwnerHealth(Arc<AtomicBool>);
impl Drop for OwnerHealth {
    fn drop(&mut self) { self.0.store(false, Ordering::Release); }
}
struct Pending {
    result: mpsc::Receiver<Result<EncodeOutput>>,
    cancelled: Arc<AtomicBool>,
}
/// A persistent connection and bounded owner queue per replica. The constructor
/// is a readiness barrier: all EncoderIds are verified before API vision is enabled.
pub struct RemoteEncoder {
    healthy: Arc<AtomicBool>,
    handshake: EncoderHandshake,
    replicas: Vec<Replica>,
    pending: HashMap<EncoderTicket, Pending>,
    next: u64,
    round_robin: usize,
}
impl RemoteEncoder {
    pub fn connect(addresses: Vec<SocketAddr>, expected: EncoderHandshake, timeout: Duration) -> Result<Self> {
        expected.validate()?;
        if addresses.is_empty() || addresses.len() > 6 { return Err(error("vision needs 1..6 replicas")); }
        let mut this = Self { healthy: Arc::new(AtomicBool::new(true)), handshake: expected.clone(), replicas: vec![], pending: HashMap::new(), next: 0, round_robin: 0 };
        for address in addresses {
            let mut stream = Wire::new(TcpStream::connect_timeout(&address, timeout).map_err(error)?, timeout)?;
            EncoderHandshake::read(&mut stream)?.compatible(&expected)?;
            expected.write(&mut stream)?;
            if get_u32(&mut stream)? != 0 { return Err(error("vision encoder rejected handshake")); }
            let socket = stream.socket.try_clone().map_err(error)?;
            let failed = Arc::new(AtomicBool::new(false));
            let failure = failed.clone();
            let health = this.healthy.clone();
            let (queue, jobs) = mpsc::sync_channel::<Work>(128);
            let owner = thread::Builder::new().name("remote-vision-owner".into()).spawn(move || {
                // Any owner exit (including panic) permanently closes vision admission.
                let _health = OwnerHealth(health);
                loop {
                    let work = match jobs.recv_timeout(Duration::from_secs(2)) {
                        Ok(work) => work,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if !failure.load(Ordering::Acquire) && ping(&mut stream).is_err() { failure.store(true, Ordering::Release); _health.0.store(false, Ordering::Release); }
                            continue;
                        }
                    };
                    if work.cancelled.load(Ordering::Acquire) { continue; }
                    let result = if failure.load(Ordering::Acquire) { Err(error("vision encoder unavailable")) }
                        else { exchange(&mut stream, &work.job) };
                    if result.is_err() { failure.store(true, Ordering::Release); _health.0.store(false, Ordering::Release); }
                    if !work.cancelled.load(Ordering::Acquire) { let _ = work.reply.send(result); }
                }
            }).map_err(error)?;
            this.replicas.push(Replica { queue: Some(queue), socket, failed, owner: Some(owner) });
        }
        Ok(this)
    }
    /// Image requests fail closed after a wire error; text needs no encoder.
    pub fn healthy(&self) -> bool { self.healthy.load(Ordering::Acquire) }
    /// Live health without a scheduler lock, including while it sleeps.
    pub fn health_handle(&self) -> Arc<AtomicBool> { self.healthy.clone() }
    pub fn ready(&self) -> bool { !self.replicas.is_empty() && self.healthy() }
}
impl EncoderClient for RemoteEncoder {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket> {
        if !self.healthy() { return Err(error("vision encoder unavailable")); }
        self.handshake.validate_job(&job)?;
        let replica = self.round_robin % self.replicas.len();
        self.round_robin = self.round_robin.wrapping_add(1);
        let replica = &self.replicas[replica];
        if replica.failed.load(Ordering::Acquire) { return Err(error("vision encoder unavailable")); }
        let ticket = EncoderTicket(self.next);
        self.next = self.next.checked_add(1).ok_or_else(|| error("vision ticket exhausted"))?;
        let (reply, result) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        replica.queue.as_ref().ok_or_else(|| error("vision encoder stopped"))?
            .try_send(Work { job, reply, cancelled: cancelled.clone() }).map_err(|_| error("vision encoder queue unavailable"))?;
        self.pending.insert(ticket, Pending { result, cancelled });
        Ok(ticket)
    }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput>> {
        let result = match self.pending.get(&ticket)?.result.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err(error("vision encoder unavailable")),
        };
        self.pending.remove(&ticket);
        Some(result)
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        if let Some(pending) = self.pending.remove(&ticket) { pending.cancelled.store(true, Ordering::Release); }
    }
}
impl Drop for RemoteEncoder {
    fn drop(&mut self) {
        for pending in self.pending.values() { pending.cancelled.store(true, Ordering::Release); }
        for r in &mut self.replicas { r.queue.take(); let _ = r.socket.shutdown(Shutdown::Both); }
        for r in &mut self.replicas { if let Some(owner) = r.owner.take() { let _ = owner.join(); } }
    }
}
fn ping(s: &mut Wire) -> Result<()> {
    s.frame();
    put_u32(s, 0)?;
    if get_u32(s)? != 0 { return Err(error("vision heartbeat failed")); }
    Ok(())
}
fn exchange(s: &mut Wire, job: &EncodeJob) -> Result<EncodeOutput> {
    let (grid, rgb8) = job.image_input()?;
    s.frame();
    put_u32(s, 1)?;
    s.write_all(job.key.bytes()).map_err(error)?;
    for n in grid { put_u32(s, n)?; }
    put_u64(s, job.tokens as u64)?;
    put_u64(s, rgb8.len() as u64)?;
    s.write_all(rgb8).map_err(error)?;
    let status = get_u32(s)?;
    let key: cuteafd_core::MediaKey = ImageKey(read_array(s)?).into();
    let elapsed_ms = get_u64(s)? as f64 / 1_000_000.0;
    let len = get_u64(s)?;
    if key != job.key { return Err(error("vision reply key mismatch")); }
    let expected = job.feature_bytes()?;
    if status == 0 && len != expected as u64 || status != 0 && len > MAX_ERROR as u64 {
        return Err(error("vision reply length mismatch"));
    }
    let mut bytes = vec![0; len as usize];
    s.read_exact(&mut bytes).map_err(error)?;
    if status != 0 { return Err(error(String::from_utf8_lossy(&bytes))); }
    Ok(EncodeOutput { key, features: bytes.into(), elapsed_ms })
}

/// Network owner is separate from the resident CUDA owner; same process/context.
/// Shutdown interrupts sockets, joins the network thread, then drains CUDA work.
pub struct EncoderServer {
    stop: Arc<AtomicBool>,
    socket: Arc<Mutex<Option<TcpStream>>>,
    owner: Option<JoinHandle<()>>,
    pub address: SocketAddr,
}
impl EncoderServer {
    pub fn start(address: SocketAddr, handshake: EncoderHandshake, service: EncoderService, lut: Arc<[f32; 768]>, timeout: Duration) -> Result<Self> {
        handshake.validate()?;
        let service = Arc::new(service);
        let health = service.clone();
        Self::start_backend(address, handshake, timeout, move || health.healthy(), move |job| {
            let started = Instant::now();
            let (grid, rgb8) = job.image_input()?;
            let ticket = service.submit(NativeJob { rgb: rgb8.clone(), grid: [grid[1] as usize, grid[2] as usize], lut: lut.clone(), output: vec![0; job.tokens * job.hidden_width] }).map_err(error)?;
            // The owner retains all buffers until its stream has drained, even
            // when a peer disappears. A deadline closes TCP, not CUDA lifetimes.
            let output = loop {
                match ticket.poll().map_err(error)? { Some(output) => break output, None => thread::sleep(Duration::from_millis(1)) }
            };
            let mut bytes = Vec::with_capacity(output.len() * 2);
            for n in output { bytes.extend_from_slice(&n.to_le_bytes()); }
            Ok(EncodeOutput { key: job.key, features: bytes.into(), elapsed_ms: started.elapsed().as_secs_f64() * 1000.0 })
        })
    }
    fn start_backend<F, H>(address: SocketAddr, handshake: EncoderHandshake, timeout: Duration, healthy: H, mut encode: F) -> Result<Self>
    where F: FnMut(&EncodeJob) -> Result<EncodeOutput> + Send + 'static, H: Fn() -> bool + Send + 'static {
        handshake.validate()?;
        let listener = TcpListener::bind(address).map_err(error)?;
        listener.set_nonblocking(true).map_err(error)?;
        let address = listener.local_addr().map_err(error)?;
        let stop = Arc::new(AtomicBool::new(false));
        let socket = Arc::new(Mutex::new(None));
        let stopped = stop.clone();
        let active = socket.clone();
        let owner = thread::Builder::new().name("vision-tcp-server".into()).spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Ok(copy) = stream.try_clone() { *active.lock().unwrap() = Some(copy); }
                        let result = Wire::new(stream, timeout).and_then(|mut stream| { handshake.write(&mut stream)
                            .and_then(|()| EncoderHandshake::read(&mut stream))
                            .and_then(|expected| handshake.compatible(&expected))
                            .and_then(|()| {
                                if !healthy() { return Err(error("vision encoder unavailable")); }
                                put_u32(&mut stream, 0)
                            })
                            .and_then(|()| serve_connection(&mut stream, &handshake, &stopped, &healthy, &mut encode)) });
                        if let Err(e) = result { tracing::debug!(%e, "vision peer closed/failed"); }
                        active.lock().unwrap().take();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(2)),
                    Err(e) => { tracing::error!(%e, "vision listener failed"); break; }
                }
            }
        }).map_err(error)?;
        Ok(Self { stop, socket, owner: Some(owner), address })
    }
}
impl Drop for EncoderServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(s) = self.socket.lock().unwrap().as_ref() { let _ = s.shutdown(Shutdown::Both); }
        if let Some(owner) = self.owner.take() { let _ = owner.join(); }
    }
}
fn serve_connection<F, H>(s: &mut Wire, handshake: &EncoderHandshake, stop: &AtomicBool, healthy: &H, encode: &mut F) -> Result<()>
where F: FnMut(&EncodeJob) -> Result<EncodeOutput>, H: Fn() -> bool {
    while !stop.load(Ordering::Acquire) {
        s.frame();
        let opcode = get_u32(s)?;
        if !healthy() { return Err(error("vision encoder unavailable")); }
        match opcode { 0 => { put_u32(s, 0)?; continue; }, 1 => {}, _ => return Err(error("invalid vision opcode")) }
        let key: cuteafd_core::MediaKey = ImageKey(read_array(s)?).into();
        let grid = [get_u32(s)?, get_u32(s)?, get_u32(s)?];
        let tokens = get_u64(s)?;
        let len = get_u64(s)?;
        let patches = u64::from(grid[1]) * u64::from(grid[2]);
        if grid[0] != 1 || patches == 0 || patches > u64::from(handshake.max_patches)
            || grid[1] % handshake.merge_size != 0 || grid[2] % handshake.merge_size != 0
            || tokens != patches / u64::from(handshake.merge_size).pow(2)
            || len != patches * u64::from(handshake.patch_size).pow(2) * 3 {
            return Err(error("invalid vision frame before allocation"));
        }
        let mut rgb = vec![0; len as usize];
        s.read_exact(&mut rgb).map_err(error)?;
        let job = EncodeJob::image(match key { cuteafd_core::MediaKey::Image(key) => key, _ => unreachable!() }, grid, rgb.into(), tokens as usize, handshake.output_width as usize);
        handshake.validate_job(&job)?;
        let result = encode(&job).and_then(|output| {
            if output.key != key || output.features.len() != job.feature_bytes()? || !output.elapsed_ms.is_finite() || output.elapsed_ms < 0.0 {
                Err(error("invalid vision backend output"))
            } else { Ok(output) }
        });
        // Encoding drains independently of the wire deadline; sending gets a fresh bound.
        s.frame();
        let (status, elapsed, bytes) = match result {
            Ok(output) => (0, (output.elapsed_ms * 1_000_000.0) as u64, output.features),
            Err(e) => (1, 0, Arc::from(e.to_string().as_bytes().iter().take(MAX_ERROR).copied().collect::<Vec<_>>())),
        };
        put_u32(s, status)?;
        s.write_all(key.bytes()).map_err(error)?;
        put_u64(s, elapsed)?;
        put_u64(s, bytes.len() as u64)?;
        s.write_all(&bytes).map_err(error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_engine::media::FakeEncoder;
    fn handshake() -> EncoderHandshake {
        EncoderHandshake { encoder_id: EncoderId([7; 32]), max_patches: 16_384, output_width: 4096, patch_size: 16, merge_size: 2, plan_hash: [9; 32] }
    }
    fn job(value: u8) -> EncodeJob {
        EncodeJob::image(ImageKey([value; 32]), [1, 4, 4], vec![value; 4*4*768].into(), 4, 4096)
    }
    fn server() -> EncoderServer {
        EncoderServer::start_backend("127.0.0.1:0".parse().unwrap(), handshake(), Duration::from_secs(2), || true, |job| Ok(EncodeOutput { key: job.key, features: FakeEncoder::features(job)?, elapsed_ms: 1.0 })).unwrap()
    }
    fn wait(client: &mut RemoteEncoder, ticket: EncoderTicket) -> Result<EncodeOutput> {
        let started = Instant::now();
        loop {
            if let Some(result) = client.poll(ticket) { return result; }
            assert!(started.elapsed() < Duration::from_secs(3));
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn loopback_replicas_are_byte_exact_cancel_and_drain() {
        let a = server(); let b = server();
        let mut client = RemoteEncoder::connect(vec![a.address,b.address], handshake(), Duration::from_secs(2)).unwrap();
        assert!(client.ready());
        let first = client.submit(job(1)).unwrap();
        let second = client.submit(job(2)).unwrap();
        assert_eq!(wait(&mut client,first).unwrap().features, FakeEncoder::features(&job(1)).unwrap());
        assert_eq!(wait(&mut client,second).unwrap().features, FakeEncoder::features(&job(2)).unwrap());
        let cancel = client.submit(job(3)).unwrap(); client.cancel(cancel);
        assert!(client.poll(cancel).is_none());
        let next = client.submit(job(4)).unwrap();
        assert_eq!(wait(&mut client,next).unwrap().features, FakeEncoder::features(&job(4)).unwrap());
    }
    #[test]
    fn trickle_reads_do_not_extend_frame_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            for _ in 0..20 {
                if socket.write_all(&[0]).is_err() { break; }
                thread::sleep(Duration::from_millis(20));
            }
        });
        let socket = TcpStream::connect(address).unwrap();
        let mut wire = Wire::new(socket, Duration::from_millis(100)).unwrap();
        let started = Instant::now();
        assert!(read_array::<20>(&mut wire).is_err());
        assert!(started.elapsed() < Duration::from_millis(350));
        drop(wire);
        peer.join().unwrap();
    }
    #[test]
    fn identity_placement_and_capacity_mismatches_fail_readiness() {
        for field in 0..3 {
            let server = server(); let mut expected = handshake();
            match field { 0 => expected.encoder_id.0[0] ^= 1, 1 => expected.plan_hash[0] ^= 1, _ => expected.output_width += 1 }
            assert!(RemoteEncoder::connect(vec![server.address], expected, Duration::from_secs(1)).is_err());
        }
    }
    #[test]
    fn idle_replica_failure_reaches_shared_health_and_fails_all_images() {
        let a = server(); let b = server();
        let mut client = RemoteEncoder::connect(vec![a.address, b.address], handshake(), Duration::from_millis(500)).unwrap();
        let health = client.health_handle();
        assert!(health.load(Ordering::Acquire));
        drop(b);
        let start = Instant::now();
        while health.load(Ordering::Acquire) {
            assert!(start.elapsed() < Duration::from_secs(4));
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!client.ready());
        assert!(client.submit(job(1)).is_err(), "the surviving replica cannot reopen image admission");
    }
    #[test]
    fn native_owner_failure_closes_idle_heartbeat() {
        let alive = Arc::new(AtomicBool::new(true));
        let server_alive = alive.clone();
        let server = EncoderServer::start_backend("127.0.0.1:0".parse().unwrap(), handshake(), Duration::from_secs(2),
            move || server_alive.load(Ordering::Acquire), |job| Ok(EncodeOutput {
                key: job.key, features: FakeEncoder::features(job)?, elapsed_ms: 1.0 })).unwrap();
        let client = RemoteEncoder::connect(vec![server.address], handshake(), Duration::from_millis(500)).unwrap();
        let health = client.health_handle();
        alive.store(false, Ordering::Release);
        let start = Instant::now();
        while health.load(Ordering::Acquire) {
            assert!(start.elapsed() < Duration::from_secs(4));
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!client.ready());
    }
    #[test]
    fn owner_exit_and_panic_fail_closed() {
        for panic in [false, true] {
            let health = Arc::new(AtomicBool::new(true));
            let owner_health = health.clone();
            let result = thread::spawn(move || {
                let _health = OwnerHealth(owner_health);
                if panic { panic!("injected owner failure"); }
            }).join();
            assert_eq!(result.is_err(), panic);
            assert!(!health.load(Ordering::Acquire));
        }
    }
    #[test]
    fn invalid_geometry_rejected_before_queue_and_rank_failure_visible() {
        let server = server();
        let mut client = RemoteEncoder::connect(vec![server.address], handshake(), Duration::from_secs(1)).unwrap();
        let mut invalid = job(0);
        if let cuteafd_engine::media::EncodeInput::Image { grid, .. } = &mut invalid.input { grid[1] = u32::MAX; }
        assert!(client.submit(EncodeJob::audio(cuteafd_core::AudioKey([0;32]), vec![0.0;481].into(), 1, 4096)).is_err());
        assert!(client.submit(invalid).is_err());
        drop(server);
        let ticket = client.submit(job(1)).unwrap();
        assert!(wait(&mut client,ticket).is_err());
        assert!(!client.healthy());
        assert!(client.submit(job(2)).is_err());
    }
}
