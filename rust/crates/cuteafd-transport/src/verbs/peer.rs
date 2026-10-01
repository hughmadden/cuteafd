//! Spark-side reduction: a full mesh of RC queue pairs between the Spark ranks
//! of one TP group, one per rail. For a wave flagged with
//! [`crate::expert::SPARK_ROW_SHARD_FLAGS`] every rank sends each peer the rows
//! that peer owns ([`crate::expert::spark_row_shard`]) straight from the
//! partial buffer its experts wrote, receives the other ranks' copies of its
//! own rows into mapped receive slots, and sums them in rank order on its GPU.
//! The coordinator then lands one reduced slice per rank instead of every
//! rank's whole partial plane.
//!
//! Messages carry the wave's request id (low 32 bits) as immediate data, so a
//! rank that is still on another wave keeps early slices until it gets there;
//! nothing blocks the worker's poll loop.
use super::*;
use crate::expert::spark_row_shard;
use cuteafd_ffi::CuteafdRdmaCompletion;

/// Reduction waves one worker can have in flight at once (one partial buffer
/// and one receive slot per link each). Coordinators keep at most this many
/// reduced waves outstanding per worker (one per lane).
pub const SPARK_REDUCE_WAVES: usize = 4;

/// Where the partial rows start in a partial buffer, as in a response slot.
const PARTIAL_OFFSET: usize = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;

/// How one Spark rank joins its group's reduction mesh.
#[derive(Debug, Clone)]
pub struct SparkReduceMeshConfig {
    pub rank: usize,
    pub world: usize,
    /// `rails[k][r]`: rank r's reduction listener on rail k. Each rail runs on
    /// the RDMA device that owns this rank's address on it.
    pub rails: Vec<Vec<SocketAddr>>,
    pub capacity_rows: u32,
    pub row_bytes: usize,
    /// Give up forming the mesh after this long.
    pub timeout: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
struct MeshHello {
    message: String,
    rank: usize,
    rail: usize,
    world: usize,
    capacity_rows: u32,
    row_bytes: usize,
    slot_bytes: usize,
    endpoint: VerbsHostNativeEndpointDescriptor,
}

struct Link {
    peer: usize,
    rail: usize,
    endpoint: NativeRdmaEndpoint,
    _control: TcpStream,
    /// Device address of receive slot 0 (mapped host memory).
    recv_device: usize,
    /// Region index of each partial buffer on this link's endpoint.
    regions: Vec<u32>,
}

struct Partial {
    host: CuteafdHostBuffer,
    device: CuteafdDeviceBuffer,
    /// Sends from this buffer not yet completed.
    sends: usize,
    busy: bool,
}

#[derive(Debug, Clone, Copy)]
struct Arrival {
    tag: u32,
    link: usize,
    slot: usize,
    bytes: usize,
}

/// One reduction step: `rows` rows starting `row_offset` rows into this rank's
/// share, summed from `slices` (rank order) into the output.
#[derive(Debug, Clone, Copy)]
pub struct SparkReduceSlices {
    pub row_offset: u32,
    pub rows: u32,
    pub slices: [*const u16; 8],
}

pub struct SparkReduceMesh {
    library: Arc<NativeLibrary>,
    rank: usize,
    world: usize,
    rails: usize,
    capacity_rows: u32,
    row_bytes: usize,
    slot_stride: usize,
    /// `links[rail * world + peer]`; `None` for this rank.
    links: Vec<Option<Link>>,
    partials: Vec<Partial>,
    arrivals: Vec<Arrival>,
    scratch: Vec<CuteafdRdmaCompletion>,
}

// The endpoints, mapped buffers and their registrations are owned here and
// touched only through `&mut self`; the mesh moves from its bootstrap thread to
// the GPU owner once and is never shared.
unsafe impl Send for SparkReduceMesh {}

/// Rows `[a, b)` of an `n`-row share that rail `rail` of `rails` carries.
fn rail_split(n: u32, rails: usize, rail: usize) -> (u32, u32) {
    let at = |k: usize| (u64::from(n) * k as u64 / rails as u64) as u32;
    (at(rail), at(rail + 1))
}

fn slot_bytes(capacity_rows: u32, world: usize, rails: usize, row_bytes: usize) -> usize {
    let share = capacity_rows.div_ceil(world as u32);
    share.div_ceil(rails as u32) as usize * row_bytes
}

/// The RDMA device whose RoCE v2 GID carries `ip`.
fn device_for(ip: IpAddr) -> Result<String> {
    let report = crate::fabric::discover()?;
    report
        .ports
        .iter()
        .find(|port| port.roce_v2.iter().any(|(_, address)| IpAddr::V4(*address) == ip))
        .map(|port| port.device.clone())
        .with_context(|| format!("no RDMA device carries reduction address {ip}"))
}

impl SparkReduceMesh {
    /// Forms this rank's links: connects to every lower rank and accepts every
    /// higher one on each rail. Lower ranks only wait on ranks below them, so
    /// the mesh forms whatever order the workers start in. Blocks until every
    /// link is up or `config.timeout` passes.
    pub fn establish(config: SparkReduceMeshConfig) -> Result<Self> {
        let SparkReduceMeshConfig { rank, world, rails, capacity_rows, row_bytes, timeout } = config;
        anyhow::ensure!(matches!(world, 2 | 3 | 4 | 6) && rank < world, "reduction mesh needs rank < world of 2, 3, 4 or 6");
        anyhow::ensure!(!rails.is_empty() && rails.iter().all(|rail| rail.len() == world),
            "every reduction rail lists one address per rank");
        anyhow::ensure!(capacity_rows > 0 && row_bytes > 0 && row_bytes % 16 == 0, "invalid reduction geometry");
        verbs_host_preflight()?;
        let library = load_verbs_host_native_library()?;
        let rail_count = rails.len();
        let slot = slot_bytes(capacity_rows, world, rail_count, row_bytes);
        let slot_stride = slot.next_multiple_of(4096);
        let deadline = Instant::now() + timeout;
        let mut partials = Vec::with_capacity(SPARK_REDUCE_WAVES);
        let partial_bytes = PARTIAL_OFFSET + capacity_rows as usize * row_bytes;
        let mut mesh = Self {
            library: Arc::clone(&library),
            rank,
            world,
            rails: rail_count,
            capacity_rows,
            row_bytes,
            slot_stride,
            links: (0..rail_count * world).map(|_| None).collect(),
            partials: Vec::new(),
            arrivals: Vec::with_capacity(4 * SPARK_REDUCE_WAVES * world * rail_count),
            scratch: vec![CuteafdRdmaCompletion::default(); 32],
        };
        for _ in 0..SPARK_REDUCE_WAVES {
            let host = library.alloc_host_buffer(partial_bytes)?;
            let device = match library.cuda_host_buffer_device_alias(host) {
                Ok(device) => device,
                Err(error) => {
                    let mut host = host;
                    let _ = library.free_host_buffer(&mut host);
                    return Err(error);
                }
            };
            partials.push(Partial { host, device, sends: 0, busy: false });
        }
        mesh.partials = partials;
        for (rail, addresses) in rails.iter().enumerate() {
            let local = addresses[rank];
            let device = device_for(local.ip())?;
            let listener = TcpListener::bind(local)
                .with_context(|| format!("binding reduction rail {rail} listener {local}"))?;
            listener.set_nonblocking(true)?;
            for peer in 0..rank {
                let stream = loop {
                    match TcpStream::connect_timeout(&addresses[peer], Duration::from_secs(1)) {
                        Ok(stream) => break stream,
                        Err(error) => {
                            anyhow::ensure!(Instant::now() < deadline,
                                "reduction rail {rail}: rank {peer} at {} never answered: {error}", addresses[peer]);
                            thread::sleep(Duration::from_millis(200));
                        }
                    }
                };
                let link = mesh.join(stream, &device, rail, Some(peer), slot, deadline)?;
                mesh.links[rail * world + peer] = Some(link);
            }
            let mut pending = world - rank - 1;
            while pending > 0 {
                match listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_nonblocking(false)?;
                        let link = mesh.join(stream, &device, rail, None, slot, deadline)?;
                        let index = rail * world + link.peer;
                        anyhow::ensure!(link.peer > rank && mesh.links[index].is_none(),
                            "reduction rail {rail}: unexpected or duplicate rank {}", link.peer);
                        mesh.links[index] = Some(link);
                        pending -= 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        anyhow::ensure!(Instant::now() < deadline,
                            "reduction rail {rail}: {pending} higher ranks never connected to {local}");
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(error) => return Err(error).context("accepting a reduction peer"),
                }
            }
        }
        tracing::info!(rank, world, rails = rail_count, slot_bytes = slot,
            partial_bytes = partial_bytes * SPARK_REDUCE_WAVES,
            receive_bytes = slot_stride * SPARK_REDUCE_WAVES * (world - 1) * rail_count,
            "Spark reduction mesh ready");
        Ok(mesh)
    }

    /// One link's handshake: `peer` is the rank this side dialed (client), or
    /// `None` when accepting (server; the hello names the peer).
    fn join(&self, stream: TcpStream, device: &str, rail: usize, peer: Option<usize>, slot: usize,
        deadline: Instant) -> Result<Link> {
        let remaining = deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(1));
        stream.set_read_timeout(Some(remaining))?;
        stream.set_write_timeout(Some(remaining))?;
        stream.set_nodelay(true)?;
        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let depth = SPARK_REDUCE_WAVES * 2;
        let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_with_buffer_flags(
            Arc::clone(&self.library),
            "client",
            64,
            slot,
            4096,
            self.slot_stride * SPARK_REDUCE_WAVES,
            next_local_psn("client"),
            CUTEAFD_HOST_BUFFER_FLAG_PINNED | CUTEAFD_HOST_BUFFER_FLAG_MAPPED,
            depth.max(VERBS_HOST_RDMA_RING_DEPTH) as u32,
            Some(device),
            1,
        )?;
        let hello = |endpoint: &NativeRdmaEndpoint| MeshHello {
            message: "spark_reduce_hello".into(),
            rank: self.rank,
            rail,
            world: self.world,
            capacity_rows: self.capacity_rows,
            row_bytes: self.row_bytes,
            slot_bytes: slot,
            endpoint: endpoint.native_descriptor(),
        };
        let theirs: MeshHello = if peer.is_some() {
            write_control(&mut writer, &hello(&endpoint))?;
            read_control(&mut reader)?
        } else {
            read_control(&mut reader)?
        };
        anyhow::ensure!(theirs.message == "spark_reduce_hello" && theirs.rail == rail
            && theirs.world == self.world && theirs.capacity_rows == self.capacity_rows
            && theirs.row_bytes == self.row_bytes && theirs.slot_bytes == slot,
            "reduction peer on rail {rail} disagrees on the mesh (rank {} world {} rows {} row bytes {})",
            theirs.rank, theirs.world, theirs.capacity_rows, theirs.row_bytes);
        if let Some(peer) = peer {
            anyhow::ensure!(theirs.rank == peer, "dialed rank {peer} but rank {} answered", theirs.rank);
        }
        anyhow::ensure!(theirs.rank < self.world && theirs.rank != self.rank, "invalid reduction peer rank");
        endpoint.connect(&theirs.endpoint)?;
        for slot_index in 0..SPARK_REDUCE_WAVES {
            endpoint.post_recv_at(slot_index * self.slot_stride, slot, slot_index as u64)?;
        }
        if peer.is_none() {
            write_control(&mut writer, &hello(&endpoint))?;
        }
        let view = endpoint.recv_buffer_view()?;
        anyhow::ensure!(!view.device_ptr.is_null(), "reduction receive slots are not device-visible");
        let mut regions = Vec::with_capacity(self.partials.len());
        for partial in &self.partials {
            // SAFETY: the partial buffers outlive every link (`Drop` destroys
            // the links, and so their registrations, first).
            regions.push(unsafe {
                self.library.rdma_rc_endpoint_register_region(endpoint.info.handle, partial.host.ptr,
                    partial.host.bytes)?
            });
        }
        stream.set_read_timeout(None)?;
        Ok(Link {
            peer: theirs.rank,
            rail,
            endpoint,
            _control: stream,
            recv_device: view.device_ptr as usize,
            regions,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// Claims a free partial buffer for a new wave; its experts write the
    /// partial rows [`EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN`] bytes into the
    /// returned slot, exactly as into a response slot.
    pub fn acquire(&mut self) -> Result<(usize, CuteafdDeviceBuffer)> {
        let deadline = Instant::now() + default_control_timeout();
        loop {
            if let Some(index) = self.partials.iter().position(|p| !p.busy && p.sends == 0) {
                self.partials[index].busy = true;
                return Ok((index, self.partials[index].device));
            }
            anyhow::ensure!(self.partials.iter().any(|p| !p.busy),
                "more than {SPARK_REDUCE_WAVES} Spark reduction waves in flight");
            anyhow::ensure!(Instant::now() < deadline, "reduction sends never completed");
            self.poll()?;
        }
    }

    /// Posts every peer's rows of a `rows`-row wave from `partial`, tagged
    /// with `tag`.
    pub fn post(&mut self, partial: usize, tag: u32, rows: u32) -> Result<()> {
        anyhow::ensure!(partial < self.partials.len() && self.partials[partial].busy, "unclaimed partial buffer");
        anyhow::ensure!(rows as usize >= self.world && rows <= self.capacity_rows,
            "a {rows}-row wave cannot be reduce-scattered over {} ranks", self.world);
        for rail in 0..self.rails {
            for peer in (0..self.world).filter(|peer| *peer != self.rank) {
                let (lo, hi) = spark_row_shard(rows, self.world, peer);
                let (a, b) = rail_split(hi - lo, self.rails, rail);
                if b == a {
                    continue;
                }
                let link = self.links[rail * self.world + peer].as_ref().context("missing reduction link")?;
                self.library.rdma_rc_endpoint_post_send_region(
                    link.endpoint.info.handle,
                    link.regions[partial],
                    PARTIAL_OFFSET + (lo + a) as usize * self.row_bytes,
                    (b - a) as usize * self.row_bytes,
                    partial as u64,
                    tag,
                )?;
                self.partials[partial].sends += 1;
            }
        }
        Ok(())
    }

    /// Drains every link's completions: sends free their partial buffer,
    /// receives become arrivals for their wave's tag.
    pub fn poll(&mut self) -> Result<()> {
        for index in 0..self.links.len() {
            let Some(link) = self.links[index].as_ref() else { continue };
            loop {
                let count = self.library.rdma_rc_endpoint_poll_completions(link.endpoint.info.handle,
                    &mut self.scratch)?;
                for completion in &self.scratch[..count] {
                    if completion.recv != 0 {
                        anyhow::ensure!(completion.has_imm != 0, "reduction slice arrived without its wave tag");
                        self.arrivals.push(Arrival {
                            tag: completion.imm,
                            link: index,
                            slot: completion.wr_id as usize,
                            bytes: completion.byte_len as usize,
                        });
                    } else {
                        let partial = &mut self.partials[completion.wr_id as usize];
                        partial.sends = partial.sends.checked_sub(1).context("unexpected reduction send completion")?;
                    }
                }
                if count < self.scratch.len() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// When every peer's copy of this rank's rows of wave `tag` (`rows` rows)
    /// has arrived: the reduction steps, one per rail, reading this rank's own
    /// rows from `partial`. Output row offsets are relative to this rank's
    /// share.
    pub fn ready(&self, partial: usize, tag: u32, rows: u32) -> Result<Option<Vec<SparkReduceSlices>>> {
        let (lo, hi) = spark_row_shard(rows, self.world, self.rank);
        let mut steps = Vec::with_capacity(self.rails);
        for rail in 0..self.rails {
            let (a, b) = rail_split(hi - lo, self.rails, rail);
            if b == a {
                continue;
            }
            let bytes = (b - a) as usize * self.row_bytes;
            let mut slices = [std::ptr::null::<u16>(); 8];
            for rank in 0..self.world {
                if rank == self.rank {
                    let base = self.partials[partial].device.ptr as usize;
                    slices[rank] = (base + PARTIAL_OFFSET + (lo + a) as usize * self.row_bytes) as *const u16;
                    continue;
                }
                let link_index = rail * self.world + rank;
                let Some(arrival) = self.arrivals.iter().find(|x| x.tag == tag && x.link == link_index) else {
                    return Ok(None);
                };
                anyhow::ensure!(arrival.bytes == bytes,
                    "rank {rank} sent {} bytes of wave {tag:#x} on rail {rail}; {bytes} expected", arrival.bytes);
                let link = self.links[link_index].as_ref().context("missing reduction link")?;
                slices[rank] = (link.recv_device + arrival.slot * self.slot_stride) as *const u16;
            }
            steps.push(SparkReduceSlices { row_offset: a, rows: b - a, slices });
        }
        Ok(Some(steps))
    }

    /// After the reduction of wave `tag` completed on the device: gives its
    /// receive slots back to their links and frees `partial` once its own
    /// sends complete.
    pub fn finish(&mut self, partial: usize, tag: u32) -> Result<()> {
        let mut index = 0;
        while index < self.arrivals.len() {
            let arrival = self.arrivals[index];
            if arrival.tag != tag {
                index += 1;
                continue;
            }
            let link = self.links[arrival.link].as_ref().context("missing reduction link")?;
            link.endpoint.post_recv_at(arrival.slot * self.slot_stride,
                slot_bytes(self.capacity_rows, self.world, self.rails, self.row_bytes), arrival.slot as u64)?;
            self.arrivals.swap_remove(index);
        }
        self.partials[partial].busy = false;
        Ok(())
    }

    /// Releases `partial` of a wave abandoned before its reduction (its own
    /// sends still complete through [`Self::poll`]).
    pub fn abandon(&mut self, partial: usize) {
        if let Some(partial) = self.partials.get_mut(partial) {
            partial.busy = false;
        }
    }

    /// Whether any link still owes this rank a completion.
    pub fn idle(&self) -> bool {
        self.arrivals.is_empty() && self.partials.iter().all(|p| !p.busy && p.sends == 0)
    }
}

impl Drop for SparkReduceMesh {
    fn drop(&mut self) {
        // Endpoints (and their registrations of the partial buffers) first.
        self.links.clear();
        for partial in &mut self.partials {
            let _ = self.library.free_host_buffer(&mut partial.host);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rails_and_shards_cover_every_row_once() {
        for world in [2usize, 3, 4, 6] {
            for rails in [1usize, 2] {
                for rows in [world as u32, 7, 64, 1000, 4096] {
                    if (rows as usize) < world {
                        continue;
                    }
                    let mut covered = vec![0u8; rows as usize];
                    for rank in 0..world {
                        let (lo, hi) = spark_row_shard(rows, world, rank);
                        assert!(hi > lo);
                        for rail in 0..rails {
                            let (a, b) = rail_split(hi - lo, rails, rail);
                            assert!((b - a) as usize * 16 <= slot_bytes(4096, world, rails, 16));
                            for row in lo + a..lo + b {
                                covered[row as usize] += 1;
                            }
                        }
                    }
                    assert!(covered.iter().all(|c| *c == 1), "world {world} rails {rails} rows {rows}");
                }
            }
        }
    }
}
