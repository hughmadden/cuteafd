//! A [`SparkExperts`] on a thread of its own, so a prefill lane's request
//! assembly, RDMA post, polling and partial copies overlap the inference
//! owner's GPU work. The transport is created on that thread and never
//! leaves it (its verbs endpoints are not `Send`); jobs carry `Send` closures
//! that build the request and consume the response rows. One job is in
//! flight per lane. The decode hot path keeps using [`SparkExperts`] inline.
use super::{SparkExperts, SparkTopology};
use crate::{DeviceLanding, ExpertProtocolV2Request, TcpTransportConfig, VerbsHostProtocolV2ResponsePayload};
use anyhow::{anyhow, ensure, Context, Result};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Builds the wave's request on the lane thread.
pub type LaneBuild = Box<dyn FnOnce() -> Result<ExpertProtocolV2Request> + Send>;
/// Receives each `(rank, first row, payload)` of the wave on the lane thread;
/// the payload must be consumed before it returns.
pub type LaneSink = Box<dyn FnMut(usize, u32, &[u8]) -> Result<()> + Send>;
/// Receives each payload's owner on the lane thread (see
/// [`SparkExperts::receive_wave_owned`]); retained receive slots must be
/// released before the lane's next job.
pub type LaneOwnedSink = Box<dyn FnMut(usize, u32, VerbsHostProtocolV2ResponsePayload) -> Result<()> + Send>;

enum Sink {
    Borrowed(LaneSink),
    Owned(LaneOwnedSink),
}

struct Job {
    build: LaneBuild,
    sink: Sink,
}

/// Seconds a finished job spent building its request and posting it, and
/// from the post until its last row arrived.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaneTimes {
    pub build_post: f64,
    pub receive: f64,
    /// [`super::WaveReceipt::landed`]: ranks whose plane landed in device memory.
    pub landed: u8,
}

pub struct SparkExpertLane {
    jobs: Option<mpsc::Sender<Job>>,
    done: mpsc::Receiver<Result<LaneTimes>>,
    world: usize,
    pending: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SparkExpertLane {
    /// Canonical transport whose rank payloads land in `landing`.
    ///
    /// # Safety
    /// [`SparkExperts::set_gpu_landing`]'s contract, with a lane's wave running
    /// from [`Self::submit`] until [`Self::wait`] returns.
    pub unsafe fn spawn_with_landing(peers: Vec<SocketAddr>, executors: Vec<u64>, capacity: u32,
        config: TcpTransportConfig, landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        // SAFETY: forwarded from this constructor's contract.
        unsafe { Self::spawn_transport(move || SparkExperts::new_ranks(&peers, &executors, capacity, config), landing) }
    }

    /// Ownership-bound transport; rank identities come only from the topology.
    ///
    /// # Safety
    /// The same landing lifetime and wave contract as [`Self::spawn_with_landing`].
    pub unsafe fn spawn_topology_with_landing(topology: SparkTopology, peers: Vec<SocketAddr>, capacity: u32,
        config: TcpTransportConfig, landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        // SAFETY: forwarded from this constructor's contract.
        unsafe { Self::spawn_transport(move || SparkExperts::new_topology(topology, &peers, capacity, config), landing) }
    }

    unsafe fn spawn_transport(build: impl FnOnce() -> Result<SparkExperts> + Send + 'static,
        landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (report, done) = mpsc::channel::<Result<LaneTimes>>();
        let (ready_tx, ready) = mpsc::channel::<Result<usize>>();
        let thread = std::thread::Builder::new().name("v41-roce-lane".into()).spawn(move || {
            let setup = build().and_then(|mut transport| {
                if landing.is_some() {
                    // SAFETY: forwarded from this constructor's contract.
                    unsafe { transport.set_gpu_landing(landing)? };
                }
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                Ok((transport, runtime))
            });
            let (mut transport, runtime) = match setup {
                Ok(setup) => setup,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(transport.world_size()));
            while let Ok(Job { build, sink }) = inbox.recv() {
                let started = Instant::now();
                let result = build().and_then(|request| transport.dispatch_wave(&request)).and_then(|wave| {
                    let posted = Instant::now();
                    let receipt = match sink {
                        Sink::Borrowed(sink) => runtime.block_on(transport.receive_wave(wave, sink))?,
                        Sink::Owned(sink) => runtime.block_on(transport.receive_wave_owned(wave, sink))?,
                    };
                    Ok(LaneTimes { build_post: (posted - started).as_secs_f64(),
                        receive: posted.elapsed().as_secs_f64(), landed: receipt.landed })
                });
                if report.send(result).is_err() {
                    break;
                }
            }
        })?;
        let world = ready.recv().map_err(|_| anyhow!("RoCE lane thread exited during setup"))??;
        Ok(Self { jobs: Some(jobs), done, world, pending: false, thread: Some(thread) })
    }

    pub fn world_size(&self) -> usize {
        self.world
    }

    /// Queues one wave: `build` runs, the request is posted to every rank
    /// and each response row goes to `sink`, all on the lane thread.
    pub fn submit(&mut self, build: LaneBuild, sink: LaneSink) -> Result<()> {
        self.submit_job(Job { build, sink: Sink::Borrowed(sink) })
    }

    /// [`Self::submit`] with a sink that takes each payload's owner.
    pub fn submit_owned(&mut self, build: LaneBuild, sink: LaneOwnedSink) -> Result<()> {
        self.submit_job(Job { build, sink: Sink::Owned(sink) })
    }

    fn submit_job(&mut self, job: Job) -> Result<()> {
        ensure!(!self.pending, "a RoCE lane holds one wave at a time");
        self.jobs.as_ref().context("RoCE lane closed")?.send(job)
            .map_err(|_| anyhow!("RoCE lane thread exited"))?;
        self.pending = true;
        Ok(())
    }

    /// Waits for the submitted wave's last row (or its error).
    pub fn wait(&mut self, timeout: Duration) -> Result<LaneTimes> {
        ensure!(self.pending, "no RoCE lane wave in flight");
        self.pending = false;
        match self.done.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!("RoCE lane wave timed out after {timeout:?}")),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(anyhow!("RoCE lane thread exited")),
        }
    }
}

impl Drop for SparkExpertLane {
    fn drop(&mut self) {
        // Closing the job channel ends the thread after its current wave.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_lane_preserves_rank_contract_and_refuses_canonical_requests() -> Result<()> {
        let topology = SparkTopology::NATIVE_TP3_EP1;
        let peers = (1..=3).map(|port| format!("127.0.0.1:{port}").parse())
            .collect::<std::result::Result<Vec<SocketAddr>, _>>()?;
        let config = TcpTransportConfig { timing: false, timeout: Duration::from_secs(1), max_frame_bytes: 64 << 20 };
        // SAFETY: no landing buffers are installed.
        let mut lane = unsafe { SparkExpertLane::spawn_topology_with_landing(topology, peers.clone(), 1, config.clone(), None)? };
        assert_eq!(lane.world_size(), 3);
        lane.submit(Box::new(|| {
            let mut request = ExpertProtocolV2Request::new(1, 1, 0, 5120,
                crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                vec![crate::ExpertProtocolV2RowDescriptor { row_id: 0, source_kind: crate::ExpertV2SourceKind::Prefill,
                    source_request_id: 1, token_position: 0, route_offset: 0, route_count: 6 }],
                (0..6).map(|expert| crate::ExpertProtocolV2RouteEntry { row_index: 0, expert_id: expert, gate_weight: 0.125 }).collect(),
                vec![0; 5280])?;
            request.header.flags |= super::super::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(request)
        }), Box::new(|_, _, _| Ok(())))?;
        let error = lane.wait(Duration::from_secs(2)).unwrap_err();
        assert!(error.to_string().contains("requires a native group request"), "{error:#}");
        // SAFETY: no landing buffers are installed; mismatched worlds fail setup.
        assert!(unsafe { SparkExpertLane::spawn_topology_with_landing(topology, peers[..2].to_vec(), 1, config, None) }.is_err());
        Ok(())
    }
}
