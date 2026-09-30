//! A [`V41Tp4Roce`] on a thread of its own, so a prefill lane's request
//! assembly, RDMA post, polling and partial copies overlap the inference
//! owner's GPU work. The transport is created on that thread and never
//! leaves it (its verbs endpoints are not `Send`); jobs carry `Send` closures
//! that build the request and consume the response rows. One job is in
//! flight per lane. The decode hot path keeps using [`V41Tp4Roce`] inline.
use super::V41Tp4Roce;
use crate::{DeviceLanding, ExpertProtocolV2Request, TcpTransportConfig, VerbsHostProtocolV2ResponsePayload};
use anyhow::{anyhow, ensure, Context, Result};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Builds the wave's request on the lane thread.
pub type V41LaneBuild = Box<dyn FnOnce() -> Result<ExpertProtocolV2Request> + Send>;
/// Receives each `(rank, first row, payload)` of the wave on the lane thread;
/// the payload must be consumed before it returns.
pub type V41LaneSink = Box<dyn FnMut(usize, u32, &[u8]) -> Result<()> + Send>;
/// Receives each payload's owner on the lane thread (see
/// [`V41Tp4Roce::receive_wave_owned`]); retained receive slots must be
/// released before the lane's next job.
pub type V41LaneOwnedSink = Box<dyn FnMut(usize, u32, VerbsHostProtocolV2ResponsePayload) -> Result<()> + Send>;

enum Sink {
    Borrowed(V41LaneSink),
    Owned(V41LaneOwnedSink),
}

struct Job {
    build: V41LaneBuild,
    sink: Sink,
}

/// Seconds a finished job spent building its request and posting it, and
/// from the post until its last row arrived.
#[derive(Debug, Clone, Copy, Default)]
pub struct V41LaneTimes {
    pub build_post: f64,
    pub receive: f64,
    /// [`super::V41WaveReceipt::landed`]: ranks whose plane landed in device memory.
    pub landed: u8,
}

pub struct V41Tp4RoceLane {
    jobs: Option<mpsc::Sender<Job>>,
    done: mpsc::Receiver<Result<V41LaneTimes>>,
    world: usize,
    pending: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl V41Tp4RoceLane {
    /// Connects a [`V41Tp4Roce::new_ranks`] transport on a new thread.
    pub fn spawn(peers: Vec<SocketAddr>, executors: Vec<u64>, capacity: u32, config: TcpTransportConfig)
        -> Result<Self> {
        // SAFETY: no GPU landing ranges are given.
        unsafe { Self::spawn_with_landing(peers, executors, capacity, config, None) }
    }

    /// [`Self::spawn`] whose transport lands rank payloads in `landing`.
    ///
    /// # Safety
    /// [`V41Tp4Roce::set_gpu_landing`]'s contract, with a lane's wave running
    /// from [`Self::submit`] until [`Self::wait`] returns.
    pub unsafe fn spawn_with_landing(peers: Vec<SocketAddr>, executors: Vec<u64>, capacity: u32,
        config: TcpTransportConfig, landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (report, done) = mpsc::channel::<Result<V41LaneTimes>>();
        let (ready_tx, ready) = mpsc::channel::<Result<usize>>();
        let thread = std::thread::Builder::new().name("v41-roce-lane".into()).spawn(move || {
            let setup = V41Tp4Roce::new_ranks(&peers, &executors, capacity, config).and_then(|mut transport| {
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
                    Ok(V41LaneTimes { build_post: (posted - started).as_secs_f64(),
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
    pub fn submit(&mut self, build: V41LaneBuild, sink: V41LaneSink) -> Result<()> {
        self.submit_job(Job { build, sink: Sink::Borrowed(sink) })
    }

    /// [`Self::submit`] with a sink that takes each payload's owner.
    pub fn submit_owned(&mut self, build: V41LaneBuild, sink: V41LaneOwnedSink) -> Result<()> {
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
    pub fn wait(&mut self, timeout: Duration) -> Result<V41LaneTimes> {
        ensure!(self.pending, "no RoCE lane wave in flight");
        self.pending = false;
        match self.done.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!("RoCE lane wave timed out after {timeout:?}")),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(anyhow!("RoCE lane thread exited")),
        }
    }
}

impl Drop for V41Tp4RoceLane {
    fn drop(&mut self) {
        // Closing the job channel ends the thread after its current wave.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
