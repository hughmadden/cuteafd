//! Coordinator (RTX) ranks of a push group, receive side: each coordinator GPU
//! receives its TP2 expert halves (and, where the family lands them, its dense
//! operands) on its own thread and ring, under the same control protocol as a
//! Spark rank. Publication follows the whole-group commit.
#[cfg(test)]
mod tests;
use crate::shared::memory::{push_consumer::PushDestination, push_coordinator::Tp2ExpertPlan,
    weight_arena::{DeliveredArena, LoadingArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::CheckedPlan, OfficialV41Catalog, V41Exl3Layer};

/// A coordinator rank's logical space: its dense operands (a family-provided
/// destination) in `[0, base)`, then the TP2 expert arena (device bytes, then
/// its host validation tail) from `base`, then the dense destination's staged
/// bytes (its host part) from `host_base`. Gaps separate the three, so no
/// fragment spans two.
pub(crate) struct Combined<'a, D> {
    dense: D,
    base: u64,
    host_base: u64,
    tp2: LoadingArena<'a>,
}

impl<'a, D: PushDestination<Delivered = ()>> Combined<'a, D> {
    pub fn new(dense: D, base: u64, tp2: LoadingArena<'a>) -> Result<Self> {
        ensure!(dense.device_bytes() as u64 <= base && base % 256 == 0, "dense operands overlap the TP2 arena");
        let (host_base, _) = rank_bytes(base, PushDestination::logical_bytes(&tp2) as u64, 1);
        Ok(Self { dense, base, host_base, tp2 })
    }
    fn dense_host(&self) -> u64 { (self.dense.logical_bytes() - self.dense.device_bytes()) as u64 }
    /// The dense destination's own offset of a staged fragment at `offset`.
    fn staged(&self, offset: u64) -> Option<u64> {
        (self.dense_host() > 0 && offset >= self.host_base).then(|| offset - self.host_base + self.dense.device_bytes() as u64)
    }
    fn tp2_offset(&self, offset: u64) -> Option<u64> { offset.checked_sub(self.base) }
}

impl<'a, D: PushDestination<Delivered = ()>> PushDestination for Combined<'a, D> {
    type Delivered = DeliveredArena<'a>;
    fn belongs_to(&self, device: crate::shared::memory::device::Device<'_>) -> bool {
        self.dense.belongs_to(device) && PushDestination::belongs_to(&self.tp2, device)
    }
    fn logical_bytes(&self) -> usize {
        match self.dense_host() { 0 => self.base as usize + PushDestination::logical_bytes(&self.tp2),
            host => (self.host_base + host) as usize }
    }
    fn device_bytes(&self) -> usize { self.base as usize + PushDestination::device_bytes(&self.tp2) }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        if let Some(at) = self.staged(offset) { return self.dense.check_fragment(at, bytes); }
        match self.tp2_offset(offset) {
            Some(at) => PushDestination::check_fragment(&self.tp2, at, bytes),
            None => self.dense.check_fragment(offset, bytes),
        }
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        match self.tp2_offset(offset as u64) {
            Some(at) => PushDestination::destination(&self.tp2, at as usize, bytes),
            None => self.dense.destination(offset, bytes),
        }
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        if let Some(at) = self.staged(offset) { return self.dense.copy_validation(at, bytes); }
        let at = self.tp2_offset(offset).context("host tail fragment in the dense operands")?;
        PushDestination::copy_validation(&mut self.tp2, at, bytes)
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        match self.tp2_offset(offset) {
            Some(at) => PushDestination::copied(&mut self.tp2, at, bytes),
            None => self.dense.copied(offset, bytes),
        }
    }
    fn done(self) -> Result<DeliveredArena<'a>> {
        self.dense.done()?;
        PushDestination::done(self.tp2)
    }
}

/// A family's dense operands for one GPU: their destination, and where the
/// TP2 arena starts in the rank's logical space.
pub(crate) struct DenseRank<D> {
    pub destination: D,
    pub base: u64,
}

/// One coordinator GPU's TP2 halves alone (the TP2-only group replay).
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_tp2<'a>(
    device: crate::shared::memory::device::Device<'a>, catalog: &OfficialV41Catalog, plan: Tp2ExpertPlan,
    checked: &CheckedPlan, rank: usize, client: &cuteafd_loader::push_load::PushClient,
    stream: &mut std::net::TcpStream, credentials: (u64, u64), slots: usize, slot_bytes: usize,
) -> Result<Option<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>>> {
    receive_rank::<NoDense>(device, Some((catalog, plan, checked)), None, rank, client, stream, credentials,
        slots, slot_bytes)
}

/// No dense operands on this rank (families whose dense weights load by mount).
pub(crate) struct NoDense;
impl PushDestination for NoDense {
    type Delivered = ();
    fn belongs_to(&self, _: crate::shared::memory::device::Device<'_>) -> bool { true }
    fn logical_bytes(&self) -> usize { 0 }
    fn device_bytes(&self) -> usize { 0 }
    fn check_fragment(&self, _: u64, _: u64) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn destination(&self, _: usize, _: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        anyhow::bail!("no dense operands on this rank")
    }
    fn copy_validation(&mut self, _: u64, _: &[u8]) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn copied(&mut self, _: u64, _: u64) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn done(self) -> Result<()> { Ok(()) }
}

/// One coordinator GPU's rank of a push group, received on its own thread and
/// ring under the same control protocol as a Spark rank (a loopback stream to
/// the group driver): its TP2 halves (`tp2`: their entries alone from offset
/// 0, the arena's own footprint) and/or its dense operands (`dense`, in
/// `[0, dense.base)` of the rank). `Ok(None)`: the whole group fell back; every
/// arena reference is dropped before the drained acknowledgment. With no TP2,
/// a committed rank returns no expert weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_rank<'a, D: PushDestination<Delivered = ()>>(
    device: crate::shared::memory::device::Device<'a>,
    tp2: Option<(&OfficialV41Catalog, Tp2ExpertPlan, &CheckedPlan)>, dense: Option<DenseRank<D>>, rank: usize,
    client: &cuteafd_loader::push_load::PushClient, stream: &mut std::net::TcpStream, credentials: (u64, u64),
    slots: usize, slot_bytes: usize,
) -> Result<Option<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>>> {
    use crate::shared::memory::{push_consumer, push_initialize, weight_arena::ExpectedCoverage};
    use cuteafd_ffi::push_ring::RegisteredRing;
    use cuteafd_transport::worker_selection::push;
    use std::os::fd::AsRawFd;
    ensure!(tp2.is_some() || dense.is_some(), "coordinator rank receives nothing");
    let library = device.library;
    let (group, token) = credentials;
    tracing::info!(group, rank, device = device.id, tp2 = tp2.is_some(), dense = dense.is_some(),
        "coordinator push rank joining load group");
    let joined = device.run(|| -> Result<_> {
        let arena = tp2.as_ref().map(|(_, plan, checked)| -> Result<_> {
            let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank)?)?;
            LoadingArena::new(device, plan.layout.clone(), expected)
        }).transpose()?;
        let ring = RegisteredRing::new(library, slots, slot_bytes)?;
        let joined = client.join(cuteafd_loader::push_load::JoinArgs { group, token,
            rank: u32::try_from(rank)?, ring_fd: ring.fd().as_raw_fd(),
            slots: u32::try_from(ring.slots())?, slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, joined))
    });
    if let Err(error) = &joined { tracing::info!(group, rank, %error, "coordinator push rank join failed"); }
    push::joined(stream, joined.is_ok()).context("coordinator push joined frame")?;
    let candidate = joined.and_then(|(arena, ring, joined)| {
        let delivered = match (arena, dense) {
            (Some(arena), Some(DenseRank { destination, base })) =>
                Some(push_consumer::consume(device, Combined::new(destination, base, arena)?, ring, joined, slots)?),
            (Some(arena), None) => Some(push_consumer::consume(device, arena, ring, joined, slots)?),
            (None, Some(DenseRank { destination, .. })) => {
                push_consumer::consume(device, destination, ring, joined, slots)?;
                None
            }
            (None, None) => unreachable!("checked above"),
        };
        let Some(((catalog, plan, _), delivered)) = tp2.zip(delivered) else { return Ok(Vec::new()) };
        let initialize = plan.residencies.iter().enumerate().map(|(region, residency)|
            push_initialize::Layer::Exl3 { region, residency }).collect::<Vec<_>>();
        let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, Some(&plan.validation), 0)?);
        plan.residencies.into_iter().enumerate().map(|(region, layout)| {
            let budget = crate::shared::experts::exl3::Exl3Weights::plan(catalog,
                crate::shared::experts::layer::ExpertLayer::BackboneTp2 {
                    layer: match layout.layer { V41Exl3Layer::Backbone(l) => l, _ => anyhow::bail!("draft layer in coordinator TP2") },
                    rank: layout.rank })?;
            crate::shared::experts::exl3::Exl3Weights::from_push(ready.clone(), region, layout, budget)
        }).collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate { tracing::info!(group, rank, %error, "coordinator push candidate rejected"); }
    if push::completed(stream, candidate.is_ok()).context("coordinator push completion")? {
        let weights = candidate?;
        tracing::info!(group, rank, layers = weights.len(), "coordinator push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe mount retry");
    push::drained(stream).context("coordinator push drained barrier")?;
    tracing::info!(group, rank, "whole push group discarded; coordinator loads through mount");
    Ok(None)
}

/// A family's dense plan for one GPU: device entries from 0 of the rank's
/// logical space up to `base` (where the TP2 arena starts), and `host_bytes`
/// of staged entries placed at the `host_base` the driver passes (after the
/// TP2 arena's device and host bytes).
pub(crate) struct DensePlan {
    pub base: u64,
    pub host_bytes: u64,
    pub append: Box<dyn Fn(&mut cuteafd_loader::push_load::builder::Builder, u32, u64) -> Result<()> + Send>,
}

/// A rank's logical bytes: dense device `[0, base)`, the TP2 arena, then
/// (after a gap) the dense staged steps: (where those start, the total).
fn rank_bytes(base: u64, tp2: u64, host: u64) -> (u64, u64) {
    let host_base = if tp2 == 0 { base } else { (base + tp2 + 1).next_multiple_of(256) };
    (host_base, if host == 0 { base + tp2 } else { host_base + host })
}

type DenseSlot = std::sync::Arc<std::sync::Mutex<Option<Vec<DensePlan>>>>;

/// The coordinator side of one Spark push group: one rank per coordinator GPU
/// (one or two), each carrying its admitted TP2 expert halves and/or the
/// family's dense operands. Per-GPU TP2 plans, the in-process loopback
/// streams, and the entries the group driver appends after the Spark ranks.
pub(crate) struct CoordinatorPush {
    gpus: Vec<(Option<Tp2ExpertPlan>, std::net::TcpStream)>,
    host: String,
    snapshot: std::path::PathBuf,
    dense: DenseSlot,
}

/// Group rank of coordinator GPU `gpu` after `sparks` Spark ranks.
fn group_rank(sparks: usize, gpu: usize) -> usize { sparks + gpu }

impl CoordinatorPush {
    /// Metadata only, before any allocation: `gpus` coordinator ranks, with each
    /// GPU's admitted TP2 halves when `tp2` is (layers, per-GPU resident budgets).
    pub(crate) fn prepare(snapshot: &std::path::Path, catalog: Option<&OfficialV41Catalog>, gpus: usize,
        tp2: Option<(&[usize], Vec<usize>)>)
        -> Result<(Self, Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)> {
        ensure!((1..=2).contains(&gpus), "one or two coordinator GPUs");
        let (_, host) = cuteafd_loader::push_load::namespace(snapshot)?;
        let plans = match &tp2 {
            Some((layers, budgets)) => {
                ensure!(gpus == 2 && budgets.len() == 2, "coordinator TP2 needs two GPUs");
                let catalog = catalog.context("coordinator TP2 push needs the expert catalog")?;
                (0..2).map(|gpu| Tp2ExpertPlan::new(catalog, layers, gpu, budgets[gpu]).map(Some)).collect::<Result<Vec<_>>>()?
            }
            None => (0..gpus).map(|_| None).collect(),
        };
        let (mut driver, mut ranks) = (Vec::new(), Vec::new());
        for _ in 0..gpus {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let rank = std::net::TcpStream::connect(listener.local_addr()?)?;
            let (accepted, _) = listener.accept()?;
            driver.push(accepted);
            ranks.push(rank);
        }
        let dense: DenseSlot = Default::default();
        // TP2 entries are re-derived from headers; dense entries (set before the
        // offers are sent) carry no device state either.
        let append_tp2 = tp2.map(|(layers, budgets)| (layers.to_vec(), budgets));
        let append_dense = dense.clone();
        let append = Box::new(move |builder: &mut cuteafd_loader::push_load::builder::Builder,
            catalog: &OfficialV41Catalog, offers: &[cuteafd_transport::worker_selection::push::Offer]| -> Result<()> {
            ensure!(offers.len() == gpus, "coordinator push offer count differs");
            let dense = append_dense.lock().map_err(|_| anyhow::anyhow!("coordinator dense plan poisoned"))?;
            for (gpu, offer) in offers.iter().enumerate() {
                let rank = u32::try_from(offer.rank)?;
                let (base, host) = dense.as_ref().map_or((0, 0), |dense| (dense[gpu].base, dense[gpu].host_bytes));
                let tp2 = append_tp2.as_ref().map(|(layers, budgets)| Tp2ExpertPlan::new(catalog, layers, gpu, budgets[gpu]))
                    .transpose()?;
                let (host_base, total) = rank_bytes(base, tp2.as_ref().map_or(0, |p| p.layout.logical_bytes() as u64), host);
                if let Some(dense) = dense.as_ref() { (dense[gpu].append)(builder, rank, host_base)?; }
                if let Some(plan) = &tp2 { plan.append_at(builder, catalog, rank, base)?; }
                ensure!(offer.placement.is_empty() && offer.arena_bytes == total,
                    "coordinator push offer differs from its admitted arenas");
            }
            Ok(())
        });
        let ranks_spec = crate::shared::memory::push_exl3::CoordinatorRanks { ranks: gpus, append };
        Ok((Self { gpus: plans.into_iter().zip(ranks).collect(), host, snapshot: snapshot.to_path_buf(), dense },
            driver, ranks_spec))
    }

    /// Receives every coordinator GPU (two: each on its own thread) while
    /// `beside` runs on this thread. `dense`: each GPU's plan (published to the
    /// group driver before the offers) and destination (unaliased descriptors of
    /// operand allocations the caller keeps alive and unread until this returns
    /// and the group commits). `Ok(None)`: the whole group fell back. A dense
    /// planning error makes every coordinator GPU offer mount, so the whole
    /// group (Sparks and TP2 halves too) loads through the mount.
    #[allow(clippy::type_complexity)]
    pub(crate) fn receive_with<'a, D: PushDestination<Delivered = ()>, U>(self,
        devices: Vec<crate::shared::memory::device::Device<'a>>, catalog: Option<&OfficialV41Catalog>, sparks: usize,
        dense: Option<Result<(Vec<DensePlan>, Vec<D>)>>, beside: impl FnOnce() -> U)
        -> (Result<Option<Vec<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>>>>, U) {
        use cuteafd_ffi::push_ring::{DEFAULT_SLOTS, DEFAULT_SLOT_BYTES};
        let Self { gpus, host, snapshot, dense: slot } = self;
        if devices.len() != gpus.len() || dense.as_ref().is_some_and(|d| d.as_ref()
            .is_ok_and(|(p, d)| p.len() != gpus.len() || d.len() != gpus.len())) {
            return (Err(anyhow::anyhow!("coordinator push GPUs differ from their plans")), beside());
        }
        let world = sparks + gpus.len();
        let declined = match &dense {
            Some(Err(error)) => {
                tracing::info!(%error, "coordinator dense push planning failed; whole group uses mount");
                true
            }
            _ => false,
        };
        let (bases, destinations): (Vec<(u64, u64)>, Vec<Option<D>>) = match dense.and_then(Result::ok) {
            Some((plans, destinations)) => {
                let bases = plans.iter().map(|p| (p.base, p.host_bytes)).collect();
                match slot.lock() {
                    Ok(mut slot) => *slot = Some(plans),
                    Err(_) => return (Err(anyhow::anyhow!("coordinator dense plan poisoned")), beside()),
                }
                (bases, destinations.into_iter().map(Some).collect())
            }
            None => (vec![(0, 0); gpus.len()], (0..gpus.len()).map(|_| None).collect()),
        };
        let client = cuteafd_loader::push_load::PushClient::default();
        let (snapshot, namespace_host) = (&snapshot, &host);
        let receive = |gpu: usize, (tp2, mut stream, dense): (Option<Tp2ExpertPlan>, std::net::TcpStream, Option<D>)| {
            let rank = group_rank(sparks, gpu);
            let (base, host) = bases[gpu];
            let planned = (|| -> Result<_> {
                ensure!(!declined, "coordinator dense operands were not planned");
                let Some(plan) = &tp2 else { return Ok(None) };
                let catalog = catalog.context("coordinator TP2 push needs the expert catalog")?;
                let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
                let checked = plan.checked(catalog, &root, world, rank)?;
                ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
                    "coordinator push source is not a SparkNest shard");
                Ok(Some(checked))
            })();
            if let Err(error) = &planned { tracing::info!(gpu, %error, "coordinator push unavailable; offering mount"); }
            let tp2_bytes = tp2.as_ref().map_or(0, |p| p.layout.logical_bytes() as u64);
            let (host_base, arena_bytes) = rank_bytes(base, tp2_bytes, host);
            let offer = planned.as_ref().ok().map(|_| cuteafd_transport::worker_selection::push::Offer {
                rank, host: namespace_host.clone(), arena_bytes, placement: Vec::new() });
            let credentials = cuteafd_transport::worker_selection::push::offer(&mut stream, offer)?;
            let Some(credentials) = credentials else { return Ok(None) };
            let checked = planned?;
            let tp2 = match (tp2, checked.as_ref()) {
                (Some(plan), Some(checked)) => Some((catalog.context("expert catalog")?, plan, checked)),
                _ => None,
            };
            receive_rank(devices[gpu], tp2, dense.map(|destination| DenseRank { destination, base }), rank,
                &client, &mut stream, credentials, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)
        };
        let inputs: Vec<_> = gpus.into_iter().zip(destinations).map(|((tp2, stream), dense)| (tp2, stream, dense)).collect();
        let (results, value) = if inputs.len() == 2 {
            let [first, second]: [_; 2] = match inputs.try_into() { Ok(pair) => pair, Err(_) => unreachable!() };
            // SAFETY: each GPU's input (its plan, stream and dense destination) is used by
            // that GPU's thread only; the dense descriptors refer to allocations the caller
            // keeps alive and unread until this returns. Each GPU result is a freshly
            // committed, drained rank arena and its views, unaliased (every Rc clone stays
            // inside that GPU's Vec); a fallback returns None after dropping its arena.
            let (pair, value) = unsafe { cuteafd_ffi::synchronized_load::load_pair_with([first, second], receive, beside) };
            (pair.map(Vec::from), value)
        } else {
            let mut inputs = inputs;
            let result = inputs.pop().context("one coordinator GPU").and_then(|input| receive(0, input)).map(|r| vec![r]);
            (result, beside())
        };
        let pushed = results.and_then(|results| {
            let committed = results.iter().filter(|r| r.is_some()).count();
            match committed {
                0 => Ok(None),
                n if n == results.len() => Ok(Some(results.into_iter().map(|r| r.unwrap_or_default()).collect())),
                _ => anyhow::bail!("coordinator push GPUs disagree on the group commit"),
            }
        });
        (pushed, value)
    }
}
