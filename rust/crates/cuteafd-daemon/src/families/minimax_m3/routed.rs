//! Routed experts: RTX-resident layers (TP1 NVFP4 on the coordinator) and the
//! Spark ranks' layers over one shared exchange link ([`SparkLink`] and its
//! [`crate::shared::spark_waves::Waves`]).
use super::{MiniMaxError, execution::ExpertExecutor, native::Native};
use crate::shared::{experts::{fp8::Fp8Experts, staging::RouteStaging}, memory::device::Device,
    prefill_depth::{Depth, PrefillDepth}, spark_intake::SparkLink, spark_waves::WaveTicket};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::serving_capacity::prefill_waves::MAX_WAVES;
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertV2SourceKind, TcpTransportConfig,
    expert::SparkWorld};
use std::{collections::VecDeque, ffi::c_void, net::SocketAddr, sync::Arc};

const ROW: usize = 6144 * 2;
const TOPK: usize = 4;

/// MiniMax serves its Spark experts from an RTX PRO 6000 coordinator, whose replies land in GPU
/// memory. A GeForce coordinator (no dma-buf landing) is refused by name, never served on a
/// fallback path.
pub(crate) const RTX_5090_UNSUPPORTED: &str = "MiniMax on an RTX 5090 is not supported on this engine yet";

/// Refuses a GeForce coordinator ([`RTX_5090_UNSUPPORTED`]), named by its CUDA device name.
pub(super) fn require_device_landing(library: &NativeLibrary, device: i32) -> Result<()> {
    let info = library.cuda_device_info(device)?;
    let bytes = info.name.iter().map(|&c| c as u8).collect::<Vec<_>>();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    refuse_geforce(device, &String::from_utf8_lossy(&bytes[..end]))
}

fn refuse_geforce(device: i32, name: &str) -> Result<()> {
    ensure!(!name.contains("GeForce"), "{RTX_5090_UNSUPPORTED}: coordinator GPU {device} is {name}; run MiniMax \
        on an RTX PRO 6000 coordinator");
    Ok(())
}

/// The family's protocol request builder (identity, layer, rows, routes, wire, kind).
pub(super) type Build=fn(u64,usize,usize,Vec<ExpertProtocolV2RouteEntry>,bytes::Bytes,ExpertV2SourceKind)
    ->Result<ExpertProtocolV2Request>;

/// The widest wave the link carries (`rows` rows of routed layer `layer`), built from a zeroed
/// staging arena with every row routed to four distinct experts: it opens each rank's session,
/// sizes the registered rings, the egress and every return slot's staging, and is exchanged
/// once before the engine runs.
fn bootstrap_request(layer: usize, rows: usize, build: Build) -> Result<ExpertProtocolV2Request> {
    let layout = RouteStaging::new(rows, TOPK, ROW)?;
    let mut staged = vec![0u8; layout.bytes];
    for (route, word) in staged[..layout.route_bytes].chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&((route % 128) as u32).to_le_bytes());
    }
    build(1, layer, rows, layout.routes(&staged)?, bytes::Bytes::copy_from_slice(layout.wire(&staged)?),
        ExpertV2SourceKind::Prefill)
}

/// A sent Spark wave: routed layer `layer`'s `rows` rows from slot `slot`'s rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Wave { pub layer: usize, pub slot: usize, pub rows: usize }

/// Sent waves not yet collected, oldest first. A step collects its waves in dispatch order with
/// at most `waves` in flight, which is exactly what the link's exchange admits (one return slot per
/// wave of its current cut, landed oldest first): the order is checked here before the exchange sees it.
pub(super) struct WaveOrder<T> { waves: usize, sent: VecDeque<(Wave, T)> }

impl<T> WaveOrder<T> {
    pub fn new(waves: usize) -> Self { Self { waves, sent: VecDeque::new() } }
    pub fn is_empty(&self) -> bool { self.sent.is_empty() }
    /// Follows the link's re-cut return slots (#762), at rest only.
    pub fn resize(&mut self, waves: usize) -> Result<()> {
        ensure!(self.sent.is_empty(), MiniMaxError::Invalid("Spark return slots re-cut with a wave in flight"));
        self.waves = waves;
        Ok(())
    }
    /// Refuses a wave past the link's return slots.
    pub fn admit(&self) -> Result<()> {
        ensure!(self.sent.len() < self.waves, MiniMaxError::Invalid("Spark waves in flight exceed the link's return slots"));
        Ok(())
    }
    pub fn push(&mut self, wave: Wave, ticket: T) -> Result<()> {
        self.admit()?;
        self.sent.push_back((wave, ticket));
        Ok(())
    }
    /// The oldest wave's ticket, which must be `wave`'s.
    pub fn pop(&mut self, wave: Wave) -> Result<T> {
        let (sent, ticket) = self.sent.pop_front().context("MiniMax Spark collect without a sent wave")?;
        ensure!(sent == wave, MiniMaxError::Invalid("Spark wave collected out of dispatch order"));
        Ok(ticket)
    }
    pub fn clear(&mut self) { self.sent.clear(); }
}

/// The Spark experts' exchange: one link (rank connections, intake return slots, request
/// staging and the exchange worker) for decode and prefill alike, with up to `slots` waves in
/// flight (each prefill wave's while other rows' GPU work runs). Its return slots are cut from
/// one reserve range per rank, re-cut at rest to each prefill call's depth (#762). Every send,
/// land and reduce is queued on the execution's one stream; this thread never waits for the wire.
pub(super) struct Spark<'a> {
    /// `None` once released after a failed step. Its terminal-owned intake retains the library.
    link: Option<SparkLink<'a>>,
    device: Device<'a>,
    build: Build,
    /// Rows of the admitted unit: the widest wave any cut carries.
    rows: usize,
    /// Admitted return slots: the deepest cut.
    slots: usize,
    /// The current cut's waves in flight.
    order: WaveOrder<WaveTicket>,
    /// The last request identity (the bootstrap's is 1).
    identity: u64,
    /// Quiesced after a failed step: the link never sends again.
    terminal: bool,
}

impl Drop for Spark<'_> {
    fn drop(&mut self) {
        // The execution drains its stream before every return (a failed step quiesces the
        // link first), so no queued work reads the link's planes or staging here; they are
        // released on their device.
        let link = self.link.take();
        crate::shared::decode_graph::fatal_drain(self.device.run(|| { drop(link); Ok(()) }),
            "MiniMax Spark link device-scoped release");
    }
}

impl<'a> Spark<'a> {
    /// Connects up to `slots` return slots of up to `rows` rows, cut from one `reserve`-row range per
    /// rank (#762), bootstraps with the widest wave (routed layer `first`) and exchanges it once on
    /// one admitted unit, so connections, rings and the ranks' expert programs are warm before the
    /// engine runs. A static link's reserve is exactly its slots' units, and it keeps that cut.
    #[allow(clippy::too_many_arguments)]
    pub fn connect(library_owner: &'a Arc<NativeLibrary>, device: i32, world: SparkWorld, peers: &[SocketAddr],
        rows: usize, config: TcpTransportConfig, first: usize, build: Build, slots: usize, reserve: usize) -> Result<Self> {
        let library = library_owner.as_ref();
        let device = Device { library, id: device };
        require_device_landing(library, device.id)?;
        // The compact reducer sums rows of the process expert geometry.
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::MINIMAX_M3,
            "MiniMax Spark helpers require the process expert geometry before the link");
        ensure!((1..=4096).contains(&rows) && (1..=MAX_WAVES as usize).contains(&slots)
            && (rows..=slots * rows).contains(&reserve), MiniMaxError::Invalid("Spark link rows, slots or reserve"));
        let warm = bootstrap_request(first, rows, build)?;
        let mut link = device.run(|| {
            library.v41_compact_reducer()?.require_plane_reduction(u32::try_from(peers.len())?)?;
            // Each wave is at most one unit's rows. Several slots are cut from one reserve range per
            // rank, so the prefill depth can change between calls within them (#762); one slot is a
            // plain single-slot link, as Step-5's.
            let mut link = if slots > 1 {
                SparkLink::new_reserve(library, world, peers, u32::try_from(rows)?, config, ROW, slots,
                    u32::try_from(rows)?, reserve)?
            } else { SparkLink::new(library, world, peers, u32::try_from(rows)?, config, ROW)? };
            link.enable_terminal_ownership(Arc::clone(library_owner))?;
            link.bootstrap(&warm)?;
            Ok(link)
        })?;
        ensure!(link.world_size() == peers.len() && link.slots() == slots,
            "MiniMax Spark link differs from its admitted world or slots");
        ensure!(link.installed_grouping()?.len() == peers.len(),
            "MiniMax Spark bootstrap requires installed grouping from every rank");
        // One admitted unit takes the widest wave until the first prefill call re-cuts to its depth.
        let cut = link.exchange()?.geometry();
        if cut.rows < rows { device.run(|| link.set_geometry(1, rows))?; }
        let waves = link.waves();
        device.run(|| warm_up(library, &mut link, warm, rows))?;
        Ok(Self { link: Some(link), device, build, rows, slots, order: WaveOrder::new(waves), identity: 1, terminal: false })
    }

    /// Return slots the link admits in flight at its deepest cut.
    pub fn slots(&self) -> usize { self.slots }
    pub fn world_size(&self) -> Option<usize> { self.link.as_ref().map(SparkLink::world_size) }

    /// No wave is in flight (between steps).
    fn idle(&self) -> Result<()> {
        ensure!(self.order.is_empty(), MiniMaxError::Invalid("Spark wave in flight outside its step"));
        Ok(())
    }

    /// Re-cuts the return slots into `waves` slots of `rows` rows, at rest (#762): offsets only;
    /// nothing is allocated, registered or reconnected. An unchanged cut is left alone (a one-slot
    /// link has none to change).
    fn recut(&mut self, waves: usize, rows: usize) -> Result<()> {
        self.idle()?;
        let device = self.device;
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        let cut = link.exchange()?.geometry();
        if (cut.waves, cut.rows) != (waves, rows) { device.run(|| link.set_geometry(waves, rows))?; }
        self.order.resize(waves)
    }

    /// Makes the current cut hold a step's `waves` waves of up to `rows` rows and checks its request
    /// staging holds the widest. A one-wave step wider than the last prefill call's cut runs on one
    /// admitted unit (re-cut at rest); a unit call's depth was cut at its start, so a multi-wave step
    /// the cut cannot hold is refused.
    fn fit(&mut self, waves: usize, rows: usize) -> Result<()> {
        ensure!((1..=self.rows).contains(&rows), MiniMaxError::Invalid("Spark wave rows exceed its return slot"));
        let cut = self.link.as_mut().context("MiniMax Spark link released")?.exchange()?.geometry();
        if waves > cut.waves || rows > cut.rows {
            ensure!(waves == 1, MiniMaxError::Invalid("Spark waves in flight exceed the link's return slots"));
            self.recut(1, self.rows)?;
        }
        let bytes = RouteStaging::new(rows, TOPK, ROW)?.bytes;
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        ensure!(link.exchange()?.fits(bytes), MiniMaxError::Invalid("Spark exchange staging is narrower than a prefill unit"));
        Ok(())
    }

    /// A unit call's boundary (#762): reports every finished prefill call's pace to `depth` (no host
    /// wait: a call whose last wave has not landed stays for a later boundary), takes this call's
    /// depth (sequence `key` from row `start`, `rows` rows), re-cuts the return slots to it at rest
    /// and starts summing this call's waves. Returns the depth the call runs.
    pub fn start_prefill(&mut self, depth: &PrefillDepth, key: u64, start: usize, rows: usize) -> Result<Depth> {
        ensure!(!self.terminal, MiniMaxError::Invalid("Spark link stopped after a failed step"));
        self.idle()?;
        let device = self.device;
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        depth.report(device.run(|| link.exchange()?.take_prefill())?);
        let next = depth.start(key, start, rows);
        ensure!(next.waves <= self.slots && next.unit_rows <= self.rows,
            "MiniMax prefill depth {next:?} outside the link's {} admitted slots of {} rows", self.slots, self.rows);
        self.recut(next.waves, next.unit_rows)?;
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        device.run(|| link.exchange()?.begin_prefill(next.call))?;
        Ok(next)
    }

    /// The unit call's last wave was sent (or the call failed): later waves are not its.
    pub fn stop_prefill(&mut self) {
        if let Some(Ok(exchange)) = self.link.as_mut().map(SparkLink::exchange) { exchange.stop_prefill(); }
    }

    /// Queues `wave` into the next return slot: its route ids, gate weights and BF16 expert input
    /// rows copy into the slot's staging on `stream`, and the link's worker builds, posts and
    /// lands it. Never waits on the host.
    ///
    /// # Safety
    /// `stream` is the execution's live stream; `ids`, `weights` and `input` are live device
    /// buffers of the wave's extents, written in `stream` order before this call.
    pub unsafe fn send(&mut self, wave: Wave, kind: ExpertV2SourceKind, ids: CuteafdDeviceBuffer,
        weights: CuteafdDeviceBuffer, input: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        ensure!(!self.terminal, MiniMaxError::Invalid("Spark link stopped after a failed step"));
        ensure!((1..=self.rows).contains(&wave.rows), MiniMaxError::Invalid("Spark wave rows exceed its return slot"));
        self.order.admit()?;
        let identity = self.identity.checked_add(1).ok_or(MiniMaxError::Overflow("Spark wave identity"))?;
        let layout = RouteStaging::new(wave.rows, TOPK, ROW)?;
        let (device, build, Wave { layer, rows, .. }) = (self.device, self.build, wave);
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        let ticket = device.run(|| {
            // SAFETY: forwarded from this function's contract; the copies read the sources in
            // stream order, so later stream work may rewrite them.
            unsafe { link.exchange()?.send(rows, layout, [ids, weights, input], stream,
                Box::new(move |routes, wire| build(identity, layer, rows, routes, wire, kind))) }
        })?;
        self.identity = identity;
        self.order.push(wave, ticket)
    }

    /// Lands the oldest wave, which must be `wave`, and queues `output = sum of its rank planes`
    /// on `stream`, which marks its return slot read.
    ///
    /// # Safety
    /// `stream` is the execution's live stream; `output` is a live `[rows, 6144]` BF16 buffer
    /// ordered on it, disjoint from the link's planes.
    pub unsafe fn land(&mut self, wave: Wave, output: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let ticket = self.order.pop(wave)?;
        let (device, slot, rows) = (self.device, ticket.slot(), ticket.rows());
        let link = self.link.as_mut().context("MiniMax Spark link released")?;
        device.run(|| {
            let exchange = link.exchange()?;
            exchange.land(ticket, stream)?;
            // SAFETY: `land` ordered the slot's planes on this stream; the output is the
            // caller's disjoint scratch view, live through the stream's work.
            unsafe { exchange.reduce(slot, std::ptr::null(), output.ptr.cast(), rows, stream) }
        })
    }

    /// Terminal teardown after a failed step, before its stream drains: closes the exchange,
    /// which releases every wait the stream still holds (a drain first could sit out the
    /// worker's no-progress timeout), then quiesces the ranks' queue pairs.
    fn quiesce(&mut self) -> Result<()> {
        self.order.clear();
        if self.terminal { return Ok(()); }
        let device = self.device;
        if let Some(link) = &mut self.link { device.run(|| link.terminal_quiesce())?; }
        self.terminal = true;
        Ok(())
    }

    /// Terminal teardown once the stream drained: the intake's copies, then the payloads the
    /// quiesced transport held, then the link. A link whose quiescence failed is retained.
    fn release(&mut self) -> Result<()> {
        if !self.terminal { return Ok(()); }
        let device = self.device;
        device.run(|| match &self.link { Some(link) => link.terminal_drain_copies(), None => Ok(()) })?;
        device.run(|| {
            if let Some(link) = &mut self.link { link.terminal_release()?; }
            drop(self.link.take());
            Ok(())
        })
    }
}

/// One widest wave landed and drained on a temporary stream before the engine runs.
fn warm_up(library: &NativeLibrary, link: &mut SparkLink<'_>, warm: ExpertProtocolV2Request, rows: usize) -> Result<()> {
    let started = std::time::Instant::now();
    let warm_stream = crate::shared::spark_intake::WarmStream::new(library)?;
    let exchange = link.exchange()?;
    let wave = exchange.send_request(warm)?;
    exchange.land_drained(wave, warm_stream.raw())
        .with_context(|| format!("warming MiniMax Spark experts with {rows} rows"))?;
    warm_stream.finish()?;
    tracing::info!(waves = exchange.waves(), ranks = exchange.ranks(), rows,
        elapsed_ms = started.elapsed().as_millis() as u64, "MiniMax Spark exchange warm");
    Ok(())
}

/// A dispatched wave not yet collected: a local one keeps its operand views
/// (its rows' scratch) until it runs at collect.
enum Pending {
    Local { layer:usize,rows:usize,index:usize,ids:CuteafdDeviceBuffer,weights:CuteafdDeviceBuffer,input:CuteafdDeviceBuffer },
    Spark(Wave),
}
pub(super) struct Routed<'a> {
    local: Option<Fp8Experts<'a>>,
    spark: Option<Spark<'a>>,
    /// Per slot, the dispatched wave not yet collected.
    pending: Vec<Option<Pending>>,
    kind:ExpertV2SourceKind,
    poisoned:bool,
}
impl<'a> Routed<'a> {
    pub fn new(local:Option<Fp8Experts<'a>>,spark:Option<Spark<'a>>)->Result<Self> {
        ensure!(local.is_some() || spark.is_some(),MiniMaxError::Unsupported("missing routed expert backend"));
        if let Some(local)=&local {
            let info=local.module.info();
            ensure!(local.tp==1 && !local.wire_input() && info.hidden==6144 && info.experts==128
                && info.topk==4 && info.intermediate==3072 && info.slice==3072
                && info.weights==cuteafd_ffi::fp8_moe::Fp8MoeWeights::Nvfp4
                && info.activation==cuteafd_ffi::fp8_moe::Fp8MoeActivation::SwigluOai && info.swiglu_limit==7.0,
                MiniMaxError::Unsupported("RTX experts require MiniMax TP1 NVFP4 W4A4"));
        }
        // The serve and golden rank gates admit these worlds (a rank policy derived from the
        // shipped packages owns them); any other link world is a bug.
        if let Some(spark)=&spark { ensure!(spark.world_size().is_some_and(|ranks|matches!(ranks,2|3|4|6)),
            MiniMaxError::Unsupported("Spark experts require TP2, TP3, TP4 or TP6")); }
        // Local layers run on the stream at collect, from any slot; each Spark wave in flight takes
        // one of the link's admitted return slots.
        let slots=spark.as_ref().map_or(MAX_WAVES as usize,Spark::slots);
        Ok(Self{local,spark,pending:(0..slots).map(|_|None).collect(),kind:ExpertV2SourceKind::Decode,poisoned:false})
    }
}
impl<'a> ExpertExecutor<'a> for Routed<'a> {
    fn begin(&mut self,routed:usize,slots:usize,rows:usize,kind:ExpertV2SourceKind)->Result<()> {
        ensure!(!self.poisoned,MiniMaxError::Invalid("previous expert failure; restart required"));
        ensure!(self.pending.iter().all(Option::is_none),MiniMaxError::Invalid("pending expert wave at step start"));
        ensure!((1..=self.pending.len()).contains(&slots),MiniMaxError::Invalid("expert slots"));
        ensure!(matches!(kind,ExpertV2SourceKind::Decode|ExpertV2SourceKind::Prefill),
            MiniMaxError::Unsupported("expert source kind"));
        self.kind=kind;
        let local=self.local.as_ref().map_or(0,|l|l.layers.len());
        routed.checked_sub(local).context("MiniMax expert inventory exceeds routed layers")?;
        if let Some(spark)=&mut self.spark { spark.idle()?; spark.fit(slots,rows)?; }
        Ok(())
    }
    fn start_prefill(&mut self,depth:&PrefillDepth,key:u64,start:usize,rows:usize)->Result<Option<Depth>> {
        ensure!(!self.poisoned,MiniMaxError::Invalid("previous expert failure; restart required"));
        self.spark.as_mut().map(|spark|spark.start_prefill(depth,key,start,rows)).transpose()
    }
    fn stop_prefill(&mut self) { if let Some(spark)=&mut self.spark { spark.stop_prefill(); } }
    fn local(&self,layer:usize)->bool { self.local.as_ref().is_some_and(|l|l.index_of(layer).is_ok()) }
    fn dispatch(&mut self,layer:usize,slot:usize,rows:usize,ids:CuteafdDeviceBuffer,weights:CuteafdDeviceBuffer,
        input:CuteafdDeviceBuffer,native:&Native<'a>)->Result<()> {
        ensure!(self.pending.get(slot).is_some_and(Option::is_none),MiniMaxError::Invalid("overlapping expert waves"));
        let pending=if let Some(index)=self.local.as_ref().and_then(|l|l.index_of(layer).ok()) {
            Pending::Local{layer,rows,index,ids,weights,input}
        } else {
            ensure!(rows>0 && ids.bytes>=rows*16 && weights.bytes>=rows*16 && input.bytes>=rows*ROW,
                "MiniMax Spark expert operand extent");
            native.check_device(&[ids,weights,input])?;
            let spark=self.spark.as_mut().context("MiniMax routed layer has neither local experts nor Spark transport")?;
            let wave=Wave{layer,slot,rows};
            // SAFETY: the router and norm wrote these row views on the execution's stream,
            // whose staging copies read them in order; the scratch outlives the step's work.
            unsafe { spark.send(wave,self.kind,ids,weights,input,native.raw())? };
            Pending::Spark(wave)
        };
        self.pending[slot]=Some(pending);
        Ok(())
    }
    fn collect(&mut self,layer:usize,slot:usize,rows:usize,output:CuteafdDeviceBuffer,native:&Native<'a>)->Result<()> {
        match self.pending.get_mut(slot).and_then(Option::take).context("MiniMax expert collect without dispatch")? {
            Pending::Local{layer:expected,rows:count,index,ids,weights,input}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("local expert wave identity"));
                let owner=self.local.as_ref().context("MiniMax local owner")?;
                ensure!(rows>0 && ids.bytes>=rows*16 && weights.bytes>=rows*16
                    && input.bytes>=rows*ROW && output.bytes>=rows*ROW, "MiniMax local expert operand extent");
                native.check_device(&[ids,weights,input,output])?;
                // SAFETY: the weight owner and every operand's owner outlive the
                // stream's work; the shared expert binding verifies its package.
                unsafe { owner.run_bf16(index,rows,input.ptr,ids.ptr,weights.ptr,output.ptr,native.raw()) }
            }
            Pending::Spark(wave)=>{
                ensure!((layer,rows)==(wave.layer,wave.rows),MiniMaxError::Invalid("Spark expert wave identity"));
                ensure!(rows>0 && output.bytes>=rows*ROW,"MiniMax Spark expert output extent");
                native.check_device(&[output])?;
                let spark=self.spark.as_mut().context("MiniMax Spark owner")?;
                // SAFETY: the output is this slot's rows' scratch view on the execution's stream,
                // disjoint from the link's planes and live through the step's work.
                unsafe { spark.land(wave,output,native.raw()) }
            }
        }
    }
    fn finish(&mut self)->Result<()> {
        ensure!(self.pending.iter().all(Option::is_none),MiniMaxError::Invalid("uncollected expert wave at step end"));
        if let Some(spark)=&self.spark {
            spark.idle()?;
            // After the step's drain: a wire failure released its waits over garbage planes.
            // Serving's token selector checks too; golden and diagnostics read results directly.
            crate::shared::spark_waves::healthy()?;
        }
        Ok(())
    }
    fn abort(&mut self) {
        self.poisoned=true;
        self.pending.iter_mut().for_each(|p|{p.take();});
        if let Some(spark)=&mut self.spark {
            if let Err(error)=spark.quiesce() {
                tracing::error!(error=%format!("{error:#}"),"MiniMax Spark link quiescence failed; its owners stay retained");
            }
        }
    }
    fn release(&mut self) {
        if let Some(spark)=&mut self.spark {
            if let Err(error)=spark.release() {
                tracing::error!(error=%format!("{error:#}"),"MiniMax Spark link release failed; its owners stay retained");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::dense_attention::{RouterBiasOrder, SigmoidRouter};
    #[test] fn minimax_router_selects_with_bias_and_weights_without_bias() {
        let logits=[-1.0f32,0.0,1.0,2.0,3.0];
        let bias=[10.0f32,0.0,0.0,0.0,0.0];
        // The minimax_route program's arithmetic: plain-sum renormalization, routed scale 2.
        let router=SigmoidRouter { topk:4, factor:2.0, epsilon:0.0, bias_order:RouterBiasOrder::GatherOriginal };
        let selected=router.evaluate(&logits,&bias).unwrap();
        assert_eq!(selected.iter().map(|(id,_)|*id).collect::<Vec<_>>(),[0,4,3,2]);
        let scores:Vec<_>=logits.iter().map(|x|1.0/(1.0+(-x).exp())).collect();
        let total:f32=selected.iter().map(|(id,_)|scores[*id]).sum();
        for (id,weight) in selected { assert!((weight-2.0*scores[id]/total).abs()<1e-6); }
    }

    /// The link is the shared exchange: sends and lands through the link's `Waves`, terminal
    /// ownership before the bootstrap before the grouping check, no inline exchange, and the
    /// terminal teardown's order (quiesce; then copies, payloads, the link).
    #[test]
    fn minimax_routed_owner_sends_and_lands_on_the_links_exchange() {
        let source = include_str!("routed.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("link: Option<SparkLink<'a>>"));
        assert!(source.contains("link.enable_terminal_ownership(Arc::clone(library_owner))"));
        assert!(source.contains("link.exchange()?.send("));
        assert!(source.contains("exchange.land(ticket, stream)"));
        assert!(source.contains("exchange.reduce(slot, std::ptr::null(), output.ptr.cast(), rows, stream)"));
        for banned in [["block", "_on"].concat(), ["Spark", "DeviceLink"].concat(), ["Lane", "Links"].concat(),
            [".arm", "()"].concat(), [".expect", "(remote"].concat(), ["record_", "wave"].concat()] {
            assert!(!source.contains(&banned), "{banned}");
        }
        let connect = source.split("pub fn connect").nth(1).unwrap().split("pub fn slots").next().unwrap();
        let order = ["SparkLink::new_reserve(", "link.enable_terminal_ownership(", "link.bootstrap(&warm)",
            "link.installed_grouping()", "link.set_geometry(1, rows)", "warm_up(library, &mut link, warm, rows)"]
            .map(|needle| connect.find(needle).unwrap_or_else(|| panic!("{needle}")));
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        assert!(connect.find("require_device_landing(").unwrap() < order[0], "a GeForce card is refused before the link");
        let release = source.split("fn release(&mut self) -> Result<()>").nth(1).unwrap().split("fn warm_up").next().unwrap();
        let steps = ["link.terminal_drain_copies()", "link.terminal_release()?", "self.link.take()"]
            .map(|needle| release.find(needle).unwrap_or_else(|| panic!("{needle}")));
        assert!(steps.windows(2).all(|w| w[0] < w[1]), "{steps:?}");
        assert!(source.contains("link.terminal_quiesce()"));
    }

    fn request(identity:u64,layer:usize,rows:usize,routes:Vec<ExpertProtocolV2RouteEntry>,wire:bytes::Bytes,
        kind:ExpertV2SourceKind)->Result<ExpertProtocolV2Request> {
        super::super::serve::request(identity,layer,rows,routes,wire,kind)
    }

    /// The bootstrap is the widest wave a link of `rows` carries: the request a real wave of those
    /// rows builds from its staging (rows, routes, BF16 wire), opening every rank's session.
    #[test]
    fn minimax_widest_bootstrap_matches_real_request_storage() -> Result<()> {
        use cuteafd_transport::{expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertV2Dtype};
        for rows in [1, 64, 2752, 4096] {
            let warm = bootstrap_request(3, rows, request)?;
            let layout = RouteStaging::new(rows, TOPK, ROW)?;
            assert_eq!((warm.header.layer_id, warm.header.row_count as usize, warm.header.hidden_dim), (3, rows, 6144));
            assert_eq!((warm.header.hidden_dtype, warm.header.request_id), (ExpertV2Dtype::Bf16, 1));
            assert_eq!(warm.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16);
            assert_eq!((warm.routes.len(), warm.hidden_payload.len()), (rows * TOPK, layout.wire_bytes));
            assert_eq!(warm.routes.len() * 8 + warm.hidden_payload.len(), layout.bytes);
            assert!(warm.routes.iter().enumerate().all(|(i, route)| route.row_index as usize == i / TOPK
                && route.expert_id as usize == i % 128 && route.gate_weight == 0.0));
            assert!(warm.rows.iter().all(|row| row.route_count == 4 && row.source_kind == ExpertV2SourceKind::Prefill));
            // The registered egress slot this request sizes holds a whole wave's staging.
            let slot = ExpertGeometry::MINIMAX_M3.compact_wire_bytes(rows as u32, true).unwrap().0;
            assert!(layout.bytes <= slot, "{rows} rows: staging {} > slot {slot}", layout.bytes);
        }
        assert!(bootstrap_request(2, 64, request).is_err(), "dense layers carry no Spark wave");
        assert!(bootstrap_request(60, 64, request).is_err());
        assert!(bootstrap_request(3, 4097, request).is_err());
        Ok(())
    }

    #[test]
    fn minimax_refuses_geforce_coordinators_by_name() {
        let refused = refuse_geforce(0, "NVIDIA GeForce RTX 5090").unwrap_err().to_string();
        assert!(refused.starts_with(RTX_5090_UNSUPPORTED) && refused.contains("NVIDIA GeForce RTX 5090"), "{refused}");
        assert!(refuse_geforce(1, "NVIDIA GeForce RTX 5080").is_err());
        for name in ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition"] {
            refuse_geforce(0, name).unwrap();
        }
    }

    /// A unit call's boundary (#762) runs in the shared order, at rest: finished calls' paces reported (no host
    /// wait), this call's depth taken, the return slots re-cut to it, then its waves flagged as the call's. A
    /// one-wave step wider than the cut runs on one admitted unit; a re-cut moves the wave order's slots, at rest.
    #[test]
    fn minimax_prefill_calls_report_take_recut_and_flag_in_order() -> Result<()> {
        let source = include_str!("routed.rs").split("#[cfg(test)]").next().unwrap();
        let start = source.split("pub fn start_prefill").nth(1).unwrap().split("pub fn stop_prefill").next().unwrap();
        let order = ["self.idle()?", "depth.report(device.run(|| link.exchange()?.take_prefill())?)",
            "depth.start(key, start, rows)", "self.recut(next.waves, next.unit_rows)?",
            "link.exchange()?.begin_prefill(next.call)"].map(|needle| start.find(needle).unwrap_or_else(|| panic!("{needle}")));
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        let recut = source.split("fn recut(").nth(1).unwrap().split("fn fit(").next().unwrap();
        for needle in ["self.idle()?", "link.set_geometry(waves, rows)", "self.order.resize(waves)"] {
            assert!(recut.contains(needle), "{needle}");
        }
        let fit = source.split("fn fit(").nth(1).unwrap().split("pub fn start_prefill").next().unwrap();
        assert!(fit.contains("ensure!(waves == 1,") && fit.contains("self.recut(1, self.rows)?"));
        let mut order = WaveOrder::new(1);
        order.push(Wave { layer: 3, slot: 0, rows: 64 }, 0)?;
        assert!(order.resize(4).is_err(), "never with a wave in flight");
        order.pop(Wave { layer: 3, slot: 0, rows: 64 })?;
        order.resize(3)?;
        for slot in 0..3 { order.push(Wave { layer: 3, slot, rows: 64 }, slot)?; }
        assert!(order.admit().is_err(), "three return slots");
        Ok(())
    }

    #[test]
    fn minimax_wave_order_admits_link_slots_and_lands_oldest_first() -> Result<()> {
        let wave = |layer, slot| Wave { layer, slot, rows: 64 };
        let mut order = WaveOrder::new(2);
        order.push(wave(3, 0), 1)?;
        order.push(wave(3, 1), 2)?;
        assert!(order.admit().is_err() && order.push(wave(4, 0), 3).is_err(), "two return slots");
        assert!(order.pop(wave(3, 1)).is_err(), "a younger wave never lands first");
        let mut order = WaveOrder::new(2);
        order.push(wave(3, 0), 1)?;
        order.push(wave(3, 1), 2)?;
        assert_eq!(order.pop(wave(3, 0))?, 1);
        order.push(wave(4, 0), 3)?;
        assert_eq!((order.pop(wave(3, 1))?, order.pop(wave(4, 0))?), (2, 3));
        assert!(order.is_empty() && order.pop(wave(4, 1)).is_err());
        Ok(())
    }
}
