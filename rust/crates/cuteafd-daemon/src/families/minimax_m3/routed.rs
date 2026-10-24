use super::{MiniMaxError, execution::ExpertExecutor};
use crate::shared::{experts::fp8::Fp8Experts,minimax_ffi,prefill_lanes::LaneLinks,
    spark_intake::{self,IntakePreference,SparkDeviceLink}};

/// Whether this GPU's Spark partials land in pinned host planes (no dma-buf
/// landing, as on GeForce cards): the ledger and the links take the same answer.
pub(super) fn host_landing(library:&NativeLibrary)->Result<bool> {
    Ok(!spark_intake::preferred_device_exchange_available(library,IntakePreference::Probed)?)
}
use cuteafd_loader::serving_capacity::prefill_lanes::MAX_PREFILL_LANES;
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{NativeLibrary,minimax::{Buffer,Commands}};
use cuteafd_transport::{ExpertProtocolV2Request,ExpertProtocolV2RouteEntry,ExpertV2SourceKind,TcpTransportConfig,
    expert::SparkWorld};
use std::{net::SocketAddr,rc::Rc};

const ROW:usize=6144*2;
const TOPK:usize=4;

/// The family's protocol request builder (identity, layer, rows, routes, wire, kind).
pub(super) type Build=fn(u64,usize,usize,Vec<ExpertProtocolV2RouteEntry>,bytes::Bytes,ExpertV2SourceKind)
    ->Result<ExpertProtocolV2Request>;

/// Spark expert transport: the device-driven exchange, one link per prefill
/// lane (each its own proxy, mailbox, egress, planes and rank connections,
/// so every lane's wave can be in flight at once). Partials land in GPU
/// memory where the NIC can write it (dma-buf), otherwise in pinned host
/// planes the reducer reads in place (GeForce cards have no dma-buf); either
/// way the stream never drains, and the inference thread never parses or
/// copies, per routed layer.
pub(super) struct Spark<'a>(LaneLinks<Rc<SparkDeviceLink<'a>>>);

impl<'a> Spark<'a> {
    /// `first` is a Spark layer for the warm wave that sizes every rank's rings;
    /// `host` is [`host_landing`].
    #[allow(clippy::too_many_arguments)]
    pub fn connect(library:&'a NativeLibrary,device:i32,world:SparkWorld,peers:&[SocketAddr],rows:usize,
        config:TcpTransportConfig,first:usize,build:Build,lanes:usize,host:bool)->Result<Self> {
        if host { tracing::info!("MiniMax Spark partials land in pinned host planes (no GPU landing)"); }
        Ok(Self(LaneLinks::connect(lanes,|lane| {
            let routes=(0..rows*TOPK).map(|i| ExpertProtocolV2RouteEntry {
                row_index:(i/TOPK) as u32, expert_id:(i%128) as u32, gate_weight:0.0 }).collect();
            // Each link numbers its own waves; lanes keep apart in the high bits.
            let base=(lane as u64)<<40 | 1;
            let warm=build(base,first,rows,routes,vec![0;rows*ROW].into(),ExpertV2SourceKind::Prefill)?;
            let mut identity=base;
            let wave_build:cuteafd_transport::expert::DeviceBuild=Box::new(move |wave,routes,wire| {
                let kind=match wave.kind {
                    1=>ExpertV2SourceKind::Decode,
                    2=>ExpertV2SourceKind::Prefill,
                    _=>return Err(MiniMaxError::Invalid("Spark wave source kind").into()),
                };
                identity=identity.checked_add(1).ok_or(MiniMaxError::Overflow("Spark wave identity"))?;
                build(identity,wave.layer as usize,wave.rows as usize,routes,wire,kind)
            });
            Ok(Rc::new(SparkDeviceLink::new_zero_copy(library,device,world,peers,rows,TOPK,ROW,ROW,config.clone(),
                Some(warm),wave_build,host)?))
        })?))
    }
}

enum Pending<'a> {
    Local { layer:usize,rows:usize,index:usize,ids:Buffer<'a>,weights:Buffer<'a>,input:Buffer<'a> },
    Spark { layer:usize,rows:usize },
}
pub(super) struct Routed<'a> {
    local: Option<Rc<Fp8Experts<'a>>>,
    spark: Option<Spark<'a>>,
    /// Per lane, the dispatched wave not yet collected.
    pending: Vec<Option<Pending<'a>>>,
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
        if let Some(Spark(links))=&spark { ensure!(!links.is_empty() && links.iter().all(|l|matches!(l.world_size(),2|3|4)),
            MiniMaxError::Unsupported("Spark experts require TP2, TP3 or TP4")); }
        // Local layers run on the stream at collect, in any lane.
        let lanes=spark.as_ref().map_or(MAX_PREFILL_LANES,|Spark(links)|links.len());
        Ok(Self{local:local.map(Rc::new),spark,pending:(0..lanes).map(|_|None).collect(),kind:ExpertV2SourceKind::Decode,poisoned:false})
    }
    fn link(&self,lane:usize)->Result<&Rc<SparkDeviceLink<'a>>> {
        self.spark.as_ref().context("MiniMax routed layer has neither local experts nor Spark transport")?.0.lane(lane)
    }
}
impl<'a> ExpertExecutor<'a> for Routed<'a> {
    fn begin(&mut self,waves:usize,lanes:usize,kind:ExpertV2SourceKind)->Result<()> {
        ensure!(!self.poisoned,MiniMaxError::Invalid("previous expert failure; restart required"));
        ensure!(self.pending.iter().all(Option::is_none),MiniMaxError::Invalid("pending expert wave at step start"));
        ensure!((1..=self.lanes()).contains(&lanes),MiniMaxError::Invalid("expert lanes"));
        ensure!(matches!(kind,ExpertV2SourceKind::Decode|ExpertV2SourceKind::Prefill),
            MiniMaxError::Unsupported("expert source kind"));
        self.kind=kind;
        if let Some(Spark(links))=&self.spark {
            let local=self.local.as_ref().map_or(0,|l|l.layers.len());
            let remote=waves.checked_sub(local).context("MiniMax expert inventory exceeds routed layers")? as u64;
            for link in &links[..lanes] { link.arm()?; link.expect(remote); }
        }
        Ok(())
    }
    fn lanes(&self)->usize { self.pending.len() }
    fn dispatch(&mut self,layer:usize,lane:usize,rows:usize,ids:&Buffer<'a>,weights:&Buffer<'a>,input:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
        ensure!(self.pending.get(lane).is_some_and(Option::is_none),MiniMaxError::Invalid("overlapping expert waves"));
        let pending=if let Some(index)=self.local.as_ref().and_then(|l|l.index_of(layer).ok()) {
            Pending::Local{layer,rows,index,ids:ids.clone(),weights:weights.clone(),input:input.clone()}
        } else {
            minimax_ffi::dispatch(self.link(lane)?,layer,rows,self.kind,ids,weights,input,commands)?;
            Pending::Spark{layer,rows}
        };
        self.pending[lane]=Some(pending);
        Ok(())
    }
    fn collect(&mut self,layer:usize,lane:usize,rows:usize,output:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
        match self.pending.get_mut(lane).and_then(Option::take).context("MiniMax expert collect without dispatch")? {
            Pending::Local{layer:expected,rows:count,index,ids,weights,input}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("local expert wave identity"));
                minimax_ffi::local(self.local.as_ref().context("MiniMax local owner")?,index,rows,&ids,&weights,&input,output,commands)
            }
            Pending::Spark{layer:expected,rows:count}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("Spark expert wave identity"));
                minimax_ffi::collect(self.link(lane)?,rows,output,commands)
            }
        }
    }
    fn finish(&mut self)->Result<()> {
        ensure!(self.pending.iter().all(Option::is_none),MiniMaxError::Invalid("uncollected expert wave at step end"));
        if let Some(Spark(links))=&self.spark { for link in links.iter() { link.check()?; } }
        Ok(())
    }
    fn abort(&mut self) {
        self.poisoned=true;
        self.pending.iter_mut().for_each(|p|{p.take();});
        // Execution drained its stream first; closing the links joins their
        // proxies and cancels any announced waves the failed step never sent.
        self.spark.take();
    }
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::families::minimax_m3::reference;
    #[test] fn minimax_router_selects_with_bias_and_weights_without_bias() {
        let logits=[-1.0f32,0.0,1.0,2.0,3.0];
        let bias=[10.0f32,0.0,0.0,0.0,0.0];
        let selected=reference::route(&logits,&bias,4,2.0);
        assert_eq!(selected.iter().map(|(id,_)|*id).collect::<Vec<_>>(),[0,4,3,2]);
        let scores:Vec<_>=logits.iter().map(|x|1.0/(1.0+(-x).exp())).collect();
        let total:f32=selected.iter().map(|(id,_)|scores[*id]).sum();
        for (id,weight) in selected { assert!((weight-2.0*scores[id]/total).abs()<1e-6); }
    }
}
