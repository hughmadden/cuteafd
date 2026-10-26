use super::{MiniMaxError, execution::ExpertExecutor, native::Native};
use crate::shared::{experts::fp8::Fp8Experts,spark_intake::{self,IntakePreference}};

/// Whether this GPU's Spark partials land in pinned host planes (no dma-buf
/// landing, as on GeForce cards): the ledger and the links take the same answer.
pub(super) fn host_landing(library:&NativeLibrary)->Result<bool> {
    Ok(!spark_intake::preferred_device_exchange_available(library,IntakePreference::Probed)?)
}
use cuteafd_loader::serving_capacity::prefill_lanes::MAX_PREFILL_LANES;
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{CuteafdDeviceBuffer,NativeLibrary};
use cuteafd_transport::{ExpertProtocolV2Request,ExpertProtocolV2RouteEntry,ExpertV2SourceKind,TcpTransportConfig,
    expert::SparkWorld};
use std::net::SocketAddr;

const ROW:usize=6144*2;

/// Why MiniMax cannot reach its Spark experts on this branch.
pub(crate) const SPARK_EXCHANGE_PENDING:&str="MiniMax runtime is not yet on the v4 Spark exchange (phase 2): \
    its routed experts ran on the device-driven Spark lane, which the shared per-link exchange \
    (spark_waves) replaced; serving and Spark goldens refuse until MiniMax is ported onto it";

/// The family's protocol request builder (identity, layer, rows, routes, wire, kind).
pub(super) type Build=fn(u64,usize,usize,Vec<ExpertProtocolV2RouteEntry>,bytes::Bytes,ExpertV2SourceKind)
    ->Result<ExpertProtocolV2Request>;

/// Spark expert transport. MiniMax exchanged its routed waves over the device-driven Spark lane
/// (`SparkDeviceLink`), which upstream deleted for the shared per-link exchange
/// (`shared::spark_waves`). Until MiniMax is ported onto that exchange no link can be made: this
/// type has no values, [`Spark::connect`] refuses with [`SPARK_EXCHANGE_PENDING`], and every
/// Spark arm below is statically unreachable. RTX-local routed layers keep working.
pub(super) struct Spark<'a> { never:std::convert::Infallible, _link:std::marker::PhantomData<&'a ()> }

impl<'a> Spark<'a> {
    /// Refuses: see [`SPARK_EXCHANGE_PENDING`]. The signature is the one the port will keep.
    #[allow(clippy::too_many_arguments)]
    pub fn connect(_library:&'a NativeLibrary,_device:i32,_world:SparkWorld,_peers:&[SocketAddr],_rows:usize,
        _config:TcpTransportConfig,_first:usize,_build:Build,_lanes:usize,_host:bool)->Result<Self> {
        Err(MiniMaxError::Unsupported(SPARK_EXCHANGE_PENDING).into())
    }
}

/// A dispatched wave not yet collected: a local one keeps its operand views
/// (lane scratch) until it runs at collect.
enum Pending {
    Local { layer:usize,rows:usize,index:usize,ids:CuteafdDeviceBuffer,weights:CuteafdDeviceBuffer,input:CuteafdDeviceBuffer },
}
pub(super) struct Routed<'a> {
    local: Option<Fp8Experts<'a>>,
    spark: Option<Spark<'a>>,
    /// Per lane, the dispatched wave not yet collected.
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
        if let Some(spark)=&spark { match spark.never {} }
        // Local layers run on the stream at collect, in any lane.
        let lanes=MAX_PREFILL_LANES;
        Ok(Self{local,spark,pending:(0..lanes).map(|_|None).collect(),kind:ExpertV2SourceKind::Decode,poisoned:false})
    }
    /// A layer that is not RTX-local needs the Spark transport, which cannot exist yet.
    fn spark(&self)->Result<&Spark<'a>> {
        self.spark.as_ref().context("MiniMax routed layer has neither local experts nor Spark transport")
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
        let local=self.local.as_ref().map_or(0,|l|l.layers.len());
        waves.checked_sub(local).context("MiniMax expert inventory exceeds routed layers")?;
        if let Some(spark)=&self.spark { match spark.never {} }
        Ok(())
    }
    fn record(&mut self,kind:ExpertV2SourceKind)->Result<()> {
        ensure!(!self.poisoned,MiniMaxError::Invalid("previous expert failure; restart required"));
        // A recording queues nothing: an uncollected dispatch it left owns no work.
        self.pending.iter_mut().for_each(|p|{p.take();});
        self.kind=kind;
        Ok(())
    }
    fn lanes(&self)->usize { self.pending.len() }
    fn dispatch(&mut self,layer:usize,lane:usize,rows:usize,ids:CuteafdDeviceBuffer,weights:CuteafdDeviceBuffer,
        input:CuteafdDeviceBuffer,_native:&Native<'a>)->Result<()> {
        ensure!(self.pending.get(lane).is_some_and(Option::is_none),MiniMaxError::Invalid("overlapping expert waves"));
        let pending=if let Some(index)=self.local.as_ref().and_then(|l|l.index_of(layer).ok()) {
            Pending::Local{layer,rows,index,ids,weights,input}
        } else {
            // A routed layer that is not RTX-local is a Spark wave: it needs the shared
            // exchange, and no Spark link exists until MiniMax is ported onto it.
            match self.spark()?.never {}
        };
        self.pending[lane]=Some(pending);
        Ok(())
    }
    fn collect(&mut self,layer:usize,lane:usize,rows:usize,output:CuteafdDeviceBuffer,native:&Native<'a>)->Result<()> {
        match self.pending.get_mut(lane).and_then(Option::take).context("MiniMax expert collect without dispatch")? {
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
        }
    }
    fn finish(&mut self)->Result<()> {
        ensure!(self.pending.iter().all(Option::is_none),MiniMaxError::Invalid("uncollected expert wave at step end"));
        if let Some(spark)=&self.spark { match spark.never {} }
        Ok(())
    }
    fn abort(&mut self) {
        self.poisoned=true;
        self.pending.iter_mut().for_each(|p|{p.take();});
        self.spark.take();
    }
}

#[cfg(test)]
mod tests {
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
}
