use super::{MiniMaxError, execution::ExpertExecutor};
use crate::shared::{experts::{fp8::Fp8Experts,staging::RouteStaging},memory::{HostAllocation,device::Device},minimax_ffi,
    spark_intake::{SparkDeviceLink,SparkLink}};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{NativeLibrary,minimax::{Buffer,Commands}};
use cuteafd_transport::{ExpertProtocolV2Request,ExpertProtocolV2RouteEntry,ExpertV2SourceKind,TcpTransportConfig,
    expert::{SparkExpertWave,SparkWorld}};
use std::{ffi::c_void,net::SocketAddr,rc::Rc,sync::Arc,time::Duration};

const ROW:usize=6144*2;
const TOPK:usize=4;

/// The family's protocol request builder (identity, layer, rows, routes, wire, kind).
pub(super) type Build=fn(u64,usize,usize,Vec<ExpertProtocolV2RouteEntry>,bytes::Bytes,ExpertV2SourceKind)
    ->Result<ExpertProtocolV2Request>;

/// Spark expert transport: device-driven where the GPU can land RDMA
/// replies, otherwise the pinned-host intake (GeForce cards have no dma-buf).
pub(super) enum Spark<'a> { Device(Rc<SparkDeviceLink<'a>>), Pinned(PinnedSpark<'a>) }

impl<'a> Spark<'a> {
    /// `first` is a Spark layer for the warm wave that sizes every rank's rings.
    #[allow(clippy::too_many_arguments)]
    pub fn connect(owner:&'a Arc<NativeLibrary>,device:i32,world:SparkWorld,peers:&[SocketAddr],rows:usize,
        config:TcpTransportConfig,first:usize,build:Build)->Result<Self> {
        let library=owner.as_ref();
        let routes=(0..rows*TOPK).map(|i| ExpertProtocolV2RouteEntry {
            row_index:(i/TOPK) as u32, expert_id:(i%128) as u32, gate_weight:0.0 }).collect();
        let warm=build(1,first,rows,routes,vec![0;rows*ROW].into(),ExpertV2SourceKind::Prefill)?;
        if !crate::shared::spark_intake::device_exchange_available(library)? {
            return Ok(Self::Pinned(PinnedSpark::connect(owner,world,peers,rows,config,&warm,build)?));
        }
        let mut identity=1u64;
        let wave_build:cuteafd_transport::expert::DeviceBuild=Box::new(move |wave,routes,wire| {
            let kind=match wave.kind {
                1=>ExpertV2SourceKind::Decode,
                2=>ExpertV2SourceKind::Prefill,
                _=>return Err(MiniMaxError::Invalid("Spark wave source kind").into()),
            };
            identity=identity.checked_add(1).ok_or(MiniMaxError::Overflow("Spark wave identity"))?;
            build(identity,wave.layer as usize,wave.rows as usize,routes,wire,kind)
        });
        Ok(Self::Device(Rc::new(SparkDeviceLink::new(library,device,world,peers,rows,TOPK,ROW,ROW,config,Some(warm),wave_build)?)))
    }
    fn world_size(&self)->usize {
        match self { Self::Device(link)=>link.world_size(), Self::Pinned(link)=>link.world_size() }
    }
}

/// Pinned-host exchange: route ids, weights and BF16 rows copy to one pinned
/// arena, the host posts the wave, and rank planes upload into the intake.
pub(super) struct PinnedSpark<'a> {
    link: Option<SparkLink<'a>>,
    staging: HostAllocation<'a>,
    runtime: tokio::runtime::Runtime,
    build: Build,
    capacity: usize,
    identity: u64,
    /// Compute stream that last consumed the intake planes; drained before release.
    stream: *mut c_void,
}

impl<'a> PinnedSpark<'a> {
    fn connect(owner:&'a Arc<NativeLibrary>,world:SparkWorld,peers:&[SocketAddr],capacity:usize,
        config:TcpTransportConfig,warm:&ExpertProtocolV2Request,build:Build)->Result<Self> {
        let library=owner.as_ref();
        library.v41_compact_reducer()?.require_plane_reduction(peers.len() as u32)?;
        let staging=HostAllocation::named(library,RouteStaging::new(capacity,TOPK,ROW)?.bytes,"experts/workspace/route_staging")?;
        let mut link=SparkLink::new(library,world,peers,u32::try_from(capacity)?,config,ROW)?;
        link.enable_terminal_ownership(Arc::clone(owner))?;
        link.bootstrap(warm)?;
        let runtime=tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(Self{link:Some(link),staging,runtime,build,capacity,identity:1,stream:std::ptr::null_mut()})
    }
    fn world_size(&self)->usize { self.link.as_ref().map_or(0,SparkLink::world_size) }

    /// Stages and posts one wave. The stream drains so the host can parse the
    /// routes; the wave's planes stay with the link until `collect`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(&mut self,layer:usize,rows:usize,kind:ExpertV2SourceKind,ids:&Buffer<'a>,weights:&Buffer<'a>,
        input:&Buffer<'a>,commands:&Commands<'a>)->Result<SparkExpertWave> {
        ensure!((1..=self.capacity).contains(&rows) && ids.raw().bytes>=rows*TOPK*4
            && weights.raw().bytes>=rows*TOPK*4 && input.raw().bytes>=rows*ROW,
            "MiniMax pinned Spark operand extent");
        let layout=RouteStaging::new(rows,TOPK,ROW)?;
        commands.retain(&[ids,weights,input])?;
        let device=Device{library:commands.library(),id:commands.device()};
        device.run(|| {
            // SAFETY: the staging arena is exclusively this link's until the
            // drain below; the three sources are retained by the stream.
            unsafe { layout.enqueue(commands.library(),self.staging.buffer,[ids.raw(),weights.raw(),input.raw()],commands.raw()) }
        })?;
        commands.drain()?;
        let staged=self.staging.bytes();
        let identity=self.identity.checked_add(1).ok_or(MiniMaxError::Overflow("Spark wave identity"))?;
        let request=(self.build)(identity,layer,rows,layout.routes(staged)?,bytes::Bytes::copy_from_slice(layout.wire(staged)?),kind)?;
        self.identity=identity;
        self.stream=commands.raw();
        let link=self.link.as_mut().context("MiniMax Spark link retired")?;
        device.run(|| link.dispatch(&request))
    }

    fn collect(&mut self,wave:SparkExpertWave,rows:usize,output:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
        ensure!(output.raw().bytes>=rows*ROW,"MiniMax Spark expert output extent");
        commands.retain(&[output])?;
        let stream=commands.raw();
        let (runtime,link)=(&self.runtime,self.link.as_mut().context("MiniMax Spark link retired")?);
        Device{library:commands.library(),id:commands.device()}.run(|| {
            runtime.block_on(async { tokio::time::timeout(Duration::from_secs(120),link.receive(wave,rows,stream)).await
                .context("MiniMax Spark receive timed out")? })?;
            // SAFETY: receive ordered every rank plane before `stream`; output is
            // a retained operand disjoint from the intake.
            unsafe { link.reduce(std::ptr::null(),output.raw().ptr.cast(),rows,stream) }
        })
    }
}

impl Drop for PinnedSpark<'_> {
    fn drop(&mut self) {
        let Some(mut link)=self.link.take() else { return };
        let (library,stream)=(self.staging.library,self.stream);
        // Quiesce the QPs, drain uploads and the consuming stream, then release
        // held payloads; on any failure the link retains its intake instead.
        let retired=link.terminal_quiesce().and_then(|()| link.terminal_drain_copies())
            // SAFETY: the stream belongs to the execution that outlives its experts.
            .and_then(|()| if stream.is_null() {Ok(())} else {unsafe { library.cuda_stream_synchronize(stream) }})
            .and_then(|()| link.terminal_release());
        if let Err(error)=retired { tracing::error!(%error,"MiniMax Spark link retirement incomplete"); }
        drop(link);
    }
}

enum Pending<'a> {
    Local { layer:usize,rows:usize,index:usize,ids:Buffer<'a>,weights:Buffer<'a>,input:Buffer<'a> },
    Spark { layer:usize,rows:usize },
    Pinned { layer:usize,rows:usize,wave:SparkExpertWave },
}
pub(super) struct Routed<'a> {
    local: Option<Rc<Fp8Experts<'a>>>,
    spark: Option<Spark<'a>>,
    pending: Option<Pending<'a>>,
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
        if let Some(spark)=&spark { ensure!(matches!(spark.world_size(),2|3|4),
            MiniMaxError::Unsupported("Spark experts require TP2, TP3 or TP4")); }
        Ok(Self{local:local.map(Rc::new),spark,pending:None,kind:ExpertV2SourceKind::Decode,poisoned:false})
    }
}
impl<'a> ExpertExecutor<'a> for Routed<'a> {
    fn begin(&mut self,waves:usize,kind:ExpertV2SourceKind)->Result<()> {
        ensure!(!self.poisoned,MiniMaxError::Invalid("previous expert failure; restart required"));
        ensure!(self.pending.is_none(),MiniMaxError::Invalid("pending expert wave at step start"));
        ensure!(matches!(kind,ExpertV2SourceKind::Decode|ExpertV2SourceKind::Prefill),
            MiniMaxError::Unsupported("expert source kind"));
        self.kind=kind;
        if let Some(Spark::Device(spark))=&self.spark {
            spark.arm()?;
            let local=self.local.as_ref().map_or(0,|l|l.layers.len());
            spark.expect(waves.checked_sub(local).context("MiniMax expert inventory exceeds routed layers")? as u64);
        }
        Ok(())
    }
    fn dispatch(&mut self,layer:usize,rows:usize,ids:&Buffer<'a>,weights:&Buffer<'a>,input:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
        ensure!(self.pending.is_none(),MiniMaxError::Invalid("overlapping expert waves"));
        if let Some(index)=self.local.as_ref().and_then(|l|l.index_of(layer).ok()) {
            self.pending=Some(Pending::Local{layer,rows,index,ids:ids.clone(),weights:weights.clone(),input:input.clone()});
        } else {
            match self.spark.as_mut().context("MiniMax routed layer has neither local experts nor Spark transport")? {
                Spark::Device(spark)=>{
                    minimax_ffi::dispatch(spark,layer,rows,self.kind,ids,weights,input,commands)?;
                    self.pending=Some(Pending::Spark{layer,rows});
                }
                Spark::Pinned(spark)=>{
                    let wave=spark.dispatch(layer,rows,self.kind,ids,weights,input,commands)?;
                    self.pending=Some(Pending::Pinned{layer,rows,wave});
                }
            }
        }
        Ok(())
    }
    fn collect(&mut self,layer:usize,rows:usize,output:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
        match self.pending.take().context("MiniMax expert collect without dispatch")? {
            Pending::Local{layer:expected,rows:count,index,ids,weights,input}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("local expert wave identity"));
                minimax_ffi::local(self.local.as_ref().context("MiniMax local owner")?,index,rows,&ids,&weights,&input,output,commands)
            }
            Pending::Spark{layer:expected,rows:count}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("Spark expert wave identity"));
                let Some(Spark::Device(spark))=&self.spark else { anyhow::bail!("MiniMax Spark owner") };
                minimax_ffi::collect(spark,rows,output,commands)
            }
            Pending::Pinned{layer:expected,rows:count,wave}=>{
                ensure!((layer,rows)==(expected,count),MiniMaxError::Invalid("Spark expert wave identity"));
                let Some(Spark::Pinned(spark))=&mut self.spark else { anyhow::bail!("MiniMax pinned Spark owner") };
                spark.collect(wave,rows,output,commands)
            }
        }
    }
    fn finish(&mut self)->Result<()> {
        ensure!(self.pending.is_none(),MiniMaxError::Invalid("uncollected expert wave at step end"));
        if let Some(Spark::Device(spark))=&self.spark { spark.check()?; }
        Ok(())
    }
    fn abort(&mut self) {
        self.poisoned=true;
        self.pending.take();
        // Execution drained its stream first; closing the link joins the
        // proxy and cancels any announced waves the failed step never sent.
        self.spark.take();
    }
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::families::minimax_m3::reference;
    #[test] fn minimax_spark_falls_back_to_pinned_intake_with_terminal_ownership() {
        let source=include_str!("routed.rs").split("#[cfg(test)]").next().unwrap();
        let connect=source.split("pub fn connect(").nth(1).unwrap().split("fn world_size").next().unwrap();
        assert!(connect.find("device_exchange_available(library)").unwrap() < connect.find("PinnedSpark::connect").unwrap());
        assert!(connect.find("PinnedSpark::connect").unwrap() < connect.find("SparkDeviceLink::new").unwrap());
        let pinned=source.split("impl<'a> PinnedSpark<'a>").nth(1).unwrap();
        assert!(pinned.find("link.enable_terminal_ownership").unwrap() < pinned.find("link.bootstrap(warm)").unwrap());
        assert!(pinned.find("commands.drain()?").unwrap() < pinned.find("layout.routes(staged)").unwrap());
        let drop=source.split("impl Drop for PinnedSpark").nth(1).unwrap();
        for (a,b) in [("terminal_quiesce","terminal_drain_copies"),("terminal_drain_copies","cuda_stream_synchronize"),
            ("cuda_stream_synchronize","terminal_release")] { assert!(drop.find(a).unwrap() < drop.find(b).unwrap()); }
        for serve in [include_str!("serve.rs"),include_str!("golden.rs")] {
            assert!(serve.contains("routed::Spark::connect(&library") && !serve.contains("SparkDeviceLink::new"));
        }
    }
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
