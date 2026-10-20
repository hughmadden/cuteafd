//! Inert K6 projection owners and projected attention/FFN seams.
//! Serving admission remains closed until full-layer and TP2 gates qualify it.
use super::weights::CoreWeights;
use crate::shared::memory::device::{Allocation,Device};
use anyhow::{ensure,Result};
use cuteafd_core::memory_layout::{Basis,Category};
use cuteafd_loader::placement::Demand;
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer,programs::Programs,
    dense_trellis::{DenseTrellisBuffers,DenseTrellisDispatch,DenseTrellisPolicy,TrellisDType},
    KimiKdaCore,KimiKdaCoreBatch,KimiKdaCoreWeights,KimiMlaSplit,KimiMlaRankBatch,KimiSitu,F32Add,V41PeerCopy};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// All mutable projection storage is private to a serialized execution lane.
/// Weight storage is separate and may be shared by lanes on the same device.
#[derive(Debug,Clone,PartialEq,Eq)]
pub(crate) struct ProjectionLayout { extents:BTreeMap<&'static str,usize>, rows:usize }
impl ProjectionLayout {
    pub(crate) fn new(program:&DenseTrellisDispatch<'_>,rows:usize)->Result<Self> {
        let p=program.policy();p.route(rows)?;
        let w=program.workspace();
        let multiply=|a:usize,b:usize|a.checked_mul(b).ok_or_else(||anyhow::anyhow!("K6 projection scratch overflow"));
        let mut extents=BTreeMap::from([("rotated",w.rotated),("table",w.pointer_table),("scales",w.block_scales),
            ("partial",w.partial),("accumulator",w.accumulator)]);
        if rows>=64 {
            extents.insert("chunk weights",p.weight_scratch);
            extents.insert("chunk output",multiply(rows,multiply(p.chunk_columns,4)?)?);
        }
        Ok(Self{extents,rows})
    }
    fn merge(&mut self,other:&Self)->Result<()> {
        ensure!(self.rows==other.rows,"K6 scratch row capacities differ");
        for (&name,&bytes) in &other.extents {
            self.extents.entry(name).and_modify(|value|*value=(*value).max(bytes)).or_insert(bytes);
        }
        self.bytes()?;Ok(())
    }
    pub(crate) fn bytes(&self)->Result<usize> {
        self.extents.values().try_fold(0usize,|sum,&v|sum.checked_add(v).ok_or_else(||anyhow::anyhow!("K6 scratch sum overflow")))
    }
    pub(crate) fn demand(&self,device:u8)->Result<Demand> {
        Ok(Demand::new(device,Category::Workspace,"Kimi K6 projection scratch",self.bytes()? as u64,Basis::Exact))
    }
    pub(crate) fn allocate<'a>(&self,device:Device<'a>)->Result<ProjectionScratch<'a>> {
        let mut owners=BTreeMap::new();
        for (&name,&bytes) in &self.extents {owners.insert(name,Allocation::new(device,bytes)?);}
        let table=owners["table"].buffer;
        let pointer=(owners["rotated"].buffer.ptr as u64).to_le_bytes();
        device.run(||device.library.copy_h2d(table,&pointer))?;
        Ok(ProjectionScratch{owners,rows:self.rows})
    }
}
pub(crate) struct ProjectionScratch<'a> {owners:BTreeMap<&'static str,Allocation<'a>>,rows:usize}
impl ProjectionScratch<'_> {
    fn buffers(&self,input:Buffer,weights:&CoreWeights<'_>)->DenseTrellisBuffers {
        let get=|name|self.owners[name].buffer;
        // Decode never touches chunk storage; a valid existing owner is used
        // as the unused placeholder instead of a null device pointer.
        DenseTrellisBuffers {input,payload:weights.payload.buffer,suh:weights.suh.buffer,svh:weights.svh.buffer,
            rotated:get("rotated"),pointer_table:get("table"),scales:get("scales"),partial:get("partial"),
            accumulator:get("accumulator"),chunk_weights:self.owners.get("chunk weights").map_or(get("partial"),|v|v.buffer),
            chunk_output:self.owners.get("chunk output").map_or(get("partial"),|v|v.buffer)}
    }
}
pub(crate) struct Projection<'a> {pub program:DenseTrellisDispatch<'a>}
impl<'a> Projection<'a> {
    pub(crate) fn load(programs:&'a Programs<'a>,shape:&str,book:&str,rows:usize,output:TrellisDType)->Result<Self> {
        let policy=DenseTrellisPolicy::from_package(programs,shape,book)?;
        Ok(Self{program:DenseTrellisDispatch::load(programs,policy,rows,output)?})
    }
    pub(crate) fn load_geometry(programs:&'a Programs<'a>,input:usize,output:usize,side:usize,book:&str,
        rows:usize,dtype:TrellisDType)->Result<Self> {
        let policy=DenseTrellisPolicy::from_package_geometry(programs,input,output,side,book)?;
        Ok(Self{program:DenseTrellisDispatch::load(programs,policy,rows,dtype)?})
    }
    pub(crate) fn core(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        stream:*mut c_void)->Result<Buffer> {
        ensure!(rows>0 && rows<=scratch.rows,"K6 scratch live-row ceiling");
        let policy=self.program.policy();
        let side=match weights.side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
            cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2};
        let book=match weights.codebook {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
            cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"};
        ensure!(weights.shape==[policy.output,policy.input] && side==policy.side && book==policy.codebook,
            "K6 projection and resident weight geometry/codebook differ");
        let b=scratch.buffers(input,weights);self.program.core(rows,b,stream)?;Ok(b.accumulator)
    }
    pub(crate) fn local(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        output:Buffer,stream:*mut c_void)->Result<()> {
        ensure!(self.program.policy().side!=2,"K-split output requires F32 peer reduction before egress");
        let core=self.core(rows,input,weights,scratch,stream)?;
        self.program.output(rows,core,weights.svh.buffer,output,stream)?;Ok(())
    }
    pub(crate) fn reduced(&self,add:&F32Add<'_>,rows:usize,local:Buffer,published_peer:Buffer,reduced:Buffer,
        weights:&CoreWeights<'_>,output:Buffer,stream:*mut c_void)->Result<()> {
        self.program.reduce_output(add,rows,local,published_peer,reduced,weights.svh.buffer,output,stream)?;Ok(())
    }
}

pub(crate) struct Projected<'a,'b> {pub projection:&'b Projection<'a>,pub weights:&'b CoreWeights<'a>,pub scratch:&'b ProjectionScratch<'a>}
impl Projected<'_,'_> {
    fn run(&self,rows:usize,input:Buffer,output:Buffer,stream:*mut c_void)->Result<()> {
        self.projection.local(rows,input,self.weights,self.scratch,output,stream)
    }
}

/// q/k/v/g are independent calibrations: four rotations, four projections.
/// The existing recurrence owns convolution, state, replay and normalization.
pub(crate) fn kda_projected(core:&KimiKdaCore<'_>,projections:[Projected<'_,'_>;4],
    weights:KimiKdaCoreWeights,batch:KimiKdaCoreBatch,stream:*mut c_void)->Result<()> {
    for (projection,output) in projections.iter().zip([batch.q,batch.k,batch.v,batch.gate]) {
        projection.run(batch.rows,batch.input,output,stream)?;
    }
    core.launch(weights,batch,stream)?;Ok(())
}
/// Stored kv_a N640 is transformed in full; the split norm consumes its N576 prefix.
pub(crate) fn mla_rank_projected(core:&KimiMlaSplit<'_>,qa:Projected<'_,'_>,kv:Projected<'_,'_>,input:Buffer,
    batch:KimiMlaRankBatch,stream:*mut c_void)->Result<()> {
    qa.run(batch.rows,input,batch.q_a,stream)?;kv.run(batch.rows,input,batch.kv_a,stream)?;
    core.rank_norm(batch,stream)?;Ok(())
}
pub(crate) fn mla_query_projected(core:&KimiMlaSplit<'_>,qb:Projected<'_,'_>,rows:usize,q_residual:Buffer,
    q_b:Buffer,uk:Buffer,query:Buffer,stream:*mut c_void)->Result<()> {
    qb.run(rows,q_residual,q_b,stream)?;core.query(rows,q_b,uk,query,stream)?;Ok(())
}
pub(crate) fn mla_gate_projected(core:&KimiMlaSplit<'_>,gate:Projected<'_,'_>,rows:usize,input:Buffer,
    attention:Buffer,uv:Buffer,gates:Buffer,normalized:Buffer,stream:*mut c_void)->Result<()> {
    gate.run(rows,input,gates,stream)?;core.gate_output(rows,attention,uv,gates,normalized,stream)?;Ok(())
}
/// Separate gate/up outputs are interleaved by row for the unchanged Situ op;
/// down remains F32 until the caller publishes and reduces the peer partial.
pub(crate) fn ffn_projected(gate:Projected<'_,'_>,up:Projected<'_,'_>,down:Projected<'_,'_>,situ:&KimiSitu<'_>,
    copy:&V41PeerCopy<'_>,rows:usize,input:Buffer,gate_output:Buffer,up_output:Buffer,gate_up:Buffer,
    activated:Buffer,stream:*mut c_void)->Result<Buffer> {
    let width=gate.projection.program.policy().output;
    ensure!(width==up.projection.program.policy().output && width==down.projection.program.policy().input,
        "K6 FFN intermediate widths differ");
    gate.run(rows,input,gate_output,stream)?;up.run(rows,input,up_output,stream)?;
    copy.interleave_bf16(gate_up,gate_output,up_output,rows,width,stream)?;
    situ.launch(rows,gate_up,activated,stream)?;
    down.projection.core(rows,activated,down.weights,down.scratch,stream)
}

/// Resolve both possible MLA ownership geometries before placement samples CUDA.
/// Serialized projections reuse one lane-private scratch set, never one set per
/// layer. Their BF16 outputs remain distinct until the downstream core consumes them.
pub(crate) struct CorePrograms<'a> {
    projections:BTreeMap<(usize,usize,usize,String),Projection<'a>>,
    pub layout:ProjectionLayout,
    pub kda:[KimiKdaCore<'a>;2],
    pub mla:BTreeMap<(usize,usize,usize),KimiMlaSplit<'a>>,
    pub situ:KimiSitu<'a>,pub copy:V41PeerCopy<'a>,pub add:F32Add<'a>,
}
impl<'a> CorePrograms<'a> {
    pub(crate) fn load(programs:&'a Programs<'a>,cfg:&cuteafd_loader::families::kimi_k3::config::KimiK3Config,
        inventory:&cuteafd_loader::families::kimi_k3::core::CoreInventory,rows:usize)->Result<Option<Self>> {
        use cuteafd_loader::placement::{LayerMode,FfnMode};
        inventory.validate_runtime_groups(cfg)?;
        if inventory.projections.is_empty() {return Ok(None);}
        let mut projections=BTreeMap::new();
        let mut layout:Option<ProjectionLayout>=None;
        for layer in 0..cfg.layers {
            let modes=if cfg.attention[layer]==cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla {
                vec![LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}]
            } else {vec![LayerMode::HeadSplit]};
            for mode in modes {
                for rank in super::weights::LayerLayout::pair_with_core(cfg,layer,mode,inventory)? {
                    for weight in rank.core_layouts() {
                        let key=projection_key(weight.shape,weight.side,weight.codebook);
                        if projections.contains_key(&key) {continue;}
                        let projection=Projection::load_geometry(programs,key.0,key.1,key.2,&key.3,rows,TrellisDType::Bf16)?;
                        let required=ProjectionLayout::new(&projection.program,rows)?;
                        if let Some(layout)=&mut layout {layout.merge(&required)?;} else {layout=Some(required);}
                        projections.insert(key,projection);
                    }
                }
            }
        }
        let Some(layout)=layout else {anyhow::bail!("Kimi K6 inventory has no bound layer projections");};
        let kda=[KimiKdaCore::load(programs,"kda_core_h48_m64",64,true)?,
            KimiKdaCore::load(programs,"kda_core_h48_m4096",4096,false)?];
        let mut mla=BTreeMap::new();
        for heads in [48,96] {for stride in [576,640] {for capacity in [64,4096] {
            let names=[format!("mla_rank_norm_h{heads}_kv{stride}_m{capacity}"),
                format!("mla_query_h{heads}_m{capacity}"),format!("mla_gate_output_h{heads}_m{capacity}")];
            mla.insert((heads,stride,capacity),KimiMlaSplit::load(programs,
                [&names[0],&names[1],&names[2]],heads,capacity,stride)?);
        }}}
        Ok(Some(Self{projections,layout,kda,mla,situ:KimiSitu::load(programs,3072)?,
            copy:programs.library().v41_peer_copy()?,add:programs.library().f32_add()?}))
    }
    pub(crate) fn projection(&self,weights:&CoreWeights<'_>)->Result<&Projection<'a>> {
        self.projections.get(&projection_key(weights.shape,weights.side,weights.codebook))
            .ok_or_else(||anyhow::anyhow!("Kimi K6 weight geometry not resolved before admission"))
    }
    pub(crate) fn projected<'b>(&'b self,weights:&'b CoreWeights<'a>,scratch:&'b ProjectionScratch<'a>)->Result<Projected<'a,'b>> {
        Ok(Projected{projection:self.projection(weights)?,weights,scratch})
    }
}
fn projection_key(shape:[usize;2],side:cuteafd_loader::formats::dense_trellis::TrellisSide,
    book:cuteafd_loader::formats::exl3_storage::Exl3Codebook)->(usize,usize,usize,String) {
    (shape[1],shape[0],match side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
        cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2},
        match book {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
            cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"}.into())
}

#[cfg(test)]
mod runtime_layout_tests {
    use super::*;
    #[test]
    fn serialized_projections_charge_memberwise_high_water_not_per_layer_sum()->Result<()> {
        let mut shared=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",256),("partial",4096),
            ("table",8),("accumulator",512)])};
        let other=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",512),("partial",2048),
            ("table",8),("accumulator",1024),("chunk weights",8192)])};
        shared.merge(&other)?;
        assert_eq!(shared.bytes()?,512+4096+8+1024+8192);
        let once=shared.clone();shared.merge(&other)?;assert_eq!(shared,once);
        assert!(shared.merge(&ProjectionLayout{rows:63,..other}).is_err());
        Ok(())
    }
}
