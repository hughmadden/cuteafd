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
