//! Inert K6 projection owners and projected attention/FFN seams.
//! Serving admission remains closed until full-layer and TP2 gates qualify it.
use super::weights::CoreWeights;
use crate::shared::memory::device::{Allocation,Device};
use anyhow::{ensure,Context,Result};
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
    fn merge(&mut self,other:&Self)->Result<()> {
        let mut layout=cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout {
            extents:self.extents.clone(), rows:self.rows };
        layout.merge(&cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout {
            extents:other.extents.clone(),rows:other.rows })?;
        self.extents=layout.extents;Ok(())
    }
    pub(crate) fn bytes(&self)->Result<usize> {
        cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout {
            extents:self.extents.clone(),rows:self.rows }.bytes()
    }
    pub(crate) fn new(program:&DenseTrellisDispatch<'_>,rows:usize)->Result<Self> {
        let p=program.policy();p.route(rows)?;
        let layout=cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout::new(p.input,p.output,rows,p.chunk_columns,p.weight_scratch)?;
        let w=program.workspace();
        ensure!([("rotated",w.rotated),("table",w.pointer_table),("scales",w.block_scales),
            ("accumulator",w.accumulator)].iter()
            .all(|(name,bytes)|layout.extents.get(name)==Some(bytes))
            && layout.extents["partial"]==w.partial/rows*rows.min(63),"K6 package/planner scratch mismatch");
        Ok(Self{extents:layout.extents,rows:layout.rows})
    }
    pub(crate) fn demand(&self,device:u8)->Result<Demand> {
        Ok(Demand::new(device,Category::Workspace,"Kimi K6 projection scratch",self.bytes()? as u64,Basis::Exact))
    }
    pub(crate) fn allocate<'a>(&self,device:Device<'a>)->Result<ProjectionScratch<'a>> {
        let storage=cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout {
            extents:self.extents.clone(),rows:self.rows }.storage()?;
        let mut owners=BTreeMap::new();
        for (&name,&bytes) in &storage.owners {owners.insert(name,Allocation::new(device,bytes)?);}
        let mut buffers=BTreeMap::new();
        for (&name,&bytes) in &self.extents {
            let buffer=if let Some(&offset)=storage.route_offsets.get(name) {
                crate::shared::prefix::view(owners["route arena"].buffer,offset,bytes)?
            } else {owners[name].buffer};
            buffers.insert(name,buffer);
        }
        let pointer=(buffers["rotated"].ptr as u64).to_le_bytes();
        device.run(||device.library.copy_h2d(buffers["table"],&pointer))?;
        Ok(ProjectionScratch{_owners:owners,buffers,rows:self.rows})
    }
}
pub(crate) struct ProjectionScratch<'a> {
    _owners:BTreeMap<&'static str,Allocation<'a>>,
    buffers:BTreeMap<&'static str,Buffer>,rows:usize,
}
impl ProjectionScratch<'_> {
    fn buffers(&self,input:Buffer,weights:&CoreWeights<'_>)->DenseTrellisBuffers {
        let get=|name|self.buffers[name];
        // Decode never touches chunk storage; a valid existing view is used
        // as the unused placeholder instead of a null device pointer.
        DenseTrellisBuffers {input,payload:weights.payload.buffer,suh:weights.suh.buffer,svh:weights.svh.buffer,
            rotated:get("rotated"),pointer_table:get("table"),scales:get("scales"),partial:get("partial"),
            accumulator:get("accumulator"),chunk_weights:self.buffers.get("chunk weights").copied().unwrap_or(get("partial")),
            chunk_output:self.buffers.get("chunk output").copied().unwrap_or(get("partial"))}
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
    pub norm:cuteafd_ffi::KimiRmsNorm<'a>,pub situ:KimiSitu<'a>,pub copy:V41PeerCopy<'a>,pub add:F32Add<'a>,
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
        if let Some(head)=inventory.get_weight("language_model.lm_head") {
            let weight=super::weights::CoreWeightLayout::new(head,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Output,0,1)?;
            let key=projection_key(weight.shape,weight.side,weight.codebook);
            let projection=Projection::load_geometry(programs,key.0,key.1,key.2,&key.3,rows.min(64),TrellisDType::F32)?;
            let mut required=ProjectionLayout::new(&projection.program,rows.min(64))?;
            // Head logits are consumed in <=64-row spans, while this scratch
            // owner also serves full-capacity serialized layer projections.
            required.rows=rows;
            if let Some(layout)=&mut layout {layout.merge(&required)?;}else{layout=Some(required);}
            projections.insert(key,projection);
        }
        let Some(layout)=layout else {anyhow::bail!("Kimi K6 inventory has no bound projections");};
        let planned=cuteafd_loader::families::kimi_k3::core::projection_scratch(cfg,inventory,rows,
            programs.family_metadata("kimi_k6_dense")?)?.context("K6 planned scratch absent")?;
        ensure!(layout.extents==planned.extents && layout.rows==planned.rows,
            "K6 runtime/planner aggregate scratch differs");
        let kda=[KimiKdaCore::load(programs,"kda_core_h48_m64",64,true)?,
            KimiKdaCore::load(programs,"kda_core_h48_m4096",4096,false)?];
        let mut mla=BTreeMap::new();
        for heads in [48,96] {for stride in [576,640] {for capacity in [64,4096] {
            let names=[format!("mla_rank_norm_h{heads}_kv{stride}_m{capacity}"),
                format!("mla_query_h{heads}_m{capacity}"),format!("mla_gate_output_h{heads}_m{capacity}")];
            mla.insert((heads,stride,capacity),KimiMlaSplit::load(programs,
                [&names[0],&names[1],&names[2]],heads,capacity,stride)?);
        }}}
        Ok(Some(Self{projections,layout,kda,mla,norm:cuteafd_ffi::KimiRmsNorm::load(programs,3584)?,situ:KimiSitu::load(programs,3072)?,
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
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn route_views_share_one_owner_and_admission_matches_allocated_bytes()->Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::families::kimi_k3::core::ProjectionScratchLayout;
        let fixture=Fixture::build()?;
        let library=fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device=Device{library:&library,id:0};
        for rows in [1,16,63,64,4096] {
            let planned=ProjectionScratchLayout::new(128,256,rows,128,128*128*2)?;
            let layout=ProjectionLayout{extents:planned.extents.clone(),rows};
            let scratch=layout.allocate(device)?;
            assert_eq!(layout.demand(0)?.bytes as usize,
                scratch._owners.values().map(|owner|owner.buffer.bytes).sum::<usize>());
            assert_eq!(layout.bytes()?,planned.bytes()?);
            let arena=scratch._owners["route arena"].buffer;
            assert_eq!(scratch.buffers["partial"].ptr,arena.ptr);
            assert_eq!(scratch.buffers["partial"].bytes,rows.min(63)*256*4);
            if rows>=64 {
                let weights=scratch.buffers["chunk weights"];
                let output=scratch.buffers["chunk output"];
                assert_eq!(weights.ptr,arena.ptr);
                assert!((output.ptr as usize)>=weights.ptr as usize+weights.bytes);
                assert!((output.ptr as usize)+output.bytes<=arena.ptr as usize+arena.bytes);
            }
            drop(scratch);
        }
        let events=fixture.events()?;
        assert_eq!(events.matches('D').count(),25);
        assert_eq!(events.matches('d').count(),25);
        Ok(())
    }
    #[test]
    fn planner_and_runtime_native_core_bytes_agree_on_and_off()->Result<()> {
        use cuteafd_loader::{families::kimi_k3::{config::{KimiK3Config,KimiAttention},core::{self,CoreCharges,CoreInventory}},
            plan::{testing,Checkpoint},placement::{LayerMode,FfnMode}};
        use cuteafd_ffi::dense_trellis::{DenseTrellisGeometry,TrellisDType};
        use super::super::weights::{CoreWeightLayout,LayerLayout};
        use cuteafd_loader::formats::dense_trellis::TrellisSide;
        let dir=tempfile::tempdir()?;
        testing::write_kimi_core_snapshot(dir.path(),false);
        let checkpoint=Checkpoint::open(dir.path())?;
        let cfg=KimiK3Config::from_hf(&checkpoint.config)?;
        let inventory=cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?;
        let mut geometries=BTreeMap::new();
        for layer in 0..cfg.layers {
            for mode in [LayerMode::HeadSplit,LayerMode::Whole{gpu:0,ffn:FfnMode::Split}] {
                if cfg.attention[layer]==KimiAttention::Kda && mode!=LayerMode::HeadSplit {continue;}
                for rank in LayerLayout::pair_with_core(&cfg,layer,mode,&inventory)? {
                    for weight in rank.core_layouts() {
                        let projection=&inventory.projections[&weight.stem];
                        // Local native allocations contain only compressed payload and vectors.
                        assert_eq!(weight.bytes as u64,projection.resident_bytes(weight.ranks)?);
                        geometries.insert(projection_key(weight.shape,weight.side,weight.codebook),());
                    }
                }
            }
        }
        let head=CoreWeightLayout::new(inventory.get_weight("language_model.lm_head").unwrap(),TrellisSide::Output,0,1)?;
        geometries.insert(projection_key(head.shape,head.side,head.codebook),());
        let policies=geometries.keys().map(|(k,n,side,book)|{
            let chunk=if k*n*2<=256*1024*1024 {*n}else{128};
            serde_json::json!({"input_features":k,"output_features":n,"sharded_side":side,"codebook":book,
                "threshold_rows":64,"below":"shared_native","at_or_above":"chunk_dequant_bf16",
                "chunk_output_dtype":"f32","stitch_before_output_transform":true,"chunk_columns":chunk,
                "weight_scratch_bytes":k*chunk*2,"chunk_output_scratch_bytes_per_row":chunk*4})
        }).collect::<Vec<_>>();
        let mut programs=Vec::new();
        let mut program=|name:String,bytes:usize| programs.push(serde_json::json!({"name":name,
            "scratch_bytes_at_capacity":{"scratch":bytes}}));
        for capacity in [64,4096] {
            program(format!("kimik3_kda_m{capacity}"),1024);
            program(format!("kda_core_h48_m{capacity}"),2048);
            for heads in [48,96] {
                let suffix=if heads==48 {""}else{"_h96"};
                program(format!("kimik3_mla_producer{suffix}_m{capacity}"),4096);
                program(format!("kimik3_mla_output{suffix}_m{capacity}"),4096);
                for mode in if capacity == 64 { &["decode", "verify"][..] } else { &["extend"][..] } {
                    program(format!("kimik3_dense_mla_{mode}{suffix}_m{capacity}"),4096);
                }
                for stride in [576,640] {program(format!("mla_rank_norm_h{heads}_kv{stride}_m{capacity}"),2048);}
            }
        }
        let metadata=serde_json::json!({"abi_version":1,"row_policies":policies});
        let manifest=serde_json::json!({"programs":programs,"families":{"kimi_k6_dense":metadata}});
        for rows in [1,16,63,64,4096] {
            for on in [false,true] {
                let empty=CoreInventory::default();
                let core=if on {&inventory}else{&empty};
                let charged=CoreCharges::from_package(&cfg,core,rows,&manifest)?;
                let mut runtime:Option<ProjectionLayout>=None;
                if on {
                    for ((k,n,side,_),()) in &geometries {
                        let live=if *n==cfg.vocab_size {rows.min(64)}else{rows};
                        let w=DenseTrellisGeometry{input:*k,output:*n,bits:6,side:*side,rows:live,
                            input_dtype:TrellisDType::Bf16,output_dtype:TrellisDType::Bf16}.workspace()?;
                        let mut extents=BTreeMap::from([("rotated",w.rotated),("table",w.pointer_table),
                            ("scales",w.block_scales),("partial",w.partial/live*live.min(63)),("accumulator",w.accumulator)]);
                        if live>=64 {
                            let chunk=if k*n*2<=256*1024*1024 {*n}else{128};
                            extents.insert("chunk weights",k*chunk*2);
                            extents.insert("chunk output",live*chunk*4);
                        }
                        let layout=ProjectionLayout{extents,rows};
                        if let Some(runtime)=&mut runtime {runtime.merge(&layout)?;}else{runtime=Some(layout);}
                    }
                }
                assert_eq!(charged.projection.as_ref().map(|p|&p.extents),runtime.as_ref().map(|p|&p.extents));
                assert_eq!(charged.receive,super::super::startup::exchange_demands(rows,false,on)?[0].bytes as usize
                    -256-256-(((cuteafd_loader::placement::HOP_SLOTS+1)*16)as usize).max(256));
                let expected_stages=if on {core::lane_extents(rows,1024,4096,2048,2048)?}else{BTreeMap::new()};
                assert_eq!(charged.stages,expected_stages);
                let sum=charged.demands()?.iter().filter(|d|d.gpu==0).map(|d|d.bytes).sum::<u64>();
                assert_eq!(sum,runtime.as_ref().map_or(Ok(0),|p|p.bytes())? as u64
                    +charged.stage_bytes as u64+charged.receive as u64);
                if on {
                    let base = cuteafd_loader::families::kimi_k3::allocation::NativeLaneLayout::new(&cfg, rows,
                        1_048_576, cuteafd_loader::families::kimi_k3::allocation::NativeScratch::from_manifest(&manifest)?)?;
                    assert_eq!(charged.stage_bytes, base.clone().with_core_stages(expected_stages)?.bytes()? - base.bytes()?);
                }
            }
        }
        // A duplicate exact geometry is ambiguous, never selected by table order.
        let mut duplicate=manifest.clone();
        let first=duplicate["families"]["kimi_k6_dense"]["row_policies"][0].clone();
        duplicate["families"]["kimi_k6_dense"]["row_policies"].as_array_mut().unwrap().push(first);
        assert!(CoreCharges::from_package(&cfg,&inventory,64,&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn serialized_projections_charge_memberwise_high_water_not_per_layer_sum()->Result<()> {
        let mut shared=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",256),("partial",4096),
            ("table",8),("accumulator",512)])};
        let other=ProjectionLayout{rows:64,extents:BTreeMap::from([("rotated",512),("partial",2048),
            ("table",8),("accumulator",1024),("chunk weights",8192),("chunk output",512)])};
        shared.merge(&other)?;
        assert_eq!(shared.bytes()?,512+8+1024+(8192usize+512).max(4096));
        let once=shared.clone();shared.merge(&other)?;assert_eq!(shared,once);
        assert!(shared.merge(&ProjectionLayout{rows:63,..other}).is_err());
        Ok(())
    }
}
