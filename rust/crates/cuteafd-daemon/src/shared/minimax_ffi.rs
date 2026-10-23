//! MiniMax's retained bridge to the shared expert owners.
use super::{experts::fp8::Fp8Experts, spark_intake::SparkDeviceLink, memory::device::Device};
use anyhow::{ensure,Result};
use cuteafd_ffi::minimax::{Buffer,Commands};
use cuteafd_transport::ExpertV2SourceKind;
use std::rc::Rc;

pub(crate) fn local<'a>(owner: &Rc<Fp8Experts<'a>>, index:usize, rows:usize,
    ids:&Buffer<'a>, weights:&Buffer<'a>, input:&Buffer<'a>, output:&Buffer<'a>, commands:&Commands<'a>) -> Result<()> {
    ensure!(rows>0 && ids.raw().bytes>=rows*16 && weights.raw().bytes>=rows*16
        && input.raw().bytes>=rows*12288 && output.raw().bytes>=rows*12288,
        "MiniMax local expert operand extent");
    commands.retain(&[ids,weights,input,output])?;
    commands.retain_owner(owner.clone());
    // SAFETY: concrete weight owner and every operand are retained by the
    // same device stream; the shared expert binding verifies its package.
    Device{library:commands.library(),id:commands.device()}.run(||unsafe {
        owner.run_bf16(index,rows,input.raw().ptr,ids.raw().ptr,weights.raw().ptr,output.raw().ptr,commands.raw())
    })
}

pub(crate) fn dispatch<'a>(owner:&Rc<SparkDeviceLink<'a>>, layer:usize,rows:usize,kind:ExpertV2SourceKind,
    ids:&Buffer<'a>,weights:&Buffer<'a>,input:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
    ensure!(rows>0 && ids.raw().bytes>=rows*16 && weights.raw().bytes>=rows*16 && input.raw().bytes>=rows*12288,
        "MiniMax Spark expert operand extent");
    commands.retain(&[ids,weights,input])?;
    commands.retain_owner(owner.clone());
    // SAFETY: the link, its mailbox, and the three source owners survive
    // stream completion. Routed serializes dispatch/collect on this stream.
    Device{library:commands.library(),id:commands.device()}.run(||unsafe {
        owner.dispatch(layer,rows,kind as u32,ids.raw(),weights.raw(),input.raw(),commands.raw())
    })
}

pub(crate) fn collect<'a>(owner:&Rc<SparkDeviceLink<'a>>, rows:usize,output:&Buffer<'a>,commands:&Commands<'a>)->Result<()> {
    ensure!(rows>0 && output.raw().bytes>=rows*12288,"MiniMax Spark expert output extent");
    commands.retain(&[output])?;
    commands.retain_owner(owner.clone());
    // SAFETY: Routed permits collection only for its one matching pending
    // dispatch. The retained link owns every returned rank plane.
    Device{library:commands.library(),id:commands.device()}.run(||unsafe {
        owner.collect(commands.raw())?;
        owner.reduce(std::ptr::null(),output.raw().ptr.cast(),rows,commands.raw())
    })
}
