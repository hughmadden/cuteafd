use crate::shared::{memory::device::{Allocation, Device}, prefix::view};
use anyhow::{Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::plan::families::minimax::admission::{RuntimeLedger, ROW_BUFFERS};
use std::{collections::BTreeMap, marker::PhantomData};

/// The admitted step scratch, one allocation per ledger entry.
pub(super) struct Scratch<'a> { buffers: BTreeMap<&'static str, Allocation<'a>> }
impl<'a> Scratch<'a> {
    pub fn new(device: Device<'a>, ledger: &RuntimeLedger) -> Result<Self> {
        let buffers = ledger.allocations.iter().filter(|a| !matches!(a.name,"embedded"|"token_ids"|"cache_descales"|"prefix_tail"|"sampler"|"selector"|"spark_intake"))
            .map(|a| Ok((a.name,Allocation::new(device,usize::try_from(a.bytes)?)?)))
            .collect::<Result<_>>()?;
        Ok(Self { buffers })
    }
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.buffers.get(name).map(|a| a.buffer).with_context(||format!("MiniMax unadmitted scratch {name}"))
    }
    /// The scratch of a prefill lane over batch rows `first..first + rows`:
    /// every per-row buffer viewed from its row `first`, the rest shared.
    pub fn rows(&self, first: usize, rows: usize) -> Result<Rows<'_>> {
        let buffers = self.buffers.iter().map(|(&name, allocation)| {
            let Some(&(_, width, bytes)) = ROW_BUFFERS.iter().find(|(row, _, _)| *row == name) else {
                return Ok((name, allocation.buffer));
            };
            let row = usize::try_from(width * bytes)?;
            Ok((name, view(allocation.buffer, first * row, rows * row)?))
        }).collect::<Result<_>>()?;
        Ok(Rows { buffers, _scratch: PhantomData })
    }
}

/// Views of the scratch a lane's rows use, borrowed from [`Scratch`].
pub(super) struct Rows<'s> { buffers: BTreeMap<&'static str, CuteafdDeviceBuffer>, _scratch: PhantomData<&'s ()> }
impl Rows<'_> {
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.buffers.get(name).copied().with_context(||format!("MiniMax unadmitted scratch {name}"))
    }
}
