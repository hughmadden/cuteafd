use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::minimax::Buffer;
use cuteafd_loader::plan::families::minimax::admission::{RuntimeLedger, ROW_BUFFERS};
use std::collections::BTreeMap;

pub(super) struct Scratch<'a> { buffers: BTreeMap<&'static str, Buffer<'a>> }
impl<'a> Scratch<'a> {
    pub fn new(device: Device<'a>, ledger: &RuntimeLedger) -> Result<Self> {
        let buffers = ledger.allocations.iter().filter(|a| !matches!(a.name,"embedded"|"token_ids"|"prefix_tail"|"sampler"|"selector"|"spark_intake"))
            .map(|a| Ok((a.name,Buffer::new(device.library,device.id,usize::try_from(a.bytes)?)?)))
            .collect::<Result<_>>()?;
        Ok(Self { buffers })
    }
    pub fn get(&self, name: &str) -> Result<&Buffer<'a>> { self.buffers.get(name).with_context(||format!("MiniMax unadmitted scratch {name}")) }
    /// The scratch of a prefill lane over batch rows `first..first + rows`:
    /// every per-row buffer viewed from its row `first`, the rest shared.
    pub fn rows(&self, first: usize, rows: usize) -> Result<Self> {
        let buffers = self.buffers.iter().map(|(&name, buffer)| {
            let Some(&(_, width, bytes)) = ROW_BUFFERS.iter().find(|(row, _, _)| *row == name) else {
                return Ok((name, buffer.clone()));
            };
            let row = usize::try_from(width * bytes)?;
            Ok((name, buffer.view(first * row, rows * row)?))
        }).collect::<Result<_>>()?;
        Ok(Self { buffers })
    }
}
