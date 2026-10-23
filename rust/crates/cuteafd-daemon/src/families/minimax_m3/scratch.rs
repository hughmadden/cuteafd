use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::minimax::Buffer;
use cuteafd_loader::plan::families::minimax::admission::RuntimeLedger;
use std::collections::BTreeMap;

pub(super) struct Scratch<'a> { buffers: BTreeMap<&'static str, Buffer<'a>> }
impl<'a> Scratch<'a> {
    pub fn new(device: Device<'a>, ledger: &RuntimeLedger) -> Result<Self> {
        let buffers = ledger.allocations.iter().filter(|a| !matches!(a.name,"embedded"|"token_ids"|"cache_descales"|"prefix_tail"|"sampler"|"selector"|"spark_intake"))
            .map(|a| Ok((a.name,Buffer::new(device.library,device.id,usize::try_from(a.bytes)?)?)))
            .collect::<Result<_>>()?;
        Ok(Self { buffers })
    }
    pub fn get(&self, name: &str) -> Result<&Buffer<'a>> { self.buffers.get(name).with_context(||format!("MiniMax unadmitted scratch {name}")) }
}
