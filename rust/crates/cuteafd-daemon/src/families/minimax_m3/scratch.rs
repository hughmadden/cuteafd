use crate::shared::{memory::device::{Allocation, Device}, prefix::view};
use anyhow::{Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::plan::families::minimax::admission::{RuntimeLedger, ROW_BUFFERS};
use std::{collections::BTreeMap, marker::PhantomData};

/// The admitted step scratch, one allocation per ledger entry.
pub(super) struct Scratch<'a> { buffers: BTreeMap<&'static str, Allocation<'a>> }
impl<'a> Scratch<'a> {
    pub fn new(device: Device<'a>, ledger: &RuntimeLedger) -> Result<Self> {
        let buffers = ledger.allocations.iter().filter(|a| !matches!(a.name,"embedded"|"token_ids"|"prefix_tail"|"sampler"|"selector"|"spark_intake"))
            .map(|a| Ok((a.name,Allocation::new(device,usize::try_from(a.bytes)?)?)))
            .collect::<Result<_>>()?;
        Ok(Self { buffers })
    }
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.buffers.get(name).map(|a| a.buffer).with_context(||format!("MiniMax unadmitted scratch {name}"))
    }
    /// The scratch of a prefill unit (or a whole step) over batch rows `first..first + rows`:
    /// every per-row buffer viewed from its row `first`, the rest shared.
    pub fn rows(&self, first: usize, rows: usize) -> Result<Rows<'_>> {
        let buffers = self.buffers.iter().map(|(&name, allocation)| {
            Ok((name, match row_extent(name, first, rows)? {
                Some((offset, bytes)) => view(allocation.buffer, offset, bytes)?,
                None => allocation.buffer,
            }))
        }).collect::<Result<_>>()?;
        Ok(Rows { buffers, _scratch: PhantomData })
    }
}

/// The byte extent `(offset, bytes)` of batch rows `first..first + rows` in per-row buffer `name`
/// ([`ROW_BUFFERS`]); `None` for a buffer that is not per row.
pub(super) fn row_extent(name: &str, first: usize, rows: usize) -> Result<Option<(usize, usize)>> {
    let Some(&(_, width, bytes)) = ROW_BUFFERS.iter().find(|(row, _, _)| *row == name) else { return Ok(None) };
    let row = usize::try_from(width * bytes)?;
    Ok(Some((first.checked_mul(row).context("MiniMax row view offset")?,
        rows.checked_mul(row).context("MiniMax row view extent")?)))
}

/// Views of the scratch a unit's (or a step's) rows use, borrowed from [`Scratch`].
pub(super) struct Rows<'s> { buffers: BTreeMap<&'static str, CuteafdDeviceBuffer>, _scratch: PhantomData<&'s ()> }
impl Rows<'_> {
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.buffers.get(name).copied().with_context(||format!("MiniMax unadmitted scratch {name}"))
    }
}
