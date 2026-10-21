//! GLM 5.3 coordinator dense operands as push-group ranks: the class-2
//! records go through the shared coordinator dense ranks.
use crate::shared::{experts::rtx::dense as coordinator_dense, memory::device::Device};
use anyhow::Result;

/// Each GPU's dense plan and destination, ranks `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, operands: Vec<coordinator_dense::Operand>,
    devices: [Device<'a>; 2], sparks: usize)
    -> Result<(Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<coordinator_dense::DenseArena<'a>>)> {
    coordinator_dense::plan_ranks(snapshot, operands, Vec::new(), &devices, sparks)
}
