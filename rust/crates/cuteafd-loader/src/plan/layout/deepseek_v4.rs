//! V4 resident allocation inventory shared with the runtime weight loader.
use crate::families::deepseek_v4::{DeepseekV4Config, resident::resident_operands};
use crate::placement::{FfnMode, LayerMode};
use crate::plan::Checkpoint;

pub(super) fn resident_weights(checkpoint: &Checkpoint, ranks: usize, dspark: bool)
    -> Option<Vec<Vec<(String, String, u64)>>> {
    let cfg = DeepseekV4Config::read(&checkpoint.snapshot, 0).ok()?;
    let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
        else { LayerMode::HeadSplit }; cfg.n_layers];
    let operands = resident_operands(&cfg, &modes, ranks, dspark,
        checkpoint.tensors.iter().map(|t| &t.meta)).ok()?;
    let mut groups = vec![std::collections::BTreeMap::<(String, String), u64>::new(); ranks];
    // All non-embedding weights are explicit admission demands. Keeping them
    // out of the baseline lets the shared solver see owner-dependent C4 costs.
    for operand in operands.into_iter().filter(|operand| operand.group == "embedding") {
        *groups[operand.rank].entry((operand.group, operand.format)).or_default() += operand.bytes;
    }
    Some(groups.into_iter().map(|group| group.into_iter()
        .map(|((group, format), bytes)| (group, format, bytes)).collect()).collect())
}
