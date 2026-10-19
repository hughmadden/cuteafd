//! Startup-only rank-local arithmetic; cache staging and transport stay eager.
use crate::shared::{decode_graph::{GraphBank, GraphDecision, GraphOwner, GraphPolicy},
    memory::device::Stream};
use anyhow::{ensure, Result};
use cuteafd_loader::{placement::{GraphSet, Lifetime, inventory::StartupGraphs},
    plan::families::step5::ownership::ExecutionOwnership};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Op { Projection(usize), SlidingFinish(usize), Dense(usize),
    Frontend(usize), Shared(usize), Join(usize) }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key { pub lane: usize, pub rank: usize, pub rows: usize, pub op: Op }

pub(super) const ROWS: [usize; 5] = [1, 4, 16, 64, 4096];

pub(super) fn inventory(ownership: &ExecutionOwnership, capacity: usize) -> Result<StartupGraphs<Key>> {
    ensure!((1..=4096).contains(&capacity)
        && capacity as u64 <= ownership.execution().spec().rows, "Step-5 graph row capacity");
    let mut keys = Vec::new();
    let mut counts = vec![0u64; ownership.ranks()];
    for lane in 0..ownership.execution().spec().lanes as usize {
        for rank in 0..ownership.ranks() {
            for rows in ROWS.into_iter().filter(|&rows| rows <= capacity) {
                for layer in 0..92 {
                    let local = ownership.layer(layer, rank)?;
                    if local.query_heads == 0 { continue; }
                    let mut ops = vec![Op::Projection(layer)];
                    if layer % 4 != 3 { ops.push(Op::SlidingFinish(layer)); }
                    if !(3..91).contains(&layer) { ops.push(Op::Dense(layer)); }
                    else { ops.extend([Op::Frontend(layer), Op::Shared(layer), Op::Join(layer)]); }
                    for op in ops { keys.push(Key { lane, rank, rows, op }); counts[rank] += 1; }
                }
            }
        }
    }
    // Estimated until the first serving census; the same exact inventory is
    // reserved before load and sealed after startup, never grown by requests.
    GraphSet::new(&counts, 256 << 10, 10, 64 << 20, keys.len() as u64, Lifetime::Startup)
        .startup(keys).map_err(anyhow::Error::msg)
}

pub(super) struct Graphs<'a> { bank: GraphBank<Key, GraphOwner<'a, ()>> }
impl<'a> Graphs<'a> {
    pub(super) fn capture(stream: &Stream<'a>, keys: &[Key],
        mut arithmetic: impl FnMut(Key) -> Result<()>) -> Result<Self> {
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        for &key in keys { arithmetic(key)?; }
        stream.drain()?;
        bank.warm(keys, |&key| stream.device.run(|| {
            // SAFETY: exclusive startup; the containing lane pins every operand
            // and drains this exact stream before dropping graphs or storage.
            unsafe { stream.device.library.cuda_graph_begin_capture(stream.raw)?; }
            let submitted = arithmetic(key);
            // SAFETY: end the matching capture even when arithmetic submission fails.
            let ended = unsafe { stream.device.library.cuda_graph_end_capture(stream.raw) };
            match (submitted, ended) {
                (Ok(()), Ok(raw)) => {
                    // SAFETY: the lane owns all captured pointers through terminal drainage.
                    let owner = unsafe { GraphOwner::new(stream.device.library, stream.device.id, raw, ())? };
                    Ok((owner, None))
                }
                (Err(error), Ok(raw)) => {
                    // SAFETY: this failed executable has never been launched.
                    unsafe { stream.device.library.cuda_graph_exec_destroy(raw)?; }
                    Err(error)
                }
                (Err(error), Err(ended)) => Err(error.context(format!("ending failed capture: {ended:#}"))),
                (Ok(()), Err(error)) => Err(error),
            }
        }))?;
        Ok(Self { bank })
    }

    pub(super) fn launch(&mut self, key: Key, stream: &Stream<'a>) -> Result<bool> {
        match self.bank.enqueue(&key) {
            (GraphDecision::Replay, Some(owner)) => {
                ensure!(owner.device() == stream.device.id, "Step-5 graph replay device changed");
                stream.device.run(|| {
                    // SAFETY: containing lane retains immutable bindings and the exact stream.
                    unsafe { stream.device.library.cuda_graph_launch(owner.raw, stream.raw) }
                })?;
                Ok(true)
            }
            (GraphDecision::Eager, _) => Ok(false),
            _ => anyhow::bail!("Step-5 live submission attempted graph capture"),
        }
    }
}

pub(super) fn check_captures(startup: u64, current: u64) -> Result<()> {
    ensure!(startup == current, "Step-5 zero-steady-capture contract: startup={startup}, current={current}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn steady_capture_count_cannot_grow() {
        check_captures(19, 19).unwrap();
        assert!(check_captures(19, 20).is_err());
    }
}
