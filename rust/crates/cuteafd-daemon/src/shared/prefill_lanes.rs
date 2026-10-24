//! The Spark links of pipelined prefill lanes (GLM 5.3 Flash, MiniMax M3). Lane planning and
//! sizing are `cuteafd_loader::serving_capacity::prefill_lanes`.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::serving_capacity::prefill_lanes::MAX_PREFILL_LANES;

/// One Spark link per prefill lane: a lane's waves stay in flight on its own connections while
/// the other lanes' GPU layers run. Lane 0 also carries decode, verify and single-lane chunks.
pub(crate) struct LaneLinks<L>(Vec<L>);

impl<L> LaneLinks<L> {
    /// `lanes` links (1 to [`MAX_PREFILL_LANES`]), `connect(lane)` each.
    pub(crate) fn connect(lanes: usize, connect: impl FnMut(usize) -> Result<L>) -> Result<Self> {
        ensure!((1..=MAX_PREFILL_LANES).contains(&lanes), "{lanes} prefill lanes (1 to {MAX_PREFILL_LANES})");
        Ok(Self((0..lanes).map(connect).collect::<Result<_>>()?))
    }

    /// Lane `lane`'s link.
    pub(crate) fn lane(&self, lane: usize) -> Result<&L> {
        self.0.get(lane).with_context(|| format!("prefill lane {lane} of {} has no Spark link", self.0.len()))
    }
}

impl<L> std::ops::Deref for LaneLinks<L> {
    type Target = [L];
    fn deref(&self) -> &[L] { &self.0 }
}

impl<L> std::ops::DerefMut for LaneLinks<L> {
    fn deref_mut(&mut self) -> &mut [L] { &mut self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_link_per_lane_in_lane_order() -> Result<()> {
        let links = LaneLinks::connect(2, |lane| Ok(lane * 10))?;
        assert_eq!((links.len(), *links.lane(1)?), (2, 10));
        assert!(links.lane(2).is_err());
        assert!(LaneLinks::connect(0, Ok).is_err());
        assert!(LaneLinks::connect(MAX_PREFILL_LANES + 1, Ok).is_err());
        assert!(LaneLinks::<usize>::connect(2, |lane| if lane == 1 { anyhow::bail!("rank down") } else { Ok(lane) }).is_err());
        Ok(())
    }
}
