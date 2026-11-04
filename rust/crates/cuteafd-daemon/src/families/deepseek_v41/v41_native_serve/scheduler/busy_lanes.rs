//! How two busy V4.1 decode lanes order their passes
//! (`CUTEAFD_V41_BUSY_LANES`). A lone busy lane runs device-ordered and
//! unpaced in every mode.
//!
//! - `host` (default): a lane whose peer is busy runs host-ordered passes;
//!   the host waits between stages interleave the two lanes layer by layer.
//! - `paced`: both lanes stay device-ordered, and each pass is paced
//!   (`shared::memory::chain::paced`): a Spark landing and what follows it
//!   are queued only once the reply landed, and the end-of-pass drain yields,
//!   which keeps that interleave without the host waits for the lane's own GPUs.
//! - `device`: both lanes device-ordered and unpaced (the October 2026 shape,
//!   for diagnosis).
use anyhow::Result;

pub(crate) const ENV: &str = "CUTEAFD_V41_BUSY_LANES";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BusyLanes {
    Host,
    Paced,
    Device,
}

/// One round's pass order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Order {
    pub device: bool,
    pub paced: bool,
}

impl BusyLanes {
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw.map(str::trim) {
            None | Some("") | Some("host") => Self::Host,
            Some("paced") => Self::Paced,
            Some("device") => Self::Device,
            Some(other) => anyhow::bail!("{ENV}={other}: expected host, paced or device"),
        })
    }
    /// The order of a round whose other lane is busy (`shared`) or idle.
    pub(crate) fn order(self, shared: bool) -> Order {
        match (self, shared) {
            (_, false) => Order { device: true, paced: false },
            (Self::Host, true) => Order { device: false, paced: false },
            (Self::Paced, true) => Order { device: true, paced: true },
            (Self::Device, true) => Order { device: true, paced: false },
        }
    }
}

/// The process's mode, read once.
pub(crate) fn mode() -> Result<BusyLanes> {
    static MODE: std::sync::OnceLock<std::result::Result<BusyLanes, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| BusyLanes::parse(std::env::var(ENV).ok().as_deref()).map_err(|error| format!("{error:#}")))
        .clone().map_err(|error| anyhow::anyhow!(error))
}

#[cfg(test)]
mod tests {
    use super::{BusyLanes, Order};

    #[test]
    fn busy_lane_modes_parse_and_order_rounds() {
        assert_eq!(BusyLanes::parse(None).unwrap(), BusyLanes::Host);
        assert_eq!(BusyLanes::parse(Some(" ")).unwrap(), BusyLanes::Host);
        assert_eq!(BusyLanes::parse(Some("paced")).unwrap(), BusyLanes::Paced);
        assert_eq!(BusyLanes::parse(Some("device")).unwrap(), BusyLanes::Device);
        assert!(BusyLanes::parse(Some("Paced")).is_err());
        let (solo, device, host, paced) = (Order { device: true, paced: false }, Order { device: true, paced: false },
            Order { device: false, paced: false }, Order { device: true, paced: true });
        for mode in [BusyLanes::Host, BusyLanes::Paced, BusyLanes::Device] { assert_eq!(mode.order(false), solo); }
        assert_eq!(BusyLanes::Host.order(true), host);
        assert_eq!(BusyLanes::Paced.order(true), paced);
        assert_eq!(BusyLanes::Device.order(true), device);
    }

    /// Source pins for device-ordered busy lanes: the per-layer staging fence
    /// is marked only at the FFN entry, after the attention tail joined the
    /// chain (the October 2026 fix; `shared::memory::staging_check` tests the
    /// mechanism on a GPU), and a paced pass collects its Spark reply as host
    /// order does but skips host order's post-reduce stream wait.
    #[test]
    fn paced_passes_keep_the_fence_and_collection_rules() {
        let lane = include_str!("../../v41_backbone_lane.rs");
        assert_eq!(lane.matches("fence_mark(").count(), 1);
        assert!(lane.contains("chain::fence_mark(library, input.layer % 2)"));
        for source in [include_str!("../../v41_target_pass.rs"), include_str!("../../v41_target_pass/distributed.rs")] {
            assert!(!source.contains("fence_mark("));
        }
        let coordinator = include_str!("../../v41_experts/coordinator.rs");
        assert!(coordinator.contains("if host_ordered || crate::shared::memory::chain::paced() {"));
        assert!(coordinator.contains("if host_ordered && crate::shared::memory::chain::active() {"));
    }
}
