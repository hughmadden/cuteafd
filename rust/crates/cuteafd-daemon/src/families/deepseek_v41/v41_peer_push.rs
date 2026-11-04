//! How the dual-RTX decode path moves rows between the two GPUs.
//!
//! `CUTEAFD_V41_PEER_EXCHANGE=push` (default): inside a device-ordered pass
//! every decode transfer is pushed by the GPU that produced it, on the
//! producer's stream, and the consumer's stream waits on a device flag
//! (`shared::peer_split::PeerExchange`). `pull` keeps the earlier path (SM
//! pulls or copy-engine copies behind cross-device events) for comparison.
//! Only decode lanes push; prefill slots and host-ordered passes keep their
//! copy-engine path either way.
//!
//! Size rule: a push runs on the SMs up to a decode lane's rows of the
//! site's row bytes, and by the sender's copy engine plus a zero-byte flag
//! above that; `CUTEAFD_V41_PUSH_MAX_BYTES` optionally caps the SM size.
//!
//! Reuse: the push sites need no parity slots or acknowledgements because
//! each layer's FFN output gates the next layer's input, so every next write
//! of a reused destination follows its last read through the flags. The
//! regular test gate checks that invariant on a model of one decode lane
//! (`schedule_tests`, with `shared::peer_split::order::happens`); pipelining
//! layers (a GPU running a layer ahead) fails it until such sites get parity
//! slots.
use anyhow::{ensure, Context, Result};

/// The switch naming the transfer path.
pub(crate) const MODE_ENV: &str = "CUTEAFD_V41_PEER_EXCHANGE";
/// The optional SM push ceiling in bytes.
pub(crate) const MAX_BYTES_ENV: &str = "CUTEAFD_V41_PUSH_MAX_BYTES";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Push,
    Pull,
}

impl Mode {
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw.map(str::trim) {
            None | Some("") | Some("push") => Self::Push,
            Some("pull") => Self::Pull,
            Some(other) => anyhow::bail!("{MODE_ENV}={other}: expected push or pull"),
        })
    }
}

/// The process's transfer path, read once.
pub(crate) fn mode() -> Result<Mode> {
    static MODE: std::sync::OnceLock<std::result::Result<Mode, String>> = std::sync::OnceLock::new();
    MODE.get_or_init(|| Mode::parse(std::env::var(MODE_ENV).ok().as_deref()).map_err(|error| format!("{error:#}")))
        .clone().map_err(|error| anyhow::anyhow!(error))
}

pub(crate) fn parse_max_bytes(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else { return Ok(None) };
    let bytes: usize = raw.parse().with_context(|| format!("{MAX_BYTES_ENV}={raw}: expected bytes"))?;
    ensure!(bytes >= 16, "{MAX_BYTES_ENV}={raw}: at least 16 bytes");
    Ok(Some(bytes))
}

fn max_bytes() -> Result<Option<usize>> {
    static MAX: std::sync::OnceLock<std::result::Result<Option<usize>, String>> = std::sync::OnceLock::new();
    MAX.get_or_init(|| parse_max_bytes(std::env::var(MAX_BYTES_ENV).ok().as_deref()).map_err(|error| format!("{error:#}")))
        .clone().map_err(|error| anyhow::anyhow!(error))
}

/// A decode lane's rows (the verify width): the row count of the size rule
/// for sites that do not know their lane's.
pub(crate) const DECODE_ROWS: usize = cuteafd_core::V41_VERIFY_ROWS;

/// The SM push bound of a site moving `rows` (a decode lane) of `row_bytes`:
/// their product, capped by `cap` and kept a nonzero 16-byte multiple.
pub(crate) fn lane_bytes_with(rows: usize, row_bytes: usize, cap: Option<usize>) -> Result<usize> {
    let bytes = rows.checked_mul(row_bytes).context("decode lane bytes overflow")?;
    let bytes = cap.map_or(bytes, |cap| bytes.min(cap));
    let bytes = bytes / 16 * 16;
    ensure!(bytes >= 16, "decode lane bound of {rows} rows x {row_bytes} B is under 16 bytes");
    Ok(bytes)
}

/// [`lane_bytes_with`] under the process's `CUTEAFD_V41_PUSH_MAX_BYTES`.
pub(crate) fn lane_bytes(rows: usize, row_bytes: usize) -> Result<usize> {
    lane_bytes_with(rows, row_bytes, max_bytes()?)
}

#[cfg(test)]
mod schedule_tests;
#[cfg(test)]
mod cuda_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_defaults_to_push_and_rejects_unknown_values() {
        assert_eq!(Mode::parse(None).unwrap(), Mode::Push);
        assert_eq!(Mode::parse(Some("")).unwrap(), Mode::Push);
        assert_eq!(Mode::parse(Some(" push ")).unwrap(), Mode::Push);
        assert_eq!(Mode::parse(Some("pull")).unwrap(), Mode::Pull);
        let error = Mode::parse(Some("sm")).unwrap_err().to_string();
        assert!(error.contains(MODE_ENV) && error.contains("push or pull"), "{error}");
    }

    #[test]
    fn lane_bytes_follow_rows_and_the_optional_cap() {
        // A decode lane of each V4.1 site's rows: hop, hidden, routed planes.
        assert_eq!(lane_bytes_with(64, 40_976, None).unwrap(), 2_622_464);
        assert_eq!(lane_bytes_with(64, 10_240, None).unwrap(), 655_360);
        assert_eq!(lane_bytes_with(64, 5120 * 6 * 4, None).unwrap(), 7_864_320);
        // The cap lowers the SM bound and keeps it a 16-byte multiple.
        assert_eq!(lane_bytes_with(64, 40_976, Some(1 << 20)).unwrap(), 1 << 20);
        assert_eq!(lane_bytes_with(64, 40_976, Some((1 << 20) + 7)).unwrap(), 1 << 20);
        assert_eq!(lane_bytes_with(64, 10_240, Some(1 << 30)).unwrap(), 655_360);
        assert!(lane_bytes_with(1, 8, None).is_err());
        assert!(lane_bytes_with(usize::MAX, 2, None).is_err());
        assert_eq!(parse_max_bytes(None).unwrap(), None);
        assert_eq!(parse_max_bytes(Some("1048576")).unwrap(), Some(1 << 20));
        assert!(parse_max_bytes(Some("15")).is_err());
        assert!(parse_max_bytes(Some("lots")).is_err());
    }
}
