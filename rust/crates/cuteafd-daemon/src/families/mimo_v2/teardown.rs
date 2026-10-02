//! Preserve the forward failure while attempting every owned cleanup phase.
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CleanupStep {
    LeadStream,
    PeerStream,
    RestoreDevice,
}

/// Cleanup errors must not replace the error that made the engine stop.
/// This does not cancel queued peer waits; callers still own that drain.
pub(super) fn finish<T>(body: Result<T>, has_peer: bool,
    mut cleanup: impl FnMut(CleanupStep) -> Result<()>) -> Result<T> {
    let mut first_failure = None;
    for step in [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice] {
        if step == CleanupStep::PeerStream && !has_peer {
            continue;
        }
        if let Err(error) = cleanup(step) {
            tracing::error!(?step, error = %format!("{error:#}"), "MiMo engine cleanup failed");
            if first_failure.is_none() {
                first_failure = Some(error.context(format!("MiMo engine cleanup {step:?}")));
            }
        }
    }
    match body {
        Err(primary) => Err(primary),
        Ok(value) => first_failure.map_or(Ok(value), Err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("forward capacity failure")]
    struct ForwardFailure;

    #[test]
    fn forward_error_survives_every_cleanup_failure() {
        let mut seen = Vec::new();
        let result: Result<()> = finish(Err(ForwardFailure.into()), true, |step| {
            seen.push(step);
            anyhow::bail!("cleanup {step:?} failed")
        });
        assert!(result.unwrap_err().downcast_ref::<ForwardFailure>().is_some());
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice]);
    }

    #[test]
    fn lead_destroy_failure_still_attempts_peer_and_restores_device() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), true, |step| {
            seen.push(step);
            if step == CleanupStep::LeadStream { anyhow::bail!("lead destroy failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("lead destroy failed"));
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice]);
    }

    #[test]
    fn peer_destroy_failure_still_restores_device() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), true, |step| {
            seen.push(step);
            if step == CleanupStep::PeerStream { anyhow::bail!("peer destroy failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("peer destroy failed"));
        assert_eq!(seen.last(), Some(&CleanupStep::RestoreDevice));
    }

    #[test]
    fn successful_body_reports_failed_device_restore() {
        let result = finish(Ok(17), false, |step| {
            if step == CleanupStep::RestoreDevice { anyhow::bail!("device restore failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("device restore failed"));
    }

    #[test]
    fn successful_single_gpu_cleanup_preserves_body_value() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), false, |step| { seen.push(step); Ok(()) });
        assert_eq!(result.unwrap(), 17);
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::RestoreDevice]);
    }
}
